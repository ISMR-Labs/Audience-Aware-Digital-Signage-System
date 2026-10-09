#!/usr/bin/env python3
"""
Lightweight people + attention counter for Raspberry Pi.

Pipeline per frame:
  1. (optional) MobileNet-SSD (TFLite, quantised) -> "person" boxes  -> total people
  2. YuNet (OpenCV ONNX)                          -> faces + 5 landmarks
  3. solvePnP on the 5 landmarks                  -> head pose
  4. angle between the head's forward vector and the vector to the camera
     -> "looking at camera" if below --attention-angle

Only OpenCV + numpy are required. The person detector is optional and needs a
TFLite runtime (tflite-runtime, ai-edge-litert or tensorflow).
"""

import argparse
import collections
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")
FACE_MODEL = os.path.join(MODEL_DIR, "face_detection_yunet_2023mar.onnx")
PERSON_MODEL = os.path.join(MODEL_DIR, "detect.tflite")

# Generic 3-D face model, OpenCV camera convention (x right, y down, z away
# from the camera), nose tip at the origin.  Units are arbitrary (approx. mm/4).
# Order: image-left eye, image-right eye, nose, image-left mouth, image-right mouth.
FACE_3D = np.array(
    [
        [-150.0, -170.0, 170.0],
        [150.0, -170.0, 170.0],
        [0.0, 0.0, 0.0],
        [-100.0, 150.0, 90.0],
        [100.0, 150.0, 90.0],
    ],
    dtype=np.float64,
)


# --------------------------------------------------------------------------- #
# Detectors
# --------------------------------------------------------------------------- #
class FaceDetector:
    """YuNet face detector. Returns boxes, 5 landmarks and scores in frame coords."""

    def __init__(self, model_path, score_thr=0.7, nms_thr=0.3, top_k=50):
        if not os.path.exists(model_path):
            sys.exit(f"Face model not found: {model_path}\nRun: python3 download_models.py")
        self.det = cv2.FaceDetectorYN.create(model_path, "", (320, 320), score_thr, nms_thr, top_k)
        self.size = None

    def __call__(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        if self.size != (w, h):
            self.det.setInputSize((w, h))
            self.size = (w, h)
        _, faces = self.det.detect(frame_bgr)
        if faces is None:
            return np.zeros((0, 15), dtype=np.float32)
        return faces  # x, y, w, h, 5x(lx, ly), score


class PersonDetector:
    """COCO SSD-MobileNet v1 (quantised TFLite). Returns [x1, y1, x2, y2, score] for persons."""

    def __init__(self, model_path, score_thr=0.5, threads=4):
        if not os.path.exists(model_path):
            sys.exit(f"Person model not found: {model_path}\nRun: python3 download_models.py")
        Interpreter = _load_tflite_interpreter()
        self.interp = Interpreter(model_path=model_path, num_threads=threads)
        self.interp.allocate_tensors()
        inp = self.interp.get_input_details()[0]
        self.in_index = inp["index"]
        self.in_h, self.in_w = int(inp["shape"][1]), int(inp["shape"][2])
        outs = self.interp.get_output_details()
        # Outputs are boxes, classes, scores, count (sort by name to be safe).
        outs = sorted(outs, key=lambda o: o["name"])
        self.out_boxes, self.out_classes, self.out_scores = (o["index"] for o in outs[:3])
        self.score_thr = score_thr

    def __call__(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(cv2.resize(frame_bgr, (self.in_w, self.in_h)), cv2.COLOR_BGR2RGB)
        self.interp.set_tensor(self.in_index, rgb[np.newaxis])
        self.interp.invoke()
        boxes = self.interp.get_tensor(self.out_boxes)[0]
        classes = self.interp.get_tensor(self.out_classes)[0]
        scores = self.interp.get_tensor(self.out_scores)[0]
        out = []
        for (y1, x1, y2, x2), cls, sc in zip(boxes, classes, scores):
            if int(cls) != 0 or sc < self.score_thr:  # class 0 == person
                continue
            out.append([x1 * w, y1 * h, x2 * w, y2 * h, float(sc)])
        return np.array(out, dtype=np.float32).reshape(-1, 5)


def _load_tflite_interpreter():
    try:
        from tflite_runtime.interpreter import Interpreter  # Raspberry Pi (pip install tflite-runtime)
        return Interpreter
    except ImportError:
        pass
    try:
        from ai_edge_litert.interpreter import Interpreter  # newer LiteRT wheels
        return Interpreter
    except ImportError:
        pass
    try:
        from tensorflow.lite import Interpreter  # full TF fallback
        return Interpreter
    except ImportError:
        sys.exit(
            "No TFLite runtime found. Install one of:\n"
            "  pip install tflite-runtime     (Raspberry Pi)\n"
            "  pip install ai-edge-litert\n"
            "or run with --faces-only to skip the person detector."
        )


# --------------------------------------------------------------------------- #
# Head pose / attention
# --------------------------------------------------------------------------- #
class HeadPose:
    def __init__(self, frame_w, frame_h, hfov_deg=60.0):
        # focal length (px) from the camera's horizontal field of view
        f = (frame_w / 2.0) / np.tan(np.radians(hfov_deg) / 2.0)
        self.K = np.array([[f, 0, frame_w / 2.0], [0, f, frame_h / 2.0], [0, 0, 1]], dtype=np.float64)
        self.dist = np.zeros((4, 1))
        self.pnp_flag = getattr(cv2, "SOLVEPNP_SQPNP", cv2.SOLVEPNP_EPNP)

    def estimate(self, lm):
        """lm: 5x2 YuNet landmarks (eye, eye, nose, mouth, mouth) in any L/R order.

        Returns (attention_deg, yaw, pitch, roll) or None if PnP fails.
        attention_deg is the angle between where the head points and the
        direction from the head to the camera: 0 == looking straight at it.
        """
        eyes = lm[:2][np.argsort(lm[:2, 0])]
        mouth = lm[3:5][np.argsort(lm[3:5, 0])]
        pts2d = np.vstack([eyes, lm[2:3], mouth]).astype(np.float64)
        # ITERATIVE needs >= 6 points; SQPNP / EPNP work with 5. Refine with LM afterwards.
        try:
            ok, rvec, tvec = cv2.solvePnP(FACE_3D, pts2d, self.K, self.dist, flags=self.pnp_flag)
            if not ok:
                return None
            rvec, tvec = cv2.solvePnPRefineLM(FACE_3D, pts2d, self.K, self.dist, rvec, tvec)
        except cv2.error:
            return None
        R, _ = cv2.Rodrigues(rvec)
        forward = R @ np.array([0.0, 0.0, -1.0])  # face points toward -z in its own frame
        to_cam = -tvec.ravel()
        n = np.linalg.norm(to_cam)
        if n < 1e-6:
            return None
        cos_a = float(np.clip(np.dot(forward, to_cam / n), -1.0, 1.0))
        attention = float(np.degrees(np.arccos(cos_a)))
        # Euler angles for display: x=pitch, y=yaw, z=roll
        angles, *_ = cv2.RQDecomp3x3(R)
        pitch, yaw, roll = (float(a) for a in angles)
        return attention, yaw, pitch, roll


# --------------------------------------------------------------------------- #
# Video sources
# --------------------------------------------------------------------------- #
class CameraSource:
    def __init__(self, source, width, height):
        self.picam = None
        self.cap = None
        if source == "picam":
            from picamera2 import Picamera2  # Raspberry Pi CSI camera

            self.picam = Picamera2()
            cfg = self.picam.create_video_configuration(main={"size": (width, height), "format": "RGB888"})
            self.picam.configure(cfg)
            self.picam.start()
            time.sleep(0.5)
        else:
            src = int(source) if source.isdigit() else source
            self.cap = cv2.VideoCapture(src)
            if not self.cap.isOpened():
                sys.exit(f"Could not open video source: {source}")
            if isinstance(src, int):
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.is_file = isinstance(src, str) and os.path.exists(src)

    def read(self):
        if self.picam is not None:
            frame = self.picam.capture_array()
            # Picamera2 "RGB888" is BGR-ordered in memory, which is what OpenCV wants.
            return True, frame
        return self.cap.read()

    def release(self):
        if self.picam is not None:
            self.picam.stop()
        if self.cap is not None:
            self.cap.release()


# --------------------------------------------------------------------------- #
# Optional MJPEG + JSON web server (handy on a headless Pi)
# --------------------------------------------------------------------------- #
class WebState:
    def __init__(self):
        self.lock = threading.Lock()
        self.jpeg = None
        self.stats = {}


def start_web_server(port, state):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # silence
            pass

        def do_GET(self):
            if self.path.startswith("/stats"):
                with state.lock:
                    body = json.dumps(state.stats).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path.startswith("/stream"):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while True:
                        with state.lock:
                            jpeg = state.jpeg
                        if jpeg is not None:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
                        time.sleep(0.05)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                body = (
                    "<html><body style='background:#111;color:#eee;font-family:sans-serif'>"
                    "<h3 id=s>...</h3><img src='/stream' style='max-width:100%'>"
                    "<script>setInterval(async()=>{const r=await fetch('/stats');const j=await r.json();"
                    "document.getElementById('s').textContent="
                    "`people: ${j.people}  looking: ${j.looking}  fps: ${j.fps}`},500)</script>"
                    "</body></html>"
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def draw(frame, persons, faces, poses, looking_mask, stats):
    for x1, y1, x2, y2, sc in persons:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (200, 200, 0), 1)
    for face, pose, looking in zip(faces, poses, looking_mask):
        x, y, w, h = (int(v) for v in face[:4])
        color = (0, 220, 0) if looking else (0, 0, 230)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        for i in range(5):
            cv2.circle(frame, (int(face[4 + 2 * i]), int(face[5 + 2 * i])), 2, color, -1)
        label = f"{pose[0]:.0f}deg" if pose is not None else "?"
        cv2.putText(frame, label, (x, max(12, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    txt = f"people {stats['people']}  looking {stats['looking']}  {stats['fps']:.1f} fps"
    cv2.putText(frame, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
    cv2.putText(frame, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)


def write_json_atomic(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0", help="camera index, video file, RTSP/HTTP URL, or 'picam' (default 0)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--detect-width", type=int, default=640,
                    help="downscale frames to this width before detection; 320 is ~3x faster but needs "
                         "people closer to the camera; 0 = no downscale (default 640)")
    ap.add_argument("--faces-only", action="store_true",
                    help="skip the person detector; people count = face count (lightest mode)")
    ap.add_argument("--person-every", type=int, default=3,
                    help="run the person detector only every Nth frame (default 3)")
    ap.add_argument("--hfov", type=float, default=60.0,
                    help="camera horizontal field of view in degrees (Pi cam v2: 62, v3: 66, v3 wide: 102, "
                         "typical USB webcam: 55-70). Used for head-pose geometry. (default 60)")
    ap.add_argument("--min-face", type=int, default=20,
                    help="faces narrower than this (px, at detection size) are counted but their pose is not "
                         "evaluated, as 5-point pose is unreliable on tiny faces (default 20)")
    ap.add_argument("--attention-angle", type=float, default=25.0,
                    help="max angle (deg) between head direction and camera to count as 'looking' (default 25)")
    ap.add_argument("--face-score", type=float, default=0.7)
    ap.add_argument("--person-score", type=float, default=0.5)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--smooth", type=float, default=1.0,
                    help="seconds of history used to smooth the reported counts (median); 0 = raw")
    ap.add_argument("--log-interval", type=float, default=1.0, help="seconds between console log lines")
    ap.add_argument("--json-out", help="write latest stats to this JSON file (atomic)")
    ap.add_argument("--web", type=int, metavar="PORT", help="serve MJPEG preview + /stats JSON on this port")
    ap.add_argument("--show", action="store_true", help="open an OpenCV window (needs a desktop)")
    ap.add_argument("--save-frame", help="save the last annotated frame to this path on exit (or each frame for images)")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N frames (testing)")
    args = ap.parse_args()

    cv2.setNumThreads(args.threads)

    face_det = FaceDetector(FACE_MODEL, score_thr=args.face_score)
    person_det = None if args.faces_only else PersonDetector(PERSON_MODEL, args.person_score, args.threads)

    cam = CameraSource(args.source, args.width, args.height)
    state = WebState() if args.web else None
    if args.web:
        start_web_server(args.web, state)
        print(f"web preview: http://0.0.0.0:{args.web}/   stats: /stats", flush=True)

    pose_est = None
    hist_people = collections.deque()
    hist_looking = collections.deque()
    last_persons = np.zeros((0, 5), dtype=np.float32)
    fps_t, fps_n, fps = time.time(), 0, 0.0
    last_log = 0.0
    frame_i = 0
    frame = None

    try:
        while True:
            ok, frame = cam.read()
            if not ok or frame is None:
                if frame_i == 0:
                    sys.exit("No frames from source")
                break
            frame_i += 1
            fh, fw = frame.shape[:2]

            # Downscale for detection
            if args.detect_width and fw > args.detect_width:
                scale = fw / args.detect_width
                small = cv2.resize(frame, (args.detect_width, int(fh / scale)))
            else:
                scale, small = 1.0, frame
            if pose_est is None:
                pose_est = HeadPose(small.shape[1], small.shape[0], args.hfov)

            # Faces + pose
            faces = face_det(small)
            poses, looking = [], []
            for f in faces:
                if f[2] < args.min_face:
                    poses.append(None)
                    looking.append(False)
                    continue
                lm = f[4:14].reshape(5, 2)
                p = pose_est.estimate(lm)
                poses.append(p)
                looking.append(p is not None and p[0] <= args.attention_angle)
            faces_full = faces.copy()
            faces_full[:, :14] *= scale

            # Persons (every Nth frame)
            if person_det is not None and (frame_i % args.person_every == 1 or args.person_every == 1):
                last_persons = person_det(small)
                last_persons[:, :4] *= scale
            n_persons = len(last_persons) if person_det is not None else 0

            n_people_raw = max(n_persons, len(faces))
            n_looking_raw = int(sum(looking))

            # Smoothing
            now = time.time()
            if args.smooth > 0:
                hist_people.append((now, n_people_raw))
                hist_looking.append((now, n_looking_raw))
                for hq in (hist_people, hist_looking):
                    while hq and now - hq[0][0] > args.smooth:
                        hq.popleft()
                n_people = int(np.median([v for _, v in hist_people]))
                n_looking = int(np.median([v for _, v in hist_looking]))
            else:
                n_people, n_looking = n_people_raw, n_looking_raw

            # FPS
            fps_n += 1
            if now - fps_t >= 1.0:
                fps = fps_n / (now - fps_t)
                fps_t, fps_n = now, 0

            stats = {
                "people": n_people,
                "looking": n_looking,
                "people_raw": n_people_raw,
                "looking_raw": n_looking_raw,
                "faces": int(len(faces)),
                "persons": int(n_persons),
                "fps": round(fps, 1),
                "ts": round(now, 2),
            }

            need_draw = args.show or args.web or args.save_frame
            if need_draw:
                draw(frame, last_persons if person_det is not None else [], faces_full, poses, looking, stats)

            if now - last_log >= args.log_interval:
                last_log = now
                print(f"people={n_people} looking={n_looking} faces={len(faces)} fps={fps:.1f}", flush=True)
                if args.json_out:
                    write_json_atomic(args.json_out, stats)

            if state is not None:
                _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                with state.lock:
                    state.jpeg = jpg.tobytes()
                    state.stats = stats
            if args.show:
                cv2.imshow("people counter", frame)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
            if args.max_frames and frame_i >= args.max_frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        if args.save_frame and frame is not None:
            cv2.imwrite(args.save_frame, frame)
            print(f"saved {args.save_frame}")
        cam.release()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
