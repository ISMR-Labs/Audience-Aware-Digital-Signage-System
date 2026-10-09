"""Camera thread: runs the detectors from people_counter.py continuously,
tracks people, and accumulates per-impression audience metrics."""

import threading
import time

import cv2
import numpy as np

import people_counter as pc
from tracker import Tracker


class Audience(threading.Thread):
    def __init__(self, source="0", width=640, height=480, detect_width=640, faces_only=False,
                 person_every=3, threads=4, hfov=60.0, min_face=20, settings_getter=None):
        super().__init__(daemon=True, name="audience")
        self.source, self.width, self.height = source, width, height
        self.detect_width, self.faces_only = detect_width, faces_only
        self.person_every, self.threads, self.hfov, self.min_face = person_every, threads, hfov, min_face
        self.get_setting = settings_getter or (lambda k, c=float: None)
        self.lock = threading.Lock()
        self.jpeg = None
        self.live = {"people": 0, "looking": 0, "fps": 0.0, "ok": False, "error": None}
        self.tracker = Tracker()
        self.imp = None  # current impression accumulator
        self.stop_event = threading.Event()

    # ---- impression lifecycle (called by the scheduler) -------------------
    def begin_impression(self):
        with self.lock:
            self.imp = {"tracks": {}, "frames": 0, "sum_people": 0, "max_people": 0, "start": time.time()}

    def end_impression(self):
        with self.lock:
            imp, self.imp = self.imp, None
        if imp is None:
            return {"viewers": 0, "lookers": 0, "avg_people": 0.0, "max_people": 0, "frames": 0}
        min_present = self.get_setting("min_present_s", float) or 0.5
        min_look = self.get_setting("min_look_s", float) or 0.7
        viewers = sum(1 for t in imp["tracks"].values() if t["present"] >= min_present)
        lookers = sum(1 for t in imp["tracks"].values() if t["present"] >= min_present and t["look"] >= min_look)
        frames = imp["frames"]
        return {"viewers": viewers, "lookers": lookers,
                "avg_people": round(imp["sum_people"] / frames, 2) if frames else 0.0,
                "max_people": imp["max_people"], "frames": frames}

    def snapshot_impression(self):
        """Live view of the running impression (for the dashboard)."""
        with self.lock:
            imp = self.imp
            if imp is None:
                return None
            min_present = self.get_setting("min_present_s", float) or 0.5
            min_look = self.get_setting("min_look_s", float) or 0.7
            return {
                "viewers": sum(1 for t in imp["tracks"].values() if t["present"] >= min_present),
                "lookers": sum(1 for t in imp["tracks"].values()
                               if t["present"] >= min_present and t["look"] >= min_look),
                "max_people": imp["max_people"],
            }

    # ---- main loop ---------------------------------------------------------
    def run(self):
        try:
            self._run()
        except Exception as e:  # keep the web server alive even if the camera dies
            with self.lock:
                self.live["ok"] = False
                self.live["error"] = f"{type(e).__name__}: {e}"
            print(f"[audience] stopped: {e}", flush=True)

    def _run(self):
        cv2.setNumThreads(self.threads)
        face_det = pc.FaceDetector(pc.FACE_MODEL)
        person_det = None if self.faces_only else pc.PersonDetector(pc.PERSON_MODEL, threads=self.threads)
        cam = pc.CameraSource(self.source, self.width, self.height)
        pose_est = None
        last_persons = np.zeros((0, 5), dtype=np.float32)
        frame_i, fps_t, fps_n, fps = 0, time.time(), 0, 0.0
        last_t = time.time()
        with self.lock:
            self.live["ok"] = True

        while not self.stop_event.is_set():
            ok, frame = cam.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue
            frame_i += 1
            now = time.time()
            dt = min(now - last_t, 0.5)
            last_t = now
            fh, fw = frame.shape[:2]

            if self.detect_width and fw > self.detect_width:
                scale = fw / self.detect_width
                small = cv2.resize(frame, (self.detect_width, int(fh / scale)))
            else:
                scale, small = 1.0, frame
            if pose_est is None:
                pose_est = pc.HeadPose(small.shape[1], small.shape[0], self.hfov)
            att_thr = self.get_setting("attention_angle", float) or 25.0

            faces = face_det(small)
            looking, poses = [], []
            for f in faces:
                if f[2] < self.min_face:
                    poses.append(None)
                    looking.append(False)
                    continue
                p = pose_est.estimate(f[4:14].reshape(5, 2))
                poses.append(p)
                looking.append(p is not None and p[0] <= att_thr)
            faces_full = faces.copy()
            faces_full[:, :14] *= scale

            if person_det is not None and (frame_i % self.person_every == 1 or self.person_every == 1):
                last_persons = person_det(small)
                last_persons[:, :4] *= scale
            persons = last_persons if person_det is not None else np.zeros((0, 5), dtype=np.float32)

            dets = self._entities(persons, faces_full, looking, fw, fh)
            visible = self.tracker.update(dets, now)
            n_people = len(visible)
            n_looking = sum(1 for t in visible if t["looking"])

            fps_n += 1
            if now - fps_t >= 1.0:
                fps = fps_n / (now - fps_t)
                fps_t, fps_n = now, 0

            # draw
            for t in visible:
                x1, y1, x2, y2 = (int(v) for v in t["box"])
                color = (0, 220, 0) if t["looking"] else (0, 0, 230)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"#{t['id']}", (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            for face, pose, lk in zip(faces_full, poses, looking):
                x, y, w, h = (int(v) for v in face[:4])
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 220, 0) if lk else (0, 0, 230), 1)
            txt = f"people {n_people}  looking {n_looking}  {fps:.1f} fps"
            cv2.putText(frame, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
            cv2.putText(frame, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
            _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])

            with self.lock:
                self.jpeg = jpg.tobytes()
                self.live.update({"people": n_people, "looking": n_looking, "fps": round(fps, 1)})
                if self.imp is not None:
                    imp = self.imp
                    imp["frames"] += 1
                    imp["sum_people"] += n_people
                    imp["max_people"] = max(imp["max_people"], n_people)
                    for t in visible:
                        rec = imp["tracks"].setdefault(t["id"], {"present": 0.0, "look": 0.0})
                        rec["present"] += dt
                        if t["looking"]:
                            rec["look"] += dt
        cam.release()

    @staticmethod
    def _entities(persons, faces, looking, fw, fh):
        """Merge person boxes and faces into one detection per person: [x1,y1,x2,y2], looking."""
        ents = [[float(p[0]), float(p[1]), float(p[2]), float(p[3]), False] for p in persons]
        for face, lk in zip(faces, looking):
            x, y, w, h = (float(v) for v in face[:4])
            cx, cy = x + w / 2, y + h / 2
            host = None
            for e in ents:
                if e[0] <= cx <= e[2] and e[1] <= cy <= e[3]:
                    if host is None or (e[2] - e[0]) * (e[3] - e[1]) < (host[2] - host[0]) * (host[3] - host[1]):
                        host = e
            if host is not None:
                host[4] = host[4] or lk
            else:  # face without a person box: synthesise a body box
                ents.append([max(0, x - w), max(0, y - h * 0.5), min(fw, x + 2 * w), min(fh, y + 4.5 * h), lk])
        # suppress duplicates: if two boxes mostly overlap (relative to the smaller one), keep the larger
        ents.sort(key=lambda e: -(e[2] - e[0]) * (e[3] - e[1]))
        kept = []
        for e in ents:
            dup = None
            for k in kept:
                ix = max(0.0, min(e[2], k[2]) - max(e[0], k[0]))
                iy = max(0.0, min(e[3], k[3]) - max(e[1], k[1]))
                small = max(1.0, min((e[2] - e[0]) * (e[3] - e[1]), (k[2] - k[0]) * (k[3] - k[1])))
                if ix * iy / small > 0.6:
                    dup = k
                    break
            if dup is None:
                kept.append(e)
            else:
                dup[4] = dup[4] or e[4]
        return [(e[:4], e[4]) for e in kept]
