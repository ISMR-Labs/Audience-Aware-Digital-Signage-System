#!/usr/bin/env python3
"""pi-billboard: audience-aware digital signage for Raspberry Pi.

  python server.py --source picam --port 8090
  dashboard: http://<pi>:8090/      player (kiosk): http://<pi>:8090/player
"""

import argparse
import csv
import io
import os
import re
import time

import cv2
from flask import Flask, Response, abort, jsonify, request, send_from_directory

from audience import Audience
from db import DB
from scheduler import Scheduler

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "static")
IMAGE_EXT = {"jpg", "jpeg", "png", "gif", "webp"}
VIDEO_EXT = {"mp4", "webm", "mov", "mkv", "m4v"}

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024
db: DB = None
aud: Audience = None
sched: Scheduler = None
DATA_DIR = MEDIA_DIR = THUMB_DIR = None


# --------------------------------------------------------------------------- pages
@app.get("/")
def dashboard():
    return send_from_directory(STATIC, "dashboard.html")


@app.get("/player")
def player():
    return send_from_directory(STATIC, "player.html")


@app.get("/favicon.ico")
def favicon():
    return "", 204


@app.get("/media/<path:name>")
def media(name):
    return send_from_directory(MEDIA_DIR, name, conditional=True)


@app.get("/thumbs/<path:name>")
def thumbs(name):
    return send_from_directory(THUMB_DIR, name)


@app.get("/stream")
def stream():
    def gen():
        while True:
            with aud.lock:
                jpeg = aud.jpeg
            if jpeg:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            time.sleep(0.1)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


# --------------------------------------------------------------------------- status / player control
@app.get("/api/status")
def api_status():
    st = sched.state()
    with aud.lock:
        st["live"] = dict(aud.live)
    return jsonify(st)


@app.post("/api/player/<action>")
def api_player(action):
    if action == "start":
        sched.start_playback()
    elif action == "stop":
        sched.stop_playback()
    elif action == "skip":
        sched.skip()
    else:
        abort(404)
    return jsonify(sched.state())


# --------------------------------------------------------------------------- categories
@app.get("/api/categories")
def api_categories():
    return jsonify(db.q("SELECT c.*, (SELECT COUNT(*) FROM ads a WHERE a.category_id=c.id) AS ads "
                        "FROM categories c ORDER BY c.name"))


@app.post("/api/categories")
def api_add_category():
    data = request.get_json(force=True)
    name = (data.get("name") or "").strip()
    if not name:
        abort(400, "name required")
    if db.one("SELECT id FROM categories WHERE name=? COLLATE NOCASE", (name,)):
        abort(409, "category exists")
    cid = db.add_category(name, data.get("color"))
    return jsonify(db.one("SELECT * FROM categories WHERE id=?", (cid,)))


@app.delete("/api/categories/<int:cid>")
def api_del_category(cid):
    for a in db.q("SELECT filename FROM ads WHERE category_id=?", (cid,)):
        _remove_media(a["filename"])
    db.exec("DELETE FROM categories WHERE id=?", (cid,))
    return jsonify({"ok": True})


# --------------------------------------------------------------------------- ads
def _ads_with_stats():
    return db.q(
        "SELECT a.*, c.name AS category_name, "
        " COUNT(i.id) AS plays, COALESCE(SUM(i.viewers),0) AS viewers, COALESCE(SUM(i.lookers),0) AS lookers, "
        " MAX(i.ended_at) AS last_played "
        "FROM ads a JOIN categories c ON c.id=a.category_id LEFT JOIN impressions i ON i.ad_id=a.id "
        "GROUP BY a.id ORDER BY c.name, a.name"
    )


@app.get("/api/ads")
def api_ads():
    return jsonify(_ads_with_stats())


@app.post("/api/ads")
def api_upload_ad():
    f = request.files.get("file")
    if not f or not f.filename:
        abort(400, "file required")
    ext = f.filename.rsplit(".", 1)[-1].lower() if "." in f.filename else ""
    if ext in IMAGE_EXT:
        type_ = "image"
    elif ext in VIDEO_EXT:
        type_ = "video"
    else:
        abort(400, f"unsupported file type .{ext}")
    try:
        cid = int(request.form.get("category_id", ""))
    except ValueError:
        abort(400, "category_id required")
    if not db.one("SELECT id FROM categories WHERE id=?", (cid,)):
        abort(400, "unknown category")

    base = re.sub(r"[^A-Za-z0-9._-]+", "_", f.filename)[:80]
    filename = f"{int(time.time() * 1000)}_{base}"
    path = os.path.join(MEDIA_DIR, filename)
    f.save(path)

    name = (request.form.get("name") or "").strip() or os.path.splitext(f.filename)[0]
    duration = request.form.get("duration", "").strip()
    probed = _video_duration(path) if type_ == "video" else None
    if duration:
        duration = float(duration)
    else:
        duration = probed or db.setting("default_duration", float)
    ad_id = db.add_ad(cid, name, filename, type_, duration)
    _make_thumb(ad_id, path, type_)
    return jsonify(db.one("SELECT * FROM ads WHERE id=?", (ad_id,)))


@app.patch("/api/ads/<int:ad_id>")
def api_patch_ad(ad_id):
    data = request.get_json(force=True)
    allowed = {"name": str, "duration": float, "enabled": int, "category_id": int,
               "active_from": int, "active_to": int}
    sets, args = [], []
    for k, cast in allowed.items():
        if k in data:
            sets.append(f"{k}=?")
            args.append(cast(data[k]))
    if sets:
        db.exec(f"UPDATE ads SET {', '.join(sets)} WHERE id=?", (*args, ad_id))
    return jsonify(db.one("SELECT * FROM ads WHERE id=?", (ad_id,)))


@app.delete("/api/ads/<int:ad_id>")
def api_del_ad(ad_id):
    ad = db.one("SELECT filename FROM ads WHERE id=?", (ad_id,))
    if ad:
        _remove_media(ad["filename"])
        thumb = os.path.join(THUMB_DIR, f"{ad_id}.jpg")
        if os.path.exists(thumb):
            os.remove(thumb)
        db.exec("DELETE FROM ads WHERE id=?", (ad_id,))
        cur = sched.state().get("current")
        if cur and cur["ad"]["id"] == ad_id:
            sched.skip()  # do not keep showing a deleted ad
    return jsonify({"ok": True})


def _remove_media(filename):
    for p in (os.path.join(MEDIA_DIR, filename),):
        if os.path.exists(p):
            os.remove(p)


def _video_duration(path):
    cap = cv2.VideoCapture(path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        return round(n / fps, 1) if fps > 0 and n > 0 else None
    finally:
        cap.release()


def _make_thumb(ad_id, path, type_):
    try:
        if type_ == "image":
            img = cv2.imread(path)
        else:
            cap = cv2.VideoCapture(path)
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            img = None
            # try a frame ~15% in (first frames are often black), then fall back to the first frame
            for pos in ([int(n * 0.15)] if n > 10 else []) + [0]:
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                ok, frame = cap.read()
                if ok and frame is not None and frame.mean() > 8:
                    img = frame
                    break
                if ok and frame is not None and img is None:
                    img = frame
            cap.release()
        if img is None:
            return
        h, w = img.shape[:2]
        tw = 240
        img = cv2.resize(img, (tw, max(1, int(h * tw / w))))
        cv2.imwrite(os.path.join(THUMB_DIR, f"{ad_id}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    except Exception as e:  # thumbnails are cosmetic
        print(f"[thumb] {e}", flush=True)


# --------------------------------------------------------------------------- stats
@app.get("/api/stats")
def api_stats():
    cat_b, _ = sched.beliefs()
    cats = db.q(
        "SELECT c.id, c.name, COUNT(i.id) AS plays, COALESCE(SUM(i.viewers),0) AS viewers, "
        " COALESCE(SUM(i.lookers),0) AS lookers, COALESCE(SUM(i.ended_at - i.started_at),0) AS seconds "
        "FROM categories c LEFT JOIN impressions i ON i.category_id=c.id GROUP BY c.id ORDER BY c.name"
    )
    total_plays = sum(c["plays"] for c in cats) or 1
    for c in cats:
        b = cat_b.get(c["id"])
        c["attention"] = round(c["lookers"] / c["viewers"], 3) if c["viewers"] else None
        c["belief"] = round(b["alpha"] / (b["alpha"] + b["beta"]), 3) if b else None
        c["share"] = round(c["plays"] / total_plays, 3)
    since = time.time() - 24 * 3600
    hourly = db.q(
        "SELECT strftime('%H', ended_at, 'unixepoch', 'localtime') AS hour, COUNT(*) AS plays, "
        " SUM(viewers) AS viewers, SUM(lookers) AS lookers FROM impressions WHERE ended_at > ? "
        "GROUP BY hour ORDER BY hour", (since,)
    )
    recent = db.q(
        "SELECT i.*, a.name AS ad_name, c.name AS category_name FROM impressions i "
        "LEFT JOIN ads a ON a.id=i.ad_id LEFT JOIN categories c ON c.id=i.category_id "
        "ORDER BY i.id DESC LIMIT 25"
    )
    totals = db.one("SELECT COUNT(*) AS plays, COALESCE(SUM(viewers),0) AS viewers, "
                    "COALESCE(SUM(lookers),0) AS lookers FROM impressions")
    today = db.one("SELECT COUNT(*) AS plays, COALESCE(SUM(viewers),0) AS viewers, "
                   "COALESCE(SUM(lookers),0) AS lookers FROM impressions WHERE ended_at > ?",
                   (time.time() - 24 * 3600,))
    return jsonify({"categories": cats, "hourly": hourly, "recent": recent, "totals": totals, "last24h": today})


@app.get("/api/impressions.csv")
def api_csv():
    rows = db.q(
        "SELECT i.id, i.started_at, i.ended_at, c.name AS category, a.name AS ad, i.viewers, i.lookers, "
        " i.avg_people, i.max_people, i.frames, i.reason FROM impressions i "
        "LEFT JOIN ads a ON a.id=i.ad_id LEFT JOIN categories c ON c.id=i.category_id ORDER BY i.id"
    )
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "started", "ended", "category", "ad", "viewers", "lookers", "avg_people", "max_people",
                "frames", "reason"])
    for r in rows:
        w.writerow([r["id"], time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r["started_at"])),
                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r["ended_at"])), r["category"], r["ad"],
                    r["viewers"], r["lookers"], r["avg_people"], r["max_people"], r["frames"], r["reason"]])
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=impressions.csv"})


@app.post("/api/stats/reset")
def api_reset_stats():
    db.exec("DELETE FROM impressions")
    return jsonify({"ok": True})


# --------------------------------------------------------------------------- settings
@app.get("/api/settings")
def api_settings():
    return jsonify(db.all_settings())


@app.post("/api/settings")
def api_set_settings():
    data = request.get_json(force=True)
    for k, v in data.items():
        if k == "running":
            continue
        db.set_setting(k, v)
    return jsonify(db.all_settings())


# --------------------------------------------------------------------------- main
def main():
    global db, aud, sched, DATA_DIR, MEDIA_DIR, THUMB_DIR
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--data-dir", default=os.path.join(HERE, "data"))
    ap.add_argument("--source", default="0", help="camera index, 'picam', file or URL")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--detect-width", type=int, default=640)
    ap.add_argument("--faces-only", action="store_true")
    ap.add_argument("--person-every", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--hfov", type=float, default=60.0)
    ap.add_argument("--min-face", type=int, default=20)
    args = ap.parse_args()

    DATA_DIR = args.data_dir
    MEDIA_DIR = os.path.join(DATA_DIR, "media")
    THUMB_DIR = os.path.join(DATA_DIR, "thumbs")
    os.makedirs(MEDIA_DIR, exist_ok=True)
    os.makedirs(THUMB_DIR, exist_ok=True)

    db = DB(os.path.join(DATA_DIR, "billboard.db"))
    aud = Audience(args.source, args.width, args.height, args.detect_width, args.faces_only,
                   args.person_every, args.threads, args.hfov, args.min_face, settings_getter=db.setting)
    sched = Scheduler(db, aud)
    aud.start()
    sched.start()
    print(f"dashboard: http://0.0.0.0:{args.port}/   player: http://0.0.0.0:{args.port}/player", flush=True)
    app.run(host="0.0.0.0", port=args.port, threaded=True, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
