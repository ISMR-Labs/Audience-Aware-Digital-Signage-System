# Audience-Aware Digital Signage System (pi-billboard)

Audience-aware digital signage for a Raspberry Pi. Upload ads into
categories from a web dashboard, press Start, and the Pi's screen plays them
fullscreen while the camera measures who is in front of the screen and who
is actually looking at it. The scheduler then plays more of what gets
attention.

Everything runs on the Pi: Flask + SQLite server, OpenCV face detection,
a 4 MB TFLite person detector, and Chromium in kiosk mode for the screen.
No cloud, no CDN, works offline.

A one-page write-up with the architecture diagram and the scheduling
method is in [`paper/pi-billboard.pdf`](paper/pi-billboard.pdf).

## How scheduling works

1. **First pass**: on Start, one ad from every category is played (the least
   played ad in each) so every category gets a measurement.
2. **Thompson sampling** after that. Each category has a Beta belief about
   its attention rate (people who looked / people present). Before each slot
   the scheduler draws one random sample per category from its belief and
   plays the category with the highest draw. Categories with a high rate win
   most of the time, uncertain ones still get tried, and a category that
   never had an audience is neither punished nor favoured.
3. **Time decay**: evidence fades with a configurable half-life (default
   2 h), so when the lunch crowd is replaced by the evening crowd the mix
   adapts within a few plays.
4. **Exploration floor**: with probability "Explore %" (default 10 %) a random
   category is played so nothing starves and new ads get data.
5. Inside the chosen category the same sampling picks the ad, never repeating
   the ad that just played when there is an alternative.

Your example (A 12/30, B 10/29, C 22/30, D 2/30) would make the scheduler
play C around 80 % of the time, A and B occasionally, D rarely, and shift
automatically if that changes.

## How the audience is measured

Per ad play ("impression") the camera thread tracks each person with a
stable id so a person is counted once, not once per frame:

- **viewer**: a person present for at least `Viewer min` seconds (0.5)
- **looked**: a viewer whose head faced the screen for at least `Look min`
  seconds (0.7) during that ad

Head direction comes from a 5-point head-pose estimate; it is a proxy for
gaze. Someone with their head toward the screen but eyes elsewhere still
counts as looking. See `people_counter.py` (the standalone counter) for the
vision details and the tuning flags.

The camera should sit next to the screen, pointing at the audience.

## Dashboard

`http://<pi>:8090/`

- Start / Stop / Skip, now playing with live viewer count, last scheduler decision
- live annotated camera view (green box = looking, red = not)
- totals, per-category table (plays, share of plays, viewers, looked, attention, current belief)
- per-ad table (thumbnail, plays, viewers, looked, attention %, last played, enable toggle,
  seconds on screen, active hours)
- viewers per hour for the last 24 h, recent plays, CSV export, reset
- settings: image duration, half-life, explore %, viewer/look thresholds, attention angle

The player page `http://<pi>:8090/player` is what the screen shows. It is
plain HTML, so it also works on any other device on the network, or on a
Smart TV browser.

## Install on the Pi

64-bit Raspberry Pi OS with desktop (Bookworm or newer), Pi 4 or 5.

```bash
git clone https://github.com/ISMR-Labs/Audience-Aware-Digital-Signage-System.git ~/pi-billboard
ssh pi@<pi>
cd ~/pi-billboard && ./install_pi.sh
```

Run manually:

```bash
.venv/bin/python server.py --source picam        # CSI camera
.venv/bin/python server.py --source 0            # USB webcam
./start_player.sh                                # fullscreen player on the Pi's HDMI screen
```

Run at boot: copy `billboard-server.service` and `billboard-player.service`
to `/etc/systemd/system/`, adjust the user and paths, then
`sudo systemctl enable --now billboard-server billboard-player`. The player
unit needs the Pi to auto-login to the desktop (raspi-config, System
Options, Boot / Auto Login). Playback state survives reboots: if it was
running, it resumes.

Server flags: `--port`, `--data-dir`, plus the camera flags from
people_counter (`--width/--height`, `--detect-width`, `--faces-only`,
`--person-every`, `--threads`, `--hfov`, `--min-face`). On a Pi 4 use the
defaults; on a Pi 3 / Zero 2 add `--detect-width 320 --faces-only`.

## Ads

- images: jpg, png, gif, webp. Shown for "Seconds" (default from settings, 10 s).
- videos: mp4, webm, mov, mkv (H.264 mp4 is safest on the Pi). Duration is
  read from the file unless you override it. Audio plays if the kiosk is
  started with `start_player.sh` (it passes the autoplay flag to Chromium).
- active hours per ad (e.g. 8-18) for dayparting.
- disable an ad with the checkbox without deleting its stats.

Files live in `data/media`, the database in `data/billboard.db`. Back up
`data/`.

## API

| Method | Path | |
|---|---|---|
| GET | `/api/status` | running, current ad, live counts |
| POST | `/api/player/start` `/stop` `/skip` | |
| GET/POST | `/api/categories`, DELETE `/api/categories/<id>` | |
| GET | `/api/ads` | ads with stats |
| POST | `/api/ads` | multipart: file, category_id, name, duration |
| PATCH/DELETE | `/api/ads/<id>` | name, duration, enabled, category_id, active_from, active_to |
| GET | `/api/stats` | category stats, hourly, recent, totals |
| GET | `/api/impressions.csv` | export |
| GET/POST | `/api/settings` | |
| GET | `/stream` | MJPEG camera |

## Project layout

- `server.py` Flask app, API, uploads, stats
- `scheduler.py` first pass + Thompson sampling, impression lifecycle
- `audience.py` camera thread, tracking, per-impression metrics
- `tracker.py` IoU tracker
- `people_counter.py` detectors and head pose (also a standalone CLI)
- `db.py` SQLite schema and helpers
- `static/dashboard.html`, `static/player.html`
- `install_pi.sh`, `start_player.sh`, `download_models.py`, systemd units

## Ideas not built yet

- MQTT / Home Assistant publishing of live counts
- weighted or paid priority per ad, frequency caps
- age / gender bucketing (needs an extra model; privacy trade-off)
- multi-screen: one server, several player pages with their own cameras
- dwell time per ad (how long people stayed) is already recorded per track
  but not yet shown
