#!/usr/bin/env bash
# One-shot setup on 64-bit Raspberry Pi OS (Bookworm or newer) with desktop.
set -euo pipefail
cd "$(dirname "$0")"
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip python3-picamera2 libgl1 libglib2.0-0 chromium unclutter || \
sudo apt-get install -y python3-venv python3-pip python3-picamera2 libgl1 libglib2.0-0 chromium-browser unclutter
python3 -m venv --system-site-packages .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
.venv/bin/python download_models.py
echo
echo "Done. Start the server:   .venv/bin/python server.py --source picam"
echo "Dashboard: http://$(hostname -I | awk '{print $1}'):8090/"
echo "Fullscreen player on this Pi's screen:   ./start_player.sh"
echo "To run both at boot:  see README (systemd units)."
