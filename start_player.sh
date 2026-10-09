#!/usr/bin/env bash
# Opens the player page fullscreen in Chromium kiosk mode on the Pi's display.
URL="${1:-http://localhost:8090/player}"
export DISPLAY="${DISPLAY:-:0}"
BROWSER=$(command -v chromium || command -v chromium-browser || command -v google-chrome)
[ -z "$BROWSER" ] && { echo "chromium not found (sudo apt install chromium)"; exit 1; }
command -v xset >/dev/null && { xset s off; xset -dpms; xset s noblank; } 2>/dev/null || true
command -v unclutter >/dev/null && (unclutter -idle 1 -root &) 2>/dev/null || true
# wait for the server
for i in $(seq 1 60); do curl -sf "$URL" >/dev/null && break; sleep 1; done
exec "$BROWSER" --kiosk --noerrdialogs --disable-infobars --disable-session-crashed-bubble \
  --autoplay-policy=no-user-gesture-required --check-for-update-interval=31536000 \
  --overscroll-history-navigation=0 --disable-pinch "$URL"
