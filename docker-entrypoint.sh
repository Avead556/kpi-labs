#!/bin/sh
set -e

: "${DISPLAY:=:99}"
export DISPLAY

Xvfb "$DISPLAY" -screen 0 "${XVFB_SCREEN:-1280x1024x24}" -nolisten tcp &

socket="/tmp/.X11-unix/X${DISPLAY#:}"
attempt=0
while [ ! -e "$socket" ]; do
    attempt=$((attempt + 1))
    if [ "$attempt" -gt 100 ]; then
        echo "Xvfb did not come up on display $DISPLAY" >&2
        exit 1
    fi
    sleep 0.1
done

exec "$@"
