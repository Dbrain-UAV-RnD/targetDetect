#!/bin/bash

set -u

DEV="/dev/video${CAM_INDEX:-0}"

for i in $(seq 1 30); do
    if [ -e "${DEV}" ] \
       && v4l2-ctl -d "${DEV}" --all 2>/dev/null | grep -q "Video Capture"; then
        exit 0
    fi
    sleep 2
done

exit 0
