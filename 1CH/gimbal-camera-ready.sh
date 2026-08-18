#!/bin/bash

set -u

DRIVER=imx708

for dev in /sys/bus/i2c/devices/*-00*; do
    [ -e "${dev}/name" ] || continue
    case "$(cat "${dev}/name" 2>/dev/null)" in
        imx708*) ;;
        *) continue ;;
    esac
    [ -e "${dev}/driver" ] && continue
    node="${dev##*/}"
    echo "${node}" > "/sys/bus/i2c/drivers/${DRIVER}/bind" 2>/dev/null || true
    sleep 2
done

for i in $(seq 1 10); do
    if timeout 8 rpicam-hello --list-cameras 2>/dev/null | grep -q "${DRIVER}"; then
        exit 0
    fi
    sleep 2
done

exit 0
