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
    echo "${DRIVER} 미바인딩 — bind 시도: ${node}"
    echo "${node}" > "/sys/bus/i2c/drivers/${DRIVER}/bind" 2>/dev/null || true
    sleep 2
done

for i in $(seq 1 10); do
    if timeout 8 rpicam-hello --list-cameras 2>/dev/null | grep -q "${DRIVER}"; then
        echo "카메라 준비됨 (${i}회차)"
        exit 0
    fi
    sleep 2
done

echo "카메라 준비 실패 — 서비스는 그대로 진행(실패 시 systemd 가 재시도)"
exit 0
