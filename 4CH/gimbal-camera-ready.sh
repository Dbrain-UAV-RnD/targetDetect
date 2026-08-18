#!/bin/bash

set -u

I2C_BUS=10
SENSOR=10-001a
MUX_ADDR=0x24
CH_CAM1=0x02

if [ ! -e "/sys/bus/i2c/devices/${SENSOR}/driver" ]; then
    echo "imx708 미바인딩 — bind 시도"
    echo "${SENSOR}" > /sys/bus/i2c/drivers/imx708/bind 2>/dev/null || true
    sleep 2
fi

i2cset -y "${I2C_BUS}" "${MUX_ADDR}" "${MUX_ADDR}" "${CH_CAM1}" 2>/dev/null || true

for i in $(seq 1 10); do
    if timeout 8 rpicam-hello --list-cameras 2>/dev/null | grep -q imx708; then
        echo "카메라 준비됨 (${i}회차)"
        exit 0
    fi
    sleep 2
done

echo "카메라 준비 실패 — 서비스는 그대로 진행(실패 시 systemd 가 재시도)"
exit 0
