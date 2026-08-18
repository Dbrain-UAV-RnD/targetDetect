#!/bin/bash

set -u

I2C_BUS=10
SENSOR=10-001a
MUX_ADDR=0x24
CH_CAM1=0x02

if [ ! -e "/sys/bus/i2c/devices/${SENSOR}/driver" ]; then
    echo "${SENSOR}" > /sys/bus/i2c/drivers/imx708/bind 2>/dev/null || true
    sleep 2
fi

i2cset -y "${I2C_BUS}" "${MUX_ADDR}" "${MUX_ADDR}" "${CH_CAM1}" 2>/dev/null || true

for i in $(seq 1 10); do
    if timeout 8 rpicam-hello --list-cameras 2>/dev/null | grep -q imx708; then
        exit 0
    fi
    sleep 2
done

exit 0
