#!/bin/bash
# Poll AMD iGPU sysfs busy%/VRAM/GTT usage in a loop until killed.
while true; do
    b=$(cat /sys/class/drm/card0/device/gpu_busy_percent 2>/dev/null)
    v=$(cat /sys/class/drm/card0/device/mem_info_vram_used 2>/dev/null)
    g=$(cat /sys/class/drm/card0/device/mem_info_gtt_used 2>/dev/null)
    echo "$(date +%s.%N) busy=$b vram=$v gtt=$g"
    sleep 0.1
done
