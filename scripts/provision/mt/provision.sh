#!/bin/bash
# MuseTalk provision script — placeholder
# Real install steps TBD

LOG=/content/mt/provision.log
exec > >(tee -a "$LOG") 2>&1

echo "[musetalk] provision started at $(date)"

# Placeholder: create a fake install process
echo "[musetalk] downloading model weights..."
sleep 2
echo "[musetalk] setting up venv..."
sleep 2
echo "[musetalk] install complete"

# Mark as ready
echo "MuseTalk v0.1.0-placeholder" > /content/mt/READY
echo "[musetalk] provision done at $(date)"
