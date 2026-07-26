#!/bin/bash
# CosyVoice provision script — placeholder
# Real install steps TBD

LOG=/content/cosy/provision.log
exec > >(tee -a "$LOG") 2>&1

echo "[cosyvoice] provision started at $(date)"

# Placeholder: create a fake install process
echo "[cosyvoice] cloning repo..."
sleep 3
echo "[cosyvoice] installing dependencies..."
sleep 3
echo "[cosyvoice] downloading models..."
sleep 2
echo "[cosyvoice] install complete"

# Mark as ready
echo "CosyVoice v0.1.0-placeholder" > /content/cosy/COSY_READY
echo "[cosyvoice] provision done at $(date)"
