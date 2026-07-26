#!/bin/bash
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="$DIR/provision.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== CosyVoice provision $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

cd "$DIR"
rm -rf env CosyVoice cosyvoice_tts.py

echo "[cosy] base deps..."
pip install -q --upgrade setuptools wheel pip
pip install -q torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -q huggingface_hub modelscope soundfile hyperpyyaml openai-whisper wetext
pip install -q "numba>=0.61" "onnxruntime-gpu<1.21"

echo "[cosy] cloning CosyVoice..."
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git CosyVoice

# Fix cudart symlink
if [ ! -f /usr/lib/x86_64-linux-gnu/libcudart.so.13 ]; then
    ln -sf /usr/local/cuda/lib64/libcudart.so.12 /usr/lib/x86_64-linux-gnu/libcudart.so.13 2>/dev/null || true
    ldconfig 2>/dev/null || true
fi

echo "[cosy] pip install -r requirements.txt..."
cd CosyVoice
pip install -q -r requirements.txt 2>/dev/null || true

# Install matcha-tts submodule
echo "[cosy] installing matcha-tts..."
cd third_party/Matcha-TTS
pip install -q -e . 2>/dev/null || pip install -q -e . --no-build-isolation 2>/dev/null || true
cd "$DIR/CosyVoice"

echo "[cosy] downloading model..."
MODEL_DIR="$DIR/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B"
mkdir -p "$MODEL_DIR"
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='$MODEL_DIR')
"

echo "[cosy] verifying import..."
python3 -c "
import sys, os, types
# Stub matcha if not installed
if 'matcha' not in sys.modules:
    m = types.ModuleType('matcha')
    m.models = types.ModuleType('matcha.models')
    m.models.components = types.ModuleType('matcha.models.components')
    m.models.components.flow_matching = types.ModuleType('matcha.models.components.flow_matching')
    class BASECFM: pass
    m.models.components.flow_matching.BASECFM = BASECFM
    sys.modules['matcha'] = m
    sys.modules['matcha.models'] = m.models
    sys.modules['matcha.models.components'] = m.models.components
    sys.modules['matcha.models.components.flow_matching'] = m.models.components.flow_matching
sys.path.insert(0, '$DIR/CosyVoice')
os.environ['MODELSCOPE_CACHE'] = '$DIR/CosyVoice/pretrained_models'
from cosyvoice.cli.cosyvoice import AutoModel
print('CosyVoice import OK')
" 2>&1

# TTS wrapper with matcha stub
cat > "$DIR/cosyvoice_tts.py" << 'PYEOF'
#!/usr/bin/env python3
import sys, os, types, argparse, base64
import numpy as np, soundfile as sf

# Stub matcha module
m = types.ModuleType('matcha')
m.models = types.ModuleType('matcha.models')
m.models.components = types.ModuleType('matcha.models.components')
m.models.components.flow_matching = types.ModuleType('matcha.models.components.flow_matching')
class BASECFM: pass
m.models.components.flow_matching.BASECFM = BASECFM
sys.modules['matcha'] = m
sys.modules['matcha.models'] = m.models
sys.modules['matcha.models.components'] = m.models.components
sys.modules['matcha.models.components.flow_matching'] = m.models.components.flow_matching

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'CosyVoice'))
os.environ['MODELSCOPE_CACHE'] = os.path.join(os.path.dirname(__file__), 'CosyVoice', 'pretrained_models')
from cosyvoice.cli.cosyvoice import AutoModel

parser = argparse.ArgumentParser()
parser.add_argument('--ref', required=True)
parser.add_argument('--ref-text-b64', default='')
parser.add_argument('--text-b64', required=True)
parser.add_argument('--speed', type=float, default=1.0)
parser.add_argument('--out', required=True)
args = parser.parse_args()

ref_text = base64.b64decode(args.ref_text_b64).decode()
text = base64.b64decode(args.text_b64).decode()

ckpt = os.path.join(os.path.dirname(__file__), 'CosyVoice', 'pretrained_models', 'Fun-CosyVoice3-0.5B')
model = AutoModel(model_dir=ckpt)
result = list(model.inference_zero_shot(text, ref_text, args.ref))[0]
sf.write(args.out, result['tts_speech'], 24000)
print(f"OK out={args.out}")
PYEOF
chmod +x "$DIR/cosyvoice_tts.py"

echo "[cosy] DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "cosy_version=3.0" > "$DIR/COSY_READY"
echo "cosy_model=Fun-CosyVoice3-0.5B" >> "$DIR/COSY_READY"
echo "cosy_installed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$DIR/COSY_READY"
