#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CHECKPOINT_URL="${FASTVLM_CHECKPOINT_URL:-https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_0.5b_stage3.zip}"
MLX_VLM_COMMIT="${MLX_VLM_COMMIT:-1884b551bc741f26b2d54d68fa89d4e934b9a3de}"
Q_GROUP_SIZE="${Q_GROUP_SIZE:-64}"
BUILD_CONFIGURATION="${FASTVLM_BUILD_CONFIGURATION:-Release}"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
LOCAL_ROOT="${FASTVLM_LOCAL_ROOT:-$HOME/Library/Caches/FastVLM-Compare}"

VENV_DIR="$LOCAL_ROOT/venv"
WORK_DIR="$LOCAL_ROOT/work"
MLX_VLM_DIR="$WORK_DIR/mlx-vlm"
CHECKPOINT_ROOT="$WORK_DIR/checkpoint"
CHECKPOINT_ZIP="$WORK_DIR/fastvlm-0.5b-stage3.zip"
OUTPUT_8BIT="$WORK_DIR/fastvlm-0.5b-int8"
OUTPUT_4BIT="$WORK_DIR/fastvlm-0.5b-int4"
DERIVED_DATA="$LOCAL_ROOT/FastVLMDerivedData"
APP_MODEL_DIR="$REPO_ROOT/app/FastVLM/model"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "This preparation script must run natively on an Apple Silicon Mac (Darwin/arm64)." >&2
  exit 1
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Could not find $PYTHON_BIN." >&2
  echo "Install Python 3.10 or set PYTHON_BIN to a compatible Python executable." >&2
  exit 1
fi

for command_name in git curl unzip xcodebuild; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Required command is missing: $command_name" >&2
    exit 1
  fi
done

mkdir -p "$LOCAL_ROOT" "$WORK_DIR"

echo "==> Creating isolated Python environment"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
PYTHON="$VENV_DIR/bin/python"

"$PYTHON" -m pip install --upgrade pip setuptools wheel
"$PYTHON" -m pip install -e "$REPO_ROOT" --no-deps
"$PYTHON" -m pip install \
  torch==2.6.0 \
  torchvision==0.21.0 \
  transformers==4.48.3 \
  tokenizers==0.21.0 \
  sentencepiece==0.1.99 \
  accelerate==1.6.0 \
  peft==0.13.2 \
  huggingface-hub==0.28.1 \
  numpy==1.26.4 \
  scipy==1.13.1 \
  einops==0.6.1 \
  einops-exts==0.0.4 \
  timm==1.0.15 \
  coremltools==8.2 \
  mlx==0.22.1 \
  Pillow==11.1.0 \
  requests==2.32.3 \
  opencv-python==4.10.0.84 \
  shortuuid==1.0.13

echo "==> Preparing pinned mlx-vlm converter"
rm -rf "$MLX_VLM_DIR"
git clone --filter=blob:none https://github.com/Blaizzy/mlx-vlm.git "$MLX_VLM_DIR"
git -C "$MLX_VLM_DIR" checkout "$MLX_VLM_COMMIT"
git -C "$MLX_VLM_DIR" apply --check "$REPO_ROOT/model_export/fastvlm_mlx-vlm.patch"
git -C "$MLX_VLM_DIR" apply "$REPO_ROOT/model_export/fastvlm_mlx-vlm.patch"
"$PYTHON" -m pip install -e "$MLX_VLM_DIR" --no-deps

echo "==> Downloading FastVLM-0.5B Stage 3 checkpoint"
rm -rf "$CHECKPOINT_ROOT" "$CHECKPOINT_ZIP"
mkdir -p "$CHECKPOINT_ROOT"
curl --fail --location --retry 4 --retry-delay 5 \
  --output "$CHECKPOINT_ZIP" \
  "$CHECKPOINT_URL"
unzip -q "$CHECKPOINT_ZIP" -d "$CHECKPOINT_ROOT"
rm -f "$CHECKPOINT_ZIP"

CONFIG_COUNT="$(find "$CHECKPOINT_ROOT" -maxdepth 3 -type f -name config.json -print | wc -l | tr -d ' ')"
if [[ "$CONFIG_COUNT" -ne 1 ]]; then
  echo "Expected exactly one checkpoint config.json, found $CONFIG_COUNT." >&2
  find "$CHECKPOINT_ROOT" -maxdepth 3 -type f -name config.json -print >&2
  exit 1
fi

CONFIG_PATH="$(find "$CHECKPOINT_ROOT" -maxdepth 3 -type f -name config.json -print | head -n 1)"
CHECKPOINT_DIR="$(dirname "$CONFIG_PATH")"
if ! compgen -G "$CHECKPOINT_DIR/*.safetensors" >/dev/null; then
  echo "No safetensors were found in $CHECKPOINT_DIR." >&2
  exit 1
fi

echo "==> Correcting checkpoint embedding metadata"
"$PYTHON" - "$CONFIG_PATH" <<'PY'
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
config = json.loads(config_path.read_text())
config["tie_word_embeddings"] = False
config_path.write_text(json.dumps(config, indent=2) + "\n")
print("tie_word_embeddings:", config["tie_word_embeddings"])
PY

echo "==> Exporting shared Core ML vision encoder"
"$PYTHON" model_export/export_vision_encoder.py --model-path "$CHECKPOINT_DIR"

VISION_COUNT="$(find "$CHECKPOINT_DIR" -maxdepth 1 -type d -name '*.mlpackage' -print | wc -l | tr -d ' ')"
if [[ "$VISION_COUNT" -ne 1 ]]; then
  echo "Expected one Core ML vision package, found $VISION_COUNT." >&2
  find "$CHECKPOINT_DIR" -maxdepth 1 -type d -name '*.mlpackage' -print >&2
  exit 1
fi

echo "==> Converting 8-bit LLM"
rm -rf "$OUTPUT_8BIT"
"$PYTHON" -m mlx_vlm.convert \
  --hf-path "$CHECKPOINT_DIR" \
  --mlx-path "$OUTPUT_8BIT" \
  --only-llm \
  --quantize \
  --q-bits 8 \
  --q-group-size "$Q_GROUP_SIZE"

echo "==> Converting 4-bit LLM"
rm -rf "$OUTPUT_4BIT"
"$PYTHON" -m mlx_vlm.convert \
  --hf-path "$CHECKPOINT_DIR" \
  --mlx-path "$OUTPUT_4BIT" \
  --only-llm \
  --quantize \
  --q-bits 4 \
  --q-group-size "$Q_GROUP_SIZE"

echo "==> Validating quantized artifacts"
"$PYTHON" - "$OUTPUT_8BIT" "$OUTPUT_4BIT" "$Q_GROUP_SIZE" <<'PY'
import json
import sys
from pathlib import Path

group_size = int(sys.argv[3])
models = ((Path(sys.argv[1]), 8), (Path(sys.argv[2]), 4))
weight_sizes = {}

for model_dir, expected_bits in models:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise SystemExit(f"Missing config.json in {model_dir}")
    weight_files = list(model_dir.glob("*.safetensors"))
    if not weight_files:
        raise SystemExit(f"Missing safetensors in {model_dir}")
    vision = list(model_dir.glob("*.mlpackage"))
    if len(vision) != 1:
        raise SystemExit(f"Expected one .mlpackage in {model_dir}, found {len(vision)}")

    config = json.loads(config_path.read_text())
    quantization = config.get("quantization")
    if not isinstance(quantization, dict):
        raise SystemExit(f"Missing quantization metadata in {config_path}")
    if quantization.get("bits") != expected_bits:
        raise SystemExit(
            f"Expected {expected_bits}-bit config, found {quantization.get('bits')!r}"
        )
    if quantization.get("group_size") != group_size:
        raise SystemExit(
            f"Expected group_size={group_size}, found {quantization.get('group_size')!r}"
        )

    weight_sizes[expected_bits] = sum(path.stat().st_size for path in weight_files)

if weight_sizes[4] >= weight_sizes[8]:
    raise SystemExit("4-bit LLM weights are not smaller than 8-bit LLM weights.")

print("8-bit LLM weight bytes:", weight_sizes[8])
print("4-bit LLM weight bytes:", weight_sizes[4])
PY

echo "==> Staging quantized resources for the Swift app"
rm -rf "$APP_MODEL_DIR"
mkdir -p "$APP_MODEL_DIR/int8.bundle" "$APP_MODEL_DIR/int4.bundle"

VISION_8BIT="$(find "$OUTPUT_8BIT" -maxdepth 1 -type d -name '*.mlpackage' -print | head -n 1)"
VISION_4BIT="$(find "$OUTPUT_4BIT" -maxdepth 1 -type d -name '*.mlpackage' -print | head -n 1)"
if [[ -z "$VISION_8BIT" || -z "$VISION_4BIT" ]]; then
  echo "Both quantized variants must contain a Core ML vision package." >&2
  exit 1
fi
if ! diff -qr "$VISION_8BIT" "$VISION_4BIT" >/dev/null; then
  echo "The INT8 and INT4 Core ML vision packages differ." >&2
  exit 1
fi

find "$OUTPUT_8BIT" -mindepth 1 -maxdepth 1 ! -name '*.mlpackage' \
  -exec cp -R {} "$APP_MODEL_DIR/int8.bundle/" \;
find "$OUTPUT_4BIT" -mindepth 1 -maxdepth 1 ! -name '*.mlpackage' \
  -exec cp -R {} "$APP_MODEL_DIR/int4.bundle/" \;
cp -R "$VISION_8BIT" "$APP_MODEL_DIR/fastvithd.mlpackage"

echo "==> Resolving Swift packages"
xcodebuild \
  -resolvePackageDependencies \
  -project app/FastVLM.xcodeproj \
  -scheme "FastVLM App"

echo "==> Building local macOS benchmark app"
rm -rf "$DERIVED_DATA"
xcodebuild \
  -project app/FastVLM.xcodeproj \
  -scheme "FastVLM App" \
  -configuration "$BUILD_CONFIGURATION" \
  -destination "platform=macOS" \
  -derivedDataPath "$DERIVED_DATA" \
  CODE_SIGNING_ALLOWED=NO \
  CODE_SIGNING_REQUIRED=NO \
  build

BUILD_ROOT="$DERIVED_DATA/Build/Products/$BUILD_CONFIGURATION"
APP_COUNT="$(find "$BUILD_ROOT" -maxdepth 1 -type d -name '*.app' -print | wc -l | tr -d ' ')"
if [[ "$APP_COUNT" -ne 1 ]]; then
  echo "Expected exactly one built .app under $BUILD_ROOT, found $APP_COUNT." >&2
  find "$BUILD_ROOT" -maxdepth 1 -type d -name '*.app' -print >&2
  exit 1
fi
APP_PATH="$(find "$BUILD_ROOT" -maxdepth 1 -type d -name '*.app' -print | head -n 1)"

INT8_CONFIG="$(find "$APP_PATH" -type f -path '*/int8.bundle/config.json' -print | head -n 1)"
INT4_CONFIG="$(find "$APP_PATH" -type f -path '*/int4.bundle/config.json' -print | head -n 1)"
VISION_COMPILED_COUNT="$(find "$APP_PATH" -type d -name 'fastvithd.mlmodelc' -print | wc -l | tr -d ' ')"
if [[ -z "$INT8_CONFIG" || -z "$INT4_CONFIG" || "$VISION_COMPILED_COUNT" -ne 1 ]]; then
  echo "Built app does not contain the expected INT8, INT4, and shared vision resources." >&2
  exit 1
fi

echo
echo "Local FastVLM benchmark app is ready:"
echo "$APP_PATH"
echo
echo "Run the controlled benchmark with:"
printf 'python3 benchmark/run_local_benchmark.py --app %q\n' "$APP_PATH"
