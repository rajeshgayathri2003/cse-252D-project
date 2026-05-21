#!/usr/bin/env bash
# One-shot setup for AI2-THOR CloudRendering on runpod containers.
# Registers an NVIDIA Vulkan ICD via VK_ICD_FILENAMES so the loader can find
# the GPU, then prints how to activate it in your current shell.
#
# Run once per fresh container (or persist via ~/.bashrc).

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNPOD_DIR="$REPO_DIR/.runpod"

mkdir -p "$RUNPOD_DIR"

cat > "$RUNPOD_DIR/nvidia_icd.json" <<'JSON'
{
  "file_format_version": "1.0.0",
  "ICD": {
    "library_path": "libGLX_nvidia.so.0",
    "api_version": "1.3.194"
  }
}
JSON

cat > "$RUNPOD_DIR/env.sh" <<EOF
export VK_ICD_FILENAMES=$RUNPOD_DIR/nvidia_icd.json
export VK_DRIVER_FILES=$RUNPOD_DIR/nvidia_icd.json
EOF

echo "Wrote:"
echo "  $RUNPOD_DIR/nvidia_icd.json"
echo "  $RUNPOD_DIR/env.sh"
echo
echo "Next steps (run in your current shell):"
echo "  source $RUNPOD_DIR/env.sh"
echo "  vulkaninfo --summary | head -40   # confirm the RTX 4090 is listed"
echo
echo "To auto-activate in every new shell on this pod:"
echo "  echo 'source $RUNPOD_DIR/env.sh' >> ~/.bashrc"
