#!/bin/sh
# GPU-gate CUDA provisioning for the linux-cuda port (senko [nvidia] extras).
# Idempotent; installs only into the active venv. See design/gpu-provisioning.md.
# Run from the repo root. Do NOT run the CUDA suite via `uv run` afterwards -
# it re-syncs and uninstalls these extras; use .venv/bin/python|pytest directly.
set -e

export PATH="$PWD/.venv/bin:$PATH"          # venv cmake/python take precedence
export CMAKE_POLICY_VERSION_MINIMUM=3.5     # harmless when the prebuilt wheel is used

uv sync --locked --extra dev                # CPU default lock + pytest/black/ruff
uv pip install cmake                        # only the kaldifeat sdist fallback needs it
uv pip install --index-url https://download.pytorch.org/whl/cu128 torchvision==0.23.0
uv pip install pyannote-audio==3.4.0
uv pip install -f https://csukuangfj.github.io/kaldifeat/cuda.html \
    "kaldifeat==1.25.5.dev20250807+cuda12.8.torch2.8.0"

.venv/bin/python - <<'PY'
import torch, kaldifeat
opts = kaldifeat.FbankOptions(); opts.device = torch.device("cuda", 0)
kaldifeat.Fbank(opts)
print("provision OK: torch", torch.__version__, "cuda", torch.cuda.is_available(),
      "kaldifeat", kaldifeat.__version__, "Fbank on cuda:0")
PY
