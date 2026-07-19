# GPU-gate provisioning - senko [nvidia] CUDA extras (linux-cuda)

Executed at the GPU gate on whiteboxlinux (RTX 4090, driver 580.159.03, CUDA
13.0 runtime, torch cu128 stack) after David's reboot cleared the driver
mismatch. Per the overlay Scope ruling (`_build-overlay.md` #5, `slice0-ground-truth.md`
§4): the frozen "`uv sync` is the whole install contract" covers the CPU
DEFAULT install only; senko's `[nvidia]` CUDA extras are this explicit,
scripted, uv-installable step - documented here so it reproduces.

## Outcome: CLEAN. No apt/sudo/CUDA-toolkit David-step required.

The `[nvidia]` extra set (`torch torchaudio torchvision pyannote-audio kaldifeat`,
per §2) resolves fully from uv-installable sources. torch/torchaudio already
pinned in the CPU default lock (`2.8.0+cu128`); this step adds torchvision,
pyannote-audio, and kaldifeat. The one member §4 flagged as non-installable -
kaldifeat - is satisfied by the vendor's official prebuilt CUDA wheel, so the
sdist build (which does need a CUDA toolkit) is never reached.

## kaldifeat: the §4 finding resolved

§4 recorded that kaldifeat 1.24 has no PyPI wheel and its sdist fails at
`cmake: not found`, leaving "whether kaldifeat then needs nvcc/CUDA UNVERIFIED".
Both questions are now answered:

1. **cmake alone is not enough.** With `cmake` on PATH (via `uv pip install cmake`,
   giving cmake 4.4.0) the sdist build advances but fails twice more:
   - cmake 4.x removed compatibility with `cmake_minimum_required < 3.5`, which
     kaldifeat's pinned pybind11 declares. Worked around with the env var
     `CMAKE_POLICY_VERSION_MINIMUM=3.5` (cmake's own suggested fix; no sudo).
   - Then the real blocker: `Could NOT find CUDA (missing CUDA_TOOLKIT_ROOT_DIR
     CUDA_NVCC_EXECUTABLE CUDA_INCLUDE_DIRS CUDA_CUDART_LIBRARY)` /
     `Caffe2 ... uses CUDA but I cannot find the CUDA libraries`.
   So **the kaldifeat sdist genuinely requires a full CUDA toolkit (nvcc +
   headers + libs)** - not just the driver. `which nvcc` = none on this box.
   Installing the toolkit is apt/sudo, i.e. it WOULD be a named David-step.

2. **But the sdist is avoidable.** The kaldifeat author (csukuangfj, same as
   k2/sherpa) publishes prebuilt CUDA wheels at a find-links page. One matches
   this stack exactly - CUDA 12.8 (= our cu128), torch 2.8.0, cp312,
   manylinux_2_28 x86_64:
   ```
   kaldifeat==1.25.5.dev20250807+cuda12.8.torch2.8.0
   ```
   It installs with no build, imports, and runs `kaldifeat.Fbank` on `cuda:0`.
   This is a uv-installable source, so per the ruling it is preferred over the
   sdist and no toolkit David-step is incurred.

   Note the version drift from senko's unpinned `kaldifeat` (§4 saw the sdist
   resolve to 1.24): the prebuilt is `1.25.5.dev...`. senko's `[nvidia]` extra
   pins nothing, and this is the wheel built for exactly our torch+cuda, so it
   is the correct member of the pair.

## Reproducible script

`design/provision-cuda.sh` (committed). Run from the repo root with the venv's
bin on PATH first. It is idempotent and installs only into the active venv.

```sh
export PATH="$PWD/.venv/bin:$PATH"          # venv cmake/python win
export CMAKE_POLICY_VERSION_MINIMUM=3.5     # harmless if the wheel is used
uv sync --locked --extra dev                # CPU default lock + pytest
uv pip install cmake                        # only needed by the sdist fallback
uv pip install --index-url https://download.pytorch.org/whl/cu128 torchvision==0.23.0
uv pip install pyannote-audio==3.4.0
uv pip install -f https://csukuangfj.github.io/kaldifeat/cuda.html \
    "kaldifeat==1.25.5.dev20250807+cuda12.8.torch2.8.0"
```

IMPORTANT (verified footgun): do NOT run the CUDA suite via `uv run` - it
re-syncs the env to the lockfile and UNINSTALLS the extras that are not in the
resolved set (kaldifeat, pyannote-audio, torchvision). Invoke the venv directly
(`.venv/bin/python`, `.venv/bin/pytest`) or use `uv run --no-sync`.

## Residual offline gate check (PASS)

senko loads pyannote segmentation from the bundled local file, never the gated
HF repo (§7). Confirmed: the file
`.venv/.../senko/models/pyannote_segmentation_3.0/pytorch_model.bin` is present
and senko's CUDA run ("Using pyannote VAD") completes with `HF_HUB_OFFLINE=1`.
No HF token step on any path.

## Installed versions (CUDA extras)

- torch 2.8.0+cu128, torchaudio 2.8.0+cu128 (from the CPU default lock)
- torchvision 0.23.0+cu128
- pyannote-audio 3.4.0
- kaldifeat 1.25.5.dev20250807+cuda12.8.torch2.8.0 (prebuilt vendor wheel)
- cmake 4.4.0 (uv pip; only the sdist fallback path uses it)
