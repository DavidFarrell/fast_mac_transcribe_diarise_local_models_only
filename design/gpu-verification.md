# GPU verification - linux-cuda port acceptance gate

Run on whiteboxlinux after David's reboot cleared the driver/module mismatch.
This is the port's final acceptance step (`linux-cuda-dev-plan.md` "GPU gate").

## Environment

- GPU: NVIDIA GeForce RTX 4090, 24 GiB, compute 8.9.
- Driver 580.159.03, CUDA 13.0 (nvidia-smi table renders; NVML healthy).
- Python 3.12.3, uv 0.11.29.
- torch 2.8.0+cu128, torchaudio 2.8.0+cu128, `torch.cuda.is_available()` True,
  device name "NVIDIA GeForce RTX 4090".
- CUDA extras provisioned per `design/gpu-provisioning.md` (kaldifeat via the
  prebuilt vendor wheel; no toolkit David-step).
- Branch `gpu-gate` off `linux-cuda` @ aa21a4d.

## Preflight - PASS

- `nvidia-smi` renders the RTX 4090 table (blocker actually clear).
- `torch.cuda.is_available()` → True in the repo venv.

## Per-engine CUDA proof - PASS

Both backends demonstrably run on CUDA (single-stream and in isolation).

- **ASR (NeMo parakeet-tdt-0.6b-v3):** after `load_model(..., "cuda")`,
  `next(model.parameters()).device` = `cuda:0`. A warm transcribe of the 72 s
  system stream runs in 0.11 s; `torch.cuda.max_memory_allocated` ≈ 4.8 GiB.
- **Diariser (senko):** `SenkoDiarizer(device="cuda")` logs `Using device: cuda`
  and `Using pyannote VAD`. The pyannote-VAD branch is CUDA-only - senko's CPU
  path uses Silero VAD (`slice0-ground-truth.md` §2) - so that log line alone
  proves the CUDA path was taken, corroborated by kaldifeat's CUDA Fbank and the
  GPU-memory climb during the run. Produces a valid nonempty segmentation.
- **Offline residual check (§7):** senko loads pyannote from its bundled local
  file under `HF_HUB_OFFLINE=1`; no HF token / gated repo on any path.

Single-stream end-to-end on CUDA (`reprocess_stream(..., device="cuda")`,
`--stream system` and `--stream mic` each alone) → exit 0, full JSONL contract,
result line emitted.

## Golden `--stream both` on CUDA - PASS (after the co-residency fix below)

The flagship golden command
`python -m diarise_transcribe.reprocess <meeting> --stream both --diar-backend
senko --verbose` (device auto → CUDA) completes: exit 0, wall 24.6 s, peak
7778 MiB, 11 pure-JSONL lines in the exact frozen event order
(preparing → system{transcribe,diarize,merge,recover} →
mic{transcribe,diarize,merge,recover} → complete → result), outer key-set
`{type,turns,speakers,duration}`, both streams present. `Loading ASR model`
appears twice in stderr - direct evidence the mic stream reloaded a fresh
(unpoisoned) model per the fix. `pytest -m integration` is green on CUDA
(3 passed): `test_golden_contract` (the auto→cuda two-stream run) plus the two
CPU-pinned smokes (`test_cpu_transcription_smoke`, `test_real_senko_cpu_run`).

### The bug this fix closes (RCA)

Before the fix, the same command **aborted reproducibly**: the system stream
fully processed on CUDA, then the mic stream's transcribe step died with
SIGABRT (exit 134):

```
terminate called after throwing an instance of 'c10::AcceleratorError'
  what():  CUDA error: an illegal memory access was encountered
```

Not hardware/driver: GPU health was clean after the crash (no ECC errors, no
remapped rows, no Xid), and every component works in isolation.

Root-cause isolation (all runs device=cuda):

| # | Scenario | Result |
|---|---|---|
| - | each stream alone (`--stream system` / `--stream mic`) | OK |
| A/E1 | reuse one NeMo model, 2 transcribes, **no senko between** | OK |
| E2 | reuse one NeMo model, **senko-CUDA diarize between** the two transcribes | **CRASH** at 2nd transcribe |
| E3 | reuse one NeMo model, **senko on CPU** between the two transcribes | OK |
| C | two senko-CUDA diarizes back to back | OK |
| B | NeMo sys → senko-CUDA sys → **fresh** NeMo mic | OK |
| D1/D2 | real `reprocess_stream` x2 (with / without recovery) | **CRASH** at mic transcribe |
| heal | E2 + `torch.cuda.synchronize()` / `empty_cache()` / del-senko+gc before the 2nd transcribe | **CRASH** (no in-place heal exists) |

Root cause: senko's **CUDA** diarization (kaldifeat custom CUDA kernels +
pyannote on GPU) irrecoverably corrupts the CUDA state of any **co-resident**
model. A NeMo model transcribed after a CUDA diarisation then dies on its next
forward pass. Only a model **loaded after** the diarisation (scenario B) is
clean; no synchronize/empty_cache/free heals a poisoned one. All three legs are
necessary: the co-resident model is reused across the diarisation (B is fine),
the diarisation is on CUDA (E3 CPU is fine), and a transcribe follows it.

There are **two** sites where a co-resident model is transcribed after a CUDA
diarisation, both real on this port:
1. **Cross-stream:** `asr.py`'s process-level `_model_cache` hands the mic
   stream the same NeMo instance the system stream used (poisoned by the
   system stream's senko). This is what the golden fixture hit (its 0-window
   recovery masked site 2).
2. **Within-stream recovery:** `reprocess._run_recovery_pass` re-transcribes
   wordless-segment windows on the **same** `asr` instance, after that stream's
   senko already ran. Latent on the clean synthetic fixture (0 wordless
   windows) but reached on any real audio with a wordless diar segment - E2 is
   exactly this pattern and crashes.

Scope: CUDA only. All CPU paths and single-stream CUDA runs were already fine.

### The fix (this branch)

A CUDA-poison generation in `asr.py`, covering both sites without touching the
frozen `reprocess`/`cli` orchestration:

- `asr.poison_cuda_asr()` bumps a module generation. `senko_diarisation.diarise`
  calls it after any diarisation that **ran on CUDA** (`_diarization_used_cuda`:
  `cuda`, or `auto` when torch reports CUDA; never `cpu`/`coreml`).
- `ASRModel` records the generation its model was loaded at and **reloads a
  fresh model** before transcribing if the generation has advanced since -
  catching both the cross-instance (cache) and same-instance (recovery) reuse.

CPU/darwin behaviour is byte-identical: on those paths senko never runs on CUDA,
`poison_cuda_asr` is never called, the generation stays 0, and the reload branch
is never taken - so the cache path resolves no device and imports no torch (the
deterministic suite still runs in ~0.9 s). Verified: 143 deterministic tests
pass (incl. 10 new in `tests/test_cuda_poison.py` asserting reload-on-poison at
both sites and that CPU/coreml/auto-without-cuda do NOT poison), and the CUDA
golden + integration suite pass above.

Cost: on CUDA the ASR model reloads (~7 s) after each senko run that precedes a
transcribe (once for the two-stream fixture: the mic reload). Correctness over
speed; a heavier isolation/subprocess approach would avoid the reload but was
rejected as out of the overlay bounds.

Chosen over the narrower shapes considered: blanket `_model_cache.clear()`
(taxes the CPU path and misses site 2); device-aware cache-skip on CUDA (misses
site 2); a lightweight CUDA heal (empirically none works - "heal" row above);
recovery-uses-a-fresh-model inside `reprocess.py` (works but edits frozen
orchestration and needs its own CPU guard).

### Upstream note

The underlying defect is in senko's CUDA path (kaldifeat / pyannote co-residency
corrupting other CUDA contexts in the same process), not in this port - worth
reporting to `narcotic-sh/senko` eventually with this minimal repro (transcribe
a NeMo model → `SenkoDiarizer(device="cuda").diarise` → transcribe the same NeMo
model → illegal memory access). The Mac never hits it: senko there uses the
CoreML path, not CUDA. This port's fix is a robust local mitigation regardless
of whether the upstream bug is fixed.

## Timings (72 s synthetic fixture, same box, same day)

| Measurement | CPU | CUDA |
|---|---|---|
| ASR inference only (warm transcribe, system stream) | 1.84 s | 0.11 s (~17x) |
| Full single-stream pipeline wall (load + infer) | 15.9 s | 15.8 s |
| Golden `--stream both` wall (2× load+infer, incl. mic reload) | - | 24.6 s |

The full-pipeline wall time is load-dominated on this short fixture (NeMo
`restore_from` ≈ 7 s + senko init), which masks the GPU inference win; the
inference-only pair shows it. A longer input would widen the wall-time gap.
The two-stream wall includes the fix's one extra ~7 s ASR reload for the mic
stream.

## Provisioning summary

Clean, no David-step. kaldifeat's sdist genuinely needs a CUDA toolkit (nvcc),
but the vendor's prebuilt `kaldifeat==1.25.5.dev20250807+cuda12.8.torch2.8.0`
wheel matches the stack and installs without a build. Full detail +
reproducible script: `design/gpu-provisioning.md`, `design/provision-cuda.sh`.
