# Build overlay - Linux/CUDA port (linux-cuda branch)

Project-specific layer on top of the global good-taste doctrine. Read before
any slice.

## What this run is

Make `diarise-transcribe` run on David's Linux whitebox (Ubuntu 24.04, RTX
4090, 32 cores) with the SAME CLI contract it has on the Mac, so muesli-merge,
fast-transcribe and blogify skill texts work unchanged on both machines.

## Hard constraints

1. **The CLI contract is frozen.** `reprocess` / `diarise-transcribe` flags
   (`--stream both`, `--diar-backend`, ...), the JSONL event stream, and the
   final result line's `turns` schema (`speaker_id`/`stream`/`t0`/`t1`/`text`)
   must not change. The Mac and its skills are the reference consumers.
2. **One codebase, two platforms.** The Mac keeps working from this same repo:
   platform-conditional dependencies via PEP 508 `sys_platform` markers in the
   DEFAULT dependency list (no new extras on either platform - the existing
   install contract is `uv sync` on both machines), runtime backend selection.
   Never break the MLX path - it cannot be tested on this box, so it must be
   isolated, not modified beyond import-guarding.
3. **Per-channel diarisation is the quality lever.** Mic and system streams
   are transcribed/diarised separately, never mixed. Preserve this.
4. **Models:** ASR = nvidia parakeet-tdt-0.6b-v3 (native NVIDIA model; CUDA
   route). Diarisation = senko (already has a `device='cuda'` path - verify
   before replacing; only fall back to raw pyannote+CAM++ if senko's CUDA path
   is broken).
5. **GPU is temporarily unavailable** (driver/module mismatch until David
   reboots). Everything must ALSO run on CPU: build + verify CPU-first, keep
   the device flag, GPU verification is the final pending step.
   **Scope ruling (19 Jul, after the kaldifeat finding): the frozen
   "uv sync is the whole install contract" applies to the CPU DEFAULT
   install only.** CUDA provisioning (senko's [nvidia] extras incl.
   kaldifeat, which needs cmake and possibly toolkit pieces uv cannot
   supply) is an explicit, scripted GPU-gate step documented in
   design/gpu-provisioning.md - try uv-installable tools (e.g. pip cmake)
   first; anything needing apt/sudo is a named David-step executed at
   reboot time, never assumed.
6. **Secrets/gates:** pyannote weights may be HF-gated. No tokens in the repo.
   If a gate is hit, stop and route the token step to David via yoshimi.

## Bounds (escalate rather than exceed)

- No rewrite of merge.py/cli.py logic beyond what backend selection needs.
- No new services, no config frameworks, no plugin registries. Backend choice
  is a function, not an architecture.
- Torch/NeMo dependency footprint is acceptable; a second model-serving
  process is not.

## Verification bar

- Existing deterministic tests pass on Linux from the DEFAULT dependency set:
  `uv sync --locked` then `uv run pytest -m "not integration"`. The Linux set
  is expected to resolve a CUDA-capable torch build (cu12x wheel) that still
  runs on CPU pre-reboot.
- Golden run: a stereo/two-stream sample processed end-to-end; JSONL schema
  identical to Mac reference output (macbase can supply a reference run).
- blogify's dependency set (yt-dlp, ffmpeg) present and its skill steps
  executable on this box.
