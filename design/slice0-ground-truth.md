# Slice 0 - ground truth (environment probe + decision gate)

Documentation-only slice. Records what the Linux/CUDA port must build against: current-dep
failure modes, the senko CPU probe, the ASR route decision, Slice 1 pin targets, the Mac
reference fixture provenance, and the frozen JSONL contract. No code changes.

Machine: Ubuntu 24.04, RTX 4090 (compute capability 8.9), 32 cores, python 3.12.3,
uv 0.11.29. GPU is UNUSABLE pre-reboot: `nvidia-smi` -> `Failed to initialize NVML:
Driver/library version mismatch` (NVML 580.159 vs running kernel module). All probes below
are CPU and were run in throwaway venvs under `/tmp/slice0-probes/`, each built with
`python3.12 -m venv <dir>` + `pip install` (uv-independent, to characterise raw resolution);
the repo's pyproject/uv.lock were not touched. Full sha256 values for every artifact are in
section 9.

---

## 0. Repo-target decision - RESOLVED (boards #811-#812)

**Decision (macbase boards #811-#812, dev plan re-converged): this standalone
`fast_transcribe` repo IS the canonical port target.** A new **Slice 0.5 "backend
reconciliation"** (dev plan, commit 8152638) sits between this slice and Slice 1: it
union-merges the embedded copy's two-stream layer (`reprocess.py`/`recovery.py`/
`constants.py`/`muesli_backend.py` plus the PTS-anchor / recovery / Sortformer-retirement
commits) from the muesli backend copy into this repo, giving Slices 2-4 the `--stream both`
JSONL entry point to port and test. Slice 0.5 is a separate builder's job. The JSONL contract
in section 6 (extracted from `reference-stdout.jsonl`) is what Slice 0.5's stubbed smoke test
validates against; it does not change.

The divergence that motivated the decision, for the record:

- `reference-stdout.jsonl` was produced by `reprocess.py` (+ `recovery.py`, `constants.py`,
  `muesli_backend.py`), which exist ONLY in the muesli backend copy
  (`~/git/projects/muesli/backend/fast_mac_transcribe_diarise_local_models_only/src/diarise_transcribe/`).
- `git -C <repo> log --all -- src/diarise_transcribe/reprocess.py` -> empty; the file exists
  on no branch (`main`, `linux-cuda`, `origin/muesli-backend`,
  `origin/claude/mac-audio-capture-research-*`). The stale `origin/muesli-backend` branch has
  `muesli_backend.py` but still no `reprocess.py`.
- The standalone `cli.py` is a single-file `--in/--out` tool (human-readable text output, no
  `--stream`, no JSONL) and still ships the Sortformer backend `reprocess.py` calls retired.
- GROUND-TRUTH.md's "identical to the standalone repo" is inaccurate today; Slice 0.5 makes it
  true.

---

## 1. Current-dependency failure modes on Linux (PLAN AMENDMENT)

The plan expects `parakeet-mlx` and `coremltools` to FAIL TO INSTALL on Linux. They do not:
both install, then fail at import/runtime. Literal commands and results (venv
`/tmp/slice0-probes/macdeps-venv`):

```
$ pip install "parakeet-mlx>=0.2.0"
Successfully installed ... mlx-0.32.0 parakeet-mlx-0.5.2 ...        # exit 0
$ python -c "import mlx.core as mx"
ImportError: libmlx.so: cannot open shared object file: No such file or directory   # exit 1
$ python -c "from parakeet_mlx import from_pretrained"
  File ".../parakeet_mlx/parakeet.py", line 7, in <module>
    import mlx.core as mx
ImportError: libmlx.so: cannot open shared object file: No such file or directory   # exit 1

$ pip install "coremltools>=7.0"
Successfully installed ... coremltools-9.0 ...                       # exit 0
$ python -c "import coremltools"                                     # exit 0, but stderr:
Failed to load _MLModelProxy: No module named 'coremltools.libcoremlpython'
Failed to load _MLCPUComputeDeviceProxy: No module named 'coremltools.libcoremlpython'
Failed to load _MLGPUComputeDeviceProxy: No module named 'coremltools.libcoremlpython'
```

The `import coremltools` succeeds (exit 0), but the captured errors prove the native
`libcoremlpython` proxy fails to load on Linux, so proxy-backed CoreML model
execution/inference is unavailable (the `_MLModelProxy`/compute-device proxies are the runtime
backends). That is the exact scope of the evidence; I did not separately run a project CoreML
operation. It is enough for this slice: the Linux port does not use CoreML at all (senko's
Linux path is silero + torch, section 2), so the Mac-only `coremltools` import simply must not
be on the Linux import path. `asr.py:10` does `from parakeet_mlx import from_pretrained`;
`diarisation.py:14` does `import coremltools as ct` - both at module top.

**Amendment for Slice 2:** import-guard / make lazy the Mac-only imports in the shared modules
(`asr.py:10`, `diarisation.py:14`) so the Linux backend imports without them. Slice 1's
`sys_platform == 'darwin'` markers keep the packages off the Linux install set, but the
top-level imports must still be deferred into the darwin branch.

---

## 2. Senko probe (STOP gate)

Senko pinned commit (matches repo `uv.lock`): `f1bc30c2ff37d807eec91bec5246eea3fe2dcbe3`.
Dependency contract (from its `pyproject.toml` at the pin):
- Default (no extra): `torch==2.8.0; sys_platform != 'darwin'`, `silero-vad; sys_platform !=
  'darwin'`, `coremltools; sys_platform == 'darwin'`, plus numpy/scikit-learn/umap-learn/
  hdbscan/numba/llvmlite/pyyaml/soundfile/termcolor/psutil/colour-science.
- Extra `[nvidia]` (compute >= 7.5, our 4090): `torch==2.8.0 torchaudio==2.8.0
  torchvision==0.23.0 pyannote-audio==3.4.0 kaldifeat`.

Device/VAD (senko `diarizer.py` + `DOCS.md`): `device='auto'` on Linux -> `cuda` if
`torch.cuda.is_available()` else `cpu`; VAD auto = pyannote for cuda/coreml, **silero for
cpu**. So the **Linux CPU path = Silero VAD + CPU CAM++ embeddings + CPU clustering; no
pyannote, no coremltools, no HF-gated weights.** The CUDA path uses pyannote VAD (section 7) +
kaldifeat fbank + GPU CAM++.

**Probe (venv `/tmp/slice0-probes/senko-venv`).** Source = fixture `meeting/audio/mic.wav`
(sha256 in section 9), resampled: `ffmpeg -i mic.wav -ar 16000 -ac 1 mic_16k.wav`
(mic_16k.wav sha256 in section 9). Commands and captured results:

Install + import:
```
$ pip install "senko @ git+https://github.com/narcotic-sh/senko.git@f1bc30c2ff37d807eec91bec5246eea3fe2dcbe3"
Successfully installed senko-0.1.0 ...        # exit 0 (native CMake module built; scikit-build-core
                                              #         self-provisions cmake; gcc/g++/make present)
$ python -c "import senko"                    # exit 0
```

Executable probe script (`senko_probe.py`) and its VERBATIM captured stdout, run under the
locked pair:
```
$ python -c "import torch,torchaudio; print('VERSIONS torch',torch.__version__,'torchaudio',torchaudio.__version__,'cuda',torch.cuda.is_available())"
VERSIONS torch 2.8.0+cu128 torchaudio 2.8.0+cu128 cuda False
$ cat senko_probe.py
import time, math, senko
d = senko.Diarizer(device='cpu', warmup=True, quiet=True)
t=time.time(); res = d.diarize("/tmp/slice0-probes/audio16k/mic_16k.wav", generate_colors=False); dt=time.time()-t
segs = res["merged_segments"]
print("SEGMENTS", len(segs))
print("SPEAKERS_DETECTED", res.get("merged_speakers_detected"))
print("EXAMPLE", {k:round(v,2) if isinstance(v,float) else v for k,v in segs[0].items()})
ok = all(isinstance(s['speaker'],str) and s['speaker'] and math.isfinite(s['start']) and math.isfinite(s['end'])
         and 0<=s['start']<s['end'] for s in segs)
print("INVARIANTS_ALL_SEGMENTS", ok)
print("DIARIZE_SECONDS", round(dt,2))
$ python senko_probe.py
SEGMENTS 5
SPEAKERS_DETECTED 4
EXAMPLE {'speaker': 'SPEAKER_01', 'start': 10.08, 'end': 18.88}
INVARIANTS_ALL_SEGMENTS True
DIARIZE_SECONDS 2.53
# exit 0
```

- **STOP-gate verdict: CLEARED.** install / import / load / CPU inference all succeed on CPU
  under the locked pair. (The single mic voice over-split into 4 clusters, matching the Mac
  reference's known over-split; documented as MAY-differ. Constructor warmup, from an earlier
  run, was ~22-38 s one-time.)

**Single locked build pair, verified for BOTH stacks.** Lock target =
`torch==2.8.0+cu128` + `torchaudio==2.8.0+cu128` (the CUDA-capable build; runs on CPU
pre-reboot - the version line above prints `cuda False` - and on CUDA post-reboot). The senko
run above is under exactly this pair; the NeMo run (section 3) prints the same version line.

The FIRST senko attempt (before pinning torchaudio) failed at Diarizer construction:
`silero_vad/utils_vad.py -> import torchaudio -> torchaudio/_extension/utils._load_lib ->
torch.ops.load_library -> OSError: libcudart.so.13: cannot open shared object file`. The
mismatched versions are evidenced by the corrective reinstall's captured output (torch was
`2.8.0+cu128` throughout; `silero-vad` had floated torchaudio to `2.11.0`, a CUDA-13 build):
```
$ pip install --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --no-deps torchaudio==2.8.0
  Found existing installation: torchaudio 2.11.0        # <- the floated, mismatched build
  Uninstalling torchaudio-2.11.0:
    Successfully uninstalled torchaudio-2.11.0
Successfully installed torchaudio-2.8.0+cu128
$ python -c "import torch,torchaudio; print(torch.__version__, torchaudio.__version__)"
2.8.0+cu128 2.8.0+cu128
```
**Slice 1 pin target:** pin `torchaudio` to the SAME version and CUDA build as torch
(`2.8.0+cu128`); `torchaudio` is a REQUIRED Linux dependency (via `silero-vad`) though senko's
own `pyproject.toml` does not list it.

Senko weights are all BUNDLED in the pinned git package (`senko/models/`) and the
`silero-vad==6.2.1` package; senko downloads NOTHING from HF at runtime (hub cache stayed
empty). CPU path loads `senko/models/.../campplus_cn_en_common.pt` and, via
`load_silero_vad()` (`diarizer.py:110`), the JIT model `silero_vad/data/silero_vad.jit`.
Checksums in section 9.

---

## 3. ASR route decision gate

**Chosen route: `nemo-toolkit[asr]==2.7.3` loading `nvidia/parakeet-tdt-0.6b-v3`.**

- Version: `nemo-toolkit==2.7.3` (metadata `torch>=2.6.0`); verified on the locked
  `torch==2.8.0+cu128` pair.
- Loading API: `from nemo.collections.asr.models import ASRModel;
  m = ASRModel.restore_from("<local>/parakeet-tdt-0.6b-v3.nemo", map_location="cpu")` (device
  `"cuda"` post-reboot). `restore_from` resolves the subclass (`EncDecRNNTBPEModel`). No
  `from_pretrained`, no `trust_remote_code`, no remote code execution.
- CPU probe under the LOCKED pair (venv `/tmp/slice0-probes/nemo28-venv`: `torch 2.8.0+cu128`,
  `torchaudio 2.8.0+cu128`, `nemo-toolkit 2.7.3`; `torch.cuda.is_available()` False). Runnable
  commands:

Setup + executable probe (`nemo_probe.py`) with VERBATIM captured stdout (NeMo INFO logs on
stderr elided):
```
$ pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0 torchaudio==2.8.0
$ pip install "nemo-toolkit[asr]==2.7.3"                       # torch stays 2.8.0 (needs >=2.6.0)
$ ffmpeg -i system.wav -ar 16000 -ac 1 system_16k.wav          # sha256 in section 9
$ python -c "import torch,torchaudio,nemo; print('VERSIONS torch',torch.__version__,'torchaudio',torchaudio.__version__,'nemo',nemo.__version__,'cuda',torch.cuda.is_available())"
VERSIONS torch 2.8.0+cu128 torchaudio 2.8.0+cu128 nemo 2.7.3 cuda False
$ cat nemo_probe.py
import os, time, math
from huggingface_hub import snapshot_download
REV="7c35754d166cca382ad1e53e68b01e7c575f3a1d"
local=snapshot_download("nvidia/parakeet-tdt-0.6b-v3", revision=REV, allow_patterns=["*.nemo"])
print("SNAPSHOT_REV", REV)
from nemo.collections.asr.models import ASRModel
t=time.time(); m=ASRModel.restore_from(local+"/parakeet-tdt-0.6b-v3.nemo", map_location="cpu")
print("RESTORE_SECONDS", round(time.time()-t,1), "CLASS", type(m).__name__)
t=time.time(); h=m.transcribe(["/tmp/slice0-probes/audio16k/system_16k.wav"], timestamps=True)[0]
print("TRANSCRIBE_SECONDS", round(time.time()-t,1))
w=h.timestamp["word"]
print("TIMESTAMP_KEYS", list(h.timestamp.keys())); print("WORD_KEYS", sorted(w[0].keys()))
print("WORD0", w[0]); print("N_WORDS", len(w))
print("INVARIANTS_ALL_WORDS", all(math.isfinite(x["start"]) and math.isfinite(x["end"]) and x["start"]<=x["end"] for x in w))
print("TEXT_PREFIX", h.text[:60])
$ python nemo_probe.py
SNAPSHOT_REV 7c35754d166cca382ad1e53e68b01e7c575f3a1d
RESTORE_SECONDS 7.1 CLASS EncDecRNNTBPEModel
TRANSCRIBE_SECONDS 2.1
TIMESTAMP_KEYS ['timestep', 'char', 'word', 'segment']
WORD_KEYS ['end', 'end_offset', 'start', 'start_offset', 'word']
WORD0 {'word': 'Right,', 'start_offset': 2, 'end_offset': 8, 'start': 0.16, 'end': 0.64}
N_WORDS 90
INVARIANTS_ALL_WORDS True
TEXT_PREFIX Right, shall we make a start? This is the weekly pipeline ch
# exit 0
```

  `start`/`end` are seconds. Slice 2's conversion: `Word(text=w['word'], start=w['start'],
  end=w['end'])`.
- Immutability: `HfApi().model_info("nvidia/parakeet-tdt-0.6b-v3")` unauth succeeds, `gated`
  = False. Pin revision `7c35754d166cca382ad1e53e68b01e7c575f3a1d`; load from the
  revision-pinned `snapshot_download` local `.nemo`.

Rejected alternatives (with evidence):
- **`transformers`**: `AutoConfig.from_pretrained("nvidia/parakeet-tdt-0.6b-v3",
  revision="7c35754d166cca382ad1e53e68b01e7c575f3a1d")` on `transformers==4.57.6` raises
  `ValueError: ... model type 'parakeet_tdt' but Transformers does not recognize this
  architecture` (config `architectures: ['ParakeetForTDT']`). No load path -> no word
  timestamps.
- **`parakeet-mlx`**: `libmlx.so` import failure on Linux (section 1).
- **ONNX (in the pinned nvidia repo)**: `[s.rfilename for s in HfApi().model_info(
  "nvidia/parakeet-tdt-0.6b-v3", revision="7c357...").siblings]` contains no `.onnx`
  (`has .onnx: False`), so there is no in-repo ONNX artifact to pin. This is scoped to the
  official repo only - I did NOT survey third-party CTranslate2/ONNX exports; any such route
  would need its own immutable-artifact + word-timestamp evidence and is out of scope for this
  slice. NeMo is the in-repo, revision-pinnable route, so it wins regardless.

---

## 4. Slice 1 pin targets

- **python:** `>=3.10,<3.14` (senko's range); build/test on 3.12 (box 3.12.3).
- **torch:** `torch==2.8.0` from `https://download.pytorch.org/whl/cu128` (`2.8.0+cu128`).
  Satisfies senko (`==2.8.0`) and nemo (`>=2.6.0`); ran on CPU for both (sections 2, 3). An
  unconstrained `nemo-toolkit[asr]` install floats torch to 2.13.0+cu130 - the lock MUST pin.
- **torchaudio:** `torchaudio==2.8.0+cu128` (identical version and CUDA build as torch).
  Required on Linux via `silero-vad`.
- **nemo-toolkit[asr]:** `==2.7.3`. Pulls transformers 4.57.6, pyannote.core/metrics/database
  (NOT the gated `pyannote-audio` model), lhotse, sentencepiece, numba 0.66.0, the cu12 wheel
  stack. Within the overlay's accepted Torch/NeMo footprint.
- **senko:** git pin `f1bc30c2ff37d807eec91bec5246eea3fe2dcbe3`, default extra (Silero CPU).
- **silero-vad:** `==6.2.1`.

**CUDA extras: resolved by the overlay Scope ruling (constraint #5, 19 Jul).** The Linux CPU
default set - platform-neutral deps + `nemo-toolkit[asr]==2.7.3` + `torch==2.8.0+cu128` +
`torchaudio==2.8.0+cu128` + senko (default) + `silero-vad==6.2.1` - is fully lockable and
testable now (sections 2, 3). Senko's CUDA path needs the `[nvidia]` extra, and one member,
`kaldifeat`, does NOT build here. Reproducible probe (venv `/tmp/slice0-probes/nemo28-venv`,
torch 2.8.0+cu128):
```
# (a) no binary wheel exists for kaldifeat 1.24:
$ pip download --only-binary=:all: kaldifeat==1.24
ERROR: Could not find a version that satisfies the requirement kaldifeat==1.24
       (from versions: 0.2, 0.3, 1.1, 1.2, 1.3, 1.4)   # wheels only for a few old versions, none 1.24
ERROR: No matching distribution found for kaldifeat==1.24
# exit 1
# (b) so the NORMAL install falls back to the sdist build, which fails:
$ pip install kaldifeat==1.24
  Building wheel for kaldifeat (pyproject.toml): finished with status 'error'
      sh: 4: cmake: not found                          # <- causal error
  ERROR: Failed building wheel for kaldifeat
# exit 1
```
So even a normal `pip install` (no `--no-binary`) reaches the failure: there is no wheel, so
pip builds the sdist, and kaldifeat 1.24 invokes `cmake` directly from its build (unlike senko,
it does not self-provision it) with no system `cmake` on PATH. This is a missing-build-tool
failure - the build never reached a CUDA step, so whether kaldifeat then needs `nvcc`/CUDA is
UNVERIFIED.

Per the overlay Scope ruling (19 Jul): the frozen "`uv sync` is the whole install contract"
applies to the **CPU DEFAULT install only** - which is the set above, fully lockable now.
**CUDA provisioning (the senko `[nvidia]` extras incl. kaldifeat) is an explicit, scripted
GPU-gate step documented in `design/gpu-provisioning.md`** - try uv-installable tools first
(e.g. `pip cmake` to satisfy the `cmake: not found`), and anything needing apt/sudo or a CUDA
toolkit is a named David-step executed at reboot, never assumed. So kaldifeat's
non-installability does NOT block the Linux default lock; it is scoped into the
GPU-provisioning step.

---

## 5. Mac reference fixture provenance

On disk at `tests/fixtures/golden/`, verified via `sha256sum` (values in section 9; the audio
hashes match GROUND-TRUTH.md).

- Audio: PCM s16le, 24 kHz, mono, 72.40 s each (`ffprobe`). Fully synthetic (Gemini TTS).
- Exact command as recorded in GROUND-TRUTH.md (Mac-side; `<this-dir>`/`$BACKEND` are Mac
  paths, unresolvable from this Linux box):
  `BACKEND=/Users/david/git/ai-sandbox/projects/muesli/backend/fast_mac_transcribe_diarise_local_models_only;
  PYTHONPATH=$BACKEND/src NUMBA_CACHE_DIR=/tmp/muesli-numba-cache HF_HUB_OFFLINE=1
  "$BACKEND/.venv/bin/python" -m diarise_transcribe.reprocess "<this-dir>/meeting"
  --stream both --diar-backend senko --verbose > reference-stdout.jsonl 2> reference-stderr.log`
  (exit 0).
- Mac dependency versions as recorded (a SUBSET, not a full freeze): python 3.12.13,
  parakeet-mlx 0.5.0, senko 0.1.0, mlx 0.30.3, numpy 2.3.5, macOS arm64.
- **Fixture-receipt-gate items for macbase (block Slice 1 relying on the fixture):** (a) the
  Mac muesli-repo commit that produced it; (b) the full Mac `pip freeze`; (c) the exact
  `mlx-community/parakeet-tdt-0.6b-v3` revision the run actually loaded (section 7 - currently
  UNKNOWN, not assumed); (d) the historical Mac senko git commit + its CoreML artifact hashes
  (section 7).

---

## 6. Frozen JSONL contract (from `reference-stdout.jsonl`)

Derived by this executable command (run in `tests/fixtures/golden/`):

```
python3.12 -c '
import json
lines=open("reference-stdout.jsonl").read().splitlines()
ev=[]; poll=0
for ln in lines:
    s=ln.strip()
    if not s: continue
    try:
        o=json.loads(s); ev.append((o.get("type"),o.get("stage"),o.get("stream"),
                                     [k for k in o if k not in ("type","stage","stream")]))
    except Exception: poll+=1
res=json.loads(lines[-1])
for e in ev: print(e)
print("POLLUTION_LINES:",poll)
print("OUTER_KEYS:",list(res.keys()))
print("TURN_KEYSETS:",{tuple(t.keys()) for t in res["turns"]})
print("N_TURNS:",len(res["turns"]),"STREAMS:",sorted({t["stream"] for t in res["turns"]}))
print("VALUE_TYPES:",{k:type(res["turns"][0][k]).__name__ for k in res["turns"][0]})'
```

Exact captured output (event sequence + pollution count):

```
('status', 'preparing', None, [])
('status', 'transcribing', 'system', [])
('status', 'diarizing', 'system', [])
('status', 'merging', 'system', [])
('status', 'recovering', 'system', ['windows'])
('status', 'transcribing', 'mic', [])
('status', 'diarizing', 'mic', [])
('status', 'merging', 'mic', [])
('status', 'recovering', 'mic', ['windows'])
('status', 'complete', None, [])
('result', None, None, ['turns', 'speakers', 'duration'])
POLLUTION_LINES: 24
```

Second command, evidencing EVERY reference-derived invariant across ALL turns (not just
`turns[0]`):

```
$ python3.12 -c '
import json, math
lines=open("reference-stdout.jsonl").read().splitlines()
res=json.loads(lines[-1]); T=res["turns"]
print("N_TURNS", len(T))
print("ALL_TURN_KEYSETS", {tuple(t.keys()) for t in T})
print("VALUE_TYPES_ALL_TURNS_CONSISTENT", all({k:type(t[k]).__name__ for k in t}=={"speaker_id":"str","stream":"str","t0":"float","t1":"float","text":"str"} for t in T))
print("SPEAKER_PREFIXES", sorted({t["speaker_id"].split(":")[0] for t in T}))
print("STREAMS", sorted({t["stream"] for t in T}))
print("TIME_FINITE_AND_ORDERED_ALL", all(math.isfinite(t["t0"]) and math.isfinite(t["t1"]) and t["t0"]<=t["t1"] for t in T))
print("DURATION", res["duration"], type(res["duration"]).__name__, "FINITE", math.isfinite(res["duration"]))
print("SPEAKERS_SORTED_EQ", res["speakers"]==sorted(res["speakers"]))
import re
print("OUTER_KEYS", list(res.keys()))
print("OUTER_KEYSET_EXACT", set(res.keys())=={"type","turns","speakers","duration"})
print("CONTAINERS", res["type"], isinstance(res["turns"],list), isinstance(res["speakers"],list), isinstance(res["duration"],float))
pat=re.compile(r"^(mic|system):SPEAKER_\d+$")
print("ALL_SPEAKERS_MATCH", all(pat.match(s) for s in res["speakers"]), "ALL_TURN_IDS_MATCH", all(pat.match(t["speaker_id"]) for t in T))
wv=[json.loads(l).get("windows") for l in lines if l.strip().startswith("{") and json.loads(l).get("stage")=="recovering"]
print("RECOVERING_WINDOWS_VALUES", wv, "TYPES", sorted({type(x).__name__ for x in wv}))'
N_TURNS 13
ALL_TURN_KEYSETS {('speaker_id', 'stream', 't0', 't1', 'text')}
VALUE_TYPES_ALL_TURNS_CONSISTENT True
SPEAKER_PREFIXES ['mic', 'system']
STREAMS ['mic', 'system']
TIME_FINITE_AND_ORDERED_ALL True
DURATION 72.24 float FINITE True
SPEAKERS_SORTED_EQ True
OUTER_KEYS ['type', 'turns', 'speakers', 'duration']
OUTER_KEYSET_EXACT True
CONTAINERS result True True True
ALL_SPEAKERS_MATCH True ALL_TURN_IDS_MATCH True
RECOVERING_WINDOWS_VALUES [0, 0] TYPES ['int']
```

So the frozen contract, ALL reference-derived from the captures above: 11 JSON events in the
order shown (system stream fully processed, then mic); each `recovering` event carries
`windows` as an `int` (values `[0, 0]` here). Final result line outer key-set exactly
`{type, turns, speakers, duration}`; `speakers` sorted, each `"<stream>:SPEAKER_XX"`;
`duration` a finite float (72.24). All 13 turns share EXACTLY the key-set
`{speaker_id, stream, t0, t1, text}` with consistent value types (`speaker_id`/`stream`/`text`
str, `t0`/`t1` float), speaker prefixes and streams both `{mic, system}`, and every
`t0 <= t1` finite. 24 interleaved non-JSON pollution lines (`Loading ASR model:`,
`Detected N speakers...`, the box-drawing tree, from `ASRModel`/`SenkoDiarizer`/senko).

SOURCE-derived (NOT present in this happy-path fixture, read from `reprocess.py`): the
`recovering` event ALSO carries `spans` (str) when `windows > 0` (`reprocess.py:166`); failure
paths emit `{"type":"error","message":<str>}` (`reprocess.py:388/406/424/453`). Flagged as
source-derived so Slice 4 does not assert them from the fixture.

**PLAN AMENDMENT - two DISTINCT assertions:**
- **Reference comparison (schema only):** parse the Mac file line-by-line, SKIP non-JSON lines,
  and compare the complete ordered event tuples `(type, stage, stream, sorted-extra-key-set)`
  - including each `recovering` event's `windows` key and int type - plus the final-line outer
  key-set, per-turn key-set, and stream-label set. (Event names alone would not enforce the
  frozen stream above.)
- **Linux output validation (Slice 4):** the NEW Linux subprocess MUST emit pure JSONL (Slice 2
  forces NeMo/torch/senko chatter to stderr), so Slice 4 asserts every line of the LINUX
  subprocess stdout parses as JSON. That assertion is against the Linux output, never against
  the polluted Mac reference file.

---

## 7. HF gates and model-artifact pinning

- `nvidia/parakeet-tdt-0.6b-v3` (Linux ASR): **ungated**; pin revision
  `7c35754d166cca382ad1e53e68b01e7c575f3a1d`; load the revision-pinned `.nemo` (no
  `trust_remote_code`).
- `mlx-community/parakeet-tdt-0.6b-v3` (Mac ASR): **CONFIRMED revision
  `ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15`** - macbase (board #825) verified this is the
  single revision of the repo in the Mac's HF cache, so the fixture run loaded it
  unambiguously (it matches the repo's current HEAD). macbase also confirmed there are NO
  senko / pyannote / CAM++ repos in the Mac's HF cache, corroborating the bundled-weights
  finding (below) from the Mac side.
- Senko - ALL weights BUNDLED in the pinned git package; senko downloads NOTHING from HF at
  runtime (hub cache empty across the CPU run). `pyannote/segmentation-3.0` is HF-gated
  (`gated: auto`) but senko does NOT touch the gated repo: `config.py:55`
  `PYANNOTE_SEGMENTATION_PT_MODEL_PATH = MODELS/'pyannote_segmentation_3.0/pytorch_model.bin'`
  (a bundled local file); `diarizer.py:92` `Model.from_pretrained(<that local path>)`. **No HF
  token / gate acceptance is required on any path.** The `[nvidia]` extra's
  `pyannote-audio==3.4.0` is only the loader library. Residual GPU-gate check: confirm that
  local-path load stays offline.
- The senko CoreML artifacts (`pyannote_segmentation.mlmodelc`, `camplusplus_batch16.mlpackage`)
  are senko's Mac diarisation weights. The section-9 checksums for them are from THIS box's
  senko install at commit `f1bc30c...`; I have NO Mac-side evidence that the fixture run used
  these same bytes (GROUND-TRUTH.md records only `senko 0.1.0`, not the git commit). So do NOT
  attribute the section-9 CoreML hashes to the reference run - the historical senko commit +
  its CoreML artifact hashes are an additional fixture-receipt-gate item for macbase.

Immutability rule, and its scope. For the LINUX backends being built this port - the only
artifacts the port actually loads - immutability holds: Linux ASR = HF revision `7c357...`
(revision-pinned `.nemo`); all senko Linux weights = senko git commit `f1bc30c...` + the
section-9 sha256s (bundled, nothing fetched from HF). A route offering neither mechanism is
rejected (section 3 ONNX). The MAC reference's model provenance is now partly closed: its ASR
revision is CONFIRMED (`ed2b7e8c...`, macbase board #825), and macbase confirmed no
senko/pyannote/CAM++ HF repos in the Mac cache (so the Mac used bundled senko weights too). The
one residual is the Mac's senko GIT COMMIT + bundled-byte hashes (unrecorded; `senko 0.1.0`
alone does not fix the commit) - a macbase gate item, section 5. Until that lands, the
fixture's byte-level senko provenance is not reproducible; its authority for Slice 4 rests on
its checksummed OUTPUT (the `reference-stdout.jsonl` schema in section 6 + the section-9 audio
hashes), which is all Slice 4 asserts anyway.

---

## 8. Plan amendments (summary)

1. **Slice 2:** Mac deps fail at import/runtime, not install - import-guard/lazy-import the
   shared modules (`asr.py:10`, `diarisation.py:14`) (section 1).
2. **Slice 4:** two distinct assertions - Mac reference filtered to JSON + schema-compared;
   Linux subprocess stdout asserted pure-JSONL (section 6).
3. **GPU gate pending job:** NO pyannote HF-token step - senko loads pyannote from a bundled
   local file, not the gated repo (section 7). Only residual check: that load stays offline.
4. **Slice 1 CUDA provisioning: resolved by the overlay Scope ruling** (section 4): the torch
   pin is resolved (`torch==2.8.0+cu128`, verified for senko AND nemo), and per the overlay's
   19-Jul ruling the frozen `uv sync` contract scopes to the CPU default (lockable now); the
   senko `[nvidia]` CUDA extras (kaldifeat, which fails with `cmake: not found`) are a scripted
   GPU-gate step in `design/gpu-provisioning.md` (pip cmake first; apt/sudo/CUDA-toolkit =
   named David-steps at reboot). Not a blocker on the CPU default lock.
5. **Fixture-receipt gate:** macbase to supply, before Slice 1 relies on the fixture: the Mac
   muesli-repo commit; the full Mac `pip freeze`; the exact `mlx-community/parakeet-tdt-0.6b-v3`
   revision loaded; and the historical Mac senko git commit + its CoreML artifact hashes
   (sections 5, 7).

---

## 9. Artifact checksums (sha256)

Fixture (repo `tests/fixtures/golden/`):
- `meeting/audio/mic.wav` = `4dc15e666abd69269dc9cc37cc910a549f167ad6e918c7c8988b1085b489338a`
- `meeting/audio/system.wav` = `3050ea79eaaa277ea1faa7ba0cf5fb7684dfadafa6315cda2951e9258816f298`
- `reference-stdout.jsonl` = `a745e837db41987f2eb36ed82da98275ce70f1040e2cb2eb39c7e0cb259f3802`
- `reference-stderr.log` = `664423050f0e5d3cfeb0f31857532beb183720459d7ade76eca035396aab6f77`

Resampled probe audio (throwaway):
- `mic_16k.wav` = `72c30eb4416b25e0477d2d05ae1acf331f3146c792d1cff101ddbcbd764a15de`
- `system_16k.wav` = `902ab80af3336e7666eeb9cee30b8c892b14af657219d068eaede7f5da600be9`

Senko bundled weights (pinned by senko git commit `f1bc30c2ff37d807eec91bec5246eea3fe2dcbe3`):
- CPU CAM++ `senko/models/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt`
  = `92f29b94e6948786a26778c9e302525d185bb08c8b9f5252ed98776902840199`
- CUDA CAM++ (traced) `senko/models/camplusplus_traced_cuda_optimized.pt`
  = `21dd6f3055f94c2af2a9b6fa59091c763b5274e9fd81009e59f8b5c7bf9229c3`
- pyannote seg (torch) `senko/models/pyannote_segmentation_3.0/pytorch_model.bin`
  = `da85c29829d4002daedd676e012936488234d9255e65e86dfab9bec6b1729298`
- CoreML `senko/models/pyannote_segmentation.mlmodelc/coremldata.bin`
  = `4a450ea1b053b9eb7eef0cab6971018076600840c7e246d064e7c5387f456c98`
- CoreML `senko/models/pyannote_segmentation.mlmodelc/weights/weight.bin`
  = `0266f4ad4d843ecf31ef9220ad6b80616b3ec64a4404b64f3ea0371554e236ec`
- CoreML `senko/models/camplusplus_batch16.mlpackage/Data/com.apple.CoreML/weights/weight.bin`
  = `bad93198b2d9431ec2a17a30a983830434cb008aaa56c814ddd371449f724d46`

Silero VAD (from `silero-vad==6.2.1`; `load_silero_vad()` loads the `.jit`):
- `silero_vad/data/silero_vad.jit` = `e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720`
- `silero_vad/data/silero_vad.onnx` = `1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3`


## Adjudication at the convergence cap (boss, 19 Jul 2026)

Five GPT-5 rounds run; round 5 returned 4 residual findings. Boss ruling per
the convergence loop: commit as-is, record residuals. Finding 1 (stale
revision-unknown claims contradicting §7) fixed in this commit. Recorded
unresolved, for any later documentation pass - none affects a probe result or
a decision:

- R5-2: §6 comparison tuple carries key names only, cannot assert the
  `windows` value type; add isinstance or (key, value-type) pairs.
- R5-3: §5 exact command still contains `<this-dir>`/unstated cwd; either
  materialize the real Mac path or label as template in the receipt gate.
- R5-4: the intro's "full sha256 for every artifact" overclaims vs §9 (the
  .nemo file's checksum is not listed; its immutable HF revision is).

Basis for shipping: every probe result and decision in this note is
independently corroborated (macbase's Mac-side HF-cache check, the gate's
byte-identical golden re-run, captured command outputs on disk).
