# Dev plan - Linux/CUDA port

Slices, each built by a fresh builder through the convergence loop. Integration
branch is `linux-cuda`; each slice branches from the LATEST integrated
`linux-cuda` and merges back into it after review. `main` stays Mac-canonical;
nothing merges to `main` this run (see GPU gate).

## Slice 0 - environment probe + decision gate (documentation-only commit)

The only repo change this slice makes is committing its ground-truth note to
`design/slice0-ground-truth.md`. Contents required:

- uv venv probe on Linux: which current deps fail to install (parakeet-mlx,
  coremltools expected) - recorded with exact errors.
- Senko probe: install, import, model load AND a real CPU inference on a
  short sample. **STOP if ANY of those fails on CPU** - CPU operation is
  mandatory pre-reboot; revise this plan for a direct pyannote+CAM++ fallback
  before any implementation, and do not build both paths speculatively. A
  CUDA-path failure is NOT a stop - it is pending until the GPU gate. Record
  what weights senko downloads and whether any HF gate is hit.
- ASR route decision gate: pick ONE package route for
  nvidia/parakeet-tdt-0.6b-v3 with word-level timestamps. Record: chosen
  package + pinned version, the model-loading API, the timestamp result shape,
  a CPU probe result, and one-line rejection reasons for the alternatives
  considered. The plan's later slices use this record, not a re-litigation.
- Pin targets discovered here: compatible torch / (nemo|alternative) / python
  ranges for Slice 1.
- Mac reference fixture: requested from macbase (board #807). **Receipt is a
  gate for Slice 1** - no implementation slice starts until it is on disk and
  recorded in the ground-truth note with: source repo commit on the Mac, audio
  sha256, stdout sha256, exact command line, dependency versions, stderr
  captured separately.
- Contract freeze: read cli.py + the Mac reference stdout and DOCUMENT in the
  ground-truth note the exact JSONL event sequence (names, order) and the
  complete outer schema of the final result line, including that every turn
  object has exactly the five frozen keys. Slice 4 asserts THIS documented
  contract - it is not re-derived later.
- Model artifact pinning: every backend (including artifacts senko downloads
  internally) must load an immutable revision OR a checksum-verified local
  snapshot - record the revision hashes/checksums here. If a candidate route
  offers neither mechanism, REJECT it in this slice.

**Secrets rule (applies to every slice):** credentials (e.g. HF token) may be
supplied only via the Hugging Face credential store or an environment
variable - never CLI args, repo files, test fixtures, design notes, or
captured logs. Never enable remote code execution to load a model.

## Slice 1 - packaging split

pyproject: current Mac dependencies STAY in the default dependency list,
gated behind `sys_platform == 'darwin'` markers - no new extra required on
Mac, existing install contract unchanged. Linux default install = the
platform-neutral deps + the chosen ASR route + senko (works CPU-only
pre-reboot; CUDA usable after). Senko git dependency pinned to an immutable
commit. Torch/NeMo/python pinned to the ranges recorded in slice 0.
Regenerate the universal uv lockfile.

Marker spec: Mac-only deps carry `; sys_platform == 'darwin'`, Linux ASR
route deps carry `; sys_platform == 'linux'`, shared deps unmarked. No new
extras on either platform - `uv sync` remains the whole install contract.

This slice also adds the pytest config (moved here from Slice 2 so its own
exit command is self-contained): `addopts = -m "not integration"` in
pyproject `[tool.pytest.ini_options]`.

Exit commands (all outputs shown in review):
- `uv sync --locked` on this box (Linux) - clean.
- `uv export --format requirements-txt` twice: once as-is (linux) and once
  with `--python-platform` / marker evaluation for macosx_arm64 - shown to
  include the Mac deps only under darwin and the Linux route only under linux.
- `uv run pytest -m "not integration"` - green on Linux.
- **macbase gate (before Slice 2 merges, may run in parallel with Slice 2
  dev):** macbase runs `uv sync --locked`, the deterministic tests, and the
  reference fixture command on the Mac from this branch - proving the new
  lockfile + markers did not break the Mac. Recorded on the board.

## Slice 2 - ASR backend dispatch

`asr.py`: the existing MLX implementation is NOT moved or refactored - its
import becomes lazy (inside the darwin path), its calls and result conversion
stay as they are. Add ONE small Linux implementation module and dispatch with
a typed function - no ABC, no registry, no factory.

Semantics (frozen):
- `ASRModel()` default maps to `mlx-community/parakeet-tdt-0.6b-v3` on
  darwin, `nvidia/parakeet-tdt-0.6b-v3` on linux.
- Explicit model_id rule (implementable, frozen): the id is passed to the
  active backend verbatim, including local paths. ONE fail-fast check: on the
  linux backend, ids starting with `mlx-community/` raise a clear error naming
  both backends; on the mlx backend, ids starting with `nvidia/` do the same.
  Anything else (custom ids, local paths, other orgs) is attempted and the
  backend's own load error propagates untouched. Tests cover: default id,
  accepted explicit id, opposite-backend prefix, and an unknown id
  (documented as backend-error passthrough).
- No CLI flags added, removed, or renamed.
- Device (LINUX-SCOPED resolver): on linux, `auto` selects CUDA only when
  torch reports it usable, else CPU; explicit `cuda` when unavailable = clear
  error, never a silent fallback. On darwin the existing device behavior is
  preserved VERBATIM (MLX/CoreML semantics untouched) and a regression test
  proves the darwin resolution path is unchanged. The existing device option
  routes to BOTH ASR and diarisation unchanged on both platforms.
- stdout stays pure JSONL: NeMo/torch/download/progress output is forced to
  stderr. New public functions carry return annotations. Timestamp conversion
  (backend word objects -> `Word`) lives in one short, directly-tested
  function.

pytest config lives in Slice 1. Mandatory recorded exit checks before merge:
the CPU real-model smoke test (this slice), the senko CPU integration test
(Slice 3), and the golden subprocess test (Slice 4) - each slice's review
must show its check's output.

Tests (deterministic, no network): backend selection by platform + explicit
arg, lazy-import isolation (importing asr on linux must not import mlx),
device forwarding, unavailable-device error, timestamp conversion from
representative captured NeMo-shaped data. Integration (marked, needs model):
CPU transcription smoke test.

## Slice 3 - diarisation device path

`senko_diarisation.py`: device forwarded correctly on Linux ('cpu' now,
'cuda' post-reboot), no darwin assumptions. Assertions in tests are
measurable: device forwarded as given, nonempty speaker ids, finite floats,
`0 <= start < end <= duration`, stable ordering. Real-senko CPU sample run =
marked integration test AND a mandatory recorded exit check before this
slice merges. (Per-stream isolation is Slice 4's concern.)

## Slice 4 - end-to-end contract verification

Golden test, defined precisely:
- Runs the real entry point as a subprocess with existing flags on the
  two-stream fixture from macbase.
- Asserts: exit 0; every stdout line parses as JSON; the event sequence
  matches the slice-0 documented contract (exact names, exact order); the
  final line matches the slice-0 documented outer schema; every turn object
  has exactly the five frozen keys.
- The fixture's GROUND-TRUTH.md MUST/MAY contract governs: text content,
  timing tolerance and stream attribution are the hard contract; speaker
  CLUSTER COUNTS and turn counts/ordering are backend-dependent and NOT
  compared against the reference (the Mac reference itself over-splits the
  synthetic voices).
- Reference comparison uses backend-independent schema summaries only: the
  set of event names, the final-line outer key-set, per-turn key-set, and
  stream label set - each compared to the same summary of the Mac stdout.
- Invariants asserted independently (not against the reference): speaker-id
  = nonempty string; timestamps finite with `0 <= t0 < t1 <= duration`;
  per-stream turns monotonically ordered; both streams present and nonempty;
  transcribed text per stream matches the fixture's known script above a
  stated word-overlap threshold.
- Plus a regression test proving mic and system are passed to ASR/diarisation
  as separate inputs (not mixed then relabelled) - asserted at the
  orchestration seam, not by reading final `stream` fields.
Marked slow/integration; wired into pytest.

## Slice 5a - wrapper audit (audit-only, no edits)

Inventory the exact skill files: this repo's `skill/` + `skill.md`, and on
this box `~/.claude/skills/fast-transcribe/`, `~/.claude/skills/blogify/`
(+ podcast-transcribe if it shares the pipeline). Record in design/: each
file's path, the demonstrated Mac-only failures (command run + error), and a
NAMED list of files-to-edit. Amend this plan with that list. No edits in 5a.

Tooling decisions (made now, not by the builder): yt-dlp = managed tool via
`uv tool install yt-dlp`, verified with `yt-dlp --version`; ffmpeg = system
package, verified with `ffmpeg -version` (if absent, that is a David sudo
step routed via the board).

## Slice 5b - wrapper edits (per the 5a list)

Edits happen only in TRACKED canonical sources. 5a must establish where each
skill's canonical source lives (this repo's `skill/` dir is expected to be
canonical for fast-transcribe; if an installed `~/.claude/skills/` file has
no tracked source, STOP for direction rather than making unreviewable
home-directory edits). 5b edits the tracked source, and the install step
(copy into `~/.claude/skills/`) is recorded as a command in design/. Live
YouTube blogify run = recorded manual smoke test (commands + output in
design/), NOT a pytest dependency.

## GPU gate (blocked on David's reboot - registered as vault pending-job)

CPU slices may all land on `linux-cuda`, but the port is not "complete" and
nothing merges to `main` until, post-reboot: the same two-stream golden
command runs with device=cuda; evidence (torch device logs) shows BOTH ASR
and diarisation on CUDA; the JSONL contract is unchanged; versions + timings
recorded in design/.

## Review protocol for this run

David is AFK: per-slice review default = GPT-5 pass over the finished branch
diff PLUS the slice's exit-command output (a diff-only review is not evidence
that packaging/model-loading/contract work). Merge into `linux-cuda` only
after the relevant deterministic tests and lockfile checks pass. David can
override on the board.
