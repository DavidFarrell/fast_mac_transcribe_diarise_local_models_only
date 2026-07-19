# Slice 0.5 - backend reconciliation ANALYSIS (for the merge-builder)

Status: analysis only. This file records the resolution the merge-builder executes;
it makes NO repo changes and is uncommitted. Read `design/_build-overlay.md`,
`design/linux-cuda-dev-plan.md` (Slice 0.5 section) and
`~/.claude/skills/careful-build-lifecycle/doctrine/good-taste.md` first.

## 1. Pinned snapshots (reconciliation source -> target)

- TARGET (canonical, merged ONTO): standalone repo `/home/david/git/projects/fast_transcribe`,
  branch `linux-cuda`, HEAD `1e31388f90f9100696467210457e262c24a51ff3`
  ("Overlay ruling: uv-sync contract = CPU default; CUDA provisioning = explicit GPU-gate step").
  Working tree carries two untracked slice-0 artifacts (`design/slice0-ground-truth.md`,
  `tests/fixtures/`) - not part of the reconciliation, leave them.
- SOURCE (merged FROM): the embedded copy inside muesli repo
  `/home/david/git/projects/muesli`, HEAD `0d4ef71914ac2eb5317f858476dd234c0b14afe9`
  ("Merge fix-backend-readiness ..."). Embedded path:
  `backend/fast_mac_transcribe_diarise_local_models_only/src/diarise_transcribe/`.
- Nothing in this run edits the muesli repo. Everything lands on `linux-cuda`.

## 2. The three key muesli commits (what each protects)

- **ad57496** ("Fix mic PTS anchor: restore meeting-anchored PTS, surface backend drops",
  8 Jul). Two halves. (i) SWIFT half (MuesliApp `MicAudioForwarder`): the mic PTS epoch was
  regressed on 6 Jul (cafca18) to reset on every watchdog engine rebuild, so mid-meeting
  rebuilds restarted mic PTS at zero while the aligned file position kept advancing ->
  `write_aligned_audio` silently dropped every mic frame until the clock caught up (~60s+
  lost per rebuild, plus a permanent mic/system desync). Restored a meeting-scoped monotonic
  epoch. This half lives in the Swift target and is OUT OF SCOPE for this Python repo.
  (ii) PYTHON half (in scope): `write_aligned_audio` in `muesli_backend.py` now COUNTS drops
  (`frames_dropped`/`bytes_dropped` on the StreamWriter) and logs them rate-limited to stderr,
  so a future silent-drop regression is visible. Pinned by `tests/test_write_aligned_audio.py`.
  Full RCA: muesli `engineer-notes/bug-2026-07-08-mic-stall/RCA-2026-07-08.md`.
- **c5ed117** ("Retire the dead Sortformer diarisation backend", 5 Jul). Deleted the ~470-line
  CoreML SortformerDiarizer + MODEL_CONFIGS + download/inference machinery from
  `diarisation.py`, leaving only the shared `DiarSegment` dataclass. Narrowed `--diar-backend`
  choices to `["senko"]` in all three entry points, removed `--diar-model` entirely, and made
  the now-unreachable non-senko branch raise `ValueError`. Dead-code proof in the commit:
  every entry point already defaulted to senko, the Swift app never passed the flags, and
  muesli-merge passes `--diar-backend senko`. Protects: a single, real diariser; no dead
  CoreML path. On Linux this is also a correctness win - it removes the Mac-only
  `import coremltools` from the import path.
- **c36a45f** ("Make ASR recovery pass fault-isolated and fix partial-coverage duplication",
  5 Jul). (i) Each recovery window's slice+transcribe is isolated in its own try/except and
  the whole `_run_recovery_pass` call is wrapped at the call site, so a bad window (e.g. a
  zero-frame clamped slice) is skipped and logged instead of failing the whole reprocess and
  discarding the good main transcript. (ii) `drop_words_in_windows()` in `recovery.py` drops
  original words whose midpoint falls in a window that actually recovered replacements, so a
  partial-coverage (<10%) segment no longer shows words twice. Protects: recovery is
  best-effort, never regresses the main transcript; no duplicated turn text. Pinned by
  `tests/test_recovery.py` + `tests/test_reprocess_recovery.py`.

## 3. Full differing-file inventory (verified by actual diffs)

macbase's numbers reproduced exactly, PLUS one file macbase did not list (`__init__.py`).
`__main__.py` is byte-identical. Changed-line counts are `diff | grep -cE '^[<>]'`.

| File | Kind | Changed lines | Resolution | One-line why |
|---|---|---|---|---|
| `reprocess.py` | embedded-only (461 ln) | - | **take-embedded** | two-stream layer, canonical target lacks it |
| `recovery.py` | embedded-only (170 ln) | - | **take-embedded** | ASR recovery pass (c36a45f) + its fault-isolation |
| `muesli_backend.py` | embedded-only (986 ln) | - | **take-embedded** | live backend + the mic-drop accounting (ad57496 python half) |
| `constants.py` | embedded-only (2 ln) | - | **take-embedded** | shared gap/tolerance defaults referenced by cli/merge/reprocess |
| `numba_cache.py` | standalone-only (579 ln) | - | **keep-standalone** | warm-cache + concurrency hardening (65c7eab/524db9c/10bdd8f) |
| `diarisation.py` | shared-diff | 741 | **take-embedded** (23-ln stub) | honor Sortformer retirement (c5ed117); removes Mac-only coremltools import |
| `senko_diarisation.py` | shared-diff | 220 | **hand-merge** | DANGER ZONE - two independent concurrency fixes, keep both (see 4) |
| `cli.py` | shared-diff | 114 | **hand-merge** | two orthogonal evolutions: retirement (embedded) + warm-cache (standalone) |
| `audio.py` | shared-diff | 61 | **hand-merge** | embedded adds recovery slicers but DROPS standalone's UTF-8 ffmpeg fix |
| `asr.py` | shared-diff | 29 | **take-embedded** (+ see surprise S2) | adds process-wide model cache; MLX import stays eager |
| `merge.py` | shared-diff | 5 | **take-embedded** | uses constants.py default instead of literal 0.8 |
| `__init__.py` | shared-diff | 2 | **take-embedded** | module docstring: senko not Sortformer (matches retirement) |
| `__main__.py` | identical | 0 | n/a | byte-identical |

## 4. Per-file hand-merge detail (the hunks that matter)

### diarisation.py -> take-embedded (23-line stub)
Standalone (750 ln) = `DiarSegment` + `StreamingState` + `download_model` +
`compute_mel_spectrogram` + `SortformerDiarizer` (~536 ln) + `diarise_audio`, and
`import coremltools as ct` at module top. Embedded (23 ln) = just `DiarSegment`.
Cross-module importers in BOTH trees only ever import `DiarSegment`
(merge.py/recovery.py/reprocess.py/muesli_backend.py), plus standalone cli.py's
`SortformerDiarizer` import. `StreamingState`/`compute_mel_spectrogram`/`download_model`/
`diarise_audio` are imported by NOTHING outside diarisation.py -> no live behavior is lost by
retirement. Take the embedded stub verbatim.

### cli.py -> hand-merge (two orthogonal evolutions, one interaction point)
The 114-line diff is two non-overlapping edit regions:
- **From EMBEDDED (Sortformer retirement, c5ed117):** drop
  `from .diarisation import SortformerDiarizer, MODEL_CONFIGS, DiarSegment`; add
  `from .constants import DEFAULT_GAP_THRESHOLD_SECONDS, DEFAULT_SPEAKER_TOLERANCE_SECONDS`.
  Narrow `--diar-backend` choices `["senko","sortformer"]` -> `["senko"]`. REMOVE the
  `--diar-model` argument entirely. Remove `diar_model` from the `run_pipeline` signature and
  its call. Replace the `SortformerDiarizer(...)` diarise branch with the `raise ValueError`
  ("only 'senko' is supported ... the Sortformer backend was retired"). Update the two
  gap/tolerance defaults + help strings to reference the constants. Update the 4-speaker/
  Sortformer help text.
- **From STANDALONE (numba warm-cache, 524db9c/65c7eab):** KEEP the whole top-of-file numba
  cache block (lines ~8-38: per-PID `NUMBA_CACHE_DIR` + `bootstrap_senko_cache_if_empty`).
  KEEP the `--warm-cache` maintenance argument, the `run_warm_cache` handling block in
  `main()`, and the `from .numba_cache import ...` uses.
- **THE ONE INTERACTION:** embedded makes `--in` `required=True`; standalone makes it
  `default=None` (optional at parser level) + a manual `parser.error("--in/-i is required")`
  in `main()`, specifically so `--warm-cache` can run WITHOUT `--in`. Because we KEEP
  `--warm-cache`, take the STANDALONE `--in` handling (optional + `parser.error`). Taking
  embedded's `required=True` would break warm-cache. Normal-run UX is byte-identical
  (exit 2 + usage banner).
- Cosmetic follow-up: the standalone numba block's comment parenthetical "(e.g. the sortformer
  backend)" is stale post-retirement; update it. Verify the merged cli.py no longer imports
  `DiarSegment` (embedded cli.py does not use it).

### senko_diarisation.py -> hand-merge, KEEP BOTH concurrency fixes  (DANGER ZONE)
The two versions solve DISTINCT problems and evolved in parallel (verified:
`git log -S'_patch_numba_cache'` on the embedded file is EMPTY - the embedded tree never had
the standalone patch, so the embedded approach is NOT a supersession).
- **STANDALONE `_patch_numba_cache()`** monkeypatches numba `IndexDataCacheFile.save` AND
  `.load` to swallow exceptions -> CROSS-PROCESS torn reads/writes of the SHARED on-disk cache
  dir (`~/.cache/senko/numba_cache`, which senko pins for every process) degrade to a recompile
  instead of crashing. Motivated by concurrent Mac workloads (muesli-merge + strip-ads).
  Idempotency-guarded. Also: `warmup` default `True`; `result is None` -> `[]`.
- **EMBEDDED** `_restore_numba_njit()` undoes senko's global `numba.njit(cache=True)` patch
  BEFORE senko imports UMAP/HDBSCAN, preventing "ReferenceError: underlying object has vanished"
  at its SOURCE (single-process). Plus a process-wide native-diarizer cache keyed by
  `(device,warmup,quiet)`; plus a `diarise()` retry loop (on that transient ReferenceError,
  reset the cached diarizer, `warmup=False`, retry once). `warmup` default `False`; empty
  `result`/`merged_segments` -> `[]`.
- **RECOMMENDED MERGE:** base on the embedded structure (`_import_senko`/`_restore_numba_njit`,
  native cache, retry loop, empty-result handling), and ALSO install standalone's
  `_patch_numba_cache` (call it once in `_ensure_loaded` before constructing the Senko
  diarizer). They target different root causes (cross-process shared-cache-file corruption vs
  single-process njit ReferenceError), so on the Mac both are wanted and on Linux both are
  harmless.
- **Interaction to be aware of (adjudicated by GPT-5, section 5):** `_patch_numba_cache`'s
  `_safe_save` swallows ALL save-path exceptions including the "underlying object has vanished"
  ReferenceError; if it ever reaches the save path, the embedded retry loop (which waits for
  that error to propagate out of `diarise()`) would never fire. This is benign - with
  `_restore_numba_njit` present the error should not arise, and if it slips through, swallowing
  it at save achieves the same "don't crash, result still correct" outcome the retry wanted.
  Retry stays as a cheap backstop.
- **warmup default:** adopt embedded `warmup=False` (the default in the tree that actually runs
  `reprocess`, which constructs `SenkoDiarizer(quiet=...)` without specifying warmup).

### audio.py -> hand-merge (SILENT-REGRESSION TRAP)
Embedded ADDS two functions the recovery/reprocess layer needs: `is_wav_16k_mono()` and
`slice_wav_to_temp()` (used by recovery slicing). BUT the embedded copy does NOT carry
standalone's `encoding="utf-8", errors="replace"` on the two ffmpeg subprocess calls (commit
10bdd8f - ffmpeg echoes source ID3 tags to stderr; non-UTF-8 bytes otherwise raise
`UnicodeDecodeError` and kill transcription before it starts). A naive take-embedded would
silently drop that fix. Resolution: take embedded's two NEW functions AND keep standalone's
`errors="replace"` on both subprocess calls. NB the plan text lists 10bdd8f under "numba_cache.py
fixes" - it is actually an audio.py fix; carry it here.

### asr.py -> take-embedded (with caveat S2)
Embedded adds a thread-safe process-wide `_model_cache` so repeated `ASRModel` instances reuse
one loaded parakeet-mlx model (matters for reprocess: main pass + recovery pass). Take it.
The MLX import `from parakeet_mlx import from_pretrained` stays module-top/eager in BOTH trees -
Slice 2 owns making it lazy. See surprise S2.

### merge.py / __init__.py / constants.py / numba_cache.py / reprocess.py / recovery.py / muesli_backend.py
- merge.py: take-embedded (3 hunks: import `DEFAULT_GAP_THRESHOLD_SECONDS`, two default
  literals `0.8` -> constant). Cosmetic + constant-sourcing.
- __init__.py: take-embedded (docstring line: "Senko (pyannote+CAM++ CoreML)" replaces
  "Sortformer CoreML"). Matches the retirement.
- constants.py: take-embedded (2 lines).
- numba_cache.py: keep-standalone wholesale (embedded has no such file).
- reprocess.py / recovery.py / muesli_backend.py: take-embedded wholesale.

## 5. GPT-5 second opinion on the danger zone (diarisation.py + senko_diarisation.py)

**Verdict: READY TO SHIP** (GPT-5, xhigh reasoning). It returned the exact ship phrase, with no
numbered changes - i.e. it endorses both recommendations as written: diarisation.py =
take-embedded stub (retirement honored), and senko_diarisation.py = hand-merge keeping both
concurrency fixes with the interaction judged benign.

What the trace does and does not show (stated precisely - the final message is the bare phrase
with no written justification): its trace shows it framed the crux itself as "whether the cache
wrapper and retry cover the same failure boundary ... determines whether 'keep both' is
justified or merely redundant global monkeypatching", then - after a bubblewrap sandbox error
blocked local repo inspection - fell back to web searches TARGETING numba's `core/caching.py`
(`IndexDataCacheFile.save`/`.load`/`_load_index`) and senko's `diarizer.py` (import-time
`njit(cache=True)` patch). The returned page content is not visible in the trace, so this is
corroboration, not independently-verifiable proof. The load-bearing basis for keep-both is the
analysis in section 4 (the two fixes act on different failure boundaries - cross-process
cache-file I/O vs single-process njit ReferenceError); GPT-5 reviewed that and did not dissent.
No disagreements to fold in.

Prompt + raw reply archived at `/tmp/slice05_diar_gpt5_prompt.md` and
`/tmp/slice05_diar_gpt5_reply.md` (session-scratch, not committed).

## 6. Regression-test surface (what the merge-builder wires up)

The plan's "new focused deterministic regressions for the mic PTS anchor and ASR recovery
fault-isolation" come in FREE with the embedded test files - they do not need writing from
scratch, only carrying into the standalone `tests/` and confirming green on Linux CPU.

- **Mic PTS anchor (ad57496, python half):** observable at `muesli_backend.write_aligned_audio`
  via the StreamWriter counters `frames_dropped`/`bytes_dropped` + rate-limited stderr log.
  Pinned by `tests/test_write_aligned_audio.py` (148 ln): silence-pad on forward gap (no drop),
  whole-frame-behind drop counted+logged, front-trim drop counted. No models, no audio files -
  pure byte assertions. Carry this file. (The Swift MicAudioForwarder half is out of this repo.)
- **ASR recovery fault-isolation (c36a45f):** `recovery.drop_words_in_windows` +
  `reprocess._run_recovery_pass` try/except isolation. Pinned by `tests/test_recovery.py` and
  `tests/test_reprocess_recovery.py` (window-failure isolation, partial-coverage replacement
  without duplication, and the no-recovery-leaves-words case). Carry both.
- **Model-stubbed `reprocess --stream both` JSONL contract run:** the import seam is already
  clean. `reprocess.py` constructs models by the module-level NAMES `ASRModel` (line ~243) and
  `SenkoDiarizer` (line ~249). Stub exactly as the existing embedded tests do:
  `monkeypatch.setattr(reprocess, "ASRModel", lambda model_id: FakeASRModel(...))`,
  `monkeypatch.setattr(reprocess, "SenkoDiarizer", lambda **kw: FakeDiarizer(...))`, plus
  `reprocess.is_wav_16k_mono -> True`, `reprocess.get_audio_duration -> const`, and
  `reprocess.slice_wav_to_temp` for the recovery path (pattern in
  `tests/test_reprocess_recovery.py::_patch_common` and `tests/test_reprocess_sessions.py`).
  A two-stream `--stream both` run with FakeASRModel/FakeDiarizer validates the JSONL
  event/turn schema with NO real models loaded - satisfying the exit criterion on CPU.
- Existing standalone tests: `test_numba_cache.py` and `test_senko_diarisation.py` keep green
  as-is. The other two touch the retired path and need concrete edits (see S3, section 8) - one
  is a whole-file delete, one is a two-assertion edit, NOT a vague "reconcile".

## 7. Confirmation: nothing retired survives

The recommended resolution takes the embedded 23-line `diarisation.py` (no SortformerDiarizer,
no MODEL_CONFIGS) and removes standalone cli.py's Sortformer import + branch + `--diar-model`.
Grep of the merged shape: the only surviving "Sortformer" strings are the retirement notices in
cli.py/reprocess.py/muesli_backend.py help+ValueError text ("the Sortformer backend was
retired"). No `SortformerDiarizer` construction, no `MODEL_CONFIGS`, no `import coremltools` on
any import path. `--diar-backend sortformer` now fails with a clear argparse error.

## 8. Surprises requiring a plan decision

- **S1 - the plan mislabels the danger zone.** The plan singles out `diarisation.py` (741-line
  diff) for its own GPT-5 review round. But that 741 is a near-total DELETION (750 -> 23):
  diarisation.py is a clean retirement, not a divergent-behavior merge. The ACTUAL danger is
  `senko_diarisation.py` (220-line diff), where two INDEPENDENT live concurrency fixes must be
  unioned. The GPT-5 round in section 5 covers both, weighted to senko. Recommend the plan note
  that senko_diarisation.py is the file whose reconciliation carries risk.
- **S2 - the "--help on Linux" exit criterion is not reachable by reconciliation alone.**
  Slice 0.5's exit line requires "both CLIs importable and showing --help on Linux". Both
  `cli.py` and the newly-imported `reprocess.py` do a module-top `from .asr import ASRModel`,
  and `asr.py` does a module-top `from parakeet_mlx import from_pretrained`, which fails at
  import on Linux (`libmlx.so` missing - recorded in `design/slice0-ground-truth.md`). The
  diarisation.py retirement fixes cli.py's OTHER Mac-only chain (coremltools) but NOT this one.
  Making the MLX import lazy is explicitly Slice 2's job ("its import becomes lazy inside the
  darwin path"). So `python -m diarise_transcribe.reprocess --help` and `diarise-transcribe
  --help` cannot succeed on Linux at the end of 0.5 without pulling Slice 2 forward.
  DECISION NEEDED - two clean options:
  (a) Amend the 0.5 exit criterion: the Linux `--help` check for both CLIs is DEFERRED to
  Slice 2 (when the MLX import goes lazy); at 0.5, verify `--help` on the Mac and verify the
  Linux import succeeds up to the asr seam. Keeps 0.5 scope pure.
  (b) Pull the minimal MLX lazy-import guard into 0.5 (move `from parakeet_mlx import
  from_pretrained` out of module scope into the load function in asr.py) since asr.py is
  already being touched here - but this is Slice 2's stated responsibility and risks scope
  creep. Recommend (a).
- **S3 - two standalone tests reference the retired Sortformer path (concrete resolutions,
  both read in full).**
  - `test_diarisation_resolve_symlinks.py` (287 ln): **DELETE the whole file.** Every test in it
    imports and exercises `SortformerDiarizer._resolve_symlinks` (the retired CoreML model-dir
    staging/symlink machinery). Once diarisation.py is the 23-line stub, `SortformerDiarizer`
    and `_resolve_symlinks` no longer exist, so the file is untestable and cannot be "updated" -
    it must go. No senko/numba coverage is lost (this hardened the SORTFORMER model dir; the
    senko numba-cache hardening is separate, in numba_cache.py + test_numba_cache.py).
  - `test_cli_backward_compat.py` (4 tests): **UPDATE two spots only.** Remove `"diar_model"`
    from the required-flags list (line ~23) and delete `assert args.diar_model == "default"`
    (line ~36), matching the `--diar-model` removal. Leave the other two tests untouched -
    `test_missing_input_file_still_errors_same_as_before` in fact GUARDS the kept warm-cache
    `parser.error("--in/-i is required")` behavior (the cli.py interaction point in section 4),
    and the cold-cache-startup test guards the kept numba block. `diar_backend` default "senko"
    (line ~35) stays valid.
- **S4 - numba_cache.py cross-file dependency on the danger-zone senko is satisfied by
  keep-both.** numba_cache.py is kept wholesale and reaches into senko_diarisation.py two ways:
  (i) its warm-worker subprocess snippet does `SenkoDiarizer(quiet=...)` then `.diarise(path)`
  (line ~487) - the merged (embedded-based) senko satisfies both the `quiet=` constructor kwarg
  and the `.diarise(path)` method, so no interface break; (ii) its module docstring (line ~36)
  references "the `_patch_numba_cache` fault-tolerant save wrapper in senko_diarisation.py" -
  keep-both preserves `_patch_numba_cache`, so that reference stays valid. This is a further
  concrete reason the senko resolution must be keep-both, not take-embedded: a take-embedded
  senko would delete `_patch_numba_cache` and leave numba_cache.py's docstring dangling.

## 9. Merge-builder execution confidence

High. Every differing file has a concrete resolution with named hunks; the two shared-file
traps (audio.py UTF-8 fix, cli.py `--in`/warm-cache interaction) are called out; the danger
zone has an independent GPT-5 adjudication; the regression tests already exist in the embedded
tree with a clean stub seam; the two retired-path test edits are concrete (S3: one delete, one
two-assertion edit) and the one cross-file dependency on the hand-merge is confirmed satisfied
(S4). Exactly ONE item needs a team-lead/plan decision before merge - S2, the `--help`-on-Linux
exit criterion (recommend deferring that check to Slice 2, option (a)). Everything else is
executable as written. The GPU/CUDA path is out of scope for 0.5 (CPU-first, per overlay).
