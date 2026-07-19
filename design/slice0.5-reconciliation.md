# Slice 0.5 - backend reconciliation (execution record)

This records what the merge-builder actually did, per file, executing
`design/slice0.5-reconciliation-analysis.md`. It is the durable record the plan
(Slice 0.5, "Reconciliation decisions recorded in
design/slice0.5-reconciliation.md") requires.

## 1. Pinned snapshots (source -> target)

- TARGET (canonical, merged ONTO): this repo `fast_transcribe`, branch
  `slice-0.5-reconciliation` cut from `linux-cuda` @
  `0da8d02fe5443074bc8cfd8e7f20440d513bd94f`
  ("Plan: fold 0.5 analysis - senko hand-merge is the danger zone, S2 ruling on
  Linux --help"). This is two design-doc-only commits ahead of the analysis's
  pinned `1e31388` (`git diff 1e31388 0da8d02` touches only
  `design/linux-cuda-dev-plan.md`), so every code resolution in the analysis
  still applies unchanged.
- SOURCE (merged FROM): the embedded copy inside the muesli repo @
  `0d4ef71914ac2eb5317f858476dd234c0b14afe9`, path
  `backend/fast_mac_transcribe_diarise_local_models_only/src/diarise_transcribe/`.
  Verified muesli HEAD unchanged from the analysis's pin before merging.
- Nothing in this run edited the muesli repo. Everything landed on
  `slice-0.5-reconciliation`.
- The pre-existing untracked slice-0 artifacts (`design/slice0-ground-truth.md`,
  `design/slice0.5-reconciliation-analysis.md`, `design/slice5a-wrapper-audit.md`,
  `tests/fixtures/`) were left as-is - not part of this reconciliation.

## 2. Per-file resolution (every differing shared module + the brought-in files)

| File | Resolution | Why |
|---|---|---|
| `reprocess.py` | take-embedded (new file, byte-identical) | two-stream `--stream both` JSONL layer, canonical target lacked it |
| `recovery.py` | take-embedded (new file, byte-identical) | ASR recovery pass + fault-isolation (c36a45f) |
| `muesli_backend.py` | take-embedded (new file, byte-identical) | live backend + mic-drop accounting (ad57496 python half) |
| `constants.py` | take-embedded (new file, byte-identical) | shared gap/tolerance defaults referenced by cli/merge/reprocess |
| `diarisation.py` | take-embedded (overwrite, byte-identical) | 23-line `DiarSegment` stub; honors Sortformer retirement (c5ed117); removes Mac-only `import coremltools` |
| `asr.py` | take-embedded (overwrite, byte-identical) | process-wide `_model_cache`; MLX import stays eager (Slice 2 makes it lazy) |
| `merge.py` | take-embedded (overwrite, byte-identical) | sources gap default from `constants.py` instead of literal 0.8 |
| `__init__.py` | take-embedded (overwrite, byte-identical) | docstring: senko not Sortformer, matches retirement |
| `__main__.py` | untouched | byte-identical in both trees |
| `numba_cache.py` | keep-standalone (untouched) | warm-cache + concurrency hardening (65c7eab/524db9c); embedded has no such file |
| `senko_diarisation.py` | **hand-merge** | DANGER ZONE - keep BOTH concurrency fixes (section 3) |
| `cli.py` | **hand-merge** | retirement (embedded) + warm-cache/optional-`--in` (standalone) (section 4) |
| `audio.py` | **hand-merge** | embedded's recovery slicers + standalone's UTF-8 ffmpeg fix (section 5) |

The 8 take-embedded files were `cp`'d verbatim and verified byte-identical to
the embedded source with `cmp -s`.

## 3. `senko_diarisation.py` hand-merge - BOTH concurrency fixes kept

Base = embedded structure (`_import_senko` / `_restore_numba_njit`, module-level
`_native_diarizer_cache`, the single retry-on-transient-ReferenceError loop,
empty-result handling, `warmup=False` default - the default the tree that runs
`reprocess` actually constructs `SenkoDiarizer` with). ALSO installed standalone's
`_patch_numba_cache` static method verbatim (idempotency-guarded) and call it once
in `_ensure_loaded` after `_import_senko()` succeeds and before constructing the
Senko diarizer.

The two fixes act on DIFFERENT failure boundaries, so keeping both is a union not
a redundancy:
- `_restore_numba_njit` + retry: single-process njit `ReferenceError: underlying
  object has vanished` at senko import/clustering time.
- `_patch_numba_cache`: cross-process torn/corrupt reads+writes of senko's SHARED
  on-disk numba cache dir (`~/.cache/senko/numba_cache`).

GPT-5 (xhigh) adjudicated this exact "same failure boundary?" crux and returned
READY TO SHIP (analysis section 5). Interaction (benign): if a "vanished"
ReferenceError ever reached the save path, `_patch_numba_cache`'s `_safe_save`
would swallow it and the retry loop would not fire - same "don't crash, result
still correct" outcome; retry stays as a cheap backstop.

## 4. `cli.py` hand-merge - two orthogonal evolutions

KEEP from standalone: the top-of-file per-PID `NUMBA_CACHE_DIR` + `bootstrap_senko
_cache_if_empty` block; the `--warm-cache` maintenance arg + its `run_warm_cache`
handling; optional `--in` (`default=None`) + the manual `parser.error("--in/-i is
required")` in `main()`.
APPLY from embedded (Sortformer retirement, c5ed117): dropped
`from .diarisation import SortformerDiarizer, MODEL_CONFIGS, DiarSegment`; added
`from .constants import ...`; narrowed `--diar-backend` to `["senko"]`; REMOVED
`--diar-model` entirely; removed `diar_model` from `run_pipeline` signature+call;
replaced the `SortformerDiarizer(...)` branch with `raise ValueError(... the
Sortformer backend was retired)`; sourced gap/tolerance defaults from constants;
removed the Sortformer help/epilog text.
THE INTERACTION (analysis 4): kept STANDALONE's optional-`--in`, NOT embedded's
`required=True` - taking `required=True` would break `--warm-cache` (which runs
without `--in`). Normal-run UX unchanged (exit 2 + usage banner, pinned by
`test_cli_backward_compat.py::test_missing_input_file_still_errors_same_as_before`).
Also updated the stale numba-block comment (removed "(e.g. the sortformer
backend)") and confirmed the merged cli.py no longer imports `DiarSegment`.

## 5. `audio.py` hand-merge - silent-regression trap avoided

Base = embedded (adds `import wave`, `is_wav_16k_mono()`, `slice_wav_to_temp()`
that the recovery/reprocess layer needs). RESTORED standalone's `encoding="utf-8",
errors="replace"` on BOTH `subprocess.run` calls (`check_ffmpeg` and
`normalise_audio`) - commit 10bdd8f; without it, ffmpeg echoing non-UTF-8 ID3 tag
bytes to stderr raises `UnicodeDecodeError` and kills transcription before it
starts. (The plan text filed 10bdd8f under numba_cache.py; it is an audio.py fix.)

## 6. Test surface

Carried in from the embedded tree (deterministic, model-stubbed):
- `test_write_aligned_audio.py` - mic PTS drop accounting (ad57496 python half).
- `test_recovery.py` - recovery windowing/dedupe primitives (c36a45f).
- `test_reprocess_recovery.py` - recovery fault-isolation via the `_patch_common`
  ASRModel/SenkoDiarizer stub seam.
- `test_reprocess_sessions.py` - `--stream both` multi-session offset accumulation
  + the five-key turn schema (DEVIATION: beyond the three the analysis named; it
  is the direct regression test for reprocess.py `main()`, take-embedded here).
- `test_asr.py` - the process-wide `_model_cache` behaviour asr.py take-embedded
  introduces (DEVIATION: beyond the three named; covers a change this slice brings
  in). Not carried: `test_muesli_backend_live_asr_only.py` (live-model integration,
  out of CPU scope).

S3 retired-path test changes (analysis section 8):
- DELETED `tests/test_diarisation_resolve_symlinks.py` (287 ln): every test
  exercised `SortformerDiarizer._resolve_symlinks`, which no longer exists after
  the 23-line diarisation.py stub. Untestable, not "updatable".
- EDITED `tests/test_cli_backward_compat.py`: removed `"diar_model"` from the
  required-flags set and deleted `assert args.diar_model == "default"`, matching
  the `--diar-model` removal. The other two tests (missing-input parser.error,
  cold-cache startup) were left - they GUARD kept behaviour.

DEVIATION - `tests/test_senko_diarisation.py` (standalone, kept as the base):
- Fixed one assertion: `test_diarise_prints_no_speech_message_when_not_quiet`
  asserted `"no speech"`, but the merged (embedded-base) `diarise()` reports a
  None/empty result as "No speakers detected in the audio." Renamed to
  `..._no_speakers_message...` and assert `"no speakers detected"`. The test's
  intent (informative message on silent input when not quiet) is preserved; this
  is a message-wording change, unrelated to the `warmup` ruling. All other
  standalone assertions pass as-is.
- APPENDED the embedded tree's two senko tests (`_restore_numba_njit` +
  retry-once-after-transient-ReferenceError) so the danger-zone file's BOTH
  concurrency fixes are proven in one place (the standalone file only covered
  `_patch_numba_cache`). Both run cleanly on Linux (no asr seam).

## 7. Verification (all on Linux CPU; environment recorded)

Environment: no `uv sync` (pyproject still carries Mac deps - Slice 1's job).
Ran from a throwaway venv `/tmp/slice05-venv` (python 3.12.3) with
`PYTHONPATH=src` and only the deps the deterministic tests + the stub harness
need: `pytest 9.1.1`, `numba 0.66.0` (matches slice-0 ground truth), `numpy 2.4.6`,
`soundfile 0.14.0`. torch/nemo/senko/parakeet-mlx deliberately NOT installed -
every ASR/diarizer is stubbed.

- **Per-module Linux import probe** (which modules hit `parakeet_mlx` at module
  top = the S2 seam):
  - Import OK on Linux: `__init__`, `constants`, `diarisation`,
    `senko_diarisation`, `audio`, `numba_cache`.
  - Import FAILS on Linux (`ModuleNotFoundError: parakeet_mlx`): `asr` (direct),
    and transitively `merge`, `recovery`, `cli`, `reprocess`, `muesli_backend`,
    `__main__`. Expected; making the MLX import lazy is Slice 2. "Imports work up
    to the asr seam" - verified.
- **Deterministic tests, shim-free files** (no asr seam):
  - `tests/test_senko_diarisation.py`: **22 passed** - the full danger-zone proof
    (standalone `_patch_numba_cache` save/load/idempotency + None-handling + the
    appended embedded `_restore_numba_njit`/retry tests). Both concurrency fixes
    green together.
  - `tests/test_numba_cache.py`: **57 passed, 5 failed**. The 5 failures are ALL
    and ONLY `ModuleNotFoundError: parakeet_mlx` from `asr.py:11`, all in
    `TestWarmCacheArgparseWiring` (which imports `cli`) - the S2 seam, not a merge
    defect. Deferred to the macbase Mac gate (where MLX imports) and Slice 2.
  - The other carried/embedded regression tests (`test_write_aligned_audio`,
    `test_recovery`, `test_reprocess_recovery`, `test_reprocess_sessions`,
    `test_asr`) and `test_cli_backward_compat` all import the asr seam and so are
    collection-deferred to the macbase Mac gate; they were byte-compiled clean
    (`py_compile`) here.
- **Model-stubbed `reprocess --stream both`** (throwaway harness
  `/tmp/slice05_stream_both_harness.py`, injects a fake `parakeet_mlx` to clear
  the Linux import seam, stubs ASRModel/SenkoDiarizer via the `_patch_common`
  pattern): PASS on all 8 checks - the exact 11-event slice-0 sequence
  (preparing -> system{transcribing,diarizing,merging,recovering} ->
  mic{...} -> complete -> result), both `recovering` events carry an int
  `windows`, result outer key-set `{type,turns,speakers,duration}`, every turn
  has EXACTLY `{speaker_id,stream,t0,t1,text}`, both streams present, pure JSONL
  (0 pollution lines). Output saved to `/tmp/slice05_stream_both_result.txt`.
- **Retired-path proof (grep + argparse):** no `SortformerDiarizer` construction,
  no `MODEL_CONFIGS`, no `import coremltools` anywhere under
  `src/diarise_transcribe/`; the only surviving "Sortformer" strings are the
  retirement notices (help text + ValueError + diarisation.py docstring).
  `--diar-backend sortformer` fails on BOTH entry points with
  `SystemExit(2): argument --diar-backend: invalid choice: 'sortformer' (choose
  from 'senko')`.

## 8. Not merged into linux-cuda

Per the plan, the macbase gate (Mac-side reference run + full deterministic suite,
where MLX imports) runs BEFORE this slice merges into `linux-cuda`. The boss
handles that; this branch is pushed for review only.
