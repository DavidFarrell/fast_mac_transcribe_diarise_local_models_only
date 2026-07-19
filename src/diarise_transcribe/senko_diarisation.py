"""
Speaker diarisation using Senko.

Platform-neutral: the ``device`` given at construction is forwarded to
``senko.Diarizer`` unchanged, and Senko resolves it (CoreML on Apple
Silicon, CUDA or CPU on Linux). This module makes no platform assumptions.

Reference: https://github.com/narcotic-sh/senko
"""

import threading
from typing import Any, List, Optional

from .diarisation import DiarSegment


_native_diarizer_cache: dict[tuple[str, bool, bool], Any] = {}
_native_diarizer_cache_lock = threading.Lock()
_senko_import_lock = threading.Lock()


def _diarization_used_cuda(device: str) -> bool:
    """Whether a Senko diarisation on `device` ran on CUDA.

    'cuda' always does; 'auto' does when torch reports CUDA usable (Senko's
    own auto rule); 'cpu'/'coreml'/anything else does not. Used to poison
    co-resident ASR models - see asr.poison_cuda_asr and
    design/gpu-verification.md.
    """
    if device == "cuda":
        return True
    if device == "auto":
        try:
            import torch

            return bool(torch.cuda.is_available())
        except Exception:
            return False
    return False


def _restore_numba_njit(senko_module: Any) -> None:
    """
    Senko globally patches numba.njit(cache=True), which can make UMAP/HDBSCAN
    try to cache transient dispatcher objects and trigger:
      ReferenceError: underlying object has vanished
    Restore the original njit before Senko imports its clustering stack.
    """
    try:
        import numba
    except Exception:
        return

    original_njit = getattr(getattr(senko_module, "config", None), "_original_njit", None)
    if original_njit is not None:
        numba.njit = original_njit


def _import_senko() -> Any:
    with _senko_import_lock:
        import senko

        _restore_numba_njit(senko)
        return senko


class SenkoDiarizer:
    """
    Speaker diarizer wrapping Senko.

    The ``device`` given at construction is forwarded to ``senko.Diarizer``
    unchanged; Senko selects the compute backend from it.
    """

    def __init__(
        self,
        device: str = "auto",
        warmup: bool = False,
        quiet: bool = False,
    ) -> None:
        """
        Initialize the Senko diarizer.

        Args:
            device: Computation device, forwarded verbatim to Senko (which
                resolves it). Senko's values: 'auto', 'cuda', 'cpu', 'coreml'.
                This wrapper does not resolve or validate it.
            warmup: Whether to warm up models during initialization
            quiet: Suppress console output
        """
        self._device = device
        self._warmup = warmup
        self._quiet = quiet
        self._diarizer = None

    @staticmethod
    def _patch_numba_cache():
        """Monkey-patch numba cache to survive concurrent-access errors on
        both the SAVE and LOAD paths.

        This guards a DIFFERENT failure boundary from ``_restore_numba_njit``
        above: that one prevents a single-process njit ReferenceError at
        senko import time; this one makes the SHARED on-disk numba cache dir
        fault-tolerant against concurrent processes. Both are wanted (see
        the reconciliation record in design/slice0.5-reconciliation.md).

        Concurrency context: senko/config.py pins numba's JIT cache to one
        fixed, shared directory (``~/.cache/senko/numba_cache``) for every
        process on the machine (see numba_cache.py's module docstring for
        the full story) - there is no per-run isolation. That means a
        muesli-merge transcription and a podcast strip-ads job can run at
        the same moment, one writing new cache entries while the other
        reads them.

        SAVE side (pre-existing patch, broadened): numba 0.63+ has a bug
        where weak references to dispatcher objects get garbage collected
        during cache serialisation, causing ReferenceError. Separately,
        ``save()`` calls ``self._load_index()`` (the same pickle-based
        index read ``load()`` uses) before any weakref-related code runs,
        so a concurrent torn/corrupt index read during save() raises the
        same ``EOFError``/``ValueError``/``UnpicklingError``/etc as the
        load-side problem below - not just ``ReferenceError``/``KeyError``.
        This makes save() fault-tolerant against any exception, not just
        those two, so JIT compilation still works even if caching the
        result fails for any reason.

        LOAD side: ``IndexDataCacheFile.load()`` already catches OSError
        around reading the data file itself (numba's own
        ``_open_for_write`` uses a temp-file + ``os.replace`` so any
        *single* file read is atomic - readers see either the fully-old or
        fully-new bytes, never a torn one). But ``load()`` does NOT catch
        exceptions raised while unpickling the index
        (``_load_index``/``pickle.loads``): a reader whose cache-hit
        signature was written by a save() a few seconds newer than what's
        on disk, or an index left corrupt by a prior process that was
        killed mid-write, or any pickle version/format inconsistency
        between two concurrently-running numba versions, raises
        ``pickle.UnpicklingError``/``EOFError``/``ValueError``/etc. Those
        are not ``OSError`` subclasses, so they are NOT caught by numba's
        own guard and propagate straight up through
        ``Dispatcher.compile()`` - crashing the whole live run over what
        is, from the caller's point of view, just a cold cache miss.
        Wrapping load() to swallow any exception and fall back to "not
        cached" (triggering a normal recompile of that one function) makes
        concurrent reads-during-a-write fail-soft, matching the guarantee
        the save-side patch already gives for concurrent writes.
        """
        try:
            from numba.core.caching import IndexDataCacheFile

            # Idempotency guard: without this, every call re-wraps whatever
            # is currently installed on the class, so N calls in one process
            # (e.g. N SenkoDiarizer instances in a long-lived daemon) build
            # an N-deep closure chain around .load/.save - unbounded
            # reference growth and, eventually, a plausible RecursionError.
            # A single sentinel attribute makes repeated calls true no-ops.
            if getattr(IndexDataCacheFile, "_diarise_transcribe_patched", False):
                return

            _original_save = IndexDataCacheFile.save

            def _safe_save(self, key, data):
                try:
                    return _original_save(self, key, data)
                except Exception:
                    # Cache save failed but JIT compilation succeeded - that's
                    # fine. Deliberately broad (not just ReferenceError/
                    # KeyError): save() calls self._load_index() before any
                    # weakref-related code runs, and a concurrent
                    # torn/corrupt index read there can raise EOFError/
                    # ValueError/UnpicklingError/etc, none of which are
                    # ReferenceError or KeyError. Any failure here must
                    # degrade to "JIT result not cached this time", never
                    # propagate and crash the run.
                    pass

            IndexDataCacheFile.save = _safe_save

            _original_load = IndexDataCacheFile.load

            def _safe_load(self, key):
                try:
                    return _original_load(self, key)
                except Exception:
                    # Torn/corrupt read (e.g. a concurrent writer replaced
                    # the index or a data file mid-read, or a stale/
                    # partially-written entry from a killed process) -
                    # treat exactly like a cache miss so the caller
                    # recompiles this one function instead of crashing the
                    # whole run. Deliberately broad: any failure here must
                    # degrade to "recompile", never propagate.
                    return None

            IndexDataCacheFile.load = _safe_load
            IndexDataCacheFile._diarise_transcribe_patched = True
        except (ImportError, AttributeError):
            pass  # numba version without this class - no patch needed

    def _cache_key(self) -> tuple[str, bool, bool]:
        return (self._device, self._warmup, self._quiet)

    def _reset_cached_diarizer(self) -> None:
        key = self._cache_key()
        with _native_diarizer_cache_lock:
            cached = _native_diarizer_cache.get(key)
            if cached is not None and cached is self._diarizer:
                _native_diarizer_cache.pop(key, None)
        self._diarizer = None

    def _ensure_loaded(self):
        """Lazy load Senko diarizer on first use."""
        if self._diarizer is not None:
            return

        try:
            senko = _import_senko()
        except ImportError as e:
            raise ImportError(
                "Senko is not installed. Install with:\n"
                "  pip install 'git+https://github.com/narcotic-sh/senko.git'\n"
                f"Original error: {e}"
            )

        # Patch numba's shared on-disk cache before constructing the diarizer
        # (senko.Diarizer triggers numba JIT). Idempotent, so calling it on
        # every first-use is a no-op after the first process-wide install.
        self._patch_numba_cache()

        key = self._cache_key()
        with _native_diarizer_cache_lock:
            cached = _native_diarizer_cache.get(key)
            if cached is None:
                if not self._quiet:
                    print("Initializing Senko diarizer...")

                cached = senko.Diarizer(
                    device=self._device,
                    warmup=self._warmup,
                    quiet=self._quiet,
                )
                _native_diarizer_cache[key] = cached

                if not self._quiet:
                    print("Senko diarizer ready.")

            self._diarizer = cached

    @staticmethod
    def _is_transient_reference_error(error: Exception) -> bool:
        return isinstance(error, ReferenceError) and "underlying object has vanished" in str(error)

    def diarise(self, audio_path: str) -> List[DiarSegment]:
        """
        Run speaker diarisation on audio file.

        Args:
            audio_path: Path to 16kHz mono WAV file

        Returns:
            List of DiarSegment with speaker labels
        """
        for attempt in range(2):
            try:
                self._ensure_loaded()

                if not self._quiet:
                    print(f"Running Senko diarisation on: {audio_path}")

                # Run Senko diarisation
                result = self._diarizer.diarize(audio_path, generate_colors=False)
                break
            except Exception as error:
                if attempt == 0 and self._is_transient_reference_error(error):
                    if not self._quiet:
                        print("Senko hit a transient Numba cache error; retrying without warmup...")
                    self._reset_cached_diarizer()
                    self._warmup = False
                    continue
                raise
        else:
            raise RuntimeError("Senko diarisation did not produce a result.")

        # This diarisation just did CUDA work (if it ran on CUDA), which
        # poisons the CUDA state of any co-resident ASR model. Signal that so
        # the next ASR transcribe reloads a fresh model instead of crashing.
        if _diarization_used_cuda(self._device):
            from . import asr

            asr.poison_cuda_asr()

        # Senko can return None / empty output for silent inputs.
        if not result:
            if not self._quiet:
                print("No speakers detected in the audio.")
            return []

        merged_segments = result.get("merged_segments")
        if not merged_segments:
            if not self._quiet:
                print("No speakers detected in the audio.")
            return []

        # Convert Senko segments to our DiarSegment format, in Senko's own
        # order. Coerce the timestamps to float so DiarSegment carries its
        # declared float type regardless of what Senko returns (the frozen
        # turn schema has float t0/t1).
        segments = []
        for seg in merged_segments:
            segments.append(DiarSegment(
                start=float(seg["start"]),
                end=float(seg["end"]),
                speaker=seg["speaker"],
            ))

        if not self._quiet:
            n_speakers = result.get(
                "merged_speakers_detected",
                len(set(s.speaker for s in segments)),
            )
            print(f"  Detected {n_speakers} speakers, {len(segments)} segments")

        return segments


def diarise_audio_senko(
    audio_path: str,
    device: str = "auto",
) -> List[DiarSegment]:
    """
    Convenience function to diarise audio using Senko.

    Args:
        audio_path: Path to 16kHz mono WAV file
        device: Computation device

    Returns:
        List of DiarSegment
    """
    diarizer = SenkoDiarizer(device=device)
    return diarizer.diarise(audio_path)
