"""
Speaker diarisation using Senko on Apple Silicon.

Senko uses CoreML for both VAD (pyannote segmentation-3.0) and speaker
embeddings (CAM++), running on the Apple Neural Engine.

Reference: https://github.com/narcotic-sh/senko
"""

from typing import List, Optional

from .diarisation import DiarSegment


class SenkoDiarizer:
    """
    CoreML-based speaker diarizer using Senko.

    Senko provides efficient speaker diarisation on Apple Silicon,
    processing ~1 hour of audio in ~7.7 seconds on M3.
    """

    def __init__(
        self,
        device: str = "auto",
        warmup: bool = True,
        quiet: bool = False,
    ):
        """
        Initialize the Senko diarizer.

        Args:
            device: Computation device ('auto', 'cuda', 'cpu', 'coreml')
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

    def _ensure_loaded(self):
        """Lazy load Senko diarizer on first use."""
        if self._diarizer is not None:
            return

        try:
            import senko
        except ImportError as e:
            raise ImportError(
                "Senko is not installed. Install with:\n"
                "  pip install 'git+https://github.com/narcotic-sh/senko.git'\n"
                f"Original error: {e}"
            )

        # Patch numba cache before initializing (handles 0.63+ bug)
        self._patch_numba_cache()

        if not self._quiet:
            print("Initializing Senko diarizer...")

        self._diarizer = senko.Diarizer(
            device=self._device,
            warmup=self._warmup,
            quiet=self._quiet,
        )

        if not self._quiet:
            print("Senko diarizer ready.")

    def diarise(self, audio_path: str) -> List[DiarSegment]:
        """
        Run speaker diarisation on audio file.

        Args:
            audio_path: Path to 16kHz mono WAV file

        Returns:
            List of DiarSegment with speaker labels
        """
        self._ensure_loaded()

        if not self._quiet:
            print(f"Running Senko diarisation on: {audio_path}")

        # Run Senko diarisation
        result = self._diarizer.diarize(audio_path, generate_colors=False)

        # senko.Diarizer.diarize() returns None when it detects no speech at
        # all in the input (e.g. silent or non-speech audio) rather than a
        # result dict with empty lists. Treat that as a legitimate "no
        # speakers found" outcome - not an error - so callers (and the CLI)
        # get a clean, valid, empty transcript instead of a TypeError from
        # indexing None.
        if result is None:
            if not self._quiet:
                print("  No speech detected - returning empty segment list")
            return []

        # Convert Senko segments to our DiarSegment format
        segments = []
        for seg in result["merged_segments"]:
            segments.append(DiarSegment(
                start=seg["start"],
                end=seg["end"],
                speaker=seg["speaker"],
            ))

        if not self._quiet:
            n_speakers = result.get("merged_speakers_detected", len(set(s.speaker for s in segments)))
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
