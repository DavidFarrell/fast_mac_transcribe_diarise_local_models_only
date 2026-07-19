"""
Unit tests for SenkoDiarizer (senko_diarisation.py).

Covers the silent-audio crash fix: senko.Diarizer.diarize() returns None
(not a result dict) when it detects no speech at all in the input - e.g.
a silent or pure-tone clip. Previously SenkoDiarizer.diarise() did
result["merged_segments"] unconditionally, so a None result raised
TypeError: 'NoneType' object is not subscriptable, crashing the whole CLI
run on legitimately-silent input. None must be treated as "no speech
detected" and produce an empty segment list instead.

Also covers the numba cache LOAD-side hardening in _patch_numba_cache:
senko's numba JIT cache is one fixed shared directory used by every
concurrently-running process on the machine (see numba_cache.py), so a
live run can read a cache entry while another live run is concurrently
writing it. numba's own IndexDataCacheFile.load() only catches OSError;
a torn/corrupt pickle read (index or data file) raises
pickle.UnpicklingError/EOFError/ValueError, which is NOT an OSError and
was NOT previously caught anywhere - it would propagate up through
numba's JIT dispatch and crash the whole transcription/diarisation run.
_patch_numba_cache now wraps load() the same way it already wraps save(),
so any read failure degrades to "cache miss" (recompile that function)
instead of crashing.
"""

import types
from unittest import mock

import numba
import pytest

from diarise_transcribe import senko_diarisation
from diarise_transcribe.senko_diarisation import SenkoDiarizer


@pytest.fixture(autouse=True)
def _reset_patch_idempotency_guard():
    """_patch_numba_cache() only re-wraps IndexDataCacheFile.load/save the
    first time it's called on a given class (see the
    ``_diarise_transcribe_patched`` sentinel) so repeated calls in one
    process don't build unbounded nested closures. Tests in this file
    monkeypatch load/save to a fresh fake per-test and then call
    _patch_numba_cache() expecting it to wrap THAT fake - so both the
    sentinel AND whatever _patch_numba_cache itself directly installed on
    .load/.save (via plain assignment, not monkeypatch.setattr, so
    pytest's own monkeypatch teardown never touches it) must be restored
    around every test. Otherwise a test running after an earlier one that
    already patched the class would either see _patch_numba_cache
    silently no-op (sentinel still set) or start from an already-wrapped
    .save/.load instead of the true original.

    Deliberately does the save/restore manually (plain assignment) rather
    than via monkeypatch.setattr: a monkeypatch call made from an autouse
    fixture's teardown (after its own yield) still lands on the SAME
    per-test undo stack as any monkeypatch.setattr calls the test body
    itself made, and monkeypatch unwinds that whole stack in LIFO order at
    end of test regardless of which fixture layer registered which call -
    so a fixture-teardown monkeypatch.setattr here would itself immediately
    be undone by the test's own patches unwinding after it, leaking the
    test's stub right back onto the class. Plain assignment has no such
    stack and is unconditionally the last word."""
    import numba.core.caching as caching

    true_save = caching.IndexDataCacheFile.save
    true_load = caching.IndexDataCacheFile.load
    true_sentinel = getattr(caching.IndexDataCacheFile, "_diarise_transcribe_patched", False)

    caching.IndexDataCacheFile._diarise_transcribe_patched = False

    yield

    caching.IndexDataCacheFile.save = true_save
    caching.IndexDataCacheFile.load = true_load
    caching.IndexDataCacheFile._diarise_transcribe_patched = true_sentinel


class _FakeSenkoDiarizerNone:
    """Stand-in for senko.Diarizer whose diarize() returns None (no speech)."""

    def diarize(self, audio_path, generate_colors=False):
        return None


class _FakeSenkoDiarizerNormal:
    """Stand-in for senko.Diarizer returning a normal result dict."""

    def diarize(self, audio_path, generate_colors=False):
        return {
            "merged_segments": [
                {"start": 0.0, "end": 1.5, "speaker": "SPEAKER_00"},
                {"start": 1.5, "end": 3.0, "speaker": "SPEAKER_01"},
            ],
            "merged_speakers_detected": 2,
        }


class TestSenkoDiarizerNoneResult:
    def test_diarise_returns_empty_list_when_result_is_none(self, monkeypatch):
        """The core fix: a None result (no speech detected) must produce
        an empty list of segments, not raise TypeError."""
        diarizer = SenkoDiarizer(quiet=True)
        diarizer._diarizer = _FakeSenkoDiarizerNone()

        segments = diarizer.diarise("fake_silent.wav")

        assert segments == []

    def test_diarise_does_not_raise_on_none_result(self, monkeypatch):
        diarizer = SenkoDiarizer(quiet=True)
        diarizer._diarizer = _FakeSenkoDiarizerNone()

        # Must not raise TypeError: 'NoneType' object is not subscriptable
        try:
            diarizer.diarise("fake_silent.wav")
        except TypeError as e:
            pytest.fail(f"diarise() raised TypeError on None result: {e}")

    def test_diarise_prints_no_speakers_message_when_not_quiet(self, capsys):
        diarizer = SenkoDiarizer(quiet=False)
        diarizer._diarizer = _FakeSenkoDiarizerNone()

        diarizer.diarise("fake_silent.wav")

        # The merged diarise() (embedded base) reports a None/empty result
        # as "No speakers detected in the audio." for both the no-speech and
        # empty-merged_segments cases.
        captured = capsys.readouterr()
        assert "no speakers detected" in captured.out.lower()

    def test_diarise_silent_when_quiet_true(self, capsys):
        diarizer = SenkoDiarizer(quiet=True)
        diarizer._diarizer = _FakeSenkoDiarizerNone()

        diarizer.diarise("fake_silent.wav")

        captured = capsys.readouterr()
        assert captured.out == ""

    def test_diarise_still_works_normally_with_real_result(self):
        """Regression guard: the None handling must not break the normal
        (speech-detected) path."""
        diarizer = SenkoDiarizer(quiet=True)
        diarizer._diarizer = _FakeSenkoDiarizerNormal()

        segments = diarizer.diarise("fake_speech.wav")

        assert len(segments) == 2
        assert segments[0].speaker == "SPEAKER_00"
        assert segments[0].start == 0.0
        assert segments[0].end == 1.5
        assert segments[1].speaker == "SPEAKER_01"

    def test_ensure_loaded_not_called_again_when_diarizer_preset(self):
        """Sanity check on the test doubles above: setting _diarizer
        directly (as these tests do) must bypass the real senko import/
        load path entirely, so these tests work without senko installed."""
        diarizer = SenkoDiarizer(quiet=True)
        diarizer._diarizer = _FakeSenkoDiarizerNone()

        with mock.patch("builtins.__import__", side_effect=AssertionError("must not import senko")):
            # Should not attempt to import senko since _diarizer is already set.
            segments = diarizer.diarise("fake_silent.wav")
        assert segments == []


# ---------------------------------------------------------------------------
# _patch_numba_cache: fault-tolerant SAVE (pre-existing) and LOAD (new)
# ---------------------------------------------------------------------------


class _FakeIndexDataCacheFile:
    """Minimal stand-in for numba.core.caching.IndexDataCacheFile so we can
    exercise SenkoDiarizer._patch_numba_cache's monkey-patching without a
    real numba cache directory or real JIT compilation."""

    def save(self, key, data):
        raise NotImplementedError("overridden by patch target in tests")

    def load(self, key):
        raise NotImplementedError("overridden by patch target in tests")


class TestPatchNumbaCacheSave:
    """Pre-existing behaviour: save() must swallow ReferenceError/KeyError
    (numba 0.63+ weakref-vanished bug) so a save failure never blocks JIT
    compilation from succeeding."""

    def test_save_swallows_reference_error(self, monkeypatch):
        import numba.core.caching as caching

        def boom(self, key, data):
            raise ReferenceError("weakly-referenced object no longer exists")

        monkeypatch.setattr(caching.IndexDataCacheFile, "save", boom, raising=True)

        SenkoDiarizer._patch_numba_cache()

        # Must not raise.
        result = caching.IndexDataCacheFile.save(_FakeIndexDataCacheFile(), "key", "data")
        assert result is None

    def test_save_swallows_key_error(self, monkeypatch):
        import numba.core.caching as caching

        def boom(self, key, data):
            raise KeyError("some-key")

        monkeypatch.setattr(caching.IndexDataCacheFile, "save", boom, raising=True)

        SenkoDiarizer._patch_numba_cache()

        result = caching.IndexDataCacheFile.save(_FakeIndexDataCacheFile(), "key", "data")
        assert result is None

    @pytest.mark.parametrize(
        "exc",
        [
            EOFError("Ran out of input"),
            ValueError("unsupported pickle protocol: 99"),
        ],
    )
    def test_save_swallows_load_index_errors_from_concurrent_torn_read(
        self, monkeypatch, exc
    ):
        """The concrete gap this patch closes: save() calls
        self._load_index() (the SAME pickle-based index read load() uses)
        BEFORE any weakref-related code runs - so a concurrent
        torn/corrupt index read during save() raises EOFError/ValueError/
        UnpicklingError, not just ReferenceError/KeyError. Those must be
        swallowed too, or a concurrent reader-during-write still crashes
        the whole run via save() even though load() is hardened."""
        import numba.core.caching as caching

        def boom(self, key, data):
            raise exc

        monkeypatch.setattr(caching.IndexDataCacheFile, "save", boom, raising=True)

        SenkoDiarizer._patch_numba_cache()

        result = caching.IndexDataCacheFile.save(_FakeIndexDataCacheFile(), "key", "data")
        assert result is None

    def test_save_still_returns_normally_on_success(self, monkeypatch):
        import numba.core.caching as caching

        def fine(self, key, data):
            return "saved-ok"

        monkeypatch.setattr(caching.IndexDataCacheFile, "save", fine, raising=True)

        SenkoDiarizer._patch_numba_cache()

        result = caching.IndexDataCacheFile.save(_FakeIndexDataCacheFile(), "key", "data")
        assert result == "saved-ok"

    def test_save_survives_load_index_eoferror_against_real_numba_save(self):
        """Reproduces the exact defect scenario: patch the REAL (not
        stubbed) IndexDataCacheFile.save, but make the real _load_index -
        which save() calls as its very first line, before any
        weakref-related code - raise EOFError (the exact exception cited
        in this module's own docstring for a torn/corrupt concurrent
        read). Prior to this fix, save()'s guard only caught
        (ReferenceError, KeyError), so this EOFError propagated straight
        out of save() uncaught."""
        import numba.core.caching as caching

        def boom(self):
            raise EOFError("Ran out of input")

        with mock.patch.object(caching.IndexDataCacheFile, "_load_index", boom):
            SenkoDiarizer._patch_numba_cache()
            inst = object.__new__(caching.IndexDataCacheFile)
            # Must not raise.
            result = caching.IndexDataCacheFile.save(inst, "some-key", b"data")
            assert result is None


class TestPatchNumbaCacheLoad:
    """New hardening: load() must swallow ANY exception (not just OSError,
    which numba's own code already catches internally) since a torn/
    corrupt pickle read - from a concurrently-writing process sharing the
    same numba cache dir - raises pickle.UnpicklingError/EOFError/
    ValueError, none of which are OSError subclasses and none of which
    were caught before this patch."""

    @pytest.mark.parametrize(
        "exc",
        [
            EOFError("Ran out of input"),
            ValueError("unsupported pickle protocol: 99"),
            Exception("generic unpickling failure"),
        ],
    )
    def test_load_swallows_non_oserror_exceptions_and_returns_none(self, monkeypatch, exc):
        import numba.core.caching as caching
        import pickle

        def boom(self, key):
            raise exc

        monkeypatch.setattr(caching.IndexDataCacheFile, "load", boom, raising=True)

        SenkoDiarizer._patch_numba_cache()

        # Must not raise - must degrade to "cache miss" (None), just like a
        # legitimately-absent cache entry would.
        result = caching.IndexDataCacheFile.load(_FakeIndexDataCacheFile(), "some-key")
        assert result is None

    def test_load_swallows_unpickling_error_specifically(self, monkeypatch):
        """The concrete real-world failure mode: a genuinely truncated/
        corrupt pickle stream, e.g. from reading a data/index file a
        concurrent writer only partially got through replacing (extremely
        rare given numba's own tmp+os.replace atomicity, but also covers
        genuinely corrupted cache entries left by a killed process)."""
        import numba.core.caching as caching
        import pickle

        def boom(self, key):
            raise pickle.UnpicklingError("invalid load key, 'X'.")

        monkeypatch.setattr(caching.IndexDataCacheFile, "load", boom, raising=True)

        SenkoDiarizer._patch_numba_cache()

        result = caching.IndexDataCacheFile.load(_FakeIndexDataCacheFile(), "some-key")
        assert result is None

    def test_load_still_returns_normal_cache_hit_on_success(self, monkeypatch):
        import numba.core.caching as caching

        def fine(self, key):
            return ("cached", "payload")

        monkeypatch.setattr(caching.IndexDataCacheFile, "load", fine, raising=True)

        SenkoDiarizer._patch_numba_cache()

        result = caching.IndexDataCacheFile.load(_FakeIndexDataCacheFile(), "some-key")
        assert result == ("cached", "payload")

    def test_load_still_returns_none_for_legitimate_cache_miss(self, monkeypatch):
        """A legitimate cache miss (key simply not present) already
        returns None from numba's own code - the patch must preserve that,
        not turn it into something else."""
        import numba.core.caching as caching

        def miss(self, key):
            return None

        monkeypatch.setattr(caching.IndexDataCacheFile, "load", miss, raising=True)

        SenkoDiarizer._patch_numba_cache()

        result = caching.IndexDataCacheFile.load(_FakeIndexDataCacheFile(), "some-key")
        assert result is None


class TestPatchNumbaCacheIdempotent:
    def test_patch_can_be_called_multiple_times_without_error(self):
        """_ensure_loaded calls _patch_numba_cache() on every SenkoDiarizer
        instance's first use; multiple instances in the same process (or
        multiple calls) must not error or infinitely wrap."""
        SenkoDiarizer._patch_numba_cache()
        SenkoDiarizer._patch_numba_cache()
        SenkoDiarizer._patch_numba_cache()
        # If we get here without exception, re-patching is safe.

    def test_repeated_calls_do_not_build_nested_closure_chain(self):
        """The property the docstring above actually claims ('must not...
        infinitely wrap') but which the exception-only check above never
        verifies: re-patching must be a true no-op, not just
        exception-free. Without an idempotency guard, each call re-wraps
        whatever .save currently is, so N calls produce an N-deep nested
        closure chain (unbounded reference growth in any long-lived
        process that constructs more than one SenkoDiarizer - a future
        daemon, a REPL, or an in-process embedding of this library).
        Verify directly via closure introspection that after multiple
        calls, the installed .save's closure still refers to the ORIGINAL
        (unpatched) method, not to a previously-installed wrapper."""
        import numba.core.caching as caching

        original_save = caching.IndexDataCacheFile.save

        SenkoDiarizer._patch_numba_cache()
        first_wrapped = caching.IndexDataCacheFile.save

        SenkoDiarizer._patch_numba_cache()
        SenkoDiarizer._patch_numba_cache()
        SenkoDiarizer._patch_numba_cache()
        final_wrapped = caching.IndexDataCacheFile.save

        # Idempotent: the 2nd/3rd/4th calls must not replace the wrapper
        # installed by the 1st call with a new one wrapping IT.
        assert final_wrapped is first_wrapped

        # And the one wrapper that does exist must close over the true
        # original method, never over another _safe_save.
        closure_contents = [
            cell.cell_contents for cell in (final_wrapped.__closure__ or ())
        ]
        assert original_save in closure_contents
        assert not any(
            getattr(c, "__name__", None) == "_safe_save" for c in closure_contents
        )


# ---------------------------------------------------------------------------
# Embedded (muesli-backend) concurrency fix: _restore_numba_njit + the single
# retry on a transient "underlying object has vanished" ReferenceError. This
# is the OTHER half of the danger-zone union merge (the reconciliation keeps
# BOTH this single-process njit-restore/retry path and the cross-process
# _patch_numba_cache wrapper above - see design/slice0.5-reconciliation.md).
# ---------------------------------------------------------------------------


def test_restore_numba_njit_uses_senko_original() -> None:
    original_njit = numba.njit

    def patched_njit(*args, **kwargs):
        return original_njit(*args, **kwargs)

    fake_senko = types.SimpleNamespace(
        config=types.SimpleNamespace(_original_njit=original_njit)
    )

    numba.njit = patched_njit
    try:
        senko_diarisation._restore_numba_njit(fake_senko)
        assert numba.njit is original_njit
    finally:
        numba.njit = original_njit


def test_diarise_retries_once_after_transient_reference_error(monkeypatch) -> None:
    senko_diarisation._native_diarizer_cache.clear()

    warmup_values: list[bool] = []
    attempts = {"count": 0}

    class FakeNativeDiarizer:
        def __init__(self, *, warmup: bool):
            self._warmup = warmup

        def diarize(self, _audio_path: str, generate_colors: bool = False):
            assert generate_colors is False
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise ReferenceError("underlying object has vanished")
            return {
                "merged_segments": [
                    {"start": 0.0, "end": 1.25, "speaker": "SPEAKER_01"},
                ],
                "merged_speakers_detected": 1,
            }

    class FakeSenkoModule:
        class config:
            _original_njit = numba.njit

        @staticmethod
        def Diarizer(device: str, warmup: bool, quiet: bool):
            assert device == "auto"
            assert quiet is True
            warmup_values.append(warmup)
            return FakeNativeDiarizer(warmup=warmup)

    monkeypatch.setattr(senko_diarisation, "_import_senko", lambda: FakeSenkoModule)

    diarizer = senko_diarisation.SenkoDiarizer(warmup=True, quiet=True)
    segments = diarizer.diarise("example.wav")

    assert warmup_values == [True, False]
    assert [(segment.start, segment.end, segment.speaker) for segment in segments] == [
        (0.0, 1.25, "SPEAKER_01"),
    ]
