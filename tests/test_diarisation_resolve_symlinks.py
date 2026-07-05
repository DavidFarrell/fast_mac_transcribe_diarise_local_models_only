"""
Unit tests for SortformerDiarizer._resolve_symlinks concurrency hardening.

The local model cache dir (<repo>/models/<model_name>) is shared by every
process using the sortformer backend on this machine, exactly like the
numba cache in numba_cache.py. Two concurrent runs could previously both
see the destination missing/incomplete and both shutil.copytree straight
into the same destination path, interleaving writes. This is hardened with
the same staging-dir + atomic os.rename pattern used in
numba_cache.bootstrap_senko_cache_if_empty.
"""

import shutil
import threading
import time
from pathlib import Path

import pytest

from diarise_transcribe.diarisation import SortformerDiarizer


def _make_fake_source(path: Path, *, with_manifest=True, payload=b"model-bytes"):
    path.mkdir(parents=True, exist_ok=True)
    if with_manifest:
        (path / "Manifest.json").write_text('{"fake": "manifest"}')
    (path / "weights.bin").write_bytes(payload)


@pytest.fixture
def diarizer():
    # SortformerDiarizer.__init__ doesn't touch the network or CoreML.
    return SortformerDiarizer(model_name="default")


@pytest.fixture
def local_models_dir(monkeypatch):
    """_resolve_symlinks computes local_dir from Path(__file__) of the
    diarisation module itself (<repo>/models) - not overridable via env var.
    We instead point Path(__file__).parent.parent.parent at a tmp dir by
    monkeypatching the module's __file__ is fragile; simpler: just use the
    real repo-relative models/ dir but under a uniquely-named model so
    tests never collide with each other or a real cached model, and clean
    up afterwards."""
    import diarise_transcribe.diarisation as diar_mod

    real_dir = Path(diar_mod.__file__).parent.parent.parent / "models"
    yield real_dir
    # Cleanup handled per-test via unique model names; nothing global here.


class TestResolveSymlinksBasic:
    def test_copies_fresh_model_and_returns_local_path(self, tmp_path, diarizer, local_models_dir):
        model_name = f"test-model-fresh-{id(tmp_path)}"
        source = tmp_path / model_name
        _make_fake_source(source)

        local_model_path = local_models_dir / model_name
        assert not local_model_path.exists()
        try:
            result = diarizer._resolve_symlinks(str(source))
            assert Path(result) == local_model_path
            assert (local_model_path / "Manifest.json").exists()
            assert (local_model_path / "weights.bin").read_bytes() == b"model-bytes"
        finally:
            shutil.rmtree(local_model_path, ignore_errors=True)

    def test_uses_existing_complete_cache_without_recopying(self, tmp_path, diarizer, local_models_dir, monkeypatch):
        model_name = f"test-model-cached-{id(tmp_path)}"
        source = tmp_path / model_name
        _make_fake_source(source, payload=b"source-bytes-should-not-be-used")

        local_model_path = local_models_dir / model_name
        _make_fake_source(local_model_path, payload=b"already-cached-bytes")

        def boom_copytree(*a, **k):
            raise AssertionError("must not copy when a complete cache already exists")

        monkeypatch.setattr(shutil, "copytree", boom_copytree)

        try:
            result = diarizer._resolve_symlinks(str(source))
            assert Path(result) == local_model_path
            assert (local_model_path / "weights.bin").read_bytes() == b"already-cached-bytes"
        finally:
            shutil.rmtree(local_model_path, ignore_errors=True)

    def test_incomplete_existing_dir_is_replaced_not_merged(self, tmp_path, diarizer, local_models_dir):
        model_name = f"test-model-incomplete-{id(tmp_path)}"
        source = tmp_path / model_name
        _make_fake_source(source, payload=b"fresh-bytes")

        local_model_path = local_models_dir / model_name
        # Incomplete: no Manifest.json.
        local_model_path.mkdir(parents=True)
        (local_model_path / "partial.bin").write_bytes(b"leftover-from-interrupted-copy")

        try:
            result = diarizer._resolve_symlinks(str(source))
            assert Path(result) == local_model_path
            assert (local_model_path / "Manifest.json").exists()
            assert (local_model_path / "weights.bin").read_bytes() == b"fresh-bytes"
        finally:
            shutil.rmtree(local_model_path, ignore_errors=True)

    def test_no_leftover_staging_dirs_after_success(self, tmp_path, diarizer, local_models_dir):
        model_name = f"test-model-nostaging-{id(tmp_path)}"
        source = tmp_path / model_name
        _make_fake_source(source)

        local_model_path = local_models_dir / model_name
        try:
            diarizer._resolve_symlinks(str(source))
            leftovers = [
                p for p in local_models_dir.iterdir()
                if p.name.startswith(f".{model_name}.staging")
            ]
            assert leftovers == []
        finally:
            shutil.rmtree(local_model_path, ignore_errors=True)

    def test_never_rmtrees_incomplete_dest_before_copy_completes(self, tmp_path, diarizer, local_models_dir, monkeypatch):
        """The critical invariant: an incomplete destination dir must not
        be deleted until AFTER our own copy has succeeded into staging -
        otherwise another concurrently-running process reading/writing
        that same incomplete dir could be clobbered mid-flight."""
        model_name = f"test-model-norace-{id(tmp_path)}"
        source = tmp_path / model_name
        _make_fake_source(source)

        local_model_path = local_models_dir / model_name
        local_model_path.mkdir(parents=True)
        (local_model_path / "partial.bin").write_bytes(b"do-not-delete-me-early")

        rmtree_calls = []
        real_rmtree = shutil.rmtree

        def tracking_rmtree(path, *a, **k):
            rmtree_calls.append(str(path))
            return real_rmtree(path, *a, **k)

        monkeypatch.setattr(shutil, "rmtree", tracking_rmtree)

        real_copytree = shutil.copytree

        def spy_copytree(src, dst, *a, **k):
            # At the moment of copytree, the (incomplete) dest must still
            # exist untouched - proving we didn't delete it before staging
            # the new copy.
            assert local_model_path.exists()
            assert (local_model_path / "partial.bin").read_bytes() == b"do-not-delete-me-early"
            return real_copytree(src, dst, *a, **k)

        monkeypatch.setattr(shutil, "copytree", spy_copytree)

        try:
            diarizer._resolve_symlinks(str(source))
        finally:
            shutil.rmtree(local_model_path, ignore_errors=True)


class TestResolveSymlinksConcurrency:
    def test_two_concurrent_resolves_never_leave_partial_dest(self, tmp_path, local_models_dir):
        """Two threads race to resolve the same missing model. A watcher
        polls the destination weights file throughout and must only ever
        observe "doesn't exist yet" or the full expected byte content -
        never a truncated/partial file."""
        model_name = f"test-model-race-{id(tmp_path)}"
        source = tmp_path / model_name
        payload = b"y" * (2 * 1024 * 1024)
        _make_fake_source(source, payload=payload)

        local_model_path = local_models_dir / model_name

        observations = []
        stop = threading.Event()

        def watcher():
            target = local_model_path / "weights.bin"
            while not stop.is_set():
                if target.exists():
                    try:
                        data = target.read_bytes()
                    except OSError:
                        continue
                    if len(data) != len(payload):
                        observations.append(len(data))
                time.sleep(0.0005)

        watcher_thread = threading.Thread(target=watcher)
        watcher_thread.start()

        results = []

        def run():
            d = SortformerDiarizer(model_name="default")
            results.append(d._resolve_symlinks(str(source)))

        t1 = threading.Thread(target=run)
        t2 = threading.Thread(target=run)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        stop.set()
        watcher_thread.join()

        try:
            assert observations == [], (
                f"observed a partial/truncated destination file during "
                f"concurrent resolve: sizes {observations} (expected only "
                f"0 or full {len(payload)})"
            )
            assert (local_model_path / "weights.bin").read_bytes() == payload
            leftovers = [
                p for p in local_models_dir.iterdir()
                if p.name.startswith(f".{model_name}.staging")
            ]
            assert leftovers == []
        finally:
            shutil.rmtree(local_model_path, ignore_errors=True)

    def test_swap_of_a_stale_incomplete_dest_has_a_known_transient_missing_window(
        self, tmp_path, diarizer, local_models_dir
    ):
        """Documents (rather than hides) a real, narrow gap: when replacing
        an existing-but-INCOMPLETE local_model_path, _resolve_symlinks does
        `os.rename(local_model_path, aside)` then `os.rename(staging,
        local_model_path)` as two separate syscalls - so local_model_path
        is observably MISSING from disk for a real (if tiny) wall-clock
        window between them. This contradicts the function's own comment
        claiming "never a moment where dest is observably missing/
        cleared". Verified directly: a watcher thread polling
        local_model_path.exists() throughout a real _resolve_symlinks call
        against a pre-existing incomplete dest does observe at least one
        "missing" sample.

        This is accepted (not fixed) because every caller of
        local_model_path in this module only ever acts on it after first
        confirming _is_complete(), and dest is only ever swapped once
        already confirmed incomplete - so nothing in this codebase today
        treats a momentarily-missing dest as an error. If a future reader
        starts inspecting dest without going through this module's own
        completeness gate, this window would need closing (e.g. via the
        mkdir-claim pattern used in
        numba_cache.bootstrap_senko_cache_if_empty). This test exists so
        that if the window ever widens or a new unguarded reader appears,
        someone notices."""
        model_name = f"test-model-transient-gap-{id(tmp_path)}"
        source = tmp_path / model_name
        _make_fake_source(source, payload=b"fresh-bytes-for-gap-test")

        local_model_path = local_models_dir / model_name
        # Pre-existing INCOMPLETE dest (no Manifest.json) - the only case
        # that reaches the aside-then-in swap.
        local_model_path.mkdir(parents=True)
        (local_model_path / "partial.bin").write_bytes(b"stale-incomplete-copy")

        observed_missing = []
        stop = threading.Event()

        def watcher():
            while not stop.is_set():
                if not local_model_path.exists():
                    observed_missing.append(True)
                time.sleep(0.0)

        watcher_thread = threading.Thread(target=watcher)
        watcher_thread.start()

        try:
            diarizer._resolve_symlinks(str(source))
        finally:
            stop.set()
            watcher_thread.join()
            shutil.rmtree(local_model_path, ignore_errors=True)

        # This assertion documents the CURRENT (accepted) behaviour: the
        # gap exists. If a future fix closes it, flip this to
        # `assert observed_missing == []` as part of that change.
        assert observed_missing, (
            "expected the known transient-missing window during the "
            "aside-then-in swap of a stale incomplete dest; if this is "
            "empty the window may have been closed - update this test's "
            "assertion (and the docstring/comment in diarisation.py) to "
            "match the new, stronger guarantee"
        )
