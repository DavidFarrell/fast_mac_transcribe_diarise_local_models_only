"""
Unit tests for the warm-once / bootstrap-per-machine numba cache
(numba_cache.py).

Key fact this module is built around (verified empirically against the
installed senko package): senko/config.py unconditionally overwrites
NUMBA_CACHE_DIR with its own fixed ~/.cache/senko/numba_cache the moment
senko is imported, and numba snapshots NUMBA_CACHE_DIR into
numba.core.config.CACHE_DIR at first import and never re-reads the
environment. So Senko's numba cache always lives at one fixed, shared,
already-persistent location - not a per-PID directory. The only lever that
reliably controls where senko's cache lands is $HOME (since senko computes
its cache dir via Path.home()).

Covers:
  - key derivation changes when versions change
  - bootstrap-on-start: canonical present / absent / corrupt, and the
    critical "never touch a non-empty real cache dir" invariant
  - fail-soft on copy errors
  - atomic install of a freshly warmed cache
  - --warm-cache CLI wiring (heavy subprocess run mocked out)

The real end-to-end check (--warm-cache for real, then a real warm
transcription) is done separately and reported in prose, not as an
automated test, since it takes real wall-clock time and depends on models
being downloaded.
"""

import os
import sys
from pathlib import Path
from unittest import mock

import pytest

from diarise_transcribe import numba_cache


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def canonical_root(tmp_path, monkeypatch):
    """Point the canonical cache root at a throwaway tmp dir for this test."""
    root = tmp_path / "numba-canonical"
    monkeypatch.setenv(numba_cache._ROOT_ENV_VAR, str(root))
    return root


@pytest.fixture
def senko_cache_dir(tmp_path, monkeypatch):
    """Point the (simulated) real senko cache dir at a throwaway tmp dir."""
    d = tmp_path / "senko-real-cache"
    monkeypatch.setenv(numba_cache._SENKO_CACHE_DIR_ENV_VAR, str(d))
    return d


# ---------------------------------------------------------------------------
# Key derivation
# ---------------------------------------------------------------------------


class TestCacheKey:
    def test_key_contains_python_version(self):
        key = numba_cache.cache_key()
        expected_py = f"py{sys.version_info.major}.{sys.version_info.minor}"
        assert expected_py in key

    def test_key_changes_when_numba_version_changes(self, monkeypatch):
        def fake_version_or_ref(dist_name):
            if dist_name == "numba":
                return "9.9.9"
            return "unknown"

        monkeypatch.setattr(numba_cache, "_package_version_or_ref", fake_version_or_ref)
        key_a = numba_cache.cache_key()

        def fake_version_or_ref_2(dist_name):
            if dist_name == "numba":
                return "1.2.3"
            return "unknown"

        monkeypatch.setattr(numba_cache, "_package_version_or_ref", fake_version_or_ref_2)
        key_b = numba_cache.cache_key()

        assert key_a != key_b

    def test_key_changes_when_senko_version_changes(self, monkeypatch):
        calls = {"numba": "0.64.0"}

        def fake(dist_name):
            if dist_name == "numba":
                return calls["numba"]
            if dist_name == "senko":
                return calls.get("senko", "0.1.0")
            return "unknown"

        monkeypatch.setattr(numba_cache, "_package_version_or_ref", fake)
        key_a = numba_cache.cache_key()

        calls["senko"] = "gitabc123def456"
        key_b = numba_cache.cache_key()

        assert key_a != key_b

    def test_key_changes_when_python_version_changes(self, monkeypatch):
        monkeypatch.setattr(numba_cache, "_package_version_or_ref", lambda name: "1.0")

        fake_info_a = mock.Mock(major=3, minor=10)
        with mock.patch.object(numba_cache.sys, "version_info", fake_info_a):
            key_a = numba_cache.cache_key()

        fake_info_b = mock.Mock(major=3, minor=12)
        with mock.patch.object(numba_cache.sys, "version_info", fake_info_b):
            key_b = numba_cache.cache_key()

        assert key_a != key_b

    def test_key_stable_for_identical_environment(self, monkeypatch):
        monkeypatch.setattr(numba_cache, "_package_version_or_ref", lambda name: f"v-{name}")
        assert numba_cache.cache_key() == numba_cache.cache_key()

    def test_git_pinned_package_uses_commit_id_not_static_version(self, monkeypatch, tmp_path):
        """senko is installed via a git URL and always reports version 0.1.0
        regardless of which commit is checked out - the commit id from
        direct_url.json must be preferred so that upgrading senko (a new
        commit, same static version) still changes the cache key."""

        class FakeDist:
            def __init__(self, commit_id):
                self._commit_id = commit_id
                self.version = "0.1.0"

            def read_text(self, name):
                if name == "direct_url.json":
                    return (
                        '{"url": "https://github.com/narcotic-sh/senko.git", '
                        '"vcs_info": {"vcs": "git", "commit_id": "%s"}}' % self._commit_id
                    )
                return None

        def fake_distribution(name):
            if name == "senko":
                return FakeDist("f1bc30c2ff37d807eec91bec5246eea3fe2dcbe3")
            raise numba_cache.importlib.metadata.PackageNotFoundError(name)

        monkeypatch.setattr(numba_cache.importlib.metadata, "distribution", fake_distribution)
        ref_a = numba_cache._package_version_or_ref("senko")

        def fake_distribution_2(name):
            if name == "senko":
                return FakeDist("0000000000000000000000000000000000000000")
            raise numba_cache.importlib.metadata.PackageNotFoundError(name)

        monkeypatch.setattr(numba_cache.importlib.metadata, "distribution", fake_distribution_2)
        ref_b = numba_cache._package_version_or_ref("senko")

        assert ref_a != ref_b
        assert ref_a == "gitf1bc30c2ff37"

    def test_missing_package_reports_unknown(self, monkeypatch):
        def fake_distribution(name):
            raise numba_cache.importlib.metadata.PackageNotFoundError(name)

        monkeypatch.setattr(numba_cache.importlib.metadata, "distribution", fake_distribution)
        assert numba_cache._package_version_or_ref("nonexistent-package-xyz") == "unknown"


# ---------------------------------------------------------------------------
# senko_cache_dir() resolution
# ---------------------------------------------------------------------------


class TestSenkoCacheDirResolution:
    def test_matches_senko_configs_own_computation(self, monkeypatch, tmp_path):
        """senko/config.py computes Path.home() / '.cache' / 'senko' /
        'numba_cache'. Our resolution (absent the test override) must
        agree exactly, since it's what determines whether bootstrap logic
        is even looking at the right directory."""
        monkeypatch.delenv(numba_cache._SENKO_CACHE_DIR_ENV_VAR, raising=False)
        fake_home = tmp_path / "fakehome"
        monkeypatch.setattr(numba_cache.Path, "home", classmethod(lambda cls: fake_home))
        assert numba_cache.senko_cache_dir() == fake_home / ".cache" / "senko" / "numba_cache"

    def test_override_env_var_takes_precedence(self, senko_cache_dir):
        assert numba_cache.senko_cache_dir() == senko_cache_dir


# ---------------------------------------------------------------------------
# Bootstrap-on-start: present / absent / corrupt / already-warm
# ---------------------------------------------------------------------------


class TestBootstrapSenkoCacheIfEmpty:
    def test_no_canonical_dir_returns_false_and_leaves_dest_untouched(
        self, canonical_root, senko_cache_dir
    ):
        result = numba_cache.bootstrap_senko_cache_if_empty(key="somekey")
        assert result is False
        if senko_cache_dir.exists():
            assert list(senko_cache_dir.iterdir()) == []

    def test_empty_canonical_dir_treated_as_absent(self, canonical_root, senko_cache_dir):
        key = "emptykey"
        (canonical_root / key).mkdir(parents=True)
        result = numba_cache.bootstrap_senko_cache_if_empty(key=key)
        assert result is False

    def test_present_canonical_dir_is_copied_in_when_dest_missing(
        self, canonical_root, senko_cache_dir
    ):
        key = "goodkey"
        source = canonical_root / key
        source.mkdir(parents=True)
        (source / "cache_index.pyc.i.json").write_text('{"fake": "index"}')
        subdir = source / "subdir"
        subdir.mkdir()
        (subdir / "artifact.nbi").write_bytes(b"\x00\x01\x02")

        assert not senko_cache_dir.exists()
        result = numba_cache.bootstrap_senko_cache_if_empty(key=key)

        assert result is True
        assert (senko_cache_dir / "cache_index.pyc.i.json").read_text() == '{"fake": "index"}'
        assert (senko_cache_dir / "subdir" / "artifact.nbi").read_bytes() == b"\x00\x01\x02"

    def test_present_canonical_dir_copied_in_when_dest_exists_but_empty(
        self, canonical_root, senko_cache_dir
    ):
        """senko itself does cache_dir.mkdir(parents=True, exist_ok=True)
        unconditionally - the real dest dir may already exist as an empty
        directory (created by senko or a previous no-op run)."""
        key = "goodkey2"
        source = canonical_root / key
        source.mkdir(parents=True)
        (source / "warm.txt").write_text("warm")

        senko_cache_dir.mkdir(parents=True)  # exists but empty

        result = numba_cache.bootstrap_senko_cache_if_empty(key=key)

        assert result is True
        assert (senko_cache_dir / "warm.txt").read_text() == "warm"

    def test_never_touches_already_warm_real_cache_dir(self, canonical_root, senko_cache_dir):
        """THE critical invariant: if the real senko cache dir already has
        ANY content (i.e. it's already warm, possibly from another
        concurrently-running process), bootstrap must be a strict no-op -
        never copy in, never merge, never touch it at all. This is what
        guarantees a live run can never race with or clobber shared
        state."""
        key = "somekey"
        source = canonical_root / key
        source.mkdir(parents=True)
        (source / "would_be_copied.txt").write_text("should never appear")

        senko_cache_dir.mkdir(parents=True)
        (senko_cache_dir / "already_here.txt").write_text("pre-existing warm content")

        result = numba_cache.bootstrap_senko_cache_if_empty(key=key)

        assert result is False
        assert (senko_cache_dir / "already_here.txt").read_text() == "pre-existing warm content"
        assert not (senko_cache_dir / "would_be_copied.txt").exists()
        # Canonical source must also be untouched.
        assert (source / "would_be_copied.txt").read_text() == "should never appear"

    def test_wrong_key_is_ignored_stale_cache(self, canonical_root, senko_cache_dir):
        """A canonical dir that exists for a *different* key than the one
        requested must be ignored entirely (stale-key handling)."""
        stale_key = "old-numba-0.60-senko-abc"
        current_key = "new-numba-0.64-senko-def"

        stale_source = canonical_root / stale_key
        stale_source.mkdir(parents=True)
        (stale_source / "stale.txt").write_text("stale artifact")

        result = numba_cache.bootstrap_senko_cache_if_empty(key=current_key)

        assert result is False
        if senko_cache_dir.exists():
            assert not (senko_cache_dir / "stale.txt").exists()

    def test_corrupt_canonical_dir_is_actually_a_file_fails_soft(
        self, canonical_root, senko_cache_dir, capsys
    ):
        """If the canonical 'directory' is corrupt (e.g. a stray file where
        a directory should be), bootstrap must fail soft, not crash."""
        key = "corruptkey"
        bogus = canonical_root / key
        bogus.parent.mkdir(parents=True, exist_ok=True)
        bogus.write_text("this should have been a directory")

        result = numba_cache.bootstrap_senko_cache_if_empty(key=key)

        assert result is False


# ---------------------------------------------------------------------------
# Concurrent bootstrap: two processes racing on an empty/missing dest must
# never leave dest in a truncated/partial state (regression for the
# in-place-copytree race).
# ---------------------------------------------------------------------------


class TestConcurrentBootstrapNeverLeavesPartialDest:
    def test_two_concurrent_bootstraps_never_observe_a_partial_dest(
        self, canonical_root, senko_cache_dir
    ):
        """Two threads both find dest empty/missing and race to bootstrap
        it from the same canonical source. A watcher thread polls dest
        throughout and must only ever observe "doesn't exist yet" or "the
        fully copied file with its complete byte content" - never a
        partially-written/truncated file, which is the in-place-copytree
        race this staging-then-atomic-rename design defends against."""
        import threading
        import time

        key = "racekey"
        source = canonical_root / key
        source.mkdir(parents=True)
        payload = b"x" * (2 * 1024 * 1024)  # large enough to not copy instantaneously
        (source / "cache_index.nbi").write_bytes(payload)

        observations = []
        stop = threading.Event()

        def watcher():
            target = senko_cache_dir / "cache_index.nbi"
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

        def bootstrap():
            results.append(numba_cache.bootstrap_senko_cache_if_empty(key=key))

        t1 = threading.Thread(target=bootstrap)
        t2 = threading.Thread(target=bootstrap)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        stop.set()
        watcher_thread.join()

        assert observations == [], (
            f"observed a partial/truncated destination file during concurrent "
            f"bootstrap: sizes {observations} (expected only 0 or full "
            f"{len(payload)})"
        )
        # Exactly one of the two racers should report having done the
        # bootstrap copy (the other should see dest already populated and
        # no-op); at least one must succeed since a canonical source exists.
        assert any(results)
        assert (senko_cache_dir / "cache_index.nbi").read_bytes() == payload
        # No leftover staging directories.
        leftovers = [
            p for p in senko_cache_dir.parent.iterdir()
            if p.name.startswith(".") and "bootstrap" in p.name
        ]
        assert leftovers == []


# ---------------------------------------------------------------------------
# Fail-soft on copy errors
# ---------------------------------------------------------------------------


class TestFailSoftOnCopyErrors:
    def test_copytree_exception_is_caught_and_logged(
        self, canonical_root, senko_cache_dir, capsys, monkeypatch
    ):
        key = "explodingkey"
        source = canonical_root / key
        source.mkdir(parents=True)
        (source / "file.txt").write_text("data")

        def boom(*args, **kwargs):
            raise OSError("simulated disk error during copy")

        monkeypatch.setattr(numba_cache.shutil, "copytree", boom)

        result = numba_cache.bootstrap_senko_cache_if_empty(key=key)

        assert result is False
        captured = capsys.readouterr()
        assert "Warning" in captured.err
        assert "cold" in captured.err.lower()

    def test_never_raises_even_on_unexpected_error(self, canonical_root, senko_cache_dir, monkeypatch):
        def boom(*args, **kwargs):
            raise RuntimeError("totally unexpected")

        monkeypatch.setattr(numba_cache, "canonical_dir_for_key", boom)

        # Must not raise.
        result = numba_cache.bootstrap_senko_cache_if_empty()
        assert result is False

    def test_permission_error_on_mkdir_fails_soft(self, canonical_root, senko_cache_dir, monkeypatch):
        key = "permkey"
        source = canonical_root / key
        source.mkdir(parents=True)
        (source / "file.txt").write_text("data")

        real_mkdir = Path.mkdir

        def fake_mkdir(self, *args, **kwargs):
            if self == senko_cache_dir.parent:
                raise PermissionError("simulated permission denied")
            return real_mkdir(self, *args, **kwargs)

        monkeypatch.setattr(Path, "mkdir", fake_mkdir)

        result = numba_cache.bootstrap_senko_cache_if_empty(key=key)
        assert result is False


# ---------------------------------------------------------------------------
# Atomic install
# ---------------------------------------------------------------------------


class TestAtomicInstall:
    def test_install_fresh_cache_no_prior_canonical(self, canonical_root, tmp_path):
        warm_dir = tmp_path / "warm_staging"
        warm_dir.mkdir()
        (warm_dir / "artifact.nbi").write_text("compiled")

        installed = numba_cache.install_canonical_cache(str(warm_dir), key="freshkey")

        assert installed == canonical_root / "freshkey"
        assert (installed / "artifact.nbi").read_text() == "compiled"
        # Source dir was moved (renamed), not left behind as a duplicate.
        assert not warm_dir.exists()

    def test_install_replaces_existing_canonical(self, canonical_root, tmp_path):
        key = "replacekey"
        old = canonical_root / key
        old.mkdir(parents=True)
        (old / "old.txt").write_text("old artifact")

        warm_dir = tmp_path / "warm_staging"
        warm_dir.mkdir()
        (warm_dir / "new.txt").write_text("new artifact")

        installed = numba_cache.install_canonical_cache(str(warm_dir), key=key)

        assert installed == old
        assert not (installed / "old.txt").exists()
        assert (installed / "new.txt").read_text() == "new artifact"
        # No leftover staging dirs.
        leftovers = [p for p in canonical_root.iterdir() if p.name != key]
        assert leftovers == []

    def test_install_missing_source_raises(self, canonical_root, tmp_path):
        with pytest.raises(FileNotFoundError):
            numba_cache.install_canonical_cache(str(tmp_path / "does_not_exist"), key="x")

    def test_install_rolls_back_on_rename_failure(self, canonical_root, tmp_path, monkeypatch):
        """If the final rename fails, the previous canonical dir must still
        be intact afterwards (never leave canonical missing)."""
        key = "rollbackkey"
        old = canonical_root / key
        old.mkdir(parents=True)
        (old / "old.txt").write_text("old artifact")

        warm_dir = tmp_path / "warm_staging"
        warm_dir.mkdir()
        (warm_dir / "new.txt").write_text("new artifact")

        real_rename = os.rename
        call_count = {"n": 0}

        def flaky_rename(src, dst):
            call_count["n"] += 1
            # First rename call moves old aside - let it succeed.
            # Second rename call (warm_dir -> dest) fails.
            if call_count["n"] == 2:
                raise OSError("simulated rename failure")
            return real_rename(src, dst)

        monkeypatch.setattr(numba_cache.os, "rename", flaky_rename)

        with pytest.raises(OSError):
            numba_cache.install_canonical_cache(str(warm_dir), key=key)

        # Old canonical must still be present and correct.
        assert old.exists()
        assert (old / "old.txt").read_text() == "old artifact"

    def test_prune_stale_removes_other_keys_keeps_current(self, canonical_root):
        current = canonical_root / "current-key"
        current.mkdir(parents=True)
        (current / "f.txt").write_text("keep me")

        stale1 = canonical_root / "stale-key-1"
        stale1.mkdir(parents=True)
        stale2 = canonical_root / "stale-key-2"
        stale2.mkdir(parents=True)

        removed = numba_cache.prune_stale_canonical_caches(keep_key="current-key")

        remaining = sorted(p.name for p in canonical_root.iterdir())
        assert remaining == ["current-key"]
        assert {p.name for p in removed} == {"stale-key-1", "stale-key-2"}

    def test_prune_stale_noop_when_root_missing(self, canonical_root):
        # canonical_root fixture points at a dir that doesn't exist yet.
        assert not canonical_root.exists()
        removed = numba_cache.prune_stale_canonical_caches(keep_key="whatever")
        assert removed == []


# ---------------------------------------------------------------------------
# --warm-cache CLI wiring (heavy subprocess run mocked out)
# ---------------------------------------------------------------------------


class TestWarmCacheCliWiring:
    """
    The heavy diarisation work happens in a fresh subprocess with $HOME
    redirected (see _run_diarisation_subprocess) - this is required
    because senko/config.py computes its cache dir via Path.home() and
    overwrites NUMBA_CACHE_DIR unconditionally at import time, so neither
    an in-process env var mutation nor NUMBA_CACHE_DIR itself can steer
    where senko's cache lands. These tests mock out
    _run_diarisation_subprocess (the subprocess boundary) rather than
    SenkoDiarizer directly.
    """

    def test_run_warm_cache_invokes_subprocess_and_installs(self, canonical_root, monkeypatch, tmp_path):
        """Verify run_warm_cache wires clip-generation -> diarisation
        subprocess (writing into <home>/.cache/senko/numba_cache) ->
        atomic install correctly."""

        def fake_make_clip(dest_wav, seconds=3.0):
            Path(dest_wav).write_bytes(b"RIFF....WAVEfake")

        monkeypatch.setattr(numba_cache, "_make_warmup_clip", fake_make_clip)

        def fake_run_subprocess(clip_path, warm_home_dir, *, verbose):
            # Simulate the subprocess (with $HOME=warm_home_dir) writing
            # senko's JIT artifacts to <home>/.cache/senko/numba_cache.
            cache_dir = Path(warm_home_dir) / ".cache" / "senko" / "numba_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            (cache_dir / "jit_artifact.nbi").write_text("compiled-during-warmup")

        monkeypatch.setattr(numba_cache, "_run_diarisation_subprocess", fake_run_subprocess)

        installed = numba_cache.run_warm_cache(verbose=False, prune_stale=False)

        assert installed.exists()
        assert (installed / "jit_artifact.nbi").read_text() == "compiled-during-warmup"
        assert installed == numba_cache.canonical_dir_for_key()

    def test_run_warm_cache_removes_throwaway_home_dir_on_success(
        self, canonical_root, monkeypatch, tmp_path
    ):
        """The per-invocation throwaway $HOME dir (numba_warm_home_*) must
        not be left behind after a successful --warm-cache run - only the
        .cache/senko/numba_cache subdirectory is needed (and it gets
        renamed out into the canonical location); everything else under
        that redirected $HOME (and the container dir itself) must be
        cleaned up rather than leaking under /tmp on every invocation."""

        def fake_make_clip(dest_wav, seconds=3.0):
            Path(dest_wav).write_bytes(b"fake")

        monkeypatch.setattr(numba_cache, "_make_warmup_clip", fake_make_clip)

        captured_home_dirs = []

        def fake_run_subprocess(clip_path, warm_home_dir, *, verbose):
            captured_home_dirs.append(warm_home_dir)
            cache_dir = Path(warm_home_dir) / ".cache" / "senko" / "numba_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            (cache_dir / "jit_artifact.nbi").write_text("compiled-during-warmup")
            # Simulate the subprocess also writing sibling state under the
            # redirected $HOME that isn't part of the canonical cache.
            (Path(warm_home_dir) / ".cache" / "uv").mkdir(parents=True, exist_ok=True)

        monkeypatch.setattr(numba_cache, "_run_diarisation_subprocess", fake_run_subprocess)

        numba_cache.run_warm_cache(verbose=False, prune_stale=False)

        assert len(captured_home_dirs) == 1
        assert not Path(captured_home_dirs[0]).exists(), (
            "throwaway warm-home temp dir was left behind after a successful "
            "--warm-cache run"
        )

    def test_run_warm_cache_does_not_leak_env_var_mutation(self, canonical_root, monkeypatch, tmp_path):
        """run_warm_cache itself must not mutate the parent process's HOME
        or NUMBA_CACHE_DIR - all cache-dir plumbing happens inside the
        child subprocess's own environment dict."""
        monkeypatch.setenv("HOME", "/some/prior/home")

        def fake_make_clip(dest_wav, seconds=3.0):
            Path(dest_wav).write_bytes(b"fake")

        monkeypatch.setattr(numba_cache, "_make_warmup_clip", fake_make_clip)

        def fake_run_subprocess(clip_path, warm_home_dir, *, verbose):
            cache_dir = Path(warm_home_dir) / ".cache" / "senko" / "numba_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            (cache_dir / "artifact.nbi").write_text("x")

        monkeypatch.setattr(numba_cache, "_run_diarisation_subprocess", fake_run_subprocess)

        numba_cache.run_warm_cache(verbose=False, prune_stale=False)

        assert os.environ["HOME"] == "/some/prior/home"

    def test_run_warm_cache_refuses_to_install_empty_cache(self, canonical_root, monkeypatch, tmp_path):
        def fake_make_clip(dest_wav, seconds=3.0):
            Path(dest_wav).write_bytes(b"fake")

        monkeypatch.setattr(numba_cache, "_make_warmup_clip", fake_make_clip)

        def fake_run_subprocess_noop(clip_path, warm_home_dir, *, verbose):
            # Does NOT write anything into <home>/.cache/senko/numba_cache.
            pass

        monkeypatch.setattr(numba_cache, "_run_diarisation_subprocess", fake_run_subprocess_noop)

        with pytest.raises(RuntimeError, match="no numba cache artifacts"):
            numba_cache.run_warm_cache(verbose=False, prune_stale=False)

        # Nothing should have been installed.
        assert not canonical_root.exists() or list(canonical_root.iterdir()) == []

    def test_run_warm_cache_prunes_stale_when_requested(self, canonical_root, monkeypatch, tmp_path):
        stale = canonical_root / "some-other-stale-key"
        stale.mkdir(parents=True)

        def fake_make_clip(dest_wav, seconds=3.0):
            Path(dest_wav).write_bytes(b"fake")

        monkeypatch.setattr(numba_cache, "_make_warmup_clip", fake_make_clip)

        def fake_run_subprocess(clip_path, warm_home_dir, *, verbose):
            cache_dir = Path(warm_home_dir) / ".cache" / "senko" / "numba_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            (cache_dir / "artifact.nbi").write_text("x")

        monkeypatch.setattr(numba_cache, "_run_diarisation_subprocess", fake_run_subprocess)

        numba_cache.run_warm_cache(verbose=False, prune_stale=True)

        assert not stale.exists()

    def test_run_warm_cache_propagates_subprocess_failure(self, canonical_root, monkeypatch, tmp_path):
        """If the diarisation subprocess itself fails (non-zero exit),
        run_warm_cache must raise rather than install a bogus cache."""

        def fake_make_clip(dest_wav, seconds=3.0):
            Path(dest_wav).write_bytes(b"fake")

        monkeypatch.setattr(numba_cache, "_make_warmup_clip", fake_make_clip)

        def fake_run_subprocess_fails(clip_path, warm_home_dir, *, verbose):
            raise RuntimeError("warm-up subprocess failed (exit 1):\nsome traceback")

        monkeypatch.setattr(numba_cache, "_run_diarisation_subprocess", fake_run_subprocess_fails)

        with pytest.raises(RuntimeError, match="warm-up subprocess failed"):
            numba_cache.run_warm_cache(verbose=False, prune_stale=False)

        assert not canonical_root.exists() or list(canonical_root.iterdir()) == []

    def test_subprocess_env_sets_home_and_skip_warming(self, monkeypatch, tmp_path):
        """_run_diarisation_subprocess must pass HOME=warm_home_dir and
        DIARISE_TRANSCRIBE_SKIP_CACHE_WARMING=1 to the child so the child
        never tries to bootstrap from (a possibly being-replaced)
        canonical cache while itself building one, and so its cache lands
        in the isolated $HOME rather than the real shared one."""
        captured = {}

        class FakeCompletedProcess:
            returncode = 0
            stderr = ""

        def fake_subprocess_run(cmd, env=None, **kwargs):
            captured["cmd"] = cmd
            captured["env"] = env
            return FakeCompletedProcess()

        monkeypatch.setattr(numba_cache.subprocess, "run", fake_subprocess_run)

        numba_cache._run_diarisation_subprocess("/fake/clip.wav", "/fake/warm_home", verbose=False)

        assert captured["env"]["HOME"] == "/fake/warm_home"
        assert captured["env"]["DIARISE_TRANSCRIBE_SKIP_CACHE_WARMING"] == "1"
        assert captured["cmd"][0] == sys.executable

    def test_subprocess_nonzero_exit_raises_with_stderr(self, monkeypatch):
        class FakeCompletedProcess:
            returncode = 1
            stderr = "Traceback: something exploded"

        monkeypatch.setattr(numba_cache.subprocess, "run", lambda *a, **k: FakeCompletedProcess())

        with pytest.raises(RuntimeError, match="something exploded"):
            numba_cache._run_diarisation_subprocess("/fake/clip.wav", "/fake/warm_home", verbose=False)


class TestWarmCacheArgparseWiring:
    """Verify --warm-cache is parsed and main() routes to run_warm_cache
    without requiring --in, and that ordinary invocations are unaffected."""

    def test_warm_cache_flag_parses_without_input_file(self):
        from diarise_transcribe.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(["--warm-cache"])
        assert args.warm_cache is True
        assert args.input_file is None

    def test_warm_cache_defaults_to_false(self):
        from diarise_transcribe.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(["--in", "audio.wav", "--out", "out.txt"])
        assert args.warm_cache is False
        assert args.input_file == "audio.wav"

    def test_main_routes_warm_cache_flag_to_run_warm_cache(self, monkeypatch):
        from diarise_transcribe import cli

        called = {}

        def fake_run_warm_cache(verbose=False):
            called["verbose"] = verbose
            return Path("/fake/canonical/dir")

        monkeypatch.setattr("diarise_transcribe.numba_cache.run_warm_cache", fake_run_warm_cache)
        monkeypatch.setattr(sys, "argv", ["diarise-transcribe", "--warm-cache"])

        cli.main()

        assert called == {"verbose": False}

    def test_main_still_requires_input_file_when_not_warming(self, monkeypatch, capsys):
        from diarise_transcribe import cli

        monkeypatch.setattr(sys, "argv", ["diarise-transcribe", "--out", "out.txt"])

        with pytest.raises(SystemExit) as exc_info:
            cli.main()

        assert exc_info.value.code == 2
        captured = capsys.readouterr()
        assert "--in" in captured.err

    def test_main_does_not_invoke_pipeline_when_warm_cache_requested(self, monkeypatch):
        """--warm-cache must short-circuit before run_pipeline is reached,
        even if other args are also passed."""
        from diarise_transcribe import cli

        monkeypatch.setattr("diarise_transcribe.numba_cache.run_warm_cache", lambda verbose=False: Path("/x"))

        pipeline_called = {"value": False}

        def fake_run_pipeline(*args, **kwargs):
            pipeline_called["value"] = True

        monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)
        monkeypatch.setattr(sys, "argv", ["diarise-transcribe", "--warm-cache"])

        cli.main()

        assert pipeline_called["value"] is False
