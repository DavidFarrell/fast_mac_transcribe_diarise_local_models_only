"""
Warm-once / bootstrap-per-machine numba cache for Senko diarisation.

Background - what actually controls Senko's numba cache
---------------------------------------------------------
``cli.py`` sets a unique ``NUMBA_CACHE_DIR`` per process at startup (see the
top of that file) so that concurrent transcriptions never write to the same
numba cache directory at once - a shared cache corrupted under parallel
transcriptions in the past (see git history: "Fix numba cache race condition
for parallel transcriptions").

However: the installed ``senko`` package (``senko/config.py``) unconditionally
does::

    cache_dir = Path.home() / '.cache' / 'senko' / 'numba_cache'
    os.environ['NUMBA_CACHE_DIR'] = str(cache_dir)

the moment ``senko`` (or anything importing ``senko.config``) is imported -
which clobbers whatever ``cli.py`` set beforehand, AND numba snapshots
``NUMBA_CACHE_DIR`` into ``numba.core.config.CACHE_DIR`` the first time its
config module is imported and never re-reads the environment afterwards. Net
effect, verified empirically: **Senko's numba JIT cache always lives at
``~/.cache/senko/numba_cache``, for every run, regardless of what this CLI
sets.** It is not actually per-PID for the Senko backend - it is one shared,
persistent directory that already accumulates warmth across every run on a
machine, with no action needed from this CLI. (The pre-existing per-PID
``NUMBA_CACHE_DIR`` line in ``cli.py`` is left completely untouched by this
module - it may still matter for the ``sortformer`` backend or other numba
use, and removing it is out of scope here.)

Given that, "copy a canonical cache into the per-PID dir on every run" would
be a no-op for Senko (nothing reads from the per-PID dir) and "treat
``~/.cache/senko/numba_cache`` as the canonical dir that live runs never
write to" is not applicable either - that directory is already the one live
runs write to as a matter of course (protected by the existing
``_patch_numba_cache`` fault-tolerant save wrapper in ``senko_diarisation.py``,
which this module does not touch or replace).

What IS worth solving, and what this module does
-------------------------------------------------
The expensive one-time cost is the *first* JIT compilation of Senko's
UMAP/HDBSCAN clustering warmup on a machine (or after
``~/.cache/senko/numba_cache`` has been deleted/is empty - e.g. a fresh
install, a new machine, or a wiped cache). This module lets that cost be
paid once, deliberately, via a maintenance command, and then reused:

1. A "canonical" snapshot directory, keyed by the numba/senko/python
   versions currently in use::

       ~/.cache/fast-diarise/numba-canonical/<key>/

   Live transcription runs NEVER write here. They only ever *read* (copy)
   from it, best-effort, and only ever to *bootstrap an empty/missing*
   ``~/.cache/senko/numba_cache`` - never to overwrite an existing one, so
   a live run can never race with, corrupt, or clobber another process's
   already-warm shared cache.

2. A separate, manually-invoked maintenance command
   (``diarise-transcribe --warm-cache``) is the only thing that ever writes
   to the canonical directory. It runs the JIT-heavy Senko diarisation path
   once, in a fresh subprocess whose ``$HOME`` is redirected to a throwaway
   temp dir (so senko.config's ``Path.home() / '.cache' / 'senko' /
   'numba_cache'`` resolves to an isolated, empty location - the only
   reliable way to control where Senko's cache lands, since NUMBA_CACHE_DIR
   itself is a dead lever once senko.config has been imported). It then
   atomically renames that throwaway cache dir into place as the new
   canonical cache for the current key. This command is meant to be run
   alone, never by the daemon/automation, and never concurrently with live
   transcriptions.

3. At the start of every live run, ``cli.py`` calls
   :func:`bootstrap_senko_cache_if_empty` *before* doing anything
   numba/senko-related. If ``~/.cache/senko/numba_cache`` is missing or
   empty AND a canonical cache exists for the current key, the canonical
   cache is copied in as a one-time bootstrap. If the real cache dir
   already has any content, this is a strict no-op (never touches it) -
   there is nothing to "restore" because that directory already persists
   warmth across runs on its own. Any failure (missing canonical, stale
   key, corrupt, copy error) is caught and logged as a warning; the run
   always falls back to today's behaviour. This function never raises.

Every failure mode here must degrade to "cold JIT, exactly as before this
module existed" - never a crash, and never a write to (or race with) shared
state that another live run might be using.
"""

from __future__ import annotations

import contextlib
import importlib.metadata
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Optional


# Root under which the canonical (warmed) numba caches live, keyed by
# version. Overridable via env var for tests.
_ROOT_ENV_VAR = "FAST_DIARISE_NUMBA_CANONICAL_ROOT"

# The real, shared numba cache directory that senko/config.py forces
# regardless of NUMBA_CACHE_DIR. Overridable via env var for tests (and
# internally, by pointing $HOME at a throwaway dir for the warm-up
# subprocess - see _run_diarisation_subprocess).
_SENKO_CACHE_DIR_ENV_VAR = "FAST_DIARISE_SENKO_CACHE_DIR_OVERRIDE"


def canonical_root() -> Path:
    """Return the root directory that holds per-key canonical caches."""
    override = os.environ.get(_ROOT_ENV_VAR)
    if override:
        return Path(override)
    return Path.home() / ".cache" / "fast-diarise" / "numba-canonical"


def senko_cache_dir() -> Path:
    """
    Return the directory senko's own numba cache actually lives in.

    Mirrors senko/config.py's ``Path.home() / '.cache' / 'senko' /
    'numba_cache'`` computation exactly, so this always agrees with
    wherever senko itself will look - including inside the warm-up
    subprocess, which runs with $HOME redirected to an isolated temp dir.
    """
    override = os.environ.get(_SENKO_CACHE_DIR_ENV_VAR)
    if override:
        return Path(override)
    return Path.home() / ".cache" / "senko" / "numba_cache"


def _package_version_or_ref(dist_name: str) -> str:
    """
    Best-effort version string for a distribution.

    For normal (PyPI) installs this is just the version string. For a
    package pinned via a git URL (like ``senko``), pip/uv record the
    resolved commit in ``direct_url.json`` - that commit id is a far more
    precise cache key than the static ``version`` field a git-sourced
    package tends to report (senko reports "0.1.0" regardless of which
    commit is actually checked out). We prefer the commit id when present
    and fall back to the version string, then to "unknown" if the package
    isn't installed at all.
    """
    try:
        dist = importlib.metadata.distribution(dist_name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"

    commit_id = None
    with contextlib.suppress(Exception):
        raw = dist.read_text("direct_url.json")
        if raw:
            import json

            data = json.loads(raw)
            vcs_info = data.get("vcs_info") or {}
            commit_id = vcs_info.get("commit_id")

    if commit_id:
        return f"git{commit_id[:12]}"

    try:
        return dist.version
    except Exception:
        return "unknown"


def cache_key() -> str:
    """
    Derive the cache key for the *current* environment.

    Composed of: numba version, senko (diarisation dep) version/commit, and
    the running python's major.minor. Any change to any of these components
    changes the key, which means canonical caches built under a different
    environment are simply ignored (never mixed with the current one).
    """
    numba_version = _package_version_or_ref("numba")
    senko_version = _package_version_or_ref("senko")
    py_version = f"py{sys.version_info.major}.{sys.version_info.minor}"
    return f"numba{numba_version}-senko{senko_version}-{py_version}"


def canonical_dir_for_key(key: Optional[str] = None) -> Path:
    """Path to the canonical cache dir for ``key`` (default: current key)."""
    return canonical_root() / (key or cache_key())


def _dir_has_content(path: Path) -> bool:
    try:
        return path.is_dir() and any(path.iterdir())
    except OSError:
        return False


def bootstrap_senko_cache_if_empty(
    *,
    key: Optional[str] = None,
    verbose: bool = False,
) -> bool:
    """
    Best-effort, one-time bootstrap of senko's real numba cache directory
    from the canonical snapshot - but ONLY if that real directory does not
    exist at all yet.

    Must be called BEFORE senko (or anything importing senko.config) is
    imported, so that if we do populate the directory, senko's own
    ``cache_dir.mkdir(parents=True, exist_ok=True)`` sees it already
    populated rather than racing to create it fresh.

    This is intentionally NOT a "copy in every run" step: if the real
    cache directory already has any content, this function does nothing at
    all - there is nothing to restore, since that directory already
    persists warmth across every run on its own. This guarantees a live
    run can never overwrite, race with, or corrupt another process's
    already-warm shared cache; the only thing it can ever do is create and
    fill in a directory that does not exist yet. An existing-but-EMPTY
    dest is deliberately treated the same as "already has content" for the
    purposes of the final claim step (see below) - it is indistinguishable
    from a directory senko itself just created and is about to write its
    first cache file into, so bootstrap backs off rather than claiming it.

    Concurrency: two live runs can both observe an empty/missing ``dest``
    at (near-)the same time. To avoid both processes copying into the same
    destination directory concurrently (which could leave a numba index
    file truncated mid-read for whichever process gets there first), the
    canonical snapshot is first copied into a private, per-process staging
    directory next to ``dest``, and only then swapped into place.

    That final swap is deliberately NOT a bare ``os.rename(staging, dest)``:
    on POSIX, ``rename()`` silently succeeds (replacing the target) when
    ``dest`` already exists as an *empty* directory - which is exactly the
    state senko's own ``cache_dir.mkdir(parents=True, exist_ok=True)``
    leaves it in the instant before senko writes its first real cache file.
    A plain rename could therefore win a race against senko itself: we'd
    replace the very directory another process is about to populate, with
    no error on either side. To close that, we instead atomically *claim*
    ``dest`` ourselves first via ``os.mkdir(dest)`` (which - unlike rename
    - always raises if the path already exists, empty or not) and only
    rename our staged copy onto the directory we just created. If the
    ``mkdir`` fails, we know for certain some other actor (another
    bootstrap or senko itself) already holds that path, and we back off
    completely without touching it. If it succeeds, no other process can
    also be holding it, so the rename onto our own fresh empty dir is
    provably race-free.

    Returns True if a bootstrap copy happened, False otherwise (including
    on any error, or when there was simply nothing to do). Never raises -
    every failure mode degrades to "proceed cold", matching prior
    behaviour.
    """
    staging = None
    try:
        dest = senko_cache_dir()

        if _dir_has_content(dest):
            # Already warm (or another process got here first) - never
            # touch it.
            return False

        source = canonical_dir_for_key(key)
        if not _dir_has_content(source):
            if verbose:
                print(f"  No canonical numba cache for key at {source}; starting cold.")
            return False

        dest.parent.mkdir(parents=True, exist_ok=True)

        # Copy into a private, per-call staging dir first (never a shared
        # path another process - or another THREAD in this same process -
        # could also be writing to), then atomically rename it into place.
        # This avoids two concurrent bootstraps ever interleaving writes
        # into the same destination files.
        #
        # PID alone is not sufficient: two threads in the same process
        # (e.g. two concurrent bootstrap calls in one long-lived daemon)
        # share a PID, so a PID-only name collides and lets one thread's
        # rmtree/copytree race the other's, tearing the copy. Mix in the
        # thread id and a UUID4 so every call gets a distinct staging path
        # regardless of process/thread reuse.
        staging = dest.parent / (
            f".{dest.name}.bootstrap-{os.getpid()}-{threading.get_ident()}-{uuid.uuid4().hex}"
        )
        shutil.rmtree(staging, ignore_errors=True)

        # copy_function=copy2 preserves mtimes numba may care about.
        shutil.copytree(source, staging, copy_function=shutil.copy2)

        if _dir_has_content(dest):
            # Someone else finished bootstrapping (or started a live run
            # that populated it) while we were copying - discard our
            # staging copy and treat this as the no-op it now is.
            shutil.rmtree(staging, ignore_errors=True)
            staging = None
            return False

        try:
            # Atomically CLAIM dest via mkdir (not rename): mkdir always
            # raises if the path already exists, even as an empty dir -
            # unlike os.rename, which would silently replace an existing
            # empty dest. This is what closes the race against senko's own
            # `cache_dir.mkdir(parents=True, exist_ok=True)`: if senko (or
            # another bootstrap) already created dest - populated or not -
            # our mkdir fails and we back off untouched. If it succeeds, we
            # hold the only reference to that path, so renaming our staged
            # copy onto it is guaranteed race-free.
            os.mkdir(dest)
        except OSError:
            # Lost the race - dest already exists (empty or not), created
            # by senko itself or another bootstrap. Never touch it; just
            # discard the staging copy.
            shutil.rmtree(staging, ignore_errors=True)
            staging = None
            return False

        try:
            os.rename(staging, dest)
        except OSError:
            # Should not happen (we just created dest ourselves and hold
            # the only reference to it), but fail soft regardless rather
            # than ever raising out of a bootstrap.
            shutil.rmtree(staging, ignore_errors=True)
            staging = None
            return False

        staging = None
        if verbose:
            print(f"  Bootstrapped numba cache from {source} -> {dest}")
        return True

    except Exception as e:  # noqa: BLE001 - deliberately broad: never fail a live run
        print(
            f"Warning: failed to bootstrap numba cache ({e}); continuing with a cold cache.",
            file=sys.stderr,
        )
        return False
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)


def install_canonical_cache(warmed_dir: str, *, key: Optional[str] = None) -> Path:
    """
    Atomically install ``warmed_dir`` as the canonical cache for ``key``.

    Only ever called by the ``--warm-cache`` maintenance command, never by a
    live transcription run. Uses rename (same-filesystem move) so readers
    never observe a partially-written canonical directory: the canonical
    directory either doesn't exist yet, is the old complete one, or is the
    new complete one - never a half-copied in-between state.

    If a canonical directory already exists for this key, it is swapped out
    (renamed aside) and removed after the new one is in place.
    """
    dest = canonical_dir_for_key(key)
    dest.parent.mkdir(parents=True, exist_ok=True)

    warmed_path = Path(warmed_dir)
    if not warmed_path.is_dir():
        raise FileNotFoundError(f"warmed cache source dir does not exist: {warmed_path}")

    # Stage the rename-swap within the same parent directory (same
    # filesystem) so both the "install new" and "remove old" steps are
    # simple renames/deletes, never a cross-filesystem copy.
    old_aside = dest.parent / f".{dest.name}.replacing-{os.getpid()}"

    if dest.exists():
        os.rename(dest, old_aside)

    try:
        os.rename(warmed_path, dest)
    except Exception:
        # Roll back so we never leave the canonical dir missing.
        if old_aside.exists() and not dest.exists():
            os.rename(old_aside, dest)
        raise

    if old_aside.exists():
        shutil.rmtree(old_aside, ignore_errors=True)

    return dest


def prune_stale_canonical_caches(*, keep_key: Optional[str] = None) -> list[Path]:
    """
    Remove canonical cache directories whose key doesn't match ``keep_key``
    (default: the current environment's key). Used by ``--warm-cache`` to
    keep the canonical root from accumulating caches for old numba/senko/
    python versions. Never touches senko's real cache directory or any
    live run's state - only the canonical snapshot root.

    Returns the list of directories removed. Best-effort: a removal failure
    for one stale dir is logged and skipped rather than raised, since this
    is cleanup, not correctness-critical.
    """
    keep = keep_key or cache_key()
    root = canonical_root()
    removed = []
    if not root.is_dir():
        return removed

    for entry in root.iterdir():
        if entry.is_dir() and entry.name != keep and not entry.name.startswith("."):
            try:
                shutil.rmtree(entry)
                removed.append(entry)
            except OSError as e:
                print(f"Warning: could not prune stale cache {entry}: {e}", file=sys.stderr)

    return removed


@contextlib.contextmanager
def temp_warm_cache_dir():
    """Context manager yielding a fresh temp dir to use as an isolated
    $HOME for the warm-up subprocess. By the time the caller's `with`
    block exits (success or failure), it has already renamed the senko
    cache subdir it needs out of this temp dir into the canonical
    location - so the whole temp dir (and anything else the subprocess
    wrote under the redirected $HOME, e.g. .cache/uv) is always cleaned up
    here, on both the success and failure paths."""
    d = tempfile.mkdtemp(prefix="numba_warm_home_")
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _make_warmup_clip(dest_wav: str, seconds: float = 3.0) -> None:
    """Generate a tiny 16kHz mono WAV clip used to drive the JIT-heavy
    diarisation path during --warm-cache. Avoids bundling a binary audio
    fixture in the repo.

    Uses macOS's built-in `say` to synthesize real speech: Senko's VAD
    stage needs actual voice activity to produce any speaker segments (a
    silent/tone-only clip makes senko.Diarizer.diarize() return None,
    which is a "no speech detected" outcome that produces no clustering
    work and therefore nothing to warm). Falls back to a synthetic tone
    via ffmpeg if `say` isn't available (e.g. a non-macOS dev environment),
    though the VAD stage may then legitimately detect no speech - that's
    surfaced as a clear error by run_warm_cache rather than silently
    installing an empty cache.
    """
    if shutil.which("say"):
        with tempfile.TemporaryDirectory(prefix="numba_warm_say_") as say_dir:
            aiff_path = str(Path(say_dir) / "warmup.aiff")
            say_cmd = [
                "say", "-o", aiff_path,
                "This is a short test clip used to warm up the just in time "
                "compilation cache before running real transcriptions.",
            ]
            say_result = subprocess.run(say_cmd, capture_output=True, text=True, check=False)
            if say_result.returncode == 0 and Path(aiff_path).exists():
                cmd = [
                    "ffmpeg", "-y",
                    "-i", aiff_path,
                    "-ar", "16000",
                    "-ac", "1",
                    "-f", "wav",
                    "-acodec", "pcm_s16le",
                    dest_wav,
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, check=False)
                if result.returncode == 0:
                    return
                # Fall through to the tone fallback below if ffmpeg failed.

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"sine=frequency=220:duration={seconds}:sample_rate=16000",
        "-ac", "1",
        "-f", "wav",
        "-acodec", "pcm_s16le",
        dest_wav,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to generate warmup clip: {result.stderr}")


_WARM_WORKER_SNIPPET = """
from diarise_transcribe.senko_diarisation import SenkoDiarizer
diarizer = SenkoDiarizer(quiet={quiet!r})
diarizer.diarise({clip_path!r})
"""


def _run_diarisation_subprocess(clip_path: str, warm_home_dir: str, *, verbose: bool) -> None:
    """
    Run the JIT-heavy Senko diarisation path in a FRESH child process with
    $HOME redirected to ``warm_home_dir`` from birth.

    This has to be $HOME, not NUMBA_CACHE_DIR, and it has to be a
    subprocess: senko/config.py unconditionally computes its cache
    directory as ``Path.home() / '.cache' / 'senko' / 'numba_cache'`` and
    overwrites NUMBA_CACHE_DIR with it at import time - so setting
    NUMBA_CACHE_DIR (in this process or a child's env) has no effect on
    where senko actually writes. Redirecting $HOME is the only reliable
    way to steer senko's cache to an isolated location. It must be a
    subprocess (not just an env var mutation here) both because $HOME
    needs to be set before senko.config's module-level code runs, and
    because this process may already have numba/senko imported with the
    real $HOME baked into numba.core.config.CACHE_DIR.
    """
    env = dict(os.environ)
    env["HOME"] = warm_home_dir
    # Never bootstrap from a canonical cache while building a new one, and
    # never let this subprocess touch the real (non-redirected) senko
    # cache dir.
    env["DIARISE_TRANSCRIBE_SKIP_CACHE_WARMING"] = "1"

    code = _WARM_WORKER_SNIPPET.format(quiet=not verbose, clip_path=clip_path)
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=not verbose,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr if result.stderr is not None else "(output not captured; ran with --verbose)"
        raise RuntimeError(
            f"warm-up subprocess failed (exit {result.returncode}):\n{stderr}"
        )


def run_warm_cache(*, verbose: bool = False, prune_stale: bool = True) -> Path:
    """
    Run the maintenance "--warm-cache" flow end to end:

      1. Build a small audio clip with real speech content (VAD needs
         actual voice activity to produce anything to cluster/embed).
      2. Run the JIT-heavy Senko diarisation path against it in a fresh
         subprocess with $HOME redirected to a throwaway temp dir, so
         senko's cache lands at <throwaway>/.cache/senko/numba_cache -
         never the canonical dir, never the real shared
         ~/.cache/senko/numba_cache a live run might be using.
      3. Atomically install that warmed cache dir as the new canonical
         cache for the current key.
      4. Optionally prune canonical caches for other (stale) keys.

    Must be invoked standalone - never concurrently with a live
    transcription or by automation.

    Returns the path to the installed canonical cache directory.
    """
    key = cache_key()
    if verbose:
        print(f"Warming numba cache for key: {key}")

    with temp_warm_cache_dir() as warm_home_dir, tempfile.TemporaryDirectory(prefix="numba_warm_clip_") as clip_dir:
        clip_path = str(Path(clip_dir) / "warmup.wav")
        _make_warmup_clip(clip_path)

        _run_diarisation_subprocess(clip_path, warm_home_dir, verbose=verbose)

        warmed_cache_dir = Path(warm_home_dir) / ".cache" / "senko" / "numba_cache"
        if not _dir_has_content(warmed_cache_dir):
            raise RuntimeError(
                "warm-up run produced no numba cache artifacts "
                f"in {warmed_cache_dir}; refusing to install an empty canonical cache"
            )

        installed = install_canonical_cache(str(warmed_cache_dir), key=key)

    if prune_stale:
        pruned = prune_stale_canonical_caches(keep_key=key)
        if verbose and pruned:
            print(f"Pruned {len(pruned)} stale canonical cache(s): {[str(p) for p in pruned]}")

    if verbose:
        print(f"Installed canonical numba cache at: {installed}")

    return installed
