"""
Platform dispatch, default-model mapping, the explicit-model-id prefix rule,
device forwarding, lazy-import isolation, and the darwin-unchanged regression.

All deterministic: no torch, no nemo, no mlx, no network.
"""

import os
import subprocess
import sys
import types

import pytest

from diarise_transcribe import asr

_SRC = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")


# --- default model mapping by platform -------------------------------------

def test_default_model_id_maps_by_platform() -> None:
    assert asr.default_model_id("linux") == asr.DEFAULT_MODEL_NEMO
    assert asr.default_model_id("darwin") == asr.DEFAULT_MODEL_MLX


# --- backend selection by platform -----------------------------------------

def test_select_backend_linux_is_nemo() -> None:
    from diarise_transcribe import asr_nemo

    backend = asr._select_backend("linux")
    assert backend.load is asr_nemo.load_model
    assert backend.transcribe is asr_nemo.transcribe


def test_select_backend_darwin_is_mlx() -> None:
    backend = asr._select_backend("darwin")
    assert backend.load is asr._load_mlx
    assert backend.transcribe is asr._transcribe_mlx


# --- explicit model-id prefix rule -----------------------------------------

def test_check_model_id_accepts_default_and_neutral_ids() -> None:
    # default per backend, other org, and a local path all pass through.
    asr.check_model_id(asr.DEFAULT_MODEL_NEMO, "linux")
    asr.check_model_id(asr.DEFAULT_MODEL_MLX, "darwin")
    asr.check_model_id("some-org/custom-asr", "linux")
    asr.check_model_id("/models/local.nemo", "linux")


def test_check_model_id_rejects_opposite_backend_prefix() -> None:
    with pytest.raises(ValueError):
        asr.check_model_id("mlx-community/parakeet-tdt-0.6b-v3", "linux")
    with pytest.raises(ValueError):
        asr.check_model_id("nvidia/parakeet-tdt-0.6b-v3", "darwin")


# --- device forwarding ------------------------------------------------------

def test_device_forwarded_to_backend_load(monkeypatch) -> None:
    asr._model_cache.clear()
    seen: list[tuple[str, str]] = []

    def fake_load(model_id: str, device: str):
        seen.append((model_id, device))
        return object()

    backend = asr._Backend(load=fake_load, transcribe=lambda *a, **k: None)
    monkeypatch.setattr(asr, "_select_backend", lambda *a, **k: backend)

    model = asr.ASRModel("some-org/custom-asr", device="cpu")
    model._ensure_loaded()

    assert seen == [("some-org/custom-asr", "cpu")]


# --- darwin resolution unchanged (device ignored, MLX untouched) -----------

def test_mlx_load_ignores_device(monkeypatch) -> None:
    seen: list[str] = []
    fake_mlx = types.ModuleType("parakeet_mlx")
    fake_mlx.from_pretrained = lambda model_id: seen.append(model_id) or object()
    monkeypatch.setitem(sys.modules, "parakeet_mlx", fake_mlx)

    # A cuda device on the darwin path must not raise and must not reach the
    # loader - MLX has no device selection, so behaviour is unchanged.
    model = asr._load_mlx(asr.DEFAULT_MODEL_MLX, "cuda")

    assert seen == [asr.DEFAULT_MODEL_MLX]
    assert model is not None


# --- lazy-import isolation --------------------------------------------------

def _run_import_probe(body: str) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = _SRC + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", body],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_importing_asr_pulls_in_no_heavy_backend() -> None:
    _run_import_probe(
        "import sys, diarise_transcribe.asr\n"
        "bad = [m for m in "
        "('mlx', 'mlx.core', 'nemo', 'torch', 'diarise_transcribe.asr_nemo') "
        "if m in sys.modules]\n"
        "assert not bad, bad\n"
    )


def test_importing_asr_nemo_pulls_in_no_torch_or_nemo() -> None:
    _run_import_probe(
        "import sys, diarise_transcribe.asr_nemo\n"
        "bad = [m for m in ('nemo', 'torch') if m in sys.modules]\n"
        "assert not bad, bad\n"
    )
