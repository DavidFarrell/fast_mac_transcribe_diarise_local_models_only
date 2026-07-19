"""
CUDA co-residency poison / ASR-reload behaviour (GPU-gate fix).

Senko's CUDA diarisation irrecoverably corrupts the CUDA state of any
co-resident model, so a NeMo model transcribed AFTER a CUDA diarisation dies
with an illegal memory access (design/gpu-verification.md). The fix:
senko_diarisation calls asr.poison_cuda_asr() after a CUDA diarise, and an
ASRModel whose model was loaded before that reloads a fresh one on next use.

These are deterministic - stubbed backend + fake senko module, no torch, no
GPU, no model load - and each would fail if the poison signal or the reload
were removed. The real CUDA proof is test_golden_contract (integration).
"""

import os
import sys

import pytest

from diarise_transcribe import asr, senko_diarisation
from diarise_transcribe.senko_diarisation import SenkoDiarizer


@pytest.fixture(autouse=True)
def _reset_asr_state():
    """Each test starts from a clean cache + generation 0 so cross-test
    ordering (e.g. an earlier CUDA integration run) can't leak in."""
    asr._model_cache.clear()
    saved = asr._cuda_asr_generation
    asr._cuda_asr_generation = 0
    yield
    asr._model_cache.clear()
    asr._cuda_asr_generation = saved


def _counting_backend(monkeypatch):
    """Stub ASRModel's backend so each load returns a distinct sentinel and
    loads are counted - no real NeMo/torch."""
    loads: list[tuple[str, str]] = []

    def fake_load(model_id: str, device: str):
        loads.append((model_id, device))
        return object()

    backend = asr._Backend(load=fake_load, transcribe=lambda *a, **k: None)
    monkeypatch.setattr(asr, "_select_backend", lambda *a, **k: backend)
    return loads


# --- ASRModel reload-on-poison (the two corruption sites) -------------------

def test_no_poison_preserves_single_cached_model_across_instances(monkeypatch):
    """Guard the optimisation the fix must NOT break: with no CUDA diarisation
    (generation unchanged) two instances still share one loaded model."""
    loads = _counting_backend(monkeypatch)

    first = asr.ASRModel("m")
    second = asr.ASRModel("m")
    first._ensure_loaded()
    second._ensure_loaded()

    assert len(loads) == 1
    assert first._model is second._model


def test_same_instance_reloads_after_poison(monkeypatch):
    """The recovery site: one ASRModel transcribes, a CUDA diarisation runs
    (poison), then the SAME instance transcribes again - it must reload a
    fresh model rather than reuse the poisoned one."""
    loads = _counting_backend(monkeypatch)

    model = asr.ASRModel("m")
    model._ensure_loaded()
    first_model = model._model

    asr.poison_cuda_asr()
    model._ensure_loaded()

    assert len(loads) == 2, "poisoned model was not reloaded on the same instance"
    assert model._model is not first_model


def test_second_instance_reloads_after_poison(monkeypatch):
    """The cross-stream site: instance A loads (and caches) a model, a CUDA
    diarisation poisons it, then instance B for the same key must NOT receive
    the poisoned cached model - it reloads fresh."""
    loads = _counting_backend(monkeypatch)

    a = asr.ASRModel("m")
    a._ensure_loaded()
    poisoned = a._model

    asr.poison_cuda_asr()

    b = asr.ASRModel("m")
    b._ensure_loaded()

    assert len(loads) == 2, "second instance reused the poisoned cached model"
    assert b._model is not poisoned


def test_poison_is_monotonic_and_reload_settles(monkeypatch):
    """After a reload at the current generation, a further _ensure_loaded with
    no new poison is a no-op (no unbounded reloading)."""
    loads = _counting_backend(monkeypatch)

    model = asr.ASRModel("m")
    model._ensure_loaded()
    asr.poison_cuda_asr()
    model._ensure_loaded()  # reload -> 2 loads
    model._ensure_loaded()  # no new poison -> still 2

    assert len(loads) == 2


# --- senko_diarisation raises the poison only for CUDA ----------------------

def _fake_senko_result_diarizer():
    class _Fake:
        def diarize(self, _path, generate_colors=False):
            return {"merged_segments": [{"start": 0.0, "end": 1.0, "speaker": "S0"}],
                    "merged_speakers_detected": 1}
    return _Fake()


@pytest.mark.parametrize(
    "device, cuda_available, expect_poison",
    [
        ("cuda", False, True),    # explicit cuda always poisons
        ("cpu", True, False),     # cpu never poisons even where cuda exists
        ("coreml", True, False),  # darwin path never poisons
        ("auto", True, True),     # auto poisons when torch reports cuda
        ("auto", False, False),   # auto on a cuda-less box does not
    ],
)
def test_diarise_poisons_asr_only_on_cuda(monkeypatch, device, cuda_available, expect_poison):
    senko_diarisation._native_diarizer_cache.clear()
    # torch.cuda.is_available is only consulted for device='auto'.
    fake_torch = type("T", (), {"cuda": type("C", (), {"is_available": staticmethod(lambda: cuda_available)})})
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    diarizer = SenkoDiarizer(device=device, quiet=True)
    diarizer._diarizer = _fake_senko_result_diarizer()

    before = asr._cuda_asr_generation
    diarizer.diarise("x.wav")
    advanced = asr._cuda_asr_generation > before

    assert advanced is expect_poison


def test_diarization_used_cuda_classification(monkeypatch):
    """Unit the device->cuda predicate directly (the fast tell, no diarise)."""
    assert senko_diarisation._diarization_used_cuda("cuda") is True
    assert senko_diarisation._diarization_used_cuda("cpu") is False
    assert senko_diarisation._diarization_used_cuda("coreml") is False
    fake_torch = type("T", (), {"cuda": type("C", (), {"is_available": staticmethod(lambda: True)})})
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert senko_diarisation._diarization_used_cuda("auto") is True


@pytest.mark.integration
def test_same_instance_transcribe_after_cuda_diarise_reloads_on_gpu(tmp_path):
    """Site 2 (recovery pattern) on REAL CUDA, through the ASRModel wrapper:
    one ASRModel transcribes, a real CUDA senko diarisation poisons the
    co-resident model, then the SAME instance transcribes again. Without the
    reload this is the exact SIGABRT / illegal-memory-access the gate found;
    with it, it must return words and not crash. Complements the deterministic
    mocks (the crash is real only on GPU) and the golden run (whose 0-window
    recovery never exercised this same-instance path)."""
    import shutil
    import subprocess

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a usable CUDA device")

    from diarise_transcribe.senko_diarisation import SenkoDiarizer

    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg, "ffmpeg is required to resample the fixtures to 16k"
    fixtures = (
        os.path.dirname(__file__),
        "fixtures",
        "golden",
        "meeting",
        "audio",
    )
    wavs = {}
    for name in ("system", "mic"):
        src = os.path.join(*fixtures, f"{name}.wav")
        dst = str(tmp_path / f"{name}_16k.wav")
        subprocess.run(
            [ffmpeg, "-y", "-i", src, "-ar", "16000", "-ac", "1", dst],
            check=True,
            capture_output=True,
        )
        wavs[name] = dst

    asr._model_cache.clear()

    model = asr.ASRModel(device="cuda")
    first = model.transcribe(wavs["system"])
    assert first.words
    loaded_before = model._model
    gen_before = asr._cuda_asr_generation

    SenkoDiarizer(device="cuda", quiet=True).diarise(wavs["system"])
    assert asr._cuda_asr_generation > gen_before, "CUDA diarise did not poison ASR"

    # The crash site: same instance, transcribe after the CUDA diarisation.
    second = model.transcribe(wavs["mic"])
    assert second.words, "no words after post-diarise transcribe"
    assert model._model is not loaded_before, "poisoned model was not reloaded"
