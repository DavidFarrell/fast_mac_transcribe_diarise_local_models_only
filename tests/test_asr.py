"""Model-cache behaviour of ASRModel, backend-agnostic via a stubbed backend."""

from diarise_transcribe import asr


def _stub_backend(monkeypatch, load):
    """Force ASRModel onto a fake backend with the given load fn."""
    backend = asr._Backend(load=load, transcribe=lambda *a, **k: None)
    monkeypatch.setattr(asr, "_select_backend", lambda *a, **k: backend)


def test_ensure_loaded_shares_one_model_across_instances(monkeypatch) -> None:
    asr._model_cache.clear()

    load_calls: list[tuple[str, str]] = []

    def fake_load(model_id: str, device: str):
        load_calls.append((model_id, device))
        return object()

    _stub_backend(monkeypatch, fake_load)

    first = asr.ASRModel("fake-model")
    second = asr.ASRModel("fake-model")

    first._ensure_loaded()
    second._ensure_loaded()

    assert load_calls == [("fake-model", "auto")]
    assert first._model is second._model


def test_ensure_loaded_loads_separately_per_model_id(monkeypatch) -> None:
    asr._model_cache.clear()

    load_calls: list[tuple[str, str]] = []

    def fake_load(model_id: str, device: str):
        load_calls.append((model_id, device))
        return object()

    _stub_backend(monkeypatch, fake_load)

    first = asr.ASRModel("model-a")
    second = asr.ASRModel("model-b")

    first._ensure_loaded()
    second._ensure_loaded()

    assert load_calls == [("model-a", "auto"), ("model-b", "auto")]
    assert first._model is not second._model


def test_ensure_loaded_loads_separately_per_device(monkeypatch) -> None:
    asr._model_cache.clear()

    load_calls: list[tuple[str, str]] = []

    def fake_load(model_id: str, device: str):
        load_calls.append((model_id, device))
        return object()

    _stub_backend(monkeypatch, fake_load)

    cpu = asr.ASRModel("model-a", device="cpu")
    cuda = asr.ASRModel("model-a", device="cuda")

    cpu._ensure_loaded()
    cuda._ensure_loaded()

    assert load_calls == [("model-a", "cpu"), ("model-a", "cuda")]
    assert cpu._model is not cuda._model
