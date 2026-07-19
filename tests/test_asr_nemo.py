"""
Linux NeMo backend units: the device resolver, timestamp conversion, and
model-id source resolution.

Deterministic - the resolver takes an injected cuda-availability predicate so
torch is never imported; the conversion runs on captured NeMo word shapes;
source resolution stubs snapshot_download so no network or nemo is needed.
"""

import types

import pytest

from diarise_transcribe import asr, asr_nemo


# --- device resolver --------------------------------------------------------

def test_auto_prefers_cuda_when_available() -> None:
    assert asr_nemo.resolve_device("auto", cuda_available=lambda: True) == "cuda"


def test_auto_falls_back_to_cpu_when_no_cuda() -> None:
    assert asr_nemo.resolve_device("auto", cuda_available=lambda: False) == "cpu"


def test_cpu_is_always_honoured() -> None:
    assert asr_nemo.resolve_device("cpu", cuda_available=lambda: True) == "cpu"


def test_cuda_when_available_is_cuda() -> None:
    assert asr_nemo.resolve_device("cuda", cuda_available=lambda: True) == "cuda"


def test_cuda_when_unavailable_errors_not_silent_cpu() -> None:
    with pytest.raises(RuntimeError):
        asr_nemo.resolve_device("cuda", cuda_available=lambda: False)


def test_unknown_device_rejected() -> None:
    with pytest.raises(ValueError):
        asr_nemo.resolve_device("gpu", cuda_available=lambda: True)


# --- timestamp conversion ---------------------------------------------------

# Captured NeMo hypothesis.timestamp['word'] shape (slice-0 ground truth §3):
# keys word/start/end/start_offset/end_offset, start/end in seconds.
_CAPTURED_NEMO_WORDS = [
    {"word": "Right,", "start_offset": 2, "end_offset": 8, "start": 0.16, "end": 0.64},
    {"word": "shall", "start_offset": 9, "end_offset": 12, "start": 0.72, "end": 0.96},
    {"word": "we", "start_offset": 13, "end_offset": 15, "start": 1.04, "end": 1.2},
]


def test_conversion_maps_text_and_seconds() -> None:
    words = asr_nemo.nemo_words_to_words(_CAPTURED_NEMO_WORDS)

    assert [w.text for w in words] == ["Right,", "shall", "we"]
    assert words[0].start == 0.16 and words[0].end == 0.64
    assert all(isinstance(w, asr.Word) for w in words)


def test_conversion_ignores_offset_keys_and_coerces_floats() -> None:
    # Integer start/end (offsets present) must come back as floats; the
    # *_offset keys are not carried onto Word.
    words = asr_nemo.nemo_words_to_words(
        [{"word": "x", "start_offset": 1, "end_offset": 2, "start": 1, "end": 2}]
    )

    assert isinstance(words[0].start, float)
    assert words[0].start == 1.0 and words[0].end == 2.0


# --- model-id source resolution --------------------------------------------

def _stub_snapshot_download(monkeypatch, tmp_path, chatter=None):
    """Fake huggingface_hub.snapshot_download that drops the pinned .nemo."""
    (tmp_path / asr_nemo._DEFAULT_NEMO_FILENAME).write_text("")

    def fake_download(model_id, revision=None, allow_patterns=None):
        if chatter:
            print(chatter)  # simulate cold-cache progress on stdout
        return str(tmp_path)

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.snapshot_download = fake_download
    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", fake_hub)


def test_non_default_id_passes_through_verbatim() -> None:
    # Local paths and other-org ids reach restore_from unchanged (frozen rule).
    assert asr_nemo._resolve_nemo_source("/models/custom.nemo") == "/models/custom.nemo"
    assert asr_nemo._resolve_nemo_source("some-org/asr") == "some-org/asr"


def test_default_id_resolves_to_pinned_nemo_file(monkeypatch, tmp_path) -> None:
    _stub_snapshot_download(monkeypatch, tmp_path)

    source = asr_nemo._resolve_nemo_source(asr.DEFAULT_MODEL_NEMO)

    assert source.endswith(asr_nemo._DEFAULT_NEMO_FILENAME)
    assert source == str(tmp_path / asr_nemo._DEFAULT_NEMO_FILENAME)


def test_snapshot_download_output_goes_to_stderr_not_stdout(
    monkeypatch, tmp_path, capsys
) -> None:
    _stub_snapshot_download(monkeypatch, tmp_path, chatter="DOWNLOAD_PROGRESS_42")

    asr_nemo._resolve_nemo_source(asr.DEFAULT_MODEL_NEMO)

    captured = capsys.readouterr()
    assert "DOWNLOAD_PROGRESS_42" not in captured.out
    assert "DOWNLOAD_PROGRESS_42" in captured.err
