"""
Linux NeMo backend units: the device resolver and the timestamp conversion.

Deterministic - the resolver takes an injected cuda-availability predicate so
torch is never imported; the conversion runs on captured NeMo word shapes.
"""

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
