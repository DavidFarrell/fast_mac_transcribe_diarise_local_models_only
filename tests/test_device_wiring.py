"""
Device wiring (slice 4 scope addition).

The frozen device option must route to BOTH engines - ASR and diarisation -
at every orchestration construction site (reprocess_stream, cli.run_pipeline,
muesli_backend.run_pipeline). Before this slice the diariser was constructed
without a device argument, so a caller-supplied device reached ASR's
constructor but not the diariser's; the GPU gate needs both on the same
device to prove CUDA diarisation end-to-end.

These are deterministic: the ASR and diariser constructors are replaced with
capturing stand-ins, so no model is loaded. Each test would fail if the
`device=` argument were dropped from that entry point's construction site.
"""

from diarise_transcribe import cli, muesli_backend, reprocess, senko_diarisation
from diarise_transcribe.asr import TranscriptResult, Word
from diarise_transcribe.diarisation import DiarSegment


# One diariser segment covering the two transcript words, so the real merge
# produces a turn and the pipeline runs to completion without a model.
_WORDS = [Word(text="hello", start=0.0, end=1.0), Word(text="world", start=1.0, end=2.0)]
_TRANSCRIPT = TranscriptResult(text="hello world", words=_WORDS)
_SEGMENTS = [DiarSegment(start=0.0, end=2.0, speaker="SPEAKER_00")]


class _CapturingASR:
    def __init__(self, captured: dict, model_id: str, device: str = "auto") -> None:
        captured["asr_device"] = device
        self.model_id = model_id

    def transcribe(self, path: str, language=None, **_kwargs) -> TranscriptResult:
        return _TRANSCRIPT


class _CapturingDiarizer:
    def __init__(self, captured: dict, device: str = "auto", **_kwargs) -> None:
        captured["diar_device"] = device

    def diarise(self, _path: str):
        return list(_SEGMENTS)


def _install(monkeypatch, module, captured: dict) -> None:
    """Replace `module`'s ASRModel / SenkoDiarizer with capturing stand-ins
    and neutralise the audio-format probe so no real WAV is needed."""
    # reprocess.py / muesli_backend.py short-circuit on an already-16k-mono WAV;
    # cli.py has no such probe (it always normalises), so only patch where used.
    if hasattr(module, "is_wav_16k_mono"):
        monkeypatch.setattr(module, "is_wav_16k_mono", lambda path: True)
    monkeypatch.setattr(
        module, "ASRModel", lambda model_id, device="auto": _CapturingASR(captured, model_id, device)
    )
    # cli.py and muesli_backend.py import SenkoDiarizer lazily from
    # senko_diarisation inside the function, so patch it at the source;
    # reprocess.py binds it at module top, so patch it there too.
    diarizer = lambda device="auto", **kw: _CapturingDiarizer(captured, device, **kw)
    monkeypatch.setattr(senko_diarisation, "SenkoDiarizer", diarizer)
    if hasattr(module, "SenkoDiarizer"):
        monkeypatch.setattr(module, "SenkoDiarizer", diarizer)


def test_reprocess_stream_routes_device_to_both_engines(monkeypatch, tmp_path) -> None:
    captured: dict = {}
    _install(monkeypatch, reprocess, captured)

    reprocess.reprocess_stream(
        tmp_path / "system.wav",
        "system",
        diar_backend="senko",
        asr_model=reprocess.DEFAULT_MODEL,
        language=None,
        gap_threshold=reprocess.DEFAULT_GAP_THRESHOLD_SECONDS,
        speaker_tolerance=reprocess.DEFAULT_SPEAKER_TOLERANCE_SECONDS,
        verbose=False,
        recovery=False,
        device="cuda",
    )

    assert captured["asr_device"] == "cuda"
    assert captured["diar_device"] == "cuda"


def test_reprocess_stream_device_defaults_to_auto(monkeypatch, tmp_path) -> None:
    captured: dict = {}
    _install(monkeypatch, reprocess, captured)

    reprocess.reprocess_stream(
        tmp_path / "system.wav",
        "system",
        diar_backend="senko",
        asr_model=reprocess.DEFAULT_MODEL,
        language=None,
        gap_threshold=reprocess.DEFAULT_GAP_THRESHOLD_SECONDS,
        speaker_tolerance=reprocess.DEFAULT_SPEAKER_TOLERANCE_SECONDS,
        verbose=False,
        recovery=False,
    )

    # No device passed -> both engines default to 'auto', so existing CLI
    # behaviour (which supplies no device) is unchanged.
    assert captured["asr_device"] == "auto"
    assert captured["diar_device"] == "auto"


def test_muesli_backend_run_pipeline_routes_device_to_both_engines(monkeypatch, tmp_path) -> None:
    captured: dict = {}
    _install(monkeypatch, muesli_backend, captured)

    muesli_backend.run_pipeline(
        input_path=tmp_path / "mic.wav",
        diar_backend="senko",
        asr_model=muesli_backend.DEFAULT_MODEL,
        language=None,
        gap_threshold=muesli_backend.DEFAULT_GAP_THRESHOLD_SECONDS,
        speaker_tolerance=muesli_backend.DEFAULT_SPEAKER_TOLERANCE_SECONDS,
        verbose=False,
        device="cuda",
    )

    assert captured["asr_device"] == "cuda"
    assert captured["diar_device"] == "cuda"


def test_cli_run_pipeline_routes_device_to_both_engines(monkeypatch, tmp_path) -> None:
    captured: dict = {}
    _install(monkeypatch, cli, captured)
    # cli.run_pipeline normalises audio and checks ffmpeg before ASR; stub
    # those so the test needs no ffmpeg and no real audio file.
    monkeypatch.setattr(cli, "check_ffmpeg", lambda: True)
    monkeypatch.setattr(cli, "normalise_audio", lambda path: str(tmp_path / "norm-does-not-exist.wav"))
    monkeypatch.setattr(cli, "get_audio_duration", lambda path: 2.0)

    input_file = tmp_path / "in.wav"
    input_file.write_bytes(b"RIFF")

    cli.run_pipeline(
        input_file=str(input_file),
        output_text=str(tmp_path / "out.txt"),
        device="cuda",
    )

    assert captured["asr_device"] == "cuda"
    assert captured["diar_device"] == "cuda"
