"""
Per-stream separation regression test (slice 4).

The quality lever of this pipeline is that the mic and system streams are
transcribed and diarised as SEPARATE inputs - never concatenated or mixed
into one signal and then relabelled by stream afterwards. This asserts that
property at the orchestration seam: it captures the audio path each engine
(ASR and diariser) actually receives and proves the two streams arrive as
two distinct inputs, rather than inspecting the final `stream` field (which a
mix-then-relabel bug would still populate correctly).
"""

import sys
from pathlib import Path

from diarise_transcribe import reprocess
from diarise_transcribe.asr import TranscriptResult, Word
from diarise_transcribe.diarisation import DiarSegment


class _RecordingASR:
    """Records every audio path handed to transcribe()."""

    def __init__(self, asr_inputs: list, model_id: str, device: str = "auto") -> None:
        self._asr_inputs = asr_inputs
        self.model_id = model_id

    def transcribe(self, path: str, language=None, **_kwargs) -> TranscriptResult:
        self._asr_inputs.append(path)
        return TranscriptResult(
            text="hello world",
            words=[Word(text="hello", start=0.0, end=1.0), Word(text="world", start=1.0, end=2.0)],
        )


class _RecordingDiarizer:
    """Records every audio path handed to diarise()."""

    def __init__(self, diar_inputs: list, **_kwargs) -> None:
        self._diar_inputs = diar_inputs

    def diarise(self, path: str):
        self._diar_inputs.append(path)
        return [DiarSegment(start=0.0, end=2.0, speaker="SPEAKER_00")]


def test_mic_and_system_reach_engines_as_separate_inputs(monkeypatch, tmp_path) -> None:
    meeting_dir = tmp_path / "meeting"
    (meeting_dir / "audio").mkdir(parents=True)
    (meeting_dir / "audio" / "system.wav").write_bytes(b"RIFF")
    (meeting_dir / "audio" / "mic.wav").write_bytes(b"RIFF")

    asr_inputs: list[str] = []
    diar_inputs: list[str] = []

    # Treat every input as already-16k-mono so no ffmpeg runs and the path
    # each engine sees is the stream's own file (system.wav / mic.wav), not a
    # shared normalised temp file.
    monkeypatch.setattr(reprocess, "is_wav_16k_mono", lambda path: True)
    monkeypatch.setattr(
        reprocess, "ASRModel", lambda model_id, device="auto": _RecordingASR(asr_inputs, model_id, device)
    )
    monkeypatch.setattr(
        reprocess, "SenkoDiarizer", lambda **kwargs: _RecordingDiarizer(diar_inputs, **kwargs)
    )
    monkeypatch.setattr(sys, "argv", ["reprocess.py", str(meeting_dir), "--stream", "both", "--no-recovery"])

    assert reprocess.main() == 0

    # Exactly the two stream files reached ASR, as two distinct inputs.
    assert {Path(p).name for p in asr_inputs} == {"system.wav", "mic.wav"}
    assert len(set(asr_inputs)) == 2
    # The diariser saw the same two distinct inputs - so neither engine was
    # ever fed a mixed/concatenated signal.
    assert set(diar_inputs) == set(asr_inputs)
    assert len(set(diar_inputs)) == 2
