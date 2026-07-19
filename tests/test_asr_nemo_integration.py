"""
CPU real-model transcription smoke test for the linux NeMo backend.

Marked integration: it loads nvidia/parakeet-tdt-0.6b-v3 (pinned revision) and
transcribes the golden fixture's system stream on CPU. Run in a venv with
torch+nemo+huggingface_hub (e.g. /tmp/slice0-probes/nemo28-venv) - the
deterministic suite excludes it with -m "not integration".
"""

import os
import shutil
import subprocess

import pytest

from diarise_transcribe import asr, asr_nemo

_FIXTURE = os.path.join(
    os.path.dirname(__file__), "fixtures", "golden", "meeting", "audio", "system.wav"
)

# The system stream's true script (slice-0 ground truth / fixture GROUND-TRUTH.md).
_SYSTEM_SCRIPT = (
    "Right, shall we make a start? This is the weekly pipeline check-in. "
    "First item is the transcription backend - how did the migration go? "
    "That's encouraging. Did the speaker separation hold up when two people "
    "talked over each other? Can I jump in here? I looked at the logs this "
    "morning, and the embedding cache was cold for the first three runs. That "
    "alone explains most of the slowdown. Perfect. Last thing - the golden "
    "fixture. Once this recording exists, every platform should produce the "
    "same turns from the same audio. Agreed?"
)


def _words(text: str) -> set[str]:
    return {w.strip(".,?-").lower() for w in text.split() if w.strip(".,?-")}


@pytest.mark.integration
def test_cpu_transcription_smoke(tmp_path) -> None:
    assert shutil.which("ffmpeg"), "ffmpeg required to resample the fixture to 16k"
    wav16k = tmp_path / "system_16k.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", _FIXTURE, "-ar", "16000", "-ac", "1", str(wav16k)],
        check=True,
        capture_output=True,
    )

    model = asr_nemo.load_model(asr.DEFAULT_MODEL_NEMO, "cpu")
    result = asr_nemo.transcribe(model, str(wav16k))

    # Words present with sane, ordered, finite timestamps.
    assert result.words, "no words returned"
    assert all(isinstance(w, asr.Word) for w in result.words)
    import math

    assert all(
        math.isfinite(w.start) and math.isfinite(w.end) and 0 <= w.start <= w.end
        for w in result.words
    )
    # Word start times are monotonically nondecreasing - diarisation alignment
    # downstream depends on it, so a reordering regression must fail here.
    starts = [w.start for w in result.words]
    assert starts == sorted(starts), "word start times not nondecreasing"

    # Text overlaps the known system script well above chance.
    hyp, ref = _words(result.text), _words(_SYSTEM_SCRIPT)
    overlap = len(hyp & ref) / len(ref)
    assert overlap >= 0.7, f"word overlap {overlap:.2f} too low; text={result.text!r}"
