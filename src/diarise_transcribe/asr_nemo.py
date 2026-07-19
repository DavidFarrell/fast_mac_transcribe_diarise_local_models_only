"""
Linux ASR backend: NeMo loading nvidia/parakeet-tdt-0.6b-v3 (CUDA or CPU).

The darwin/MLX backend lives in asr.py; this module is the linux counterpart
selected by asr._select_backend. torch and nemo are imported inside functions,
never at module top, so importing this module (or asr) on linux does not pull
in the heavy CUDA stack until a model is actually loaded.

stdout is kept pure JSONL: NeMo/torch load and transcribe chatter (progress
bars, INFO logs) is forced to stderr, because the reprocess entry point writes
its JSONL event stream to stdout.
"""

import glob
import logging
import os
import sys
from contextlib import redirect_stdout
from typing import Any, Callable, List, Optional

from .asr import DEFAULT_MODEL_NEMO, TranscriptResult, Word

# Immutable revision of nvidia/parakeet-tdt-0.6b-v3 pinned in slice-0 ground
# truth (design/slice0-ground-truth.md §7). The default model is always loaded
# from this exact revision, never the mutable HF 'main'.
DEFAULT_REVISION = "7c35754d166cca382ad1e53e68b01e7c575f3a1d"


def _torch_cuda_available() -> bool:
    """Whether torch reports a usable CUDA device (imports torch lazily)."""
    import torch

    return torch.cuda.is_available()


def resolve_device(
    requested: str,
    cuda_available: Callable[[], bool] = _torch_cuda_available,
) -> str:
    """
    Resolve a requested device to a concrete torch device on linux.

    'auto' -> 'cuda' when torch reports CUDA usable, else 'cpu'.
    'cuda' when CUDA is unavailable is a hard error, never a silent CPU
    fallback. 'cpu' is always honoured. Anything else is rejected.

    cuda_available is injected so the resolver is testable without torch.
    """
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not cuda_available():
            raise RuntimeError(
                "device='cuda' was requested but torch reports no usable CUDA "
                "device. Reboot to load the matching driver, or pass "
                "device='cpu'. Refusing to fall back to CPU silently."
            )
        return "cuda"
    if requested == "auto":
        return "cuda" if cuda_available() else "cpu"
    raise ValueError(
        f"Unknown device {requested!r}: expected 'auto', 'cuda' or 'cpu'."
    )


def nemo_words_to_words(nemo_words: List[dict]) -> List[Word]:
    """
    Convert NeMo word-timestamp dicts to Word objects.

    NeMo's hypothesis.timestamp['word'] entries look like
    {'word': 'Right,', 'start': 0.16, 'end': 0.64, 'start_offset': 2,
     'end_offset': 8}; 'start'/'end' are seconds. Only text/start/end are used.
    """
    return [
        Word(text=w["word"], start=float(w["start"]), end=float(w["end"]))
        for w in nemo_words
    ]


def _resolve_nemo_source(model_id: str) -> str:
    """
    Map a model id to a local .nemo path to restore from.

    The default nvidia model is fetched at its pinned immutable revision. A
    local .nemo file (or a directory containing one) is used verbatim. Any
    other id is passed through unchanged for restore_from to resolve or fail
    on - the frozen id rule keeps custom ids as backend-error passthrough.
    """
    if model_id == DEFAULT_MODEL_NEMO:
        from huggingface_hub import snapshot_download

        local = snapshot_download(
            model_id, revision=DEFAULT_REVISION, allow_patterns=["*.nemo"]
        )
        matches = glob.glob(os.path.join(local, "*.nemo"))
        if not matches:
            raise RuntimeError(
                f"No .nemo file in the pinned snapshot of {model_id}."
            )
        return matches[0]
    if os.path.isdir(model_id):
        matches = glob.glob(os.path.join(model_id, "*.nemo"))
        if matches:
            return matches[0]
    return model_id


def load_model(model_id: str, device: str) -> Any:
    """Load a parakeet-tdt NeMo model onto the resolved device."""
    resolved = resolve_device(device)
    source = _resolve_nemo_source(model_id)

    # NeMo/torch emit load logs and progress to stdout; keep stdout pure JSONL.
    with redirect_stdout(sys.stderr):
        logging.getLogger("nemo_logger").handlers = [
            logging.StreamHandler(sys.stderr)
        ]
        from nemo.collections.asr.models import ASRModel as _NemoASRModel

        print(f"Loading ASR model: {model_id}", file=sys.stderr)
        model = _NemoASRModel.restore_from(source, map_location=resolved)
        print("ASR model loaded.", file=sys.stderr)
    return model


def transcribe(
    model: Any,
    audio_path: str,
    language: Optional[str] = None,
    chunk_duration: float = 120.0,
    overlap_duration: float = 15.0,
) -> TranscriptResult:
    """
    Transcribe a 16kHz mono WAV with NeMo, returning word-level timestamps.

    language/chunk_duration/overlap_duration are accepted for signature parity
    with the darwin backend but not forwarded: NeMo's parakeet-tdt model does
    its own long-audio handling and language is auto-detected. Very long inputs
    therefore chunk differently from the darwin path - a known cross-platform
    residual, out of scope for this slice.
    """
    with redirect_stdout(sys.stderr):
        hypothesis = model.transcribe([audio_path], timestamps=True)[0]

    words = nemo_words_to_words(hypothesis.timestamp["word"])
    return TranscriptResult(text=hypothesis.text, words=words)
