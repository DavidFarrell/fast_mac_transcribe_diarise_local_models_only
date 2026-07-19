"""
ASR module with per-platform backends.

darwin: parakeet-mlx (Apple Silicon, MLX). Its loader and transcribe logic live
in this module and are unchanged; the parakeet_mlx import is lazy so this module
imports on linux without it.

linux: NeMo loading nvidia/parakeet-tdt-0.6b-v3 (see asr_nemo.py). That module
is imported lazily inside _select_backend so importing asr on linux pulls in
neither mlx nor nemo/torch until a model is loaded.

Backend choice is a single typed function, not a factory or registry.
"""

import sys
import threading
from dataclasses import dataclass
from typing import Any, Callable, List, NamedTuple, Optional


# Cache of loaded models keyed by (model_id, device), so repeated ASRModel
# instances for the same model reuse one in-memory model instead of
# re-deserialising it. Bounded by the number of distinct (id, device) pairs
# used in a process - in practice 1. Each entry is (model, generation): the
# generation it was loaded at, for the CUDA-poison invalidation below.
_model_cache: dict[tuple[str, str], tuple[Any, int]] = {}
_model_cache_lock = threading.Lock()

# CUDA co-residency poison generation. Senko's CUDA diarisation irrecoverably
# corrupts the CUDA state of any co-resident model (a torch/kaldifeat bug -
# see design/gpu-verification.md; the Mac's CoreML path never hits it), so a
# NeMo model transcribed after a CUDA diarisation dies with an illegal memory
# access. senko_diarisation calls poison_cuda_asr() after any CUDA diarise;
# an ASRModel whose model was loaded at an older generation reloads a fresh
# one before transcribing. On CPU/darwin senko never runs on CUDA, so this is
# never called, the generation stays 0, and model reuse is byte-identical to
# before - no device resolution, no torch import on the cache path.
_cuda_asr_generation = 0


def poison_cuda_asr() -> None:
    """Mark every currently-loaded ASR model as CUDA-poisoned (see above).

    Called by senko_diarisation after a diarisation that ran on CUDA. The
    next transcribe on any model loaded before this call reloads a fresh one.
    """
    global _cuda_asr_generation
    with _model_cache_lock:
        _cuda_asr_generation += 1


@dataclass
class Word:
    """A single word with timestamp."""
    text: str
    start: float  # seconds
    end: float  # seconds

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class TranscriptResult:
    """Full transcript with word-level timestamps."""
    text: str
    words: List[Word]


# Default models - Parakeet TDT 0.6B v3, one build per platform.
DEFAULT_MODEL_MLX = "mlx-community/parakeet-tdt-0.6b-v3"
DEFAULT_MODEL_NEMO = "nvidia/parakeet-tdt-0.6b-v3"


def default_model_id(platform: str = sys.platform) -> str:
    """The default model id for the active platform's backend."""
    return DEFAULT_MODEL_NEMO if platform.startswith("linux") else DEFAULT_MODEL_MLX


# Resolved once at import time on the running platform, so cli.py / reprocess.py
# get the right per-platform default without a CLI flag change.
DEFAULT_MODEL = default_model_id()


def check_model_id(model_id: str, platform: str = sys.platform) -> None:
    """
    Fail fast on a model id that belongs to the other platform's backend.

    Ids prefixed for the opposite backend (mlx-community/ on linux,
    nvidia/ on darwin) can never load and are a clear user error. Any other id
    - custom ids, local paths, other orgs - is left for the backend to attempt.
    """
    if platform.startswith("linux") and model_id.startswith("mlx-community/"):
        raise ValueError(
            f"model id {model_id!r} is an MLX (darwin) model but this is the "
            f"linux NeMo backend. Use an nvidia/ id such as {DEFAULT_MODEL_NEMO!r}."
        )
    if platform == "darwin" and model_id.startswith("nvidia/"):
        raise ValueError(
            f"model id {model_id!r} is a NeMo (linux) model but this is the "
            f"darwin MLX backend. Use an mlx-community/ id such as {DEFAULT_MODEL_MLX!r}."
        )


class _Backend(NamedTuple):
    """The loader and transcriber for one platform's ASR backend."""
    load: Callable[[str, str], Any]
    transcribe: Callable[..., TranscriptResult]


def _select_backend(platform: str = sys.platform) -> _Backend:
    """Return the load/transcribe pair for the active platform."""
    if platform.startswith("linux"):
        from . import asr_nemo

        return _Backend(asr_nemo.load_model, asr_nemo.transcribe)
    return _Backend(_load_mlx, _transcribe_mlx)


def _load_mlx(model_id: str, device: str) -> Any:
    """
    Load a parakeet-mlx model (darwin). device is accepted for signature
    parity and ignored - MLX runs on the Apple Silicon GPU/Neural Engine and
    has no device selection, so darwin behaviour is unchanged.
    """
    from parakeet_mlx import from_pretrained

    print(f"Loading ASR model: {model_id}")
    model = from_pretrained(model_id)
    print("ASR model loaded.")
    return model


def _transcribe_mlx(
    model: Any,
    audio_path: str,
    language: Optional[str] = None,
    chunk_duration: float = 120.0,
    overlap_duration: float = 15.0,
) -> TranscriptResult:
    """Transcribe with parakeet-mlx, merging BPE subword tokens into words."""
    result = model.transcribe(
        audio_path,
        chunk_duration=chunk_duration,
        overlap_duration=overlap_duration,
    )

    # Extract words by merging BPE subword tokens
    # BPE tokens starting with space (or ▁) indicate new word boundaries
    words = []
    current_word_tokens = []

    for sentence in result.sentences:
        for token in sentence.tokens:
            token_text = token.text
            if not token_text:
                continue

            # Check if this token starts a new word
            # New word indicators: leading space, leading ▁, or first token
            is_new_word = (
                token_text.startswith(" ") or
                token_text.startswith("▁") or
                not current_word_tokens
            )

            if is_new_word and current_word_tokens:
                # Finish previous word
                word_text = "".join(t.text for t in current_word_tokens)
                word_text = word_text.strip().replace("▁", "")
                if word_text:
                    words.append(Word(
                        text=word_text,
                        start=current_word_tokens[0].start,
                        end=current_word_tokens[-1].end,
                    ))
                current_word_tokens = []

            current_word_tokens.append(token)

    # Don't forget the last word
    if current_word_tokens:
        word_text = "".join(t.text for t in current_word_tokens)
        word_text = word_text.strip().replace("▁", "")
        if word_text:
            words.append(Word(
                text=word_text,
                start=current_word_tokens[0].start,
                end=current_word_tokens[-1].end,
            ))

    return TranscriptResult(
        text=result.text,
        words=words,
    )


class ASRModel:
    """
    Platform-dispatching ASR model with word-level timestamps.

    Selects the darwin (MLX) or linux (NeMo) backend at construction and defers
    the heavy model load until first transcribe.
    """

    def __init__(self, model_id: str = DEFAULT_MODEL, device: str = "auto"):
        """
        Args:
            model_id: model id for the active platform's backend.
            device: 'auto', 'cuda' or 'cpu'. Resolved by the linux backend;
                ignored on darwin (MLX has no device selection).
        """
        check_model_id(model_id)
        self.model_id = model_id
        self.device = device
        self._backend = _select_backend()
        self._model = None
        self._loaded_generation = None

    def _ensure_loaded(self):
        """Lazy load the model on first use, reusing a process-wide cache.

        Reloads if a CUDA diarisation has poisoned co-resident models since
        this instance's model was loaded (see poison_cuda_asr). When no CUDA
        diarisation has occurred the generation is unchanged and this is the
        original reuse-one-cached-model behaviour.
        """
        if self._model is not None and self._loaded_generation == _cuda_asr_generation:
            return

        key = (self.model_id, self.device)
        with _model_cache_lock:
            generation = _cuda_asr_generation
            cached = _model_cache.get(key)
            if cached is None or cached[1] != generation:
                model = self._backend.load(self.model_id, self.device)
                _model_cache[key] = (model, generation)
            else:
                model = cached[0]
            self._model = model
            self._loaded_generation = generation

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        chunk_duration: float = 120.0,
        overlap_duration: float = 15.0,
    ) -> TranscriptResult:
        """
        Transcribe a 16kHz mono WAV with word-level timestamps.

        Args:
            audio_path: Path to 16kHz mono WAV file
            language: Language code (auto-detected if None)
            chunk_duration: Duration of audio chunks for long files
            overlap_duration: Overlap between chunks
        """
        self._ensure_loaded()
        return self._backend.transcribe(
            self._model,
            audio_path,
            language=language,
            chunk_duration=chunk_duration,
            overlap_duration=overlap_duration,
        )


def transcribe_audio(
    audio_path: str,
    model_id: str = DEFAULT_MODEL,
    language: Optional[str] = None,
    device: str = "auto",
) -> TranscriptResult:
    """
    Convenience function to transcribe audio.

    Args:
        audio_path: Path to 16kHz mono WAV file
        model_id: model id for the active platform's backend
        language: Language code (auto-detected if None)
        device: 'auto', 'cuda' or 'cpu' (see ASRModel)
    """
    model = ASRModel(model_id, device=device)
    return model.transcribe(audio_path, language=language)
