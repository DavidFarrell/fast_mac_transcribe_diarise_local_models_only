# Accelerated Diarisation

Offline transcription + speaker diarisation pipeline for Apple Silicon Macs.

## Features

- **ASR**: NVIDIA Parakeet via [parakeet-mlx](https://github.com/senstella/parakeet-mlx) (MLX-accelerated)
- **Diarisation**: NVIDIA Streaming Sortformer via CoreML (Neural Engine/GPU)
- **Output**: Speaker-labelled transcripts in TXT, JSON, SRT, and RTTM formats
- **Fully offline** after initial model downloads

## Requirements

- macOS on Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- ffmpeg (`brew install ffmpeg`)

## Installation

```bash
# Clone/navigate to the project
cd /Users/david/git/ai-sandbox/projects/accelerated_diarisation

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install the package in development mode
pip install -e .
```

## Usage

```bash
# Always activate the venv first!
source .venv/bin/activate

# Basic usage - transcribe with speaker labels
python -m diarise_transcribe --in audio.wav --out transcript.txt

# All output formats
python -m diarise_transcribe --in audio.wav \
    --out transcript.txt \
    --out-json transcript.json \
    --out-srt transcript.srt \
    --out-rttm diarisation.rttm

# Use higher quality diarisation model (slower, better for offline)
python -m diarise_transcribe --in audio.wav --out transcript.txt \
    --diar-model nvidia_high

# Specify expected number of speakers (filters to top N by activity)
python -m diarise_transcribe --in audio.wav --out transcript.txt \
    --num-speakers 2

# Keep temp files for debugging
python -m diarise_transcribe --in audio.wav --out transcript.txt \
    --keep-temp --verbose
```

## CLI Options

| Option | Description |
|--------|-------------|
| `--in`, `-i` | Input audio file (any format ffmpeg supports) |
| `--out`, `-o` | Output plain text file with speaker labels |
| `--out-json` | Output JSON file with words, segments, and turns |
| `--out-srt` | Output SRT subtitle file with speaker labels |
| `--out-rttm` | Output RTTM file (diarisation segments only) |
| `--diar-model` | Diarisation model: `default`, `nvidia_low`, `nvidia_high` |
| `--asr-model` | ASR model ID (default: mlx-community/parakeet-tdt-0.6b-v3) |
| `--language` | Language code for ASR (auto-detected if not specified) |
| `--num-speakers` | Filter output to top N speakers by activity |
| `--gap-threshold` | Gap threshold (seconds) for turn splitting (default: 0.8) |
| `--speaker-tolerance` | Tolerance for word-to-speaker assignment (default: 0.25) |
| `--keep-temp` | Keep temporary normalised WAV files |
| `--verbose`, `-v` | Verbose output |

## Diarisation Models

| Model | Description | Use Case |
|-------|-------------|----------|
| `default` | Sortformer streaming, ~1s latency | General use |
| `nvidia_low` | NVIDIA Low, ~1s latency | Streaming |
| `nvidia_high` | NVIDIA High, ~30s latency | Best quality, offline |

**Note**: All Sortformer models output up to 4 speakers. Use `--num-speakers` to filter.

## Output Formats

### Plain Text (`--out`)
```
[00:00.12 - 00:03.45] SPEAKER_00: Hello, how are you today?
[00:03.67 - 00:06.89] SPEAKER_01: I'm doing great, thanks for asking.
```

### JSON (`--out-json`)
```json
{
  "turns": [
    {
      "speaker": "SPEAKER_00",
      "start": 0.12,
      "end": 3.45,
      "text": "Hello, how are you today?",
      "words": [...]
    }
  ],
  "segments": [...]
}
```

### SRT (`--out-srt`)
```
1
00:00:00,120 --> 00:00:03,450
[SPEAKER_00] Hello, how are you today?

2
00:00:03,670 --> 00:00:06,890
[SPEAKER_01] I'm doing great, thanks for asking.
```

### RTTM (`--out-rttm`)
```
SPEAKER audio 1 0.12 3.33 <NA> <NA> SPEAKER_00 <NA> <NA>
SPEAKER audio 1 3.67 3.22 <NA> <NA> SPEAKER_01 <NA> <NA>
```

## How It Works

1. **Audio Normalisation**: Converts input to 16kHz mono WAV using ffmpeg
2. **ASR**: Parakeet-MLX transcribes audio with word-level timestamps
3. **Diarisation**: Sortformer CoreML identifies speaker segments
4. **Merge**: Words are assigned to speakers based on timestamp overlap
5. **Output**: Formatted as requested (TXT/JSON/SRT/RTTM)

## Troubleshooting

### ffmpeg not found
```bash
brew install ffmpeg
```

### Model download issues
Models are downloaded from HuggingFace on first use. Ensure you have internet access.
Cache location: `~/.cache/huggingface/hub/`

### CoreML errors
Ensure you're on macOS with Apple Silicon. Intel Macs are not supported.

### Memory issues with long audio
Try processing shorter segments or use `--diar-model default` for lower memory usage.

## Architecture

```
src/diarise_transcribe/
├── __init__.py       # Package info
├── __main__.py       # Module entry point
├── cli.py            # CLI argument parsing and pipeline
├── audio.py          # Audio normalisation with ffmpeg
├── asr.py            # ASR with parakeet-mlx
├── diarisation.py    # Speaker diarisation with CoreML
└── merge.py          # Merge ASR + diarisation outputs
```

## Dependencies

- `parakeet-mlx` - NVIDIA Parakeet ASR on MLX
- `coremltools` - CoreML model loading and inference
- `librosa` - Audio processing and mel spectrogram
- `soundfile` - Audio file I/O
- `numpy` - Numerical operations
- `huggingface-hub` - Model downloading

## License

MIT
