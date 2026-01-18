# macOS Audio Capture Research

## Executive Summary

Capturing all audio on macOS (both system output and microphone input) requires navigating Apple's security model. There are **three main approaches** available today, ranging from simple (virtual audio drivers) to modern native APIs (Core Audio taps). The best choice depends on your requirements for user experience, setup complexity, and minimum macOS version support.

## The Three Approaches

| Approach | Min macOS | Setup | UX | Reliability | Best For |
|----------|-----------|-------|-----|-------------|----------|
| **1. Core Audio Taps API** | 14.2+ | None | Excellent | High | New apps, no driver install |
| **2. ScreenCaptureKit** | 12.3+ | None | Good | High | When you also need screen |
| **3. Virtual Audio Driver** | 10.13+ | Required | Moderate | High | Widest compatibility |

---

## Approach 1: Core Audio Taps API (Recommended for macOS 14.2+)

### Overview

Introduced in **macOS 14.2** (December 2023) and improved in **14.4**, this is Apple's native solution for capturing system audio without virtual drivers.

### Key Advantages

- **No driver installation** - Works out of the box
- **Pre-mixer audio** - Captures clean audio regardless of system volume (turn speakers to zero, still records)
- **Process filtering** - Can tap specific apps or all system audio
- **Muting support** - Can mute tapped audio while still recording
- **No screen recording indicator** - Unlike ScreenCaptureKit

### How It Works

1. Create a `CATapDescription` specifying which processes to tap
2. Call `AudioHardwareCreateProcessTap()` to create the tap
3. Create an aggregate device including the tap
4. Use `AudioDeviceCreateIOProcIDWithBlock` for callbacks
5. Process audio in real-time

### Permission Required

Apps need `NSAudioCaptureUsageDescription` in Info.plist. Users see a one-time permission prompt.

### Best Reference Implementations

**Swift/Objective-C:**
- [AudioCap](https://github.com/insidegui/AudioCap) - Comprehensive sample code by Guilherme Rambo
- [AudioTee](https://github.com/makeusabrew/audiotee) - CLI tool that pipes audio to stdout (ideal for streaming to ASR)

**Key AudioTee Features:**
- Outputs raw PCM to stdout (perfect for piping to transcription)
- Configurable sample rates: 8000, 16000, 22050, 24000, 32000, 44100, 48000 Hz
- Mono or stereo output
- Can mute playback while recording
- Process filtering (include/exclude specific PIDs)

### Example: Using AudioTee

```bash
# Record all system audio to a file
audiotee > recording.pcm

# Record at 16kHz mono (ideal for speech recognition)
audiotee --sample-rate 16000 > recording.pcm

# Record specific app only
audiotee --include-pids 12345 > recording.pcm

# Mute while recording
audiotee --mute > recording.pcm
```

### Python Integration Challenge

The Core Audio taps API is poorly documented and low-level. PyObjC bindings exist (`pyobjc-framework-CoreAudio`) but:
- Documentation notes: "CoreAudio is a fairly low-level framework... I'm not yet convinced that the API actually works correctly from Python"
- Recent issues reported with ScreenCaptureKit on macOS 15 (SCStreamErrorDomain -3805)

**Recommended Python approach:** Use AudioTee as a subprocess and read from stdout:

```python
import subprocess
import numpy as np

# Start audio capture
process = subprocess.Popen(
    ['audiotee', '--sample-rate', '16000'],
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL
)

# Read audio chunks
chunk_size = 3200  # 100ms at 16kHz mono, 16-bit
while True:
    data = process.stdout.read(chunk_size)
    if not data:
        break
    audio = np.frombuffer(data, dtype=np.int16)
    # Process audio...
```

---

## Approach 2: ScreenCaptureKit

### Overview

Available since **macOS 12.3**, originally for screen capture but supports audio-only capture with workarounds.

### Key Characteristics

- **Requires screen recording permission** - Shows purple indicator in Control Center
- **Post-mixer audio** - Recording level follows system volume
- **Can filter by app or window**
- **No pure audio-only mode** - Must capture screen at very low framerate as workaround

### When to Use

- If you need screen capture anyway
- If supporting macOS 12.3-14.1
- Apple recommends: "If you are not capturing the screen and only capturing audio, it would be best to use a Core Audio tap"

### Example Configuration

```swift
let streamConfig = SCStreamConfiguration()
streamConfig.capturesAudio = true
streamConfig.excludesCurrentProcessAudio = true
// Workaround: set very low framerate to minimize overhead
streamConfig.minimumFrameInterval = CMTime(value: 10, timescale: 1)  // 0.1 fps
```

---

## Approach 3: Virtual Audio Driver (BlackHole)

### Overview

[BlackHole](https://github.com/ExistentialAudio/BlackHole) is a free, open-source virtual audio driver that creates a loopback device.

### Key Advantages

- **Widest compatibility** - Works on macOS 10.13+
- **Zero latency** - Direct audio routing
- **Customizable** - 2, 16, 64, 128, or 256 channels
- **No kernel extension** - Works without disabling SIP

### Key Disadvantages

- **Requires installation** - Users must install the driver
- **Multi-Output Device setup** - Users must configure Audio MIDI Setup
- **Volume control disabled** - When using virtual output

### How It Works

1. Install BlackHole
2. Create a "Multi-Output Device" in Audio MIDI Setup combining:
   - BlackHole 2ch (for capture)
   - Your actual speakers (for hearing audio)
3. Set Multi-Output Device as system output
4. Record from BlackHole input

### Python Integration

```python
import sounddevice as sd
import numpy as np

# Find BlackHole device
devices = sd.query_devices()
blackhole_idx = None
for i, d in enumerate(devices):
    if 'BlackHole' in d['name'] and d['max_input_channels'] > 0:
        blackhole_idx = i
        break

# Record from BlackHole
def callback(indata, frames, time, status):
    # Process audio chunk
    audio_chunk = indata.copy()
    # Send to transcription...

with sd.InputStream(device=blackhole_idx,
                    channels=1,
                    samplerate=16000,
                    callback=callback):
    # Recording active
    pass
```

### Using SoX

```bash
# Simple recording
sox -t coreaudio "BlackHole 2ch" recording.wav

# Record at 16kHz mono
sox -t coreaudio "BlackHole 2ch" -r 16000 -c 1 recording.wav
```

### Using FFmpeg

```bash
# List devices
ffmpeg -f avfoundation -list_devices true -i ""

# Record from BlackHole (device index varies)
ffmpeg -f avfoundation -i ":2" -ar 16000 -ac 1 recording.wav
```

---

## Capturing BOTH Microphone and System Audio

For meeting transcription, you need both:
- **System audio** - To hear the other participants
- **Microphone** - To capture your own voice

### Option A: Separate Streams (Recommended)

Capture mic and system audio as separate streams, then process independently:

```python
# System audio via AudioTee
system_process = subprocess.Popen(
    ['audiotee', '--sample-rate', '16000'],
    stdout=subprocess.PIPE
)

# Microphone via sounddevice
import sounddevice as sd
mic_stream = sd.InputStream(samplerate=16000, channels=1)

# Process both streams
# Benefit: Better speaker diarization (you vs them)
```

### Option B: Combined Stream

Using BlackHole, create a combined aggregate device:
1. Create Multi-Output Device (speakers + BlackHole)
2. Create Aggregate Device combining:
   - BlackHole (system audio input)
   - Microphone

### How Granola Does It

According to their documentation:
- Captures system audio and microphone separately
- Shows "Me" and "Them" in transcripts
- Uses real-time transcription (Deepgram, Assembly)
- Requires "Screen & System Audio recording" permission on macOS

---

## Open Source Meeting Transcription Projects

### 1. AudioTee
**Best for:** Streaming system audio to ASR
- Language: Swift
- Uses Core Audio taps
- Outputs PCM to stdout
- [GitHub](https://github.com/makeusabrew/audiotee)

### 2. Meetily
**Best for:** Full meeting transcription
- Uses Parakeet/Whisper + Ollama
- 100% local processing
- Speaker diarization
- [GitHub](https://github.com/Zackriya-Solutions/meeting-minutes)

### 3. Recap
**Best for:** macOS-native approach
- Uses Core Audio taps + AVAudioEngine
- WhisperKit for local transcription
- Status: Incomplete/broken
- [GitHub](https://github.com/RecapAI/Recap)

### 4. Pluely (Cluely alternative)
**Best for:** Real-time AI assistance
- Built with Tauri
- System audio capture
- Multiple STT providers
- [GitHub](https://github.com/iamsrikanthnani/pluely)

---

## Recommendation for Your Project

Given you already have local transcription working (Parakeet MLX + Senko diarization), here's the recommended approach:

### Minimum Viable Solution (Quick)

**Use BlackHole + Python sounddevice:**
1. Have users install BlackHole 2ch
2. Document Multi-Output Device setup
3. Use `sounddevice` to capture from BlackHole
4. Pipe audio to your existing transcription pipeline

```python
# Add to your project
import sounddevice as sd

def capture_system_audio():
    """Capture system audio from BlackHole device."""
    # Find BlackHole
    for i, d in enumerate(sd.query_devices()):
        if 'BlackHole' in d['name'] and d['max_input_channels'] > 0:
            return sd.InputStream(device=i, samplerate=16000, channels=1)
    raise RuntimeError("BlackHole not found - please install it")
```

### Best Solution (Modern)

**Build or use AudioTee for system audio capture:**
1. Compile AudioTee (Swift)
2. Run as subprocess from Python
3. Stream output to your transcription
4. Separately capture microphone with standard APIs

This gives you:
- No driver installation required (macOS 14.2+)
- Clean separation of "me" vs "them" audio
- Pre-mixer audio quality
- Single permission prompt

### Architecture

```
┌─────────────────┐     ┌─────────────────┐
│   Microphone    │     │  System Audio   │
│  (sounddevice)  │     │   (AudioTee)    │
└────────┬────────┘     └────────┬────────┘
         │                       │
         v                       v
┌─────────────────────────────────────────┐
│           Audio Buffer/Queue            │
│         (with source labeling)          │
└────────────────┬────────────────────────┘
                 │
                 v
┌─────────────────────────────────────────┐
│        Parakeet MLX Transcription       │
└────────────────┬────────────────────────┘
                 │
                 v
┌─────────────────────────────────────────┐
│     Senko/Sortformer Diarization        │
│   (enhanced with source knowledge)      │
└────────────────────────────────────────┘
```

---

## Sources

- [AudioTee - GitHub](https://github.com/makeusabrew/audiotee)
- [AudioCap - GitHub](https://github.com/insidegui/AudioCap)
- [Apple: Capturing system audio with Core Audio taps](https://developer.apple.com/documentation/CoreAudio/capturing-system-audio-with-core-audio-taps)
- [BlackHole - GitHub](https://github.com/ExistentialAudio/BlackHole)
- [Meetily - GitHub](https://github.com/Zackriya-Solutions/meeting-minutes)
- [Pluely - GitHub](https://github.com/iamsrikanthnani/pluely)
- [Recap - GitHub](https://github.com/RecapAI/Recap)
- [Granola - How transcription works](https://help.granola.ai/article/transcription)
- [Recording system audio on macOS 2025](https://slinkp.com/record-system-audio-macos-2025.html)
- [Core Audio Tap API Gist](https://gist.github.com/directmusic/7d653806c24fe5bb8166d12a9f4422de)
- [pyobjc-framework-CoreAudio](https://pypi.org/project/pyobjc-framework-CoreAudio/)
