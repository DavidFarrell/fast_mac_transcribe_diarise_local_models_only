# Golden fixture - two-stream transcription + diarisation reference (Mac)

Built by macbase, 19 Jul 2026, for cross-platform verification of the Linux
port. **Fully synthetic** (Gemini TTS voices, no meeting content) - safe to
commit to a repo as a permanent test asset.

## The audio

`meeting/audio/mic.wav` + `meeting/audio/system.wav` - 72.4s each, PCM s16le
24kHz mono, time-aligned like a real Muesli call (when one stream speaks the
other is silent; one deliberate overlap: mic1 starts at 9.82s while sys
utterance 1's tail is closing).

sha256:
- mic.wav    `4dc15e666abd69269dc9cc37cc910a549f167ad6e918c7c8988b1085b489338a`
- system.wav `3050ea79eaaa277ea1faa7ba0cf5fb7684dfadafa6315cda2951e9258816f298`

## True content (what a perfect system would output)

Timeline (start times as constructed, ±TTS render):

| t (s) | Stream | True voice | Text |
|---|---|---|---|
| 0.00 | system | Voice A (Aoede) | "Right, shall we make a start? This is the weekly pipeline check-in. First item is the transcription backend - how did the migration go?" |
| 9.82 | mic | Voice M (Puck) | "It went better than expected, actually. The batch jobs finished overnight, and the word error rate came down by about two percent on the evaluation set." |
| 19.80 | system | Voice A | "That's encouraging. Did the speaker separation hold up when two people talked over each other?" |
| 26.58 | mic | Voice M | "Mostly, yes. There were a couple of rough patches near the end of long meetings, but nothing that broke the downstream summaries." |
| 35.04 | system | Voice B (Charon) | "Can I jump in here? I looked at the logs this morning, and the embedding cache was cold for the first three runs. That alone explains most of the slowdown." |
| 45.18 | mic | Voice M | "Good catch. Let's warm the cache before the next benchmark, then, and compare like for like. I'll write that up as an action for Thursday." |
| 54.04 | system | Voice A | "Perfect. Last thing - the golden fixture. Once this recording exists, every platform should produce the same turns from the same audio. Agreed?" |
| 65.38 | mic | Voice M | "Agreed. Same audio in, same turns out, on any machine. That's the whole point." |

Ground truth: mic stream = ONE voice (4 utterances). System stream = TWO
voices (A×3, B×1).

## The Mac reference run

Exact command (run from anywhere; BACKEND is the copy embedded in the muesli
repo - identical to the standalone repo):

```bash
BACKEND=/Users/david/git/ai-sandbox/projects/muesli/backend/fast_mac_transcribe_diarise_local_models_only
PYTHONPATH=$BACKEND/src NUMBA_CACHE_DIR=/tmp/muesli-numba-cache HF_HUB_OFFLINE=1 \
  "$BACKEND/.venv/bin/python" -m diarise_transcribe.reprocess \
  "<this-dir>/meeting" \
  --stream both --diar-backend senko --verbose \
  > reference-stdout.jsonl 2> reference-stderr.log
```

Exit 0. Output: `reference-stdout.jsonl` (stdout only; final line is
`{"type":"result","turns":[...],"speakers":[...],"duration":72.24}`),
`reference-stderr.log` (progress/verbose).

Versions: Python 3.12.13, parakeet-mlx 0.5.0 (ASR: parakeet-tdt-0.6b-v3),
senko 0.1.0 (pyannote segmentation-3.0 + CAM++ via CoreML), mlx 0.30.3,
numpy 2.3.5. macOS arm64 (Apple Silicon, Neural Engine).

## How to compare on Linux - what MUST match vs what may differ

MUST match (contract):
- JSONL protocol: same event shape, final result line with turns of
  `{speaker_id, stream, t0, t1, text}`; speaker_id prefixed `mic:`/`system:`;
  streams NEVER mixed.
- Text: near-verbatim vs the true content above (the Mac reference is
  near-perfect; treat >2-3% WER vs true text as a regression).
- Timing: turn boundaries within ~0.5s of the reference.
- Stream attribution: every turn on the correct stream.

MAY legitimately differ (document, don't chase):
- Speaker CLUSTER counts/labels. Known Mac behaviour on this synthetic
  fixture: senko OVER-SPLITS - mic's single voice came back as 4 speaker IDs,
  system's two voices as 3. Isolated prosody-flat TTS utterances are hard to
  cluster; real meetings behave better. A Linux pyannote pipeline that returns
  mic=1 / system=2 here is BETTER than the reference, not wrong. The
  meaningful diarisation check is per-stream separation + the system stream
  distinguishing voice B's utterance (35-45s) from voice A's.
- Numeric formatting ("2%" vs "two percent"), boundary-word drops (the Mac
  dropped "Good catch." and the first "Agreed." at turn edges).
