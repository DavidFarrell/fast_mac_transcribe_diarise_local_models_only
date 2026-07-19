"""
Golden end-to-end contract test (slice 4) - the port's flagship proof.

Runs the real `python -m diarise_transcribe.reprocess` entry point as a
subprocess on the committed two-stream fixture and asserts the frozen JSONL
contract documented in design/slice0-ground-truth.md and the MUST/MAY rules
in tests/fixtures/golden/GROUND-TRUTH.md.

What is asserted as HARD contract (backend-independent):
  - exit 0;
  - Linux stdout is PURE JSONL (every non-blank line parses - the Linux
    improvement over the Mac reference, whose stdout carries progress noise);
  - the ordered (type, stage, stream) event sequence matches the documented
    contract exactly, and each `recovering` event carries an int `windows`;
  - the final result line's outer key-set and every turn's key-set match the
    documented schema;
  - independent invariants: nonempty speaker ids, finite ordered timestamps
    within [0, duration], both streams present and nonempty, per-stream text
    overlapping the fixture's known script above threshold;
  - a backend-independent schema summary equal to the Mac reference's.

What is NOT asserted (MAY differ per GROUND-TRUTH.md): speaker cluster
counts/labels, turn counts, turn ordering, and the `recovering` event's extra
keys (senko over-splits the synthetic voices; Linux recovery may add a
`spans` key when it finds windows).

Marked integration: loads real ASR + senko models on CPU. The deterministic
suite excludes it via -m "not integration".
"""

import json
import math
import os
import re
import subprocess
import sys

import pytest

_HERE = os.path.dirname(__file__)
_FIXTURE_DIR = os.path.join(_HERE, "fixtures", "golden")
_MEETING = os.path.join(_FIXTURE_DIR, "meeting")
_REFERENCE = os.path.join(_FIXTURE_DIR, "reference-stdout.jsonl")

# The frozen event sequence (design/slice0-ground-truth.md section 6): system
# stream fully processed, then mic, bracketed by preparing/complete/result.
_EXPECTED_EVENTS = [
    ("status", "preparing", None),
    ("status", "transcribing", "system"),
    ("status", "diarizing", "system"),
    ("status", "merging", "system"),
    ("status", "recovering", "system"),
    ("status", "transcribing", "mic"),
    ("status", "diarizing", "mic"),
    ("status", "merging", "mic"),
    ("status", "recovering", "mic"),
    ("status", "complete", None),
    ("result", None, None),
]
_OUTER_KEYS = frozenset({"type", "turns", "speakers", "duration"})
_TURN_KEYS = frozenset({"speaker_id", "stream", "t0", "t1", "text"})
# Every speaker id is namespaced by its stream (slice-0 §6). The cluster
# COUNT after SPEAKER_ is MAY-differ and is never asserted.
_SPEAKER_ID_RE = re.compile(r"^(mic|system):SPEAKER_\d+$")

# The fixture's known scripts (tests/fixtures/golden/GROUND-TRUTH.md).
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
_MIC_SCRIPT = (
    "It went better than expected, actually. The batch jobs finished "
    "overnight, and the word error rate came down by about two percent on the "
    "evaluation set. Mostly, yes. There were a couple of rough patches near "
    "the end of long meetings, but nothing that broke the downstream "
    "summaries. Good catch. Let's warm the cache before the next benchmark, "
    "then, and compare like for like. I'll write that up as an action for "
    "Thursday. Agreed. Same audio in, same turns out, on any machine. That's "
    "the whole point."
)

# Word-overlap floor per the plan ("above a stated word-overlap threshold").
# Tolerant of the documented numeric-formatting and boundary-word variants
# (GROUND-TRUTH.md: the Mac reference itself dropped "Good catch." and the
# first "Agreed."); still far above chance. A tighter cross-backend WER cap
# is deliberately avoided: the two ASR stacks differ on exactly these
# documented variants, which would make the flagship test flaky.
_WORD_OVERLAP_THRESHOLD = 0.85


def _words(text: str) -> set[str]:
    return {w.strip(".,?!-").lower() for w in text.split() if w.strip(".,?!-")}


def _parse_jsonl(text: str, *, pure: bool) -> list[dict]:
    """Parse JSONL. With pure=True (Linux stdout) every non-blank line MUST
    parse; with pure=False (the polluted Mac reference) non-JSON lines are
    skipped."""
    events: list[dict] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            events.append(json.loads(stripped))
        except json.JSONDecodeError:
            if pure:
                raise AssertionError(f"non-JSON line on Linux stdout (expected pure JSONL): {stripped!r}")
    return events


def _event_triples(events: list[dict]) -> list[tuple]:
    return [(e.get("type"), e.get("stage"), e.get("stream")) for e in events]


def _schema_summary(events: list[dict]) -> dict:
    """Backend-independent summary: nothing here depends on cluster counts,
    turn counts, or speaker labels."""
    result = events[-1]
    return {
        "triples": _event_triples(events),
        "outer_keys": frozenset(result.keys()),
        "turn_keysets": {frozenset(t.keys()) for t in result["turns"]},
        "streams": frozenset(t["stream"] for t in result["turns"]),
    }


@pytest.mark.integration
def test_golden_contract() -> None:
    proc = subprocess.run(
        [
            sys.executable, "-m", "diarise_transcribe.reprocess", _MEETING,
            "--stream", "both", "--diar-backend", "senko", "--verbose",
        ],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert proc.returncode == 0, f"exit {proc.returncode}\nstderr tail:\n{proc.stderr[-4000:]}"

    events = _parse_jsonl(proc.stdout, pure=True)
    assert events, "no JSON emitted on stdout"

    # Event sequence: exact names and order (extra keys on `recovering` are
    # ignored here - only the triple is the frozen contract).
    assert _event_triples(events) == _EXPECTED_EVENTS

    # Every `recovering` event carries an int windows count; the value and any
    # `spans` key are backend-dependent and NOT asserted.
    for event in events:
        if event.get("stage") == "recovering":
            assert isinstance(event.get("windows"), int), event

    result = events[-1]
    assert result["type"] == "result"
    assert frozenset(result.keys()) == _OUTER_KEYS
    turns = result["turns"]
    assert turns, "no turns in result"
    for turn in turns:
        assert frozenset(turn.keys()) == _TURN_KEYS

    duration = result["duration"]
    assert isinstance(duration, float) and math.isfinite(duration) and duration > 0

    # Independent structural invariants (not compared to the reference).
    for turn in turns:
        sid = turn["speaker_id"]
        assert isinstance(sid, str) and _SPEAKER_ID_RE.match(sid), sid
        assert isinstance(turn["text"], str)
        assert turn["stream"] in {"mic", "system"}
        # A turn's speaker id carries its OWN stream's namespace, never the
        # other stream's - the streams are diarised separately.
        assert sid.startswith(turn["stream"] + ":"), (sid, turn["stream"])
        t0, t1 = turn["t0"], turn["t1"]
        assert isinstance(t0, float) and isinstance(t1, float)
        assert math.isfinite(t0) and math.isfinite(t1) and 0 <= t0 < t1 <= duration

    # The result's speakers list is the sorted set of the turns' well-formed
    # speaker ids (cluster COUNT is MAY-differ and deliberately not checked).
    speakers = result["speakers"]
    assert speakers == sorted(speakers)
    assert all(isinstance(s, str) and _SPEAKER_ID_RE.match(s) for s in speakers)
    assert set(speakers) == {t["speaker_id"] for t in turns}

    # Both streams present and nonempty; per-stream turns monotonically
    # ordered by t0 in emission order.
    assert {t["stream"] for t in turns} == {"mic", "system"}
    per_stream_text = {"mic": [], "system": []}
    per_stream_t0 = {"mic": [], "system": []}
    for turn in turns:
        per_stream_text[turn["stream"]].append(turn["text"])
        per_stream_t0[turn["stream"]].append(turn["t0"])
    for stream in ("mic", "system"):
        assert per_stream_text[stream], f"{stream} stream produced no turns"
        starts = per_stream_t0[stream]
        assert starts == sorted(starts), f"{stream} turns not ordered by t0"

    # Text per stream matches the fixture's known script well above chance.
    for stream, script in (("system", _SYSTEM_SCRIPT), ("mic", _MIC_SCRIPT)):
        hyp = _words(" ".join(per_stream_text[stream]))
        ref = _words(script)
        overlap = len(hyp & ref) / len(ref)
        assert overlap >= _WORD_OVERLAP_THRESHOLD, (
            f"{stream} word overlap {overlap:.2f} < {_WORD_OVERLAP_THRESHOLD}; "
            f"text={' '.join(per_stream_text[stream])!r}"
        )

    # Backend-independent schema equality vs the Mac reference (filtered to
    # its JSON lines). Only schema summaries - never turn/cluster counts or
    # speaker labels, which GROUND-TRUTH.md marks MAY-differ.
    with open(_REFERENCE, encoding="utf-8") as fh:
        reference_events = _parse_jsonl(fh.read(), pure=False)
    # The reference's recovering events carry an int windows too (the exact
    # value and any spans key are MAY-differ, so not compared cross-backend).
    for event in reference_events:
        if event.get("stage") == "recovering":
            assert isinstance(event.get("windows"), int), event
    linux_summary = _schema_summary(events)
    reference_summary = _schema_summary(reference_events)
    assert linux_summary["triples"] == reference_summary["triples"] == _EXPECTED_EVENTS
    assert linux_summary["outer_keys"] == reference_summary["outer_keys"] == _OUTER_KEYS
    assert linux_summary["turn_keysets"] == reference_summary["turn_keysets"] == {_TURN_KEYS}
    assert linux_summary["streams"] == reference_summary["streams"] == frozenset({"mic", "system"})
