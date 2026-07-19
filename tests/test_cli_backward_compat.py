"""
Guard rails: the CLI's existing behaviour (flags, defaults, required-input
enforcement, no-canonical-cache-present startup) must remain unchanged now
that warm-cache support has been added. These are consumed by other repos
via `uv run --directory ... diarise-transcribe ...`, so no existing flag,
default, or error path may change.
"""

import subprocess
import sys

import pytest


def test_all_prior_flags_still_present():
    from diarise_transcribe.cli import create_parser

    parser = create_parser()
    dest_names = {action.dest for action in parser._actions}

    expected = {
        "input_file", "output_text", "output_json", "output_srt", "output_rttm",
        "diar_backend", "asr_model", "language", "num_speakers",
        "gap_threshold", "speaker_tolerance", "keep_temp", "verbose", "help",
    }
    assert expected.issubset(dest_names)


def test_prior_defaults_unchanged():
    from diarise_transcribe.cli import create_parser

    parser = create_parser()
    args = parser.parse_args(["--in", "a.wav", "--out", "b.txt"])

    assert args.diar_backend == "senko"
    assert args.language is None
    assert args.num_speakers is None
    assert args.gap_threshold == 0.8
    assert args.speaker_tolerance == 0.25
    assert args.keep_temp is False
    assert args.verbose is False


def test_missing_input_file_still_errors_same_as_before(monkeypatch, capsys):
    """Previously argparse's required=True produced the missing-input
    error (exit code 2, standard argparse usage banner on stderr); now
    --in is optional at the parser level (so --warm-cache can run without
    it) and the same check is enforced manually via parser.error(), which
    reproduces the identical exit code and usage-banner behaviour."""
    from diarise_transcribe import cli

    monkeypatch.setattr(sys, "argv", ["diarise-transcribe", "--out", "out.txt"])
    with pytest.raises(SystemExit) as exc_info:
        cli.main()
    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "--in" in captured.err
    assert "usage:" in captured.err


def test_no_canonical_cache_present_startup_is_cold_and_silent(tmp_path, monkeypatch):
    """With no canonical cache present (the common case today), importing
    cli.py must behave exactly as before: NUMBA_CACHE_DIR gets set to the
    per-PID /tmp path and nothing else observable happens. Also uses an
    empty (nonexistent) simulated senko cache dir so the bootstrap check
    has nothing to find, mirroring "no canonical cache yet" - and so this
    test never touches the real ~/.cache/senko/numba_cache on the machine
    running it."""
    canonical_root = tmp_path / "does-not-exist"
    senko_cache_dir = tmp_path / "senko-cache-does-not-exist"
    env = dict(__import__("os").environ)
    env["FAST_DIARISE_NUMBA_CANONICAL_ROOT"] = str(canonical_root)
    env["FAST_DIARISE_SENKO_CACHE_DIR_OVERRIDE"] = str(senko_cache_dir)

    result = subprocess.run(
        [sys.executable, "-c", (
            "import os; "
            "from diarise_transcribe import cli; "
            "print(os.environ['NUMBA_CACHE_DIR'])"
        )],
        cwd=__import__("pathlib").Path(__file__).resolve().parent.parent,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    printed = result.stdout.strip()
    assert printed.startswith("/tmp/numba_cache_")
    # No warning should be printed when there's simply no canonical cache yet.
    assert "Warning" not in result.stderr
