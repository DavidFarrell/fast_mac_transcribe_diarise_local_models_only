# Slice 5a - wrapper audit (audit-only, no edits)

Audit of the skill "wrapper" texts that drive `diarise-transcribe` / the
transcription pipeline, so Slice 5b can make them run on the Linux whitebox
without breaking the Mac. No skill file was edited in this slice. The only
write is this file.

Box: whiteboxlinux (Ubuntu 24.04). Date: 2026-07-19. Branch: `linux-cuda`.

## TL;DR for the boss

- 5 skill surfaces audited. Every one is Mac-bound and demonstrably fails here.
- **fast-transcribe** has a tracked source in THIS repo, but the INSTALLED copy
  has drifted (3 blocks the tracked source lacks) - reconcile before editing.
- **blogify** and **podcast-transcribe** have **NO tracked source anywhere on
  this box** (`~/.claude/skills/` is not a git repo). Per the plan, editing
  those is a boss decision, not a builder edit. This is the headline blocker
  for 5b.
- `skill/SKILL.md` in this repo is an **orphan** "fast-diarize" variant - not
  installed as any skill, superseded by `skill.md`. Recommend retire, don't
  edit.
- yt-dlp: installed this slice via `uv tool install yt-dlp` -> 2026.7.4.
  ffmpeg/ffprobe present (6.1.1). **mogrify (ImageMagick) MISSING** - a
  David-sudo step for blogify.

## Files audited (5)

| # | Path | Installed name | Tracked source? |
|---|------|----------------|-----------------|
| 1 | `/home/david/git/projects/fast_transcribe/skill.md` | fast-transcribe | tracked here (and byte-identical muesli-embedded copy) |
| 2 | `/home/david/git/projects/fast_transcribe/skill/SKILL.md` | fast-diarize (ORPHAN) | tracked here; not installed anywhere |
| 3 | `~/.claude/skills/fast-transcribe/SKILL.md` | fast-transcribe | **drifted** - superset of #1, no exact tracked source |
| 4 | `~/.claude/skills/blogify/SKILL.md` | blogify | **NONE on this box** |
| 5 | `~/.claude/skills/podcast-transcribe/SKILL.md` | podcast-transcribe | **NONE on this box** |

Provenance evidence (md5):
- `skill.md` (repo root) == muesli-embedded `skill.md` == `9c4534df...` (identical, both tracked).
- installed `fast-transcribe/SKILL.md` == `deab3544...` - matches NEITHER; it is
  `skill.md` PLUS three hand-added blocks (see drift below).
- `skill/SKILL.md` == `d87459db...` (the divergent fast-diarize variant).
- `~/.claude/skills` and `~/.claude` are **not git repos** - installed skills are unversioned.
- blogify / podcast-transcribe SKILL.md: `find` over all of `$HOME` returns ONLY
  the two installed copies (`~/.claude/skills/blogify/SKILL.md`,
  `~/.claude/skills/podcast-transcribe/SKILL.md`) - no tracked copy anywhere.
  They exist only as unversioned installed files.

### Drift: installed fast-transcribe vs its tracked source

`diff skill.md ~/.claude/skills/fast-transcribe/SKILL.md` shows the installed
copy adds, and the tracked source lacks:
- after L101: `export NUMBA_CACHE_DIR=/tmp/numba_cache` in the Run block.
- a whole `### Running Multiple Transcriptions` section (numba cache-corruption
  guidance, uses `trash`).
- a `## Permissions` section (proceed-without-asking allowlist).

Consequence: if 5b edits the tracked `skill.md` and copies it over the
installed file, those three blocks are LOST unless first reconciled INTO the
tracked source. This must be a boss decision (which copy is canonical), not a
silent builder overwrite.

## Demonstrated Mac-only failures (run on this box, harmless probes)

| Skill surface | Command as written | Result on Linux |
|---|---|---|
| fast-transcribe #1/#3, podcast-transcribe | `cd /Users/david/git/ai-sandbox/projects/fast_mac_transcribe_diarise_local_models_only` | `bash: cd: ... No such file or directory` (exit 1) |
| blogify, podcast-transcribe | `[ -f "/Users/david/git/podsyncfixdocker/data/..." ]` | NOT FOUND (Mac PodSync tree absent) |
| podcast-transcribe | `jq -r '.email' ~/.config/pocketcasts/credentials.json` | `Could not open file ... No such file or directory` |
| blogify (error path) | `mogrify -resize ...` | `mogrify: command not found` (ImageMagick not installed) |

Other named Mac-bound patterns - checked across the three in-scope skill texts
(`skill.md`, installed fast-transcribe, blogify, podcast-transcribe) with a grep
for `parakeet-mlx|coremltools|mlx|/opt/homebrew|pbpaste|osascript|afplay|open -a|screencapture`:
- ALL absent as executable lines. The only hits are descriptive prose in
  fast-transcribe L3 (description frontmatter) and L11 ("Parakeet MLX (ASR) +
  Senko/pyannote+CAM++ CoreML") - already flagged as platform wording, not a
  runnable command. blogify and podcast-transcribe: zero hits. So "demonstrate
  everything Mac-bound" is complete: no parakeet-mlx import, no /opt/homebrew
  path, no pbpaste/osascript/screencapture in any in-scope wrapper.

Working on Linux (recorded as working, not speculation):
- `ffmpeg`/`ffprobe` 6.1.1 present at `/usr/bin` - the extraction/probe lines run.
- `yt-dlp` 2026.7.4 present after this slice's install - download lines run.
- `jq`, `trash` (trash-cli 0.24.5), `/tmp/fast-diarize/` all fine.
- The Linux repo DOES declare the entry point: `pyproject.toml` L34-35
  `[project.scripts] diarise-transcribe = "diarise_transcribe.cli:main"`.

The wrapper command itself - actually run, not asserted:
- `cd /home/david/git/projects/fast_transcribe && uv run diarise-transcribe --help`
  was executed. It did NOT reach `--help`: with no provisioned `.venv`, `uv`
  began resolving and downloading the CUDA torch stack (torch 846MB,
  nvidia-cublas 566MB, etc.) and timed out at 2 min mid-download. This is
  Slice 1 (packaging) territory, not a wrapper defect - the wrapper's
  `cd`-corrected `uv run` line is sound, but a live `--help` cannot be
  demonstrated until Slice 1 has provisioned the Linux dependency set. Honest
  status: entry point declared (verified), live invocation gated on Slices 1-4.
  (Side effect cleaned up: the resolve rewrote `uv.lock`; restored to HEAD with
  `git checkout -- uv.lock`. The incomplete `.venv` is gitignored. Repo left
  clean apart from this design file.)

## Tooling verification (per plan Slice 5a decisions)

- **yt-dlp = uv-managed tool.** Was ABSENT. Installed this slice:
  `uv tool install yt-dlp` -> `yt-dlp==2026.7.4`; `yt-dlp --version` = `2026.07.04`;
  appears in `uv tool list`. On PATH via `~/.local/bin`.
- **ffmpeg = system package.** PRESENT: `ffmpeg version 6.1.1-3ubuntu5`,
  `ffprobe version 6.1.1-3ubuntu5`. No action.
- **mogrify / ImageMagick = system package.** ABSENT. Only used in blogify's
  image-dimension error path (`mogrify -resize "1500x1500>"`). This is a
  **David-sudo step** (`sudo apt install imagemagick`) - do NOT auto-install.
  Route via the board. Lower priority: it is an error-recovery path, not the
  main pipeline, and the main ffmpeg extraction already caps frame size.

## NAMED files-to-edit list for Slice 5b

One line per file: [tracked-source status] specific Mac-only lines -> proposed treatment.

1. `/home/david/git/projects/fast_transcribe/skill.md` [TRACKED here] - L103 `cd /Users/.../fast_mac_transcribe...`; descriptive L8/L10-11 (Apple Silicon / Parakeet MLX + Senko CoreML) -> platform-conditional: neutral `cd` into the repo per box + a Linux tech note (NeMo parakeet + senko/CUDA). BLOCKED-ish: reconcile the installed drift (numba + Permissions blocks) into this source first.
2. `/home/david/git/projects/fast_transcribe/skill/SKILL.md` [TRACKED here, ORPHAN] - L89 same Mac `cd`; "fast-diarize" name, not installed -> NO edit; recommend RETIRE. Boss decision, not a builder edit.
3. `~/.claude/skills/fast-transcribe/SKILL.md` [INSTALLED, drifted, no exact tracked source] - L106 `cd /Users/...`; L8/L10-11 descriptive -> this is 5b's install TARGET; edit flows from the reconciled #1 then `cp` in. FLAG: reconcile drift first.
4. `~/.claude/skills/blogify/SKILL.md` [**NO TRACKED SOURCE**] - PodSync paths L29-40 & L56-73 (`/Users/david/git/podsyncfixdocker/data`, `.../complete-historical-data`); L266 `mogrify` -> BLOCKED: needs boss decision on where blogify's canonical source lives before any edit; also depends on the ImageMagick sudo step.
5. `~/.claude/skills/podcast-transcribe/SKILL.md` [**NO TRACKED SOURCE**] - L39 PodSync data dir; L257 `cd /Users/.../fast_mac_transcribe...`; L317 `VAULT_PATH=/Users/david/Library/CloudStorage/.../obsidian_vault` (Mac Drive path) -> BLOCKED: same no-tracked-source boss decision. Note credentials at `~/.config/{pocketcasts,podcastindex}` are config, not skill text - not a 5b edit.

## Plan correction (Slice 5a is meant to amend the plan - do this)

Slice 5b's text says: "this repo's `skill/` dir is expected canonical for
fast-transcribe." **That is wrong and will misdirect the 5b builder.** The
`skill/` dir holds `skill/SKILL.md`, which is the `name: fast-diarize` ORPHAN
variant (not installed as any skill). The actual canonical source of the
installed `fast-transcribe` skill is **`skill.md` at the repo ROOT** (byte-
identical to the muesli-embedded copy; the installed file is that plus untracked
drift). Amend the plan so 5b edits `skill.md` (root), not `skill/SKILL.md`.

## No-tracked-source flags (loud, per plan)

- **blogify** (`~/.claude/skills/blogify/SKILL.md`): no version-controlled source
  anywhere on this box. 5b must not edit it as a home-dir file. Boss must decide
  a canonical home (e.g. add it to this repo, or a dotfiles/skills repo) first.
- **podcast-transcribe** (`~/.claude/skills/podcast-transcribe/SKILL.md`): same -
  no tracked source. Same boss decision.
- **installed fast-transcribe drift**: the installed copy is a superset of the
  tracked `skill.md`; the delta (numba + Permissions blocks) is untracked. Decide
  canonical before 5b overwrites.

## Orphan check (referenced binaries/scripts vs what exists here)

- `mogrify` (ImageMagick) - referenced in blogify, **absent**; David-sudo install.
- `yt-dlp` - was absent, now installed (uv tool).
- Everything else referenced (`ffmpeg`, `ffprobe`, `jq`, `trash`, `curl`,
  `diarise-transcribe` entry point) resolves on this box.
- No missing custom SCRIPTS (the skills call binaries + the repo entry point, not
  bespoke helper scripts) beyond the Mac backend dir, which is the port's whole point.

## Adjacent finding (OUT of my named 5a scope - for the boss)

`~/.claude/skills/muesli-merge/SKILL.md` (a project skill under
`/home/david/obsidian/.claude/skills/`) is ALSO Mac-bound and shares this
pipeline's `reprocess` backend: L69 `BACKEND=/Users/david/git/ai-sandbox/projects/muesli/backend/fast_mac_transcribe_diarise_local_models_only`,
L35/L72/L151/L167 `/Users/david/Library/Application Support/Muesli/...`,
L150 `python3 /Users/david/obsidian/.claude/skills/muesli-merge/filler_analysis.py`.
The overlay names muesli-merge as a consumer that should work on both machines,
but the Slice 5a task named only fast-transcribe/blogify/podcast-transcribe.
Flagging so the boss can decide whether muesli-merge joins the 5b list. Note its
Muesli recordings live only on the Mac, so muesli-merge may be intentionally
Mac-only regardless.
