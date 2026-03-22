# Contributing

## Prerequisites

- [Python 3.10+](https://www.python.org/)
- [uv](https://docs.astral.sh/uv/getting-started/installation/) — fast Python package manager
- [Lefthook](https://lefthook.dev) — git hook manager

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/your-org/ai-sdk-stream-python.git
cd ai-sdk-stream-python

# 2. Install Python dependencies (includes dev tools: pytest, ruff, pyright)
uv sync --all-groups

# 3. Install Lefthook
#    macOS / Linux (Homebrew)
brew install lefthook
#    or via npm
npm install -g @evilmartians/lefthook
#    or via pip (any OS)
pip install lefthook

# 4. Activate git hooks
lefthook install
```

That's it. The hooks will run automatically on every commit and push.

## Git hooks (via Lefthook)

| Hook | What it does |
|------|-------------|
| `pre-commit` | Formats staged files with **ruff format**, auto-fixes lint issues with **ruff check --fix**, and type-checks with **pyright**. Fixed files are automatically re-staged. |
| `commit-msg` | Validates the commit message follows [Conventional Commits](#commit-messages). |
| `pre-push` | Runs the full **pytest** suite before any push reaches the remote. |

To run a hook group manually without committing:

```bash
lefthook run pre-commit
lefthook run pre-push
```

## Commit messages

This project uses [Conventional Commits](https://www.conventionalcommits.org/). Semantic versioning and the changelog are generated automatically from commit history, so the format matters.

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

| Type | When to use | Version bump |
|------|------------|-------------|
| `feat` | New feature | minor (`1.1.0`) |
| `fix` | Bug fix | patch (`1.0.1`) |
| `perf` | Performance improvement | patch |
| `feat!` or `fix!` | Breaking change (the `!`) | major (`2.0.0`) |
| `docs` | Documentation only | no release |
| `refactor` | Code change, no behaviour change | no release |
| `test` | Adding or updating tests | no release |
| `chore` / `ci` / `build` | Tooling, CI, dependencies | no release |

**Examples:**

```
feat: add write_source() helper for citation events
fix(context): prevent double finish() from emitting two finish events
feat!: rename write_event_to_stream() to write_event()
docs: add FastAPI integration example to README
```

## Running checks manually

```bash
# Format
uv run ruff format src tests

# Lint (with auto-fix)
uv run ruff check --fix src tests

# Type check
uv run pyright src

# Tests
uv run pytest

# Run everything at once (mirrors CI)
uv run ruff format src tests && uv run ruff check src tests && uv run pyright src && uv run pytest
```

## Releases

Releases are handled automatically by [python-semantic-release](https://python-semantic-release.readthedocs.io/) on push to `main`. You do not need to bump versions or write changelog entries manually — just use the correct commit types above.

| Branch | Release type | Example version |
|--------|-------------|----------------|
| `main` | Stable | `1.2.0` |
| `beta` | Beta pre-release | `1.2.0b1` |
| `alpha` | Alpha pre-release | `1.2.0a1` |
