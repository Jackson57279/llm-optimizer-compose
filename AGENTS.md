# Miniforge Agent Guide

## Purpose
Python library for MiniMax M2.7 inference, optimized for GMKtech M7 (28GB RAM constraint).

## Architecture
- Source: `src/miniforge/` — import as `miniforge`
- Public API: `Miniforge`, `InferenceEngine`, `M7Config` (from `__init__.py`)
- Backends: `llama_cpp` (preferred) with `transformers` fallback
- Config: `M7Config` + YAML files in `configs/`

## Development Commands
```bash
# Install (uv preferred; editable avoids llama-cpp-python build issues on Windows)
uv pip install -e ".[all]"

# For llama.cpp backend only (requires C++ toolchain on Windows)
uv pip install -e ".[llama-cpp]"

# Validation order
black src/ tests/
ruff check src/ tests/
mypy src/
pytest tests/test_core.py   # quick check
pytest tests/test_tools.py  # tool calling
pytest                      # full suite (slow)

# Reinstall after packaging changes
python -m pip install -e .
```

## Tooling Config (from pyproject.toml)
- **Black**: line-length 100, target py310-312
- **Ruff**: same line-length, ignores E501 (handled by black), B008, C901, N818
- **MyPy**: strict — `disallow_untyped_defs = true`, `disallow_incomplete_defs = true`
- **Pytest**: asyncio_mode = auto, addopts `-v --tb=short`

## Conventions
- All public APIs must be typed (mypy strict).
- Keep async; no blocking calls in async paths.
- Memory defaults are conservative (28GB ceiling).
- Backend errors must be clear when fallback unavailable.
- Never hardcode secrets; use env vars.

## Directory Override
- `src/` has additional rules in `src/AGENTS.md`.

## Learned User Preferences
- Develop and run this stack on Windows without relying on WSL.
- Use `scripts/start.ps1` as the primary entrypoint for starting, stopping, and checking the local stack instead of many manual steps.
- When executing an attached implementation plan, do not edit the plan file; use the existing todos.

## Learned Workspace Facts
- LiteLLM proxy listens on `http://127.0.0.1:4000` and Open WebUI on `http://127.0.0.1:8080` by default (`scripts/start.ps1`).
- Scripted service logs are written under `logs/`.
