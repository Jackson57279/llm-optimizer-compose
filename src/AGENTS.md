# Source Tree Agent Guide

Applies to: `src/miniforge/`

## API Preservation
- Keep `miniforge` import contract: `Miniforge`, `InferenceEngine`, `M7Config` exportable.
- Async consistency: no sync I/O in `async def` paths.
- Backend fallback: `llama_cpp` → `transformers` when unavailable; preserve this unless asked to change.

## Implementation
- Change the narrowest module that owns the behavior.
- Reuse existing config/memory utilities before adding abstractions.
- Memory defaults stay conservative; align with `M7Config` semantics.
- Raise clear errors for unsupported backend/config combos.

## Validation
```bash
pytest tests/test_core.py   # engine, config, backends
pytest tests/test_tools.py  # tool calling
pytest                      # full suite for broad changes
```
