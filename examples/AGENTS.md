# Examples Agent Guide

Applies to everything under `examples/`.

## Goals
- Keep examples simple, copy-paste friendly, and aligned with current APIs.
- Demonstrate one concept per file (chat, streaming, tools, vision, server).

## Constraints
- Avoid hidden dependencies beyond documented project extras.
- Keep runtime assumptions explicit (model id, backend, quantization).
- Prefer safe defaults that work on local machines.

## Maintenance
- If API signatures change, update affected examples in the same task.
- Verify examples at least import and run through basic execution paths when practical.
