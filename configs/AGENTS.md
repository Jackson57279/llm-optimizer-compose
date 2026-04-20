# Configs Agent Guide

Applies to everything under `configs/`.

## Config Change Rules
- Treat config values as user-facing defaults.
- Keep keys stable; avoid renames unless migration/docs are included.
- Prefer additive updates over breaking changes.

## Safety
- Do not include secrets in config files.
- Keep resource limits realistic for local hardware constraints.
- Ensure YAML remains valid and readable.

## Validation
- If config schema/shape changes, verify loading via existing config helpers and run `pytest`.
