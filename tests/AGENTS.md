# Tests Agent Guide

Applies to everything under `tests/`.

## Test Standards
- Keep tests deterministic and offline by default.
- Prefer unit tests with fakes/mocks over network/model downloads.
- Add coverage for bug fixes and public API behavior changes.

## Structure
- Match existing pytest style and fixture patterns.
- Keep assertions specific and user-visible (messages, return shape, exceptions).
- Avoid brittle timing-based assertions in async tests.

## Execution
- Run targeted tests for changed behavior first.
- Before handoff, ensure `pytest` passes when changes are non-trivial.
