---
name: check
description: Runs ruff lint and pytest for this repo and summarizes any failures. Use when the user asks to lint, run tests, check the code, or verify everything passes before committing.
---

## Lint

!`ruff check . 2>&1`

## Tests

!`pytest 2>&1`

## Instructions

Look at the lint and test output above.

- If both are clean, say so in one line — don't repeat the output.
- If there are lint violations, list each one with file:line and a one-line fix
  suggestion.
- If there are test failures, list each failing test name and the assertion/error
  that caused it, then point to the likely source file (`keyword_classify.py`,
  `keyword_cluster.py`, `keyword_peers.py`, or `keyword_common.py`).
- Do not attempt fixes yourself unless the user asks — just report findings clearly.
