# Contributing

Contributions must preserve the phase discipline in
`KERR_DISK_CODEX_RESEARCH_GUIDE.md`.

- Do not add physical equations without updating `docs/equation_registry.md`.
- Do not add optional dependencies without a specific requirement.
- Do not hide failed simulations or numerical warnings.
- Run `uv run ruff check .`, `uv run ruff format --check .`,
  `uv run mypy src/kerrdisk`, and `uv run pytest -q` before submitting changes.
