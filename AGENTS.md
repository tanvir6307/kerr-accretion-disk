# KerrDisk-UQ Agent Contract

Read `KERR_DISK_CODEX_RESEARCH_GUIDE.md`, this file, `docs/assumptions.md`,
and `docs/equation_registry.md` before making changes.

Work one phase at a time. Do not implement physics from a later phase early.
Every production physical equation must be registered in
`docs/equation_registry.md` before implementation.

For each phase:

1. Summarize the requested phase.
2. List files to create or modify.
3. Identify scientific risks and required validation.
4. Implement only the phase scope.
5. Run formatting, linting, typing, and tests.
6. Report unresolved warnings and uncertainties.

Generated code is provisional until it passes analytic tests, independent
comparison, convergence studies, and inference validation where applicable.
