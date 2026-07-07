# Tools

This directory stores repository-maintenance scripts for AIForge.

## Scope

- Link validation, benchmark generation, source auditing, coverage metrics, and repeatable repository checks.
- Tools should be runnable from the repository root and avoid hidden external state unless documented.
- Prefer standard-library scripts when possible so checks remain easy to reproduce.

## Existing Tools

- `validate_internal_links.py`: validates local Markdown links.
- `benchmark_repository_scale.py`: counts local files, Markdown files, text, URLs, and sampled GitHub comparators.

## Routing Rules

- Put generated reports in `../docs/` or the relevant source-atlas directory.
- Put workflow wiring in `../.github/workflows/`.
- Keep one-off manual notes out of this directory.
