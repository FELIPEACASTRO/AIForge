# GitHub Workflows

This directory stores GitHub Actions workflows for AIForge automation.

## Scope

- CI, validation, documentation publishing, link checks, benchmark refreshes, and release automation.
- Workflows should be deterministic, least-privilege, and documented with clear trigger rules.
- Secrets should be referenced by name only and never committed.

## Reference Links

- GitHub Actions docs: https://docs.github.com/en/actions
- Workflow syntax: https://docs.github.com/en/actions/writing-workflows/workflow-syntax-for-github-actions
- Security hardening: https://docs.github.com/en/actions/security-guides/security-hardening-for-github-actions

## Routing Rules

- Put issue templates in `../ISSUE_TEMPLATE/`.
- Put repo launch documentation in `../../docs/launch/`.
- Put validation scripts in `../../tools/`.
