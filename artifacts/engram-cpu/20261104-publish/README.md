# Publication checks — 2026-11-04

Both repositories are published as feature branches based on paired v0.30.0
tags. Main branches and upstream origins are unchanged.

- Independent native/operator suite rechecked: 102 passed, 30 skipped. Device,
  SVE2, model quality, multi-rank and performance acceptance remain pending.
- Changed Python files: Ruff check, Ruff format check and compileall passed in
  both repositories. Staged whitespace checks passed.
- Standard pre-commit commands were attempted: vLLM `pre-commit run` and Ascend
  `bash format.sh ci`. Their first-run environment bootstrap stalled while
  installing external hook dependencies. The task-owned initialization processes
  were stopped before hooks executed; logs are preserved here. Full pre-commit
  coverage is unverified. Commits use explicit sign-off and attribution; automatic
  hooks are bypassed for this delivery after the available checks above.
- No compiled binaries, virtual environments or unrelated local documents are
  included. Historical raw receipts retain original contents; their text paths
  disable whitespace lint so captured diff context lines remain verbatim.

Earlier build/numerical results:
[CPU pipeline/graph receipt](../20261103-cpu-pipeline-graph/README.md).
