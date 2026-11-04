# Eager model integration: local receipt

Date: 2026-11-03. Both repositories use `feat/engram-cpu-v41-integration`
from their paired v0.30.0 tags. Changes remain uncommitted and unpushed.

This snapshot adds actual NVIDIA/Ascend model lookup selection, synchronous
device-ID/host-gather/device-row staging, eager configuration guards and
platform-specific E8M0 semantics. Existing sharding/collective paths are reused.
The full model has **not** been executed in this environment.

## Completed local checks

- CMake native build/install: exit 0; the installed module is the real Torch
  dispatcher extension, not a mocked operator or benchmark JIT.
- Isolated native/bridge tests: **97 passed, 28 skipped**. All skips require
  unavailable SVE2. NEON/scalar execute on macOS arm64, Torch 2.10.0.
- Bridge tests use CPU destinations and real native gather. They cover high-water
  buffer growth/reuse, changing and empty token batches, strided IDs, concurrent
  buffer ownership, and native/legacy E8M0. They do not test GPU/NPU transfers.
- Both repositories' modified Python files pass Ruff lint/format and syntax
  checks; native sources pass clang-format; both Git diffs pass whitespace checks.
- The whole-model parity tool passes its argument-help entry check. Its model
  execution, cleanup protocol and numerical comparison remain target-only work.

## Failed attempts retained

- `runner-attempt1.*` / `runner-attempt1-*.txt`: the local receipt runner resolved
  the virtualenv Python symlink to the base interpreter, losing its packages.
  The corrected runner uses the explicit `.venv/bin/python` path. These logs
  are tooling failures, not operator failures, and have not been overwritten.
- `config-pytest-attempt.txt`: standard repository conftest requires missing
  `tblib`; exit 4 before collecting configuration tests.
- `ascend-pytest-attempt.txt`: source checkout is not installed as a package;
  exit 4. Retrying with the paired upstream source on PYTHONPATH reaches a missing
  `regex` dependency (`ascend-pytest-with-upstream-path.txt`); exit 4.

## Target work not accepted

Linux/Kunpeng 920/950 and SVE2 execution, GPU/NPU copies, actual checkpoint rows,
distributed TP/DP/EP/SP/PCP, request lifecycle, generation/logprobs, quality and
performance have not been validated. CPU source/hash are independent operators;
the eager model still uses device hash. Asynchronous CPU overlap and device graph
replay are not implemented; CPU configuration requires `--enforce-eager`.

The new device/distributed/Ascend tests and
[model validation tool](../../../../tools/validate_engram_cpu_model.py) are
prepared for the internal target environment. See the
[serving guide](../../../../docs/contributing/model/engram_cpu_serving.md).

Commands, exit codes, binary hash and repository heads are in
[manifest.json](manifest.json). [sources.json](sources.json) records the exact
source snapshot; older P0/Kunpeng receipts remain historical snapshots.
