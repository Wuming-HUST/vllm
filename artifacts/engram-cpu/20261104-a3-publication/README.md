# A3 experiment publication checks (2026-11-04)

Code commits: vLLM `5b4777525342b1e74b3ecaad36481f701d928dc4`; vLLM-Ascend `bbafe101803384346e9d7d371657e62ecbf59041`.

This is a local publication review, not A3 runtime or performance acceptance.

- `.venv/bin/python -m pytest --noconftest tests/kernels/test_engram_cpu.py -q -ra`: 102 passed, 31 skipped (28 SVE2, one CUDA, two NPU cases).
- `run_isolated.py`, run with the vLLM virtualenv Python: 18/18 current-source cases passed; the previous integration HEAD passed 5/18. Actual runner/hash methods and new tests are AST-extracted; the receipt records source hashes. NPU/runtime/model/event/collective behavior is mocked. This proves Python dispatch and cleanup behavior only.
- Full Ascend test attempt: its virtualenv lacks pytest; using the paired vLLM virtualenv reaches conftest but fails with `ModuleNotFoundError: No module named 'vllm'`. No full Ascend test executed.
- Changed Python files: Ruff check, Ruff format check, AST parsing and Git whitespace checks passed.
- Companion JSON parsed; seven CSVs checked for consistent columns. Empty sample CSVs contain headers only. Local document links checked. Launcher bash syntax, device/cpu dry runs and invalid control rejection passed; no service was started.
- vLLM pre-commit and Ascend `bash format.sh ci` were attempted. External hook installation remained blocked at `pip install .` for ruff-pre-commit for over seven minutes; the second check waited for the shared lock. Both task-owned processes were terminated. No full hook success is claimed. Saved logs show the incomplete setup; commits use explicit hook bypass after the available checks, with this limitation recorded in their messages.

Still required on A3: matching real INT8/FP32 tables/hash/rows and generation, callback snapshot lifetime, repeated FULL replay, idle ranks/slot reuse, DP4/TP4/HCCL, effective configuration, data and run-list freeze, instrumentation/replay implementation, model quality and E2E performance.

The published plan includes required companion deployment/quantization/audit documents, launch script, manifest, source-inspection record and seven CSV assets. The separate workload protocol and unrelated untracked notes are excluded.
