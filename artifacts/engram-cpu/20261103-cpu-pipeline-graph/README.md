# CPU pipeline and graph preparation receipt — 2026-11-03

Current implementation: CPU vocabulary compression, hash and embedding gather;
stream-ordered native host callback for gather; graph-external per-step input
refresh; existing platform communication and per-layer consumer event paths.
Both repositories remain on `feat/engram-cpu-v41-integration`, paired v0.30.0.
No commit or push was performed.

## Completed local checks

- Actual native CMake build/install and required NEON capability check passed.
- Final independent operator/workspace/bridge/callback-payload suite: **102 passed,
  30 skipped**. Skips: 28 SVE2, 1 CUDA graph, 1 NPU graph. macOS arm64, Torch 2.10.
- Source, hash/history/cache, capacity padding, quantized decode, tail/padded-head
  reconstruction, buffer reuse and callback error propagation checked locally.
- Ruff, format, Python compilation, native format and diff checks passed.
  Initial destructor-style lint failure is retained in `ruff-vllm.txt`; the
  correction passed in `ruff-vllm-retry.txt`. This is not a runtime failure.
- Model validation tool help/syntax checked. No checkpoint generation executed.

`host/manifest.json` records exact configure/build/install/test commands and
binary hash. `host/numerical.txt` is that build's test result. `host-final.txt`
records the final Python staging version after the style-only correction.
`checks.json` records additional commands and exit codes. `source-manifest.json`
records final source hashes across both repositories, tags/commits and acceptance.
Untracked source files are covered by hashes; tracked patches are also saved.
Historical receipts were not overwritten.

## Not accepted on target

The normal vLLM suite failed while importing conftest (`tblib` missing); the
Ascend suite failed while importing upstream runtime (`regex` missing). Their
exit code 4 and complete outputs are retained in `standard-*-attempt.txt`.
Neither collection attempt reached the actual tests. Isolated native checks
are not a replacement for those suites.

Real CUDA/NPU callback + external-event graph replay tests are written in
`tests/kernels/test_engram_cpu.py`. They were skipped here. The CUDA hash-cache
replay test compares CPU/device outputs and cache state against an independent
oracle; multi-GPU tests cover external FULL/breakable replay, DP/shared/split,
TP and SP. These device tests have not run. Ascend's producer-capability test
excludes CPU from graph-internal production; full Ascend mock/runtime tests
remain pending due to missing dependencies.

Target gates remain: runtime callback support and ordering, FULL/piecewise/eager,
TP1/2/4/8 and tag-supported DP/EP/SP/PCP, request/cancel/reorder/prefix/speculative
history, KV rebind, pinned-buffer lifetime/teardown, checkpoint rows/logits/tokens,
dataset quality, and performance/profiler measurements on Kunpeng 920/950.
Metadata copies and CPU conversion/hash are still a synchronous dispatch prefix;
only embedding retrieval is queued as native CPU callback work. No full hiding,
zero stall, end-to-end speedup or hardware acceptance is claimed.

Commands, rollback and protocol details:
[serving guide](../../../docs/contributing/model/engram_cpu_serving.md).
