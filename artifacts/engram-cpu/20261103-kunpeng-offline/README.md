# Kunpeng offline preparation — 2026-11-03

This receipt supersedes the initial portable P0 snapshot for the current
920/950 implementation. Both feature branches still start at their paired
v0.30.0 tags. Changes remain uncommitted; transfer the working files, not only
branch refs, to the internal network.

| Check | Result | Evidence |
| --- | --- | --- |
| Host build/install and native Python import | Passed on macOS arm64 | [Build manifest](build-validation/manifest.json) |
| Portable/NEON numerical suite | 93 passed, 28 SVE2 cases skipped | [Numerical output](build-validation/numerical.txt) |
| Usable local ISAs | NEON true, SVE2 false | [Capabilities](build-validation/capabilities.txt) |
| Linux ARMv8-A NEON raw kernel | Cross compilation only | [Preparation manifest](manifest.json) |
| Linux scalable SVE2 raw kernels | Cross compilation only | Preparation manifest |
| Offline build validator | Completed | Build manifest and per-command logs |
| CPU benchmark harness | Completed tiny smoke, 38 records | [Smoke manifest](benchmark-smoke/manifest.json) |
| Python/C++ style and Ascend syntax | Passed | Preparation manifest |
| Kunpeng 920/950, Linux execution and Ascend model | Not run | No target environment |

The microbenchmark smoke uses three repetitions and tiny synthetic tables solely
to check the harness. Its samples are not Kunpeng performance evidence.
The SVE2 object has not been executed locally. Ascend UT was previously blocked
by missing runtime dependencies; its adapter changes receive syntax/style
checks here, without substituting them for runtime validation.

The [preparation manifest](manifest.json) records current source hashes, commands
and outcomes. Historical [P0 receipts](../20261103-p0-local/README.md) retain
earlier source and binary hashes; they do not describe the updated binary.

Use the [internal validation guide](../../../docs/contributing/model/engram_cpu_kunpeng.md)
for required-ISA gates, NUMA/thread sweeps, table sizes and remaining model
acceptance. The implementation boundary is synchronous host kernels plus an
explicit Ascend adapter; serving selection, transfers, collectives and graph
integration remain subsequent phases.
