# Engram CPU integration delivery

Branch: `feat/engram-cpu-v41-integration`. Paired base tags: v0.30.0.

Use both modified repositories for Ascend:

- [vLLM branch](https://github.com/Wuming-HUST/vllm/tree/feat/engram-cpu-v41-integration)
- [vLLM-Ascend branch](https://github.com/Wuming-HUST/vllm-ascend/tree/feat/engram-cpu-v41-integration)
- [Build, startup, rollback and target acceptance](https://github.com/Wuming-HUST/vllm/blob/feat/engram-cpu-v41-integration/docs/contributing/model/engram_cpu_serving.md)
- [Local verification receipts](https://github.com/Wuming-HUST/vllm/tree/feat/engram-cpu-v41-integration/artifacts/engram-cpu/20261103-cpu-pipeline-graph)

`lookup_backend=cpu` selects CPU vocabulary conversion, hash and embedding
retrieval. Quantized tables are currently dequantized on CPU and BF16 rows are
transferred to the device. Projections/gates and the existing parallel
communication remain on device. CPU metadata/conversion/hash are a synchronous
dispatch prefix. Native stream callbacks queue embedding lookup work.

For Ascend layer 1/14 overlap, merge `multistream_engram_overlap: true` into
additional-config. Eager/FULL use auxiliary-stream preparation; PIECEWISE keeps
synchronous preparation. CPU production stays outside device graph capture.
NPU hash plus CPU retrieval is not an independently selectable mode.

Local independent validation: 102 passed, 30 skipped (28 SVE2, one CUDA, one
NPU). Native build/style/syntax checks passed in the recorded local environment.
Full repository tests were blocked by missing runtime dependencies. Kunpeng,
GPU/NPU graph, multi-rank, checkpoint parity, model quality and performance
acceptance remain pending; source inspection is not hardware validation.

Historical analysis/plans/receipts retain their original stage descriptions,
including statements that changes were not committed at that stage. This file
and the linked serving guide describe the current delivery.

Publication check note: full pre-commit initialization was interrupted while
installing external dependencies, before checks executed. Available changed-file
Ruff/style/syntax and staged whitespace gates passed. The publication receipt
in the vLLM repository records this limitation.

## A3 deployment and host-ID update (2026-11-04)

The paired Ascend update passes CPU hash IDs directly to CPU
lookup when local/shared tables require no per-step hash collective. It removes
hash H2D followed by lookup-ID D2H on that route; metadata D2H, BF16 H2D,
TP row communication and ready/event ordering remain. Device-collective routes
retain the existing bridge. This update requires both repositories at the versions recorded in the
[A3 experiment plan](docs/contributing/model/engram_cpu_a3_experiment_plan_zh.md) and still needs target validation.

See [A3 Docker deployment](docs/contributing/model/engram_cpu_a3_docker_zh.md) for the fixed v0.30.0 A3 DP4/TP4 recipe,
both source installations, native CPU extension build and callback/graph checks.

## Static correctness audit (2026-11-04)

The audit fixed V1 preparation failures bypassing metadata release and V1
forward/callback failures leaving Engram events unretired. CPU checks passed
102 tests with 31 skipped; 16 source-isolated runner control-flow cases passed.
Full Ascend UT and A3 runtime acceptance remain pending. See
[the audit record](docs/contributing/model/engram_cpu_static_audit_zh.md).
These fixes are included in this delivery; A3 acceptance remains pending.
