# Engram CPU 路径功能与流程静态审查

审查日期：2026-11-04。范围为两仓 `feat/engram-cpu-v41-integration`
本次配套更新，包括 host hash 直通优化；代码版本见实验计划。配套基线均为 v0.30.0。
这是源码与本机 CPU 检查记录，不是 A3、鲲鹏或模型精度验收。

## 结论与修复

发现并修复 V1 `_model_forward` 的两个异常清理缺口，均位于
`vllm-ascend/vllm_ascend/worker/model_runner_v1.py`：

1. **准备失败漏释放 metadata submission。** 原来 Engram 准备发生在
   `try/finally` 外。CPU hash、查表准备或图输入准备抛异常时，
   `executor.release()` 不会执行；调用方的正常返回后释放也到达不了。
   executor 保留 `submission_in_flight`，后续提交可能被拒绝。
   现已把准备与执行放在同一异常保护范围内。
2. **执行/回调错误没有清理未消费的 Engram event。** 原来最终只调用
   `finish_cpu_engram()`，该方法检查 CPU callback，不负责 join producer
   或 reset ExternalEvent。若模型在某个 Engram 层前退出，已 record 的
   event 可能缺少对应的 wait/reset，破坏下一次使用时的事件状态。
   现已在失败时调用 `retire_engram_lookups(reset_events=True)`，并使
   callback 检查错误也经过该路径；metadata release 始终在 finally 中执行。

正常执行继续使用原来的逐层 ready/wait 顺序。上述清理不承诺从硬件故障、
损坏的 runtime context 或分布式通信失败中恢复服务。

回归用例在 `vllm-ascend/tests/ut/models/test_engram.py` 的
`test_v1_engram_failure_retires_events_and_releases_metadata`：覆盖成功、
准备异常、forward 异常、callback 检查异常，与两种 ENPU 下发顺序、
普通准备/捕获输入准备组合，共 16 个用例。

## 当前流程与静态核对

| 环节 | 当前行为与核对结果 |
| --- | --- |
| 输入和历史 | `CPUHashWorkspace.run` 在当前 producer stream 上同步复制设备 metadata；V1 使用 prompt lookback 加物理 slot 历史，V2 使用 runner 的完整设备 lookback。不是从异步调度的 CPU placeholder 猜生成 token。 |
| 词表转换与 hash | `sources_out` 先写当前 slot，再读历史；按每个请求的 query offsets 隔离边界。image/dead 截断历史，缺失 lookback 回退 slot 或 pad；`hash_out` 一次生成所有 Engram 层的全局索引。原生算子数值测试通过。 |
| host hash 直通 | resolved shared-memory 模式或无 EDP hash exchange 时，CPU hashes 直接传给 CPU lookup。桥接器仍复制到自己的 host IDs buffer，防止 workspace 重用改写异步 callback 的输入。 |
| 分片表的设备 hash 通信 | 需要 EDP 索引通信时仍 publish hashes 到 NPU、沿用 gather、再把 IDs 复制到 CPU。没有跨设备张量直接交给 HCCL 的改动。 |
| 检索与反量化 | CPU 按 global head/vocab offset 读取本地分片；BF16 复制，INT8/FP32 或 MXFP8/E8M0 解码为 BF16。callback 与其 tensor owners 由 ticket 保持，错误由 worker 检查。 |
| rows 与通信 | host IDs 的 `embed_gathered` 显式分配 NPU 输出。BF16 H2D 后再执行原有 EDP rows exchange、TP head gather；没有把 rows collective 移到 CPU。 |
| 图与第 1/14 层 | CPU producer 不进入设备 capture。FULL 在每次 replay 前更新固定地址的 rows/mask，consumer 保留 ExternalEvent wait/reset。eager 使用普通 event；PIECEWISE 同步准备。逐层 event 在该层 H2D、通信和 padding 后 record。 |
| buffer 生命周期 | producer 先等待 main，保护上次消费及本次输入写入；lookup 的下一次复用等待 H2D completion event。host hash 源即使立即被改写，callback 应仍消费快照。真实 NPU 快照/回放用例已在测试中，但本机跳过。 |
| V2 收尾 | `EngramModelState.finish_execution(failed=...)` 已 join/reset producer；本次修复补齐 V1 对应的异常路径。 |

已检查上述调用链，暂未发现另一个可确定的正常数值/分片流程错误。
这不代表所有输入、并行拓扑及 runtime 时序已经穷尽验证。

特别注意：**metadata D2H、词表转换和 hash 当前仍是模型下发前的同步前缀**。
异步部分是后续 callback 检索、rows H2D、通信与模型层的重叠；各层检索
在一个 producer stream 上按顺序排队。不能把当前实现描述为完整 hash 异步化，
也不能保证第 1/14 层的等待时间已经被全部掩盖。

## 本次验证与边界

- vLLM：`.venv/bin/python -m pytest --noconftest tests/kernels/test_engram_cpu.py -q -ra`
  得到 **102 passed, 31 skipped**。跳过 28 个 SVE2、1 个 CUDA 和 2 个
  torch_npu host/device-ID 图测试。此结果不覆盖真实 Ascend event 或 HCCL。
- V1 新回归：隔离执行实际 `_model_forward` 和测试函数的 AST，以 mock
  替代不可导入的 Ascend 依赖。修复前 4/16 通过（12 个异常用例失败），
  修复后 16/16 通过。只验证 Python 控制流与清理调用。
- 完整 Ascend UT 在 `tests/ut/conftest.py:201` 导入阶段被
  `ModuleNotFoundError: No module named 'vllm'` 阻断，未执行测试。
- 本次修改的 Python 文件 Ruff check、语法检查及两仓 `git diff --check` 通过。
- 未运行 A3 模型、目标 CANN host callback、多 rank HCCL、真实量化 checkpoint
  对比及端到端精度测试；本次发布包含这些修复，目标验收仍待执行。

## 内网先执行的正确性门槛

性能实验前先按配套 A3 recipe，使用 V1、DP4/TP4、shared host table，比较
`lookup_backend=device/cpu`，并保留以下逐级 oracle：

1. 同一 metadata 的词表转换结果与每层 hash IDs 必须完全一致；同一 INT8
   checkpoint 的检索 rows 比较 CPU/NPU BF16 输出，明确记录相同值或误差分布。
   包括 invalid ID、image/pad、请求边界和 bucket padding。
2. whole/chunked prefill、prefix hit/miss、请求重排/移除、decode、物理 slot
   重用及 KV backing 重新绑定，均与关闭 prefix/chunking 的对应结果比较。
   若启用 speculative decoding，还要覆盖 reject 后重写 slot。
3. 连续多次 FULL replay，变更 IDs、batch/bucket 大小并插入 idle DP rank，
   确认没有 stale rows、误用他 rank 索引、event 残留和 buffer 提前覆盖。
   对照 eager，分别检查第 1/14 层消费前的数据。
4. 在基线支持范围内验证 TP 与 SP、非共享 EDP 分片；PCP 仅按支持它的
   runner/recipe 单独验收，不据源码中的 V2 分支声称 A3 V2 已支持。
   PP、DCP 等被配置检查拒绝的拓扑不纳入已支持列表。
5. rows 正确后比较固定请求的 logits/输出 token，再运行模型精度评估；
   最后才采性能。按实验计划同时保留同步前缀、检索、H2D、通信、consumer
   wait 与客户端 E2E，不能只比较 CPU kernel 和 NPU hash kernel。

部署及正确 recipe 见 [Docker 指南](engram_cpu_a3_docker_zh.md)，
完整矩阵与打点口径见 [实验计划](engram_cpu_a3_experiment_plan_zh.md)。
