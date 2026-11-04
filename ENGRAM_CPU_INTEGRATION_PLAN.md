# Engram CPU integration — v0.30.0 分阶段实施

日期：2026-11-03。分支：`feat/engram-cpu-v41-integration`。

## 1. 已冻结的配套基线

| 仓库 | tag | 完整 commit |
| --- | --- | --- |
| vLLM | v0.30.0 | ced6857afa0ea7b2e3f0846a62e1394e90f15607 |
| vLLM-Ascend | v0.30.0 | c09d3e20b9c45c80b34ae3781440d90904a44023 |

两个目录都已切换到该 tag 上的新 feature 分支；原 main commit 保留在 `feat/engram-cpu-v41-integration-main-baseline`，用户原有未跟踪文件保留。未提交、未推送。

运行栈按 Ascend tag 的 `docs/source/community/versioning_policy.md` 固定：CANN 9.2.1、PyTorch 2.10.0、torch_npu 2.10.0.post7、Triton Ascend 3.6.0.post2、Python 3.10–3.12。CPU-only 构建不证明完整 NPU 依赖解析与运行可用；目标环境必须执行 pip check 和真实加载。

此前 main 分析完整保留在 [ENGRAM_CPU_MAIN_ANALYSIS.md](ENGRAM_CPU_MAIN_ANALYSIS.md)，仅用于设计背景，其行号、特性和验收条件不得套用于 tag。

## 2. tag 的计算路径与接入点

计算仍是 token/history → LUT → n-gram hash → head-sharded table retrieval → DP/TP/SP 数据恢复 → WKV → hidden-dependent gate → residual 注入。CPU backend 迁移词表转换、hash 和 Embedding 检索；投影、gate、mHC、attention、MoE 继续执行在设备上。

实际 tag 中 NVIDIA 模型引入独立的 `vllm/models/deepseek_v41/nvidia/engram.py`，负责 DP 分片、共享 host 表、prefetch；公共 layout/hash/gate 在 `common/engram.py`。不能照 main 中合并后的单文件位置修改。

Ascend 使用 `vllm_ascend/models/deepseek_v41/engram/embedding.py`，原路径为 NPU hash + NPU UVA 读取 host 表。tag 的 `platform.py` 限定 TP=1/2/4/8、PP=DCP=1，拒绝 embedding_across_dp 和 elastic EP。保持这些限制；模型类存在 PP 接口不代表 Engram 已支持 PP。

PCP 保留“global 连续请求 hash 后按 local_token_indices 选行”；TP/SP 保持 head 轴和 token 轴的不同语义。推测接受、请求重排、prefix replay 等以 runner 的权威 token history 为准，不在算子内部维护另一个可提前提交的历史。

## 3. P0：正式同步 CPU 算子和软件契约

P0 实施范围：

- `csrc/engram_cpu.cpp`：正式 `_engram_cpu` dispatcher schema，CPU-only、out 参数、dtype/shape/边界/alias 校验；ATen 线程池，不修改全局线程设置，不使用 benchmark JIT。
- `sources_out`：INT32/INT64 原始输入、ragged query offsets、positions、两类 image dead masks，输出 INT32 `[T,K]`。缺失非负位置的 V2 lookback 是 pad，而不是 DEAD；sequence 起点和 image DEAD 才截断更老来源。
- `hash_out`：配置化层数/头数/ngram，uint64 wraparound、reciprocal multiply-high 与修正；输出全局 INT32 桶地址。默认可移植标量，SVE2 显式选择，硬件不支持时明确拒绝。
- `csrc/engram_cpu_sve.cpp`：迁移 head 维 SVE2 UMULH/Barrett；单独编译，Linux AArch64 compiler feature check + HWCAP2 runtime check。加载模块不强制要求 SVE2，不能宣称目标运行已通过。
- `gather_out`：原生 E4M3+E8M0、INT8+FP32、BF16；融合 gather/dequant/BF16 写入，无完整 FP32 中间张量；支持任意合法 head_start/vocab_start、无效行及 padded heads 清零。
- CMake 在 device early return 前定义独立 host target；setup.py 加入 source-wheel 扩展，并允许预编译 wheel 含有该组件时提取。旧预编译 wheel 没有新算子时需显式增量构建，不能声称自动具备 CPU retrieval。
- `vllm/model_executor/layers/engram_cpu.py`：host API，主动加载已编译库，不运行 JIT；P1 已增加显式 serving 配置开关。
- Ascend `AscendParallelEngramEmbedding.lookup_cpu`：显式同步入口，复用现有 host Tensor、head/vocab 边界和 caller-owned 输出；不自动 D2H、H2D 或切换默认 lookup。

P0 算子不修改持久 history、不发 collective、不 record 设备 event。不得将 PyTorch Fake/Meta pass 当作设备图支持；当前没有 Meta kernel，host producer 应在 graph 外每步执行。

测试设计：I/O 契约是原始 token/量化表到可复用 host 输出；主要防止跨请求/分片混淆、history 截断错误、整数溢出、量化误解和输出越界。最便宜有效层是原生 dispatcher + Python 整数/Torch 解码 oracle。新建 `tests/kernels/test_engram_cpu.py` 是因为现有 test_engram.py 在 module import 阶段依赖 Triton/CUDA，无法作为独立 host 扩展检查入口。

## 4. 阶段和验收状态

| 阶段 | 实现/软件检查 | 目标验收 |
| --- | --- | --- |
| 基线切换 | 两仓均在 v0.30.0 tag；main ref 已保留 | 完整 NPU stack / checkpoint baseline 未运行 |
| P0 同步 CPU 算子 | host CMake build/install；本机独立数值检查；SVE2 cross compile | Linux/SVE2 实际执行、GPU wheel、Ascend UT 待验收 |
| P1 eager 集成 | 已接入两仓真实 lookup：device hash→同步 D2H→CPU gather→同步 H2D；配置保护与持久 staging 已实现，全 CPU hash 未接入 | 真实 checkpoint rows/logits/输出序列 |
| P2 并行 | 复用现有 shard/collective/PCP 选行；已扩展 device/CPU 参数化测试，目标未执行 | TP1/2/4/8、DP/shared、EP/SP、合法 PCP；vLLM PP/EDP 另验 |
| P3 生命周期 | 未实施 | cancel/reorder/slot reuse、mixed batch、prefix replay、DSpark 接受前缀 |
| P4 异步/graph | 未实施 | bounded ticket pool、逐层 ready、consumer retire；每 replay 刷新 |
| P5 性能/质量 | 未实施 | 同资源 A/B 的数据集质量、TTFT/TPOT/吞吐/CPU核秒/PSS/拷贝/通信 |

本地 TP 测试只是在一个进程中构造各 rank shard 并重组数值，不是多卡 collective 或分布式验收。小型合成表不能证明真实规模访存或服务性能。

既有目标 `agent03` 的本次 SSH 探测在 banner exchange 超时；未重启任何任务，未向远端复制或启动测试。目标状态未知。失败信息保存在本地验收回执。

## 5. 异步后续阶段与停止条件

P1 通过模型状态扩展点提交 host producer；设备生成 token 的 D2H 必须显式管理完成事件，禁止读取 async CPU placeholder。host Future 完成后 H2D 和既有 DP/TP 通信才可发布 layer-ready；consumer 结束前不得复用输出。

FULL graph 内 producer 需要排除 CPU backend，改用每次 replay 刷新的 graph 外路线；PIECEWISE 和 bounded decoder replay 单独验证。出现 hash/rows 不一致、rank collective 顺序差异、旧 buffer 被覆盖或模型质量回退时，停止推进。

数值对齐：tag 的 NVIDIA E8M0 byte 0/255 是 zero/Inf，Ascend/Torch 是 `2^-127`/NaN。P1 CPU bridge 通过 `legacy_e8m0=True` 保留 NVIDIA 语义，Ascend 保持原生语义；两类软件 oracle 已覆盖。实际 checkpoint 与设备 rows/logits 仍需对照，不能以 host 测试替代。

每阶段保留原设备 UVA 路径。P1 新增 `lookup_backend=cpu`，默认仍为 device；强制 cpu_offload 和 enforce_eager。异步/graph 不能沿用当前同步桥接而宣称已受支持。

## 6. 本地复现

所有 Python 使用 vLLM 的 `.venv/bin/python`，工具使用 uv 安装。host 验证环境：macOS arm64、Python 3.12、Torch 2.10.0；CMake 的 `empty` 仅用于配置/构建独立纯 host target，不代表 empty-target wheel 打包已经通过。

```bash
.venv/bin/cmake -S . -B build/engram-cpu -G Ninja \
  -DVLLM_TARGET_DEVICE=empty \
  -DVLLM_PYTHON_EXECUTABLE="$PWD/.venv/bin/python" \
  -DCMAKE_MAKE_PROGRAM="$PWD/.venv/bin/ninja" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PWD"
.venv/bin/cmake --build build/engram-cpu --target engram_cpu_C -j 2
.venv/bin/cmake --install build/engram-cpu --component engram_cpu_C
.venv/bin/python -m pytest --noconftest tests/kernels/test_engram_cpu.py -q
```

`--noconftest` 只隔离上层 serving/Triton/model fixture；测试直接调用真实 CMake 编译的 torch dispatcher，并使用独立 oracle。标准仓库测试与 Ascend 完整 UT 仍需配套目标环境执行，不能用该结果替代。

## 7. 本次检查回执

初始 portable P0 回执为 **21 passed**；920/950 路径补充后的当前结果见第 8 节。初始ruff、格式、syntax、diff 检查通过。SVE2 仅 cross compile 为 Linux AArch64 object，未执行。

Ascend 新增 UT 已尝试启动，但 `tests/ut/conftest.py` 导入上游运行栈时缺少 `regex`，退出码 4；本机只安装了独立算子验证依赖，不将该失败记为通过。首次独立测试中的 4 个 BF16 oracle dtype 错误也保留了原始日志，修正测试参考后全部通过。

配置、源码 SHA256、命令、退出码及失败回执见 [P0 validation](artifacts/engram-cpu/20261103-p0-local/README.md) 和 [manifest](artifacts/engram-cpu/20261103-p0-local/manifest.json)。当前证据不包含模型质量或性能结果。

## 8. 鲲鹏 920/950 专项与内网验证交付

用户明确当前没有目标环境，先完成可移植代码与内网验证入口；不再把远程节点可用性作为继续编码的前提，也不再探测旧节点。

- 920：Linux 基础/NEON TU 强制 `-march=armv8-a`；FP8 bit decode、INT8 widening、软件 BF16 RNE，不依赖 SVE、FP16 或 BF16 扩展。
- 950：单独的 scalable SVE2 hash/gather TU，编译检查及 HWCAP_SVE + HWCAP2_SVE2 运行时保护；不按商品名称/CPU ID 或固定 VL 分派。
- `gather_out`/Ascend `lookup_cpu` 可显式选 scalar/auto/neon/sve2，auto 优先可用 NEON；SVE2 gather 保持显式。prefetch_distance 默认 0，正数仅对有界合法本地行做 read hint。没有目标证据，不预设最佳线程数、NUMA 策略、预取距离或 batch 阈值。
- 当前 macOS arm64：**93 passed, 28 skipped**。跳过项是本机未编译/不支持的 SVE2 路径；NEON 已实际执行，覆盖 FP8 全编码、E8M0 特殊值、RNE ties/NaN/signed-zero、非整向量尾部、padded heads 和 TP 分片重组。不能据此宣称 920/950 或 NPU 验收通过。
- 内网工具：`tools/validate_engram_cpu.py`（无下载的 build/install/required-ISA/numerical gate）和 `benchmarks/kernels/benchmark_engram_cpu.py`（先逐项数值检查，再记录原始 CPU 微基准样本）。保存失败、源码/二进制 hash、命令/退出码和资源配置；不覆盖旧目录。
- 本机已跑通这两个入口；微基准仅用极小 smoke 配置检查脚本，不报告其速度或推导鲲鹏收益。NEON ARMv8-A 与 scalable SVE2 raw kernels 已分别交叉编译为 Linux AArch64 object；没有执行 Linux/SVE2。

内网操作步骤：[Kunpeng guide](docs/contributing/model/engram_cpu_kunpeng.md)。本轮回执：[Kunpeng offline preparation](artifacts/engram-cpu/20261103-kunpeng-offline/README.md)。

以上第 8 节是 P0 专项历史回执；P1 eager 模型接入已继续实施，当前状态见第 9 节。collective/history/graph/checkpoint/performance 的目标验收仍未完成。

## 9. P1 eager 实际模型接入（历史回执）

- 两仓的模型构造均传递 CPU backend 配置到实际 embedding.lookup；host IDs/rows 按高水位复用，锁保护并发；阻塞 D2H/H2D 确保 host 读与复用安全。此阶段不承诺计算通信重叠收益。
- NVIDIA 复用 staged_rows 与 DP row exchange；Ascend 复用 PCP local_token_indices、DP 行交换、TP head gather 和 ready events；不改变已有并行限制。
- 新增 config/lookup/DP 分布式测试参数，以及 Ascend loader 分片测试；这些设备测试已写好，但本机完整 serving 依赖不足且无 GPU/NPU，尚未执行通过。
- `tools/validate_engram_cpu_model.py` 已准备真实 checkpoint 的双进程 device/CPU greedy token 与 logprob 对照，保留失败与 shutdown 回执。该工具目前仅完成语法/入口检查，未跑模型。
- 独立 host + staging bridge 检查：97 passed、28 skipped；包含真实扩展、buffer 增长/复用、strided IDs、并发所有权及两类 E8M0 语义。所有 skip 是 SVE2 不可用。
- CPU source/hash 仍是独立算子，尚未进入 runner；异步 producer、graph replay 和生产性能/质量尚未完成。

启动、回滚和内网验收：[eager serving guide](docs/contributing/model/engram_cpu_serving.md)。本轮独立回执：[P1 receipt](artifacts/engram-cpu/20261103-eager-integration/README.md)。

## 10. CPU 全链路与 graph/流水接入（当前）

用户已确认：词表转换、hash、Embedding 检索均在 CPU 上执行。第 8/9 节的 eager-only 限制与计数保留为历史阶段记录，以本节为当前状态。

| 范围 | 当前实现 | 验收边界 |
| --- | --- | --- |
| CPU conversion/hash | CPUHashWorkspace 已接入 common/Ascend NgramHashState；容量 padding 清空；V1 CPU slot cache 重写与 KV rebind reset；V2 使用 runner 权威 lookback | host oracle 已查；真实模型历史/推测/取消等目标待验 |
| 异步 Embedding | 原生 C++ stream host callback：D2H → CPU gather/dequant → H2D；无 Python/GIL 回调；错误在 worker 收尾抛出 | host payload/error gate 已查；CANN/CUDA 实际队列待验 |
| 生命周期 | 每表单个高水位 pinned staging、callback ticket、H2D completion lease；复用/增长前 retire；异常路径保留已提交 copy 的 lease | 跨步 device consumer 与失败恢复需目标压力验证 |
| NVIDIA graph | V1 wrapper 前刷新；V2 FULL 显式 replay 前刷新；固定输入/输出；逐层 external CUDA event；breakable 保留 | FULL/piecewise 多卡实际 capture/replay 待验 |
| Ascend graph | CPU backend 排除 graph 内 producer，选择 tag 已有 graph 外 producer；保留 ExternalEvent wait/reset 与逐层 ready | 原有 mock 路由测试已补；NPU 真机待验 |
| 并行 | 复用原有 PCP 选行、DP/shared/idle 参与、TP head gather、SP 切分与 EP 组；不扩展 tag 的 PP/DCP 等限制 | TP1/2/4/8 和合法 DP/EP/SP/PCP 组合均需目标验收 |
| 模型质量/性能 | 已准备 eager/graph 双进程 checkpoint 比较与重复 batch 入口 | 未执行；没有质量/端到端收益结论 |

CPU metadata D2H 和 conversion/hash 仍是模型下发前的同步前缀；主要 Embedding 检索已改为原生回调排队，可与后续 device graph 执行并发。是否真正被流水完全掩盖、是否更快，必须由 profiler 和同资源 A/B 验收。CPU 工作不在 device graph 内捕获，不能把“graph 外每轮刷新”称作“CPU 被捕获”。

当前不再强制 enforce_eager；默认仍为 device，切回 lookup_backend=device 即回滚。CANN/CUDA host callback 不可用时明确失败，不静默换路径。paired tag、未提交/未推送和无目标环境边界保持。

完整协议与内网命令：[serving guide](docs/contributing/model/engram_cpu_serving.md)。本轮独立回执：[CPU pipeline/graph](artifacts/engram-cpu/20261103-cpu-pipeline-graph/README.md)。

本轮本机最终独立检查：**102 passed、30 skipped**（SVE2 28、CUDA 1、NPU 1）。原生构建、格式/语法与 diff 检查通过；完整仓库测试分别在 tblib/regex fixture 依赖导入处失败，目标模型/并行/图与性能均未验收。具体失败与命令见第 10 节回执。
