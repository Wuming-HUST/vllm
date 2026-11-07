# A3 8 NPU/W4A8 与 Engram CPU 卸载可行性

日期：2026-11-07。属于[实验主计划](engram_cpu_a3_experiment_plan_zh.md)的资源分支分析；以下为当前配套源码、公开 recipe 和权重元数据证据，没有 A3 目标运行结果。

## 1. 结论与资源选择

- **16 个逻辑 NPU 全可用**：保持 `a3-16-w8a8`，固定 W8A8 revision、DP4/TP4 + EP、V1，沿用现有官方 A3 launch。
- **只有 8 个逻辑 NPU 可用**：选择 `a3-8-w4a8`，候选 DP1/TP8 + EP，先 V1/eager，再验证 V1 FULL_DECODE_ONLY。当前 v0.30.0 已包含所需 W4A8 MoE 路径，Engram CPU 接口与这种主干量化没有直接冲突；值得在现有配套分支开展准入验证。
- **尚不能保证**指定 A3 镜像/CANN/custom ops 能运行全部 W4A8 算子、8 NPU 容量足够、模型精度合格或 CPU 卸载更快。四项分别验证，不能由量化类型注册、能启动 HTTP 服务或 A2 文章推定。

这里 NPU 指逻辑设备。标准 A3 8 张双芯物理卡是16逻辑设备；本分支的8逻辑设备是半机资源。记录真实映射及互联，不能只按设备数量选择任意8个 ID。

## 2. 当前 v0.30.0 究竟支持什么

源码版本沿用主计划：vLLM `5b4777525342b1e74b3ecaad36481f701d928dc4`，vLLM-Ascend `bbafe101803384346e9d7d371657e62ecbf59041`。Ascend `.github/vllm-release-tag.commit` 指定 vLLM `v0.30.0`。以下量化实现与 Ascend `v0.30.0` tag 的对应文件无差异，属于已有能力，而非本次 CPU 补丁新增。

| 问题 | 源码证据与边界 |
| --- | --- |
| 有 W4A8 后端吗？ | `quantization/methods/w4a8/w4a8.py` 注册 `W4A8_DYNAMIC` / `moe`，支持 ModelSlim packed INT4 weights + dynamic INT8 activation，以及另一种 compressed-tensors 格式。普通 W4A8 与 `W4A8MXFP` 是不同 scheme，不能混用 |
| 接受任意 W4A8 权重吗？ | 不接受。当前此 scheme 只支持 **per-channel**；`group_size > 0` 初始化即拒绝。这里的 MoE `group_size=0` 与 Engram `group_size=32` 是两个独立参数 |
| 如何选择量化？ | `--quantization ascend` 解析 checkpoint 的 `quant_model_description.json`，按具体模块选择 scheme；它不会在线把 W8A8 checkpoint 变成 W4A8。顶层 `model_quant_type=W8A8_DYNAMIC` 也不能覆盖 routed experts 的逐模块 W4A8 标记 |
| 如何装载辅助量化张量？ | `configs/modelslim_config.py` 将 `w13_scale_bias/w2_scale_bias` 映射到 MoE loader 的 scale 路径；`method_adapters.py` 设定装载/分片属性。实际 packed 维度、scale_bias 形状必须核对 |
| 为什么 TP8 还必须 EP？ | W4A8 scheme 在 EP 时使用 MoE `tp_size=1`，`w2_scale_bias` 分配末维16；无EP时TP8分配末维2。候选权重头部样本是 `[5120,16]`，因此 DP1/TP8 必须保留 `--enable-expert-parallel`，不能只设置TP8 |
| 必须 V2 吗？ | 该量化 scheme 有 V1/V2 分支，未发现强制 V2 的量化限制；当前 Engram CPU 路径也未按 W8A8 类型限定。这只说明**源码允许验证 V1**，尚非本 checkpoint 整模型 V1 已通过 |
| 实际依赖哪些设备算子？ | W4A8 路径调用 `_C_ascend.grouped_matmul_swiglu_quant_v2`、`torch_npu.npu_grouped_matmul`、`npu_dynamic_quant` 等；路由配置还会选择其他 fused MoE/通信实现。必须在拟用 A3 image digest、CANN、torch_npu 和源码重编译 custom ops 上逐步运行验证，Python 类存在不证明内核可用 |

来源：[固定 W4A8 实现](https://github.com/Wuming-HUST/vllm-ascend/blob/bbafe101803384346e9d7d371657e62ecbf59041/vllm_ascend/quantization/methods/w4a8/w4a8.py)、[ModelSlim loader](https://github.com/Wuming-HUST/vllm-ascend/blob/bbafe101803384346e9d7d371657e62ecbf59041/vllm_ascend/quantization/configs/modelslim_config.py)。

版本文档须分开看：[v0.30.0 A3 recipe](https://docs.vllm.ai/projects/ascend/en/v0.30.0/tutorials/models/DeepSeek-V4.1-Flash.html)以 W8A8/V1/DP4TP4 为基线；[当前 latest](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/DeepSeek-V4.1-Flash.html)正文采用 v0.32.0 的 W4A8/V2 配置，但单机示例仍是16逻辑设备 DP4/TP4。它支持选择 W4A8 的方向，不证明本次 v0.30.0 的8 NPU方案已经验证，也不能把 latest 命令整体套进旧版本。

## 3. W4A8 候选 checkpoint 的实读证据

候选采用官方 latest 所链接的 [Eco-Tech/DeepSeek-V4.1-Flash-1017-w4a8](https://www.modelscope.cn/models/Eco-Tech/DeepSeek-V4.1-Flash-1017-w4a8)，冻结 revision **`4fbbf8c78aeed000add9765003bd842099844006`**。不是把微信作者的未具名下载来源认作这个仓库，也不与其他作者的定制 runtime/checkpoint 混用。

读取 immutable revision 的 config、quant description、index、文件元数据，以及2个主 Engram shard和主干样本的 safetensors JSON header；没有下载完整 tensor payload。URL、大小、SHA256和选定记录见 [证据 JSON](engram_cpu_a3/model_sources_w4a8_20261107.json)。

| 内容 | 已读取事实 | 仍需目标核验 |
| --- | --- | --- |
| 架构 | `DeepseekV41ForCausalLM`；40层；Engram层1/14，D=256，2/3/4-gram×8 heads | 当前配套整模型正确装载、旋转合同和输出 |
| 主干 routed experts | 逐模块 `W4A8_DYNAMIC`；主干 MoE `group_size=0`；scale/offset/scale_bias 样本F32，down scale_bias `[5120,16]` | 全层 packed INT4形状、loader 无缺失或错误分片、所选算子实跑 |
| 混合量化 | shared experts 样本为 W8A8，部分张量 FLOAT；不能把整个模型容量简单除以2 | 注意力、视觉、DSpark是否装载及实际dtype/占用 |
| Engram codes | layer1 `[384006168,256]`，layer14 `[384016682,256]`，均I8 | 文件完整SHA、各rank实际shard与codes数值 |
| Engram scales | 两层分别 `[384006168,8]`、`[384016682,8]`，均 `F8_E8M0`，group32 | loader转换后FP32的数值与BF16 row oracle |
| index | 187076 keys；记录的 `metadata.total_size=325256569544` B **未计入所有 Engram payload** | 不用此单个数字作为模型磁盘总量或HBM需求 |

表格式与 W8A8 分支相同，**不代表两种 checkpoint 的表内容、旋转矩阵或全部数值相同**。本分支 A/B 使用自身同一份加载后 codes/scales；跨权重质量比较另做。

## 4. Engram CPU 路径为什么原则上可复用

1. Ascend `model.py` 依据 Engram shard 的实际 dtype建立表，`embedding.py`把源I8 codes直接装入INT8 buffer、E8M0 scales转换到FP32；这个过程不由 routed experts 的 W4/W8决定。
2. B 使用 `lookup_backend=cpu`：`CPUHashWorkspace` 构造历史/压缩ID/hash，`CPUEmbeddingLookup` 通过 CANN callback做CPU INT8/FP32 group32 gather/dequant，输出BF16 rows，再H2D。模型主干、WKV、gate、残差和必要collective继续在NPU。
3. A 使用 `lookup_backend=device`：仍从同一host INT8/FP32表做NPU UVA gather/dequant。两种方式的row接口都是BF16；W4A8 MoE在其自身边界执行activation quant，不要求Engram传入INT4，也无需把CPU算子改成W4。
4. `platform.py:_validate_engram_config` 允许TP=1/2/4/8，要求PP=DCP=1；DP1/TP8是允许的候选。24 heads能均分为每rank3 heads。
5. `resolve_dp_shared_memory()` 在Engram DP size≤1时返回false，DP1无需共享DP表；推荐显式 `dp_shared_memory=false`。此时host hash仍可直通CPU lookup，不必hash H2D后再把IDs D2H。但metadata/source/history D2H、私有H2H IDs快照、BF16 rows H2D、TP head all-gather和event仍保留。

来源：[Ascend model](https://github.com/Wuming-HUST/vllm-ascend/blob/bbafe101803384346e9d7d371657e62ecbf59041/vllm_ascend/models/deepseek_v41/model.py)、[表加载及CPU分发](https://github.com/Wuming-HUST/vllm-ascend/blob/bbafe101803384346e9d7d371657e62ecbf59041/vllm_ascend/models/deepseek_v41/engram/embedding.py)、[CPU实现](https://github.com/Wuming-HUST/vllm/blob/5b4777525342b1e74b3ecaad36481f701d928dc4/vllm/model_executor/layers/engram_cpu.py)。

不能由此推定更快。W4A8可能缩短主干执行，也可能引入不同反量化/通信成本；第1层前可隐藏CPU工作的窗口必须重新测量。CPU metadata/source/hash目前仍是同步前缀；减少HBM占用不意味着消除这个前缀或降低CPU表读量。

每token两层共48行；INT8/FP32读量全head合计13824 B，BF16输出24576 B。TP8每rank两层逻辑表读1728 B、BF16 rows H2D3072 B（N×该值，不含padding/协议/TP通信）。相比TP4的每rank6 heads，TP8是3 heads；不能直接用旧TP4单rank结果评价此分支。

## 5. 容量只给预算，不给装载保证

两层主Engram表合计768022850行，源 codes+E8M0 scale逻辑payload约188.83 GiB；运行时INT8/FP32为 **206.00 GiB**，TP8按head实际分片约25.75 GiB/rank。DP1不复制DP副本，但不能把206 GiB再除以8当成整机需求。另计DSpark等其他表（如实际装载）、页锁定/注册、loader峰值、文件页缓存、工作区、进程和系统余量；采集RSS/PSS、cgroup、`/dev/shm`和NUMA。host RAM从≥300 GiB可用预算起核查，仍以峰值实测为准，不将推荐值作保证。

主干81个shard磁盘合计约302.94 GiB，平均分到8 rank约37.87 GiB，**只是磁盘字节的均分参照，不是HBM预测**：部分张量不装载或复制、量化辅助张量会转换、设备重排/临时buffer有峰值。所有safetensors磁盘约492.455 GiB，不能全部算进HBM，也不能假设所有Engram文件都进同一CPU表。

每rank实测并保存：

```text
HBM_peak = weights_and_runtime_scales + KV_pool + activations/workspaces
         + ACL_graph_pools + Engram_rows/staging + communication + allocator/reserve
```

逐rank核对，不能只看全机空闲之和。32K上下文、小batch先装载，分别记录装载、warmup、capture和稳态峰值；达到门槛后再扩64K/128K及并发。不由8×64GB、微信41GB/rank或配置 `max_model_len=1M`推定1M prompt已跑通。

## 6. 候选 launch 与 runner 风险

现有 `launch_a3_recipe.sh` 仍是已冻结的16设备W8A8脚本。8设备分支创建独立的 `runs/<run_id>/launch/`副本，按下表形成**资格验证配置**并保存diff、SHA、有效配置；不能仅换 `MODEL_PATH`。以下是待目标验证的参数设计，不标为官方8设备recipe。

| 参数 | 8 NPU分支初始值或要求 |
| --- | --- |
| 源码与镜像 | 两仓仍用本文配套版本、A3 image及实际digest；记录重编译custom ops和CPU native库SHA，不能覆盖成A2 rc2 wheel |
| 模型 | 上述固定W4A8 revision，`--quantization ascend`，实际quant description/header核验 |
| 设备/并行 | 明确限定分配到的8个逻辑设备，DP1/TP8 + EP，PP=DCP=PCP=1，SP先关闭 |
| runner/图 | 首轮 `VLLM_USE_V2_MODEL_RUNNER=0` + `--enforce-eager`；A/B同时使用eager。通过后移除eager，再成对验证 `FULL_DECODE_ONLY`，分别记图池峰值 |
| 调度 | `max-model-len=32768`、batch tokens4096、max seqs4；chunked prefill on、prefix off、无speculative。容量与正确性通过后再成对扩展 |
| Engram | `cpu_offload=true, dp_shared_memory=false, lookup_backend=device/cpu`；overlap显式开启；同一CPU总预算、NUMA/绑核、kernel和采样配置 |
| 其他编译/通信 | 资格验证先关闭npugraph_ex/static_kernel、flashcomm1、shared_expert_dp、dsa_cp，fused_mc2先设0；核对目标配置解析与实际算子路由。每项优化再独立成对启用，不能沿用原DP4 recipe的全部开关而不复验 |

选择V1是控制变量和优先复用现有CPU生命周期适配，**不是已证明W4A8/V1兼容**。若实际W4装载/算子只能在V2通过，先保留首次失败和版本证据，再为A/B同时切换V2；不在A用V2、B用V1比较。

V2已有源码接入，但仍有额外风险：

- `engram/model_state.py`每步从 `req_states.all_token_ids.gpu`生成lookback，再交CPU。即使hash/lookup移到CPU，前置NPU history gather仍会执行。
- 当前lookback Triton kernel未显式校验request-state index上下界，且upstream `gpu/states.py`仍为 `all_token_ids`使用UVA-backed staged tensor。这是微信文章提到的故障区域，属于目标需重点复验的风险，不能断言必然触发或已修复。
- CPU hash显式拒绝在capture中执行。`can_capture_engram_producer()`遇CPU hash返回false；V2 FULL通过`aclgraph_utils.py:run_fullgraph`先调用graph外`prepare_engram()`，配合ExternalEvent和`finish_execution` retirement。这说明已有接口路径，不证明多轮replay、异常和slot重用均正确。
- 若采用微信lookback补丁，必须先检查当前源码接口、bounds/mask顺序、staged-write语义和额外HBM容量；形成单独版本变更，再重新验证A/B。不要直接应用粘贴脚本，不把device torch gather称为CPU历史计算。

参考[微信文章](https://mp.weixin.qq.com/s/pYA8VyGoCQyXhH6znSAshw)：作者报告的是A2 8×910B3、v0.30.0rc2镜像/V2/TP8+EP及两项lookback修补；公开信息未给出可冻结的模型仓库revision、镜像digest和配套源码SHA。其1M为配置/容量描述，正文说明实际单请求长序列只测到32K。它是候选和风险线索，不能替代A3验收。

## 7. 8 NPU分支独立准入与失败处理

所有项目当前均为 **pending target run**；本次只完成上述source/header审核。

| 门槛 | 必须保存的验收证据 |
| --- | --- |
| Q0来源/格式 | 两仓SHA、image digest、8设备映射、完整模型文件SHA；所有主干量化shape和Engram shard、scale/旋转合同，无未知patch |
| Q1 W4设备算子与装载 | A/eager/TP8+EP装载无missing/unexpected/shape错误；各选中W4算子实际执行，短请求输出正常，通信参与和退出正常 |
| Q2容量 | host与逐rank HBM的load/warmup/capture/steady峰值；无swap或意外回退；确定上下文、KV和并发可行点，余量不足标infeasible |
| Q3基线模型质量 | 在冻结数据/解码参数下对比W4与参考W8或可信reference，报告logprob/任务分数和预先约定阈值；硬件拓扑不同则说明混杂因素 |
| Q4 CPU等价性 | 在同一W4 checkpoint/同拓扑内验证A/B hash整数、mask、row oracle、token/logprob；覆盖prefill续chunk、decode历史、取消/抢占/slot复用、图像哨兵，callback异常能报告并清理 |
| Q5图与稳定性 | V1 eager先通过，再A/B多bucket连续FULL replay及30–60分钟churn；若使用V2，独立重做lookback/UVA/slot/graph/异常门槛。不能把skip算通过 |
| Q6性能 | 进入主计划A1/B1/B2；重新测层1/14 reach/ready/wait、metadata/callback/H2D/TP通信、CPU/DDR成本、TTFT/TPOT/goodput和尾延迟；吞吐改善与CPU因果结论分开 |

若Q1失败，先区分checkpoint格式、算子/CANN和runner，保留原始日志，禁止隐式换模型/升级。若容量不足，先成对减小上下文/batch/KV和关闭speculative；仍不能装载则8 NPU分支标infeasible，等待16 NPU或另立经评审的升级方案。若CPU正确性或图回放失败，保留A设备基线、暂停B性能验收，修复成对分支并重新过门槛。恢复16 NPU时回到原固定W8A8 recipe，两个分支的配置和结果分别归档。

本次仅更新实验文档和来源证据，未修改量化/Engram代码、未应用微信补丁、未运行NPU，既有102 passed/31 skipped或隔离路由测试不能算本W4A8分支验收。
