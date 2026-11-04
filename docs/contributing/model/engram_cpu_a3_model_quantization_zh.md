# A3 DeepSeek-V4.1-Flash 权重与 Engram 反量化核对

核查日期：2026-11-04。范围：公开模型、实际 checkpoint header、当前集成分支的加载及消费路径。未进行 A3 整模型装载、精度或性能验收。

## 1. 可下载的 Ascend 量化候选

| ModelScope 模型 | 量化与实际 Engram 格式 | 使用边界 |
| --- | --- | --- |
| [Eco-Tech/DeepSeek-V4.1-Flash-w8a8](https://www.modelscope.cn/models/Eco-Tech/DeepSeek-V4.1-Flash-w8a8) | Ascend W8A8 Dynamic；第 1/14 层表实际为 I8 codes + F32 scales，group32 | 与官方 A3 recipe 所述格式一致；README 与实际文件存在冲突，必须固定 revision |
| [Eco-Tech/DeepSeek-V4.1-Flash-1017-w8a8](https://www.modelscope.cn/models/Eco-Tech/DeepSeek-V4.1-Flash-1017-w8a8) | Ascend W8A8；第 1/14 层表实际为 I8 codes + F8_E8M0 scales，group32 | 可作为本次实验的优先核验候选；当前 loader 保持 I8 codes，将 scales 转为运行时 FP32 |
| [chiro2001/DeepSeek-V4.1-Flash-1017-w4a8-Ascend](https://www.modelscope.cn/models/chiro2001/DeepSeek-V4.1-Flash-1017-w4a8-Ascend) | 发布者描述为 W4A8、独立 INT8 Engram，声称 A3 TP8 验证 | 依赖其自定义 vLLM/vLLM-Ascend 补丁及镜像；本次未读取其 tensor header，不能直接套入当前配套 tag |

vLLM-Ascend [v0.30.0 官方 recipe](https://docs.vllm.ai/projects/ascend/en/v0.30.0/tutorials/models/DeepSeek-V4.1-Flash.html) 的 A3 单机部署采用 W8A8、INT8 Engram、V1 runner、DP4/TP4 + EP、host shared table。页面“将发布 Eco-Tech 权重”的措辞落后于本次直接读取到的仓库状态。模型已公开不代表本分支或目标机已验收。

固定版本：

```text
Eco-Tech/DeepSeek-V4.1-Flash-w8a8
155aced390c4692698400b744546a8bc396699c6

Eco-Tech/DeepSeek-V4.1-Flash-1017-w8a8
12b85455a58c7df345c1abc43934624a76bac576
```

本次读取 config、quant_model_description、index 和 safetensors JSON header，未下载整模型。不可仅看仓库名或 `config.dtype=bfloat16` 判断表格式。详细 URL、SHA256、header 与固定 revision 复核结果保存在 [核验记录](engram_cpu_a3/model_sources_20261104.json)。header 的 SHA256 只覆盖 JSON header，不是完整权重数据校验。

两层表的实际 shape：

| Tensor | Shape | 无 1017 版 | 1017 版 |
| --- | --- | --- | --- |
| `layers.1.engram.embed.weight` | `[384006168, 256]` | I8 | I8 |
| `layers.14.engram.embed.weight` | `[384016682, 256]` | I8 | I8 |
| `layers.1.engram.embed.scale` | `[384006168, 8]` | F32 | F8_E8M0 |
| `layers.14.engram.embed.scale` | `[384016682, 8]` | F32 | F8_E8M0 |

特别注意：无 1017 版 README 说 Engram BF16、无 scale，而同次获取的 index、量化描述和 header 是 I8 + F32 scale。应以固定 revision 的实际文件为准，并保留此元数据冲突，不能据 README 假设无需反量化。1017 的 E8M0 是 scale 类型，其 codes 仍是 INT8，不能称为 MXFP8 表。

## 2. 已量化与仍需反量化并不矛盾

当前消费接口需要 BF16 rows；量化表保留在 CPU 内存，每步只恢复命中行，不展开整表。

```text
A：NPU hash → NPU UVA 读取 host INT8 codes/FP32 scales
   → NPU gather + dequant → BF16 rows → 原有通信/ready/wait → WKV/gate

B：CPU 词表转换/hash → CPU gather + dequant
   → BF16 rows H2D → 原有通信/ready/wait → WKV/gate
```

这些箭头说明反量化归属，不代替实验计划中的完整 runner、metadata、collective 和 event 时序。两种路径均应计入最终 rows 可消费之前的全部工作。

源码依据（路径均相对于各自仓库）：

1. vLLM-Ascend `vllm_ascend/models/deepseek_v41/engram/embedding.py`：`engram_storage_dtype()` 按 safetensors 实际 dtype 选择表存储，独立于主干 Linear 量化。INT8 storage 的 scale buffer 为 FP32。
2. 同文件 `_load_into_storage()`：I8 源分支直接复制 codes，并将 scale 数值复制到 FP32 buffer。因此 1017 的 E8M0 → FP32 是加载时 scale 表示转换，不是重新量化 INT8 codes，更不是整表反量化为 BF16。源 FP8 → INT8 是另一个分支，不应把这项启动转换成本套到已经 I8 的 checkpoint。
3. 同文件 `lookup()`：CPU backend 分发给 `CPUEmbeddingLookup`；device backend 的 host buffer 分发给 `gather_dequantize_host_uva()`。
4. vLLM-Ascend `vllm_ascend/ops/triton/engram_lookup.py`：UVA kernel 读取选中 codes/scales，相乘后写 BF16。反量化在 NPU 执行，host 存表不表示 CPU 反量化。
5. vLLM `csrc/engram_cpu.cpp:gather_out()`：CPU INT8 gather 按 group 乘 FP32 scale，输出 BF16。
6. vLLM-Ascend `vllm_ascend/models/deepseek_v41/engram/layer.py:AscendEngram.forward()`：`self.wkv(rows)` 消费上述结果。主干 W8A8 不意味着输入的 Engram rows 必须保持 INT8。

本机 PyTorch 2.10.0 的小型数值检查确认 E8M0 tensor `copy_` 到 FP32 执行数值转换，例如 E8M0 位模式 126/127/128 得到 0.5/1/2。该检查不代替 Docker 内实际 torch/safetensors 版本上的加载测试。

## 3. 原始 DeepSeek 权重与 Ascend 权重的区别

原始 [DeepSeek-V4.1-Flash-1017 config](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/raw/main/config.json) 已声明 FP8、UE8M0 scales 和 FP4 experts；不能称为全 BF16 未量化模型。但它不等同于 Ascend W8A8 ModelSlim checkpoint。

`--quantization ascend` 选择已有量化权重的加载/执行后端，不是在启动时自动量化任意原始模型。`vllm_ascend/quantization/configs/modelslim_config.py:maybe_update_config()` 要求 `quant_model_description.json`，缺失时明确报错。Engram loader 能转换某种表格式，也不意味着它能自动适配整个主干的 FP8/FP4 权重。

本次实验可先用固定版本的 Eco-Tech 1017 W8A8，按已有 A3 recipe 完成装载、输出与精度核验，再用相同的运行时 INT8 codes/FP32 scales 对照 A/B。无需先把整模型恢复 BF16，也无需重复离线量化已准备好的 Ascend checkpoint。

若希望 CPU 只 gather 压缩行，传 INT8 codes/scales 后由 NPU 反量化，可另外增加第三条实验路径；当前 CPU backend 输出 BF16，尚未实现该组合。它会减少 rows H2D 字节，但新增 NPU kernel、buffer 和 event 依赖，是否更快必须实测。
