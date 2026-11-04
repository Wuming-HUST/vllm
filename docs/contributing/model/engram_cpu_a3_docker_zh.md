# A3 Docker 部署：DeepSeek-V4.1-Flash Engram CPU 与 NPU-UVA 对照

日期：2026-11-04。部署流程来自配套 v0.30.0 源码 recipe；CPU 新路径仍需内网 A3 验收。

已公开模型的固定 revision、实际 Engram dtype 和反量化位置，见 [权重量化核对](engram_cpu_a3_model_quantization_zh.md)。下载时按该记录固定版本，不根据仓库名或 README 单独推断表格式。

本实验固定 `Eco-Tech/DeepSeek-V4.1-Flash-1017-w8a8`，revision `12b85455a58c7df345c1abc43934624a76bac576`；先采NPU/UVA基线，再做CPU替换及E2E。源码recipe参数保留，模型目录指向该固定权重；详细步骤见 [实验计划](engram_cpu_a3_experiment_plan_zh.md)。

## 1. Recipe 来源及本次修正

以 vLLM-Ascend 提交 `59a979ef168500ba12682cdb6f669f96be450e7b` 的
`docs/source/tutorials/models/DeepSeek-V4.1-Flash.md` 为固定来源：

- §4.1 A3 Docker：16 个逻辑设备、host IPC、driver/device 挂载。
- §5.1 A3 单机 P/D 合设：DP4/TP4、EP、V1 runner、W8A8、INT8 Engram、shared host table、async scheduling、FULL_DECODE_ONLY。
- §5.2 的 DP4/TP4 Prefill + DP8/TP2 Decode 是双机 PD，不能拿来作为只有一台机器的单机启动命令。

官方版本页面：[v0.30.0 DeepSeek-V4.1-Flash](https://docs.vllm.ai/projects/ascend/en/v0.30.0/tutorials/models/DeepSeek-V4.1-Flash.html)。
本地固定源码：[recipe](https://github.com/Wuming-HUST/vllm-ascend/blob/59a979ef168500ba12682cdb6f669f96be450e7b/docs/source/tutorials/models/DeepSeek-V4.1-Flash.md)。

已删除实验计划中自行拼装的 TP8 启动示例，改用
[launch_a3_recipe.sh](engram_cpu_a3/launch_a3_recipe.sh)。脚本从上述 §5.1 A3 block 提取，保留 topology、量化、tokenizer、通信和编译参数；只新增 backend/kernel 选择、显式 Engram overlap 及指定虚拟环境入口。

源码 recipe 给出的 `max-model-len=150000`、`max-num-batched-tokens=8192`、`max-num-seqs=32` 是初始配置；实验中若改这些值，A/B共同修改，记录为实验因素。A3 recipe没有设置 DSpark speculative-config，因此初始单机对照也不额外加入。

## 2. 哪些传输必须，哪些可以删除

| 开销 | 当前处理 | 必要性和后续方向 |
| --- | --- | --- |
| runner metadata/IDs/history D2H | 仍保留 | 当前接口传入设备张量。Prefill可进一步接入runner的host原件；Decode需等真实采样/接受结果可用。须同时维护重排、取消、slot复用、spec accepted-prefix，不能直接删复制 |
| CPU hash H2D再lookup IDs D2H | 本次在Ascend无每步hash交换的路径消除 | `dp_shared_memory=true`或Engram DP size≤1时，CPU hashes直接交给CPU lookup；TP仍可在rows阶段通信 |
| 需要设备hash collective的布局 | 保留hash H2D和collective后的IDs D2H | 本次不改通信ownership。以后可研究CPU通信或host replica选择；必须单独验收idle rank和通信序 |
| pinned IDs staging | 保留H2H快照 | 防止CPU workspace被下一步覆盖时callback仍读取旧指针；它不是D2H |
| BF16 rows H2D | 保留 | WKV/gate仍在NPU，需要交付结果。若改为搬INT8再NPU反量化，是另一实验方案 |
| TP等rows collective、ready、wait/reset | 保留 | 数据/通信真正完成前消费者不能运行；删除会破坏正确性 |

本次改动没有实现“整个CPU路径零D2H”，也没有实现runner host输入直通。Decode在设备采样时，总有某种方式将已接受token信息交给CPU；目标是复用已有同步/host结果，而非重复搬运，且不提前读取未接受draft。

两种已实现路径：

```text
A: runner → NPU source/hash → NPU UVA gather/dequant(host表)
   → TP/必要rows通信 → layer-ready → NPU消费
B（本次无hash collective的快速路径）:
   runner metadata D2H → CPU source/hash → H2H IDs快照
   → CANN callback CPU gather/dequant → BF16 H2D
   → TP/必要rows通信 → layer-ready → NPU消费
```

预期收益是减少往返和相关同步；是否实际持平或更优仍由完整step/E2E测量判断。

## 3. 宿主机准备

推荐基于官方 A3 镜像做源码覆盖安装，保留其 CANN/torch_npu/自定义算子配套；不要从普通CPU Python镜像开始拼依赖。

1. 确认 `npu-smi info` 正常、主机驱动/固件与镜像兼容、可见16逻辑设备。记录CPU型号、NUMA、RAM和host `/dev/shm`容量；`--ipc=host`用的是宿主IPC空间，不会凭空扩容。
2. 准备 `quay.io/ascend/vllm-ascend:v0.30.0-a3` 对应内网镜像，记录实际digest。若内部已验证镜像有附加patch，保存版本清单，以其配套为准，不能只按tag断言兼容。
3. 两仓完整源码置于宿主目录，例如 `/data/engram-src/vllm`、`/data/engram-src/vllm-ascend`；包含本次修改和新增文件。使用实验计划列出的两仓分支与配套代码版本；旧集成提交不包含本次优化/文档。
4. 模型目录例 `/data/weights/DeepSeek-V4.1-Flash-1017-w8a8`；不要把macOS构建的 `engram_cpu_C.abi3.so` 当成Linux库使用。源码挂载可读写，模型可只读。
5. 准备内网wheelhouse和构建工具：uv、GCC≥11.3/C++20、CMake≥3.26.1、Ninja、pytest、setuptools等。保留镜像已装torch/torch_npu/CANN，不用公共CUDA依赖清单覆盖它们。

## 4. 启动容器

以下设备和driver挂载取自源码recipe §4.1 A3，额外加入源码、结果、wheelhouse挂载。宿主路径如不同按实际修改。内网预先 `docker load` 后无需pull。

```bash
export IMAGE=quay.io/ascend/vllm-ascend:v0.30.0-a3
export MODEL_ROOT=/data/weights
export SOURCE_ROOT=/data/engram-src
export RESULTS_ROOT=/data/engram-results
export WHEELHOUSE=/data/wheelhouse
mkdir -p "$RESULTS_ROOT"

docker run --rm -it --name engram-a3 \
  --net=host --ipc=host --privileged=true \
  --device /dev/davinci0 --device /dev/davinci1 \
  --device /dev/davinci2 --device /dev/davinci3 \
  --device /dev/davinci4 --device /dev/davinci5 \
  --device /dev/davinci6 --device /dev/davinci7 \
  --device /dev/davinci8 --device /dev/davinci9 \
  --device /dev/davinci10 --device /dev/davinci11 \
  --device /dev/davinci12 --device /dev/davinci13 \
  --device /dev/davinci14 --device /dev/davinci15 \
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /etc/hccn.conf:/etc/hccn.conf \
  -v "$MODEL_ROOT:$MODEL_ROOT:ro" \
  -v "$SOURCE_ROOT/vllm:/workspace/vllm" \
  -v "$SOURCE_ROOT/vllm-ascend:/workspace/vllm-ascend" \
  -v "$RESULTS_ROOT:/results" -v "$WHEELHOUSE:/wheelhouse:ro" \
  "$IMAGE" bash
```

容器内后续命令都使用同一个虚拟环境。镜像的 `--rm` 会删除容器内未挂载的环境，源码/结果不受影响；验收后的环境可制作派生镜像，或把虚拟环境也挂载持久目录。不同CPU/torch镜像不能共用同一native build目录。

## 5. 容器内安装两仓与编译CPU算子

**需要编译。**CPU后端依赖 `vllm/engram_cpu_C.abi3.so`；只切Python分支、只安装vllm-ascend或使用stock镜像，都不足以启用它。

### 5.1 创建继承镜像依赖的虚拟环境

```bash
set -euo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
uv venv --system-site-packages --python "$(command -v python3)" /workspace/.venv
export PATH="/workspace/.venv/bin:$PATH"
export ENGRAM_PYTHON=/workspace/.venv/bin/python

"$ENGRAM_PYTHON" - <<'PY'
import sys, torch, torch_npu
print('python', sys.executable)
print('torch', torch.__version__, 'torch_npu', torch_npu.__version__)
print('device count', torch.npu.device_count())
assert torch.npu.is_available()
PY
```

如 `set_env.sh` 路径不同，以镜像真实安装路径为准。先保存包清单及镜像已装两仓的版本/路径。`system-site-packages`用于复用已验证加速栈，不是允许混用任意torch版本。

构建辅助包若缺失，通过内网wheelhouse补齐；示例不包含torch/torch_npu，以免替换镜像配套：

```bash
uv pip install --python "$ENGRAM_PYTHON" --offline --no-index \
  --find-links /wheelhouse \
  'cmake>=3.26.1' ninja packaging 'setuptools>=77.0.3,<81' \
  'setuptools-scm>=8' setuptools-rust wheel jinja2 pytest
```

两仓构建还需各自pyproject里其余已安装依赖、CANN开发文件和系统编译器；缺包时按错误在wheelhouse补齐相容版本，不执行会改动整个runtime的无约束upgrade。vLLM通用build requirements与Ascend适配torch版本可能不同，采用已验证Ascend配套并关闭build isolation。

### 5.2 源码覆盖安装

```bash
cd /workspace/vllm
VLLM_TARGET_DEVICE=empty uv pip install --python "$ENGRAM_PYTHON" \
  --offline --no-index --no-build-isolation --no-deps -e .

cd /workspace/vllm-ascend
SOC_VERSION=ascend910_9391 COMPILE_CUSTOM_KERNELS=1 \
  uv pip install --python "$ENGRAM_PYTHON" \
  --offline --no-index --no-build-isolation --no-deps -e .
```

`empty`是vLLM上游在Ascend环境中的构建目标，不是将推理改成CPU backend；加速卡由Ascend插件提供。Ascend的 `COMPILE_CUSTOM_KERNELS=1` 重建其NPU扩展。不要以 `COMPILE_CUSTOM_KERNELS=0` 装载不匹配的新Python/旧NPU扩展；若复用镜像已有扩展，需要另外证明同版本ABI和实际导出接口一致。

本次CPU优化是Python路由改动，没有新增C++ ABI，但CPU库仍必须按容器内实际torch构建。若后续重装torch/torch_npu，清理独立build目录后重新编译和验收。

### 5.3 单独构建、安装并检查CPU扩展

```bash
cd /workspace/vllm
"$ENGRAM_PYTHON" tools/validate_engram_cpu.py \
  --build-dir /workspace/vllm/build/engram-cpu-a3 \
  --jobs 4 --require-isa neon --output /results/build-a3-neon-001
```

脚本实际执行：CMake `VLLM_TARGET_DEVICE=empty`、target `engram_cpu_C`、component install到当前源码的`vllm/`，随后能力探测与数值测试。receipt目录必须全新；失败看configure/build/install日志和退出码，不跳过继续启动。

若host确实支持SVE2，再单独加 `--require-isa neon sve2`；不要根据“A3”名称假设CPU有SVE2。NEON是920/950均可使用的baseline候选。编译器支持SVE2不等于运行CPU可执行SVE2。

### 5.4 确认实际import及callback可用

```bash
cd /workspace/vllm
"$ENGRAM_PYTHON" - <<'PY'
from pathlib import Path
import torch, torch_npu, vllm, vllm_ascend
from vllm.model_executor.layers.engram_cpu import capabilities
print('vllm', vllm.__file__)
print('ascend', vllm_ascend.__file__)
assert Path(vllm.__file__).resolve().is_relative_to('/workspace/vllm')
assert Path(vllm_ascend.__file__).resolve().is_relative_to('/workspace/vllm-ascend')
print('CPU ISA', capabilities())
assert torch.ops._engram_cpu.callback_available('npu')
print('NPU host callback symbol available; runtime graph test still required')
PY

"$ENGRAM_PYTHON" -m pytest --noconftest tests/kernels/test_engram_cpu.py \
  -k npu_callback -v -ra
```

`callback_available`只是动态符号探测，不是运行验收。新增host IDs/device IDs两种callback+ExternalEvent graph用例均应实际运行通过；如果skip，先修环境，不算成功。检查 `.so` 的Linux架构/torch ABI及 `ldd`；不在容器里使用源目录遗留的macOS `.so`。

Ascend侧再用同一个解释器运行：

```bash
cd /workspace/vllm-ascend
"$ENGRAM_PYTHON" -m pytest tests/ut/models/test_engram.py \
  tests/ut/models/test_engram_multistream.py \
  tests/ut/models/test_engram_v2_model_state.py -q -ra
```

UT的V2覆盖属于源码契约检查，A3正式recipe仍是V1；不能因此切换A3到V2。

## 6. 使用固定recipe分别启动A/B

两实例顺序执行，先确认前一个进程组/设备已退出，再启动下一个。完整启动命令见提取脚本。

```bash
export MODEL_PATH=/data/weights/DeepSeek-V4.1-Flash-1017-w8a8
export ENGRAM_PYTHON=/workspace/.venv/bin/python
export OMP_NUM_THREADS=4
export CPU_KERNEL=auto
cd /workspace/vllm
set -o pipefail
bash docs/contributing/model/engram_cpu_a3/launch_a3_recipe.sh device \
  2>&1 | tee /results/A-device-startup.log
```

完成A并退出后，B使用同一环境，仅替换backend：

```bash
bash docs/contributing/model/engram_cpu_a3/launch_a3_recipe.sh cpu \
  2>&1 | tee /results/B-cpu-startup.log
```

OMP=4是实验线程候选，不是官方recipe承诺的最佳值；按核预算比较1/2/4/8等。两边均显式启用Engram overlap，并保留原recipe中的其余additional-config。DP shared table路径应使用本次host-hash直通；用trace确认无对应hash H2D/lookup IDs D2H，不仅看配置值。

另一个容器终端使用 `docker exec -it engram-a3 bash`；显式使用同一个 `/workspace/.venv/bin/python`，设置相同CANN环境。先检查 `/health`、模型列表和生成；完整A/B token/logprob/graph正确性按实验计划G1执行，再跑性能。健康接口成功不是模型/Engram正确性的证明。

## 7. 常见失败定位及持久化

| 现象 | 优先检查 |
| --- | --- |
| CPU operators unavailable | 两仓import路径、目标.so是否真正安装、构建日志、torch ABI、Linux架构 |
| host callback unavailable | 容器CANN路径/实际版本和已加载runtime库；符号存在还须跑callback测试 |
| 原来的代码仍执行 | editable install实际路径、同名site-packages覆盖、启动解释器、进程是否重启 |
| head不能均匀分片/内存不足 | 正式DP4/TP4 + dp_shared_memory=true、共享IPC、checkpoint完整性和host RAM |
| source/install构建依赖冲突 | 继承镜像的torch/torch_npu、no-build-isolation、build辅助包版本，不用CUDA requirements替换Ascend依赖 |
| 旧rows/图死锁 | 停止性能验收，保留trace和event generation，确认host IDs快照、copy retire、ready/wait/reset |
| CPU无加速 | 首先核对metadata同步前缀、callback有效线程、实际NUMA及TP/DP核预算；按实验计划分项测量 |

部署通过后保存image digest、包清单、源码commit+dirty patch、native `.so` hash、实际启动和有效配置、G1日志。离线交付可用 `git bundle` 带已提交历史，未提交变更另带patch和新增文件；导出完整源码归档时排除旧build/.venv/平台不符.so。结果目录挂载宿主并保留失败记录。

实验计划：[A3实验计划](engram_cpu_a3_experiment_plan_zh.md)。本地完成的是源码、文档和非NPU检查，未在本机运行Docker/A3模型；内网需完成上述目标验收。
