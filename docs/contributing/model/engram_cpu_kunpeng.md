# Engram CPU: Kunpeng 920 / 950 offline validation

This phase prepares host kernels and reproducible validation for the internal
network. CPU producer and graph selection are described in the
[serving guide](engram_cpu_serving.md). Both repositories remain on their
paired v0.30.0 feature branches. No Kunpeng runtime or performance result has
been obtained for this patch.

## CPU dispatch and build boundary

| Target | Portable/hash path | Gather candidates |
| --- | --- | --- |
| Kunpeng 920 | ARMv8-A integer multiply-high and reciprocal modulo | NEON software FP8/INT8 decode, software BF16 rounding; portable loop |
| Kunpeng 950 | Same baseline; optional SVE2 head hash | Same NEON baseline; explicit scalable SVE2 decode |

Selection uses compiler support and OS capability bits, not CPU model strings
or a fixed SVE width. Linux baseline and NEON objects are compiled with
`-march=armv8-a`. Only the separate SVE object gets
`-march=armv8.5-a+sve2 -msve-vector-bits=scalable`. Do not add global
`-march=native` or a fixed vector width when building a shared 920/950 artifact.
The installed PyTorch itself must also support the destination CPU; isolating
these kernels cannot make an incompatible PyTorch wheel portable.

NEON needs neither hardware BF16 nor FP16 conversion. FP8 decoding handles
subnormals, signed zero and NaN; output conversion matches ATen's software BF16
round-to-nearest-even. SVE loads/stores are predicated; NEON has a bounded scalar
tail. The FP8 lookup in SVE uses byte indices in `[0,255]` into a read-only
256-entry table. The caller's output is checked for contiguity and aliasing.

Auto gather selects NEON when available, otherwise portable C++. SVE2 gather
and prefetch remain explicit candidates. Hash retains its existing explicit
`use_sve2` switch. No thread count, affinity or NUMA policy is changed inside
the operators. Cross compilation checks syntax and object generation only.

## Prepare the internal environment

Transfer the modified source files as well as the base tags: this work is not
committed, so transferring only the feature branch ref omits the implementation.
Include the new `csrc/engram_cpu*` files, Python API, tests, tools, benchmarks,
and both tracked build-file changes. Preserve the paired Ascend adapter changes.
The receipts record source hashes so internal runs can identify their source.

Provide a Python 3.10–3.12 virtual environment, compatible PyTorch, pytest,
CMake, Ninja and a C++20 compiler in the internal environment. For the paired
Ascend stack use its documented PyTorch 2.10.0 runtime; GPU validation must use
that backend's own compatible runtime. Keep the operator library matched to its
runtime PyTorch. Full Ascend serving dependencies are still required for its UT
and model tests. The validation scripts never download or install dependencies.

If dependencies are supplied by a wheelhouse, install them through uv and the
project virtual environment, for example:

```bash
uv pip install --python .venv/bin/python --offline --no-index \
  --find-links /path/to/wheelhouse torch==2.10.0 pytest cmake ninja
```

## Build and numerical acceptance

From the vLLM checkout, use a fresh receipt directory for every run:

```bash
# Kunpeng 920
.venv/bin/python tools/validate_engram_cpu.py \
  --output /path/to/results/920-host-001 --require-isa neon

# Kunpeng 950
.venv/bin/python tools/validate_engram_cpu.py \
  --output /path/to/results/950-host-001 --require-isa neon sve2
```

Each command configures only the host CMake build, builds and installs the
`engram_cpu_C` component into this checkout, checks the required ISAs, then runs
the isolated numerical suite. An empty-device setup.py wheel does not build
this component automatically. A missing compiler feature cannot silently pass
950 validation: the required-ISA check fails if SVE2 is unavailable.

The script saves each command, exit status, output, source hashes, base commit
and library hash. A failed run remains marked `failed` with its logs intact.
Record skips: on 920, SVE2 numerical cases should skip; on a 950 build requiring
SVE2, they must execute. The probe's reported vector length belongs to the
calling thread. Worker-thread vector-length variants require separate launches
under the target OS policy; the kernels do not assume all workers share a width.

After the paired runtime is installed, run the adapter test from the Ascend
checkout with that same runtime environment:

```bash
/path/to/venv/bin/python -m pytest tests/ut/models/test_engram.py \
  -k cpu_native_lookup -q -ra
```

No skipped adapter or kernel case counts as target execution. Preserve missing
dependency errors as environment failures. In the present macOS environment,
Ascend UT cannot import the full runtime because `regex` is absent.

## Controlled microbenchmarks

First pass numerical acceptance. Then bind CPUs and memory explicitly to a
chosen NUMA node and use a fresh output directory. Choose cores based on the
actual topology; the following node number is an example:

```bash
numactl --cpunodebind=0 --membind=0 .venv/bin/python \
  benchmarks/kernels/benchmark_engram_cpu.py \
  --output /path/to/results/950-micro-001 \
  --tp 8 --rank 0 --tokens 1,8,128,1024 --threads 1,2,4,8 \
  --formats mxfp8 int8 bf16 --backends scalar neon sve2 \
  --prefetch 0 2 4 8 --rows-per-head 65536 --warmup 10 --repeats 100
```

On 920 use `--backends scalar neon`. Repeat with `--tp 1`, `2`, `4`, `8` and
the ranks of interest, keeping resource limits and topology fixed. Do not
launch concurrent sweeps on overlapping CPUs. The operator itself uses ATen's
pool; the benchmark's `--threads` changes the pool only for this process.

The harness checks each output before timing, reuses output buffers, and saves
raw samples, median/p95, table/output bytes, ISA availability, calling-thread
SVE width, CPU affinity, topology and source/library hashes. Inputs are identical
across backend choices and thread candidates. Compare equal format, table size,
token count, TP rank, threads and prefetch settings. Include regressions and
tail latency before selecting a default or a batch threshold.

These timings include Python dispatcher calls. Hash includes reciprocal setup;
gather includes synchronous validation and prefetch. The benchmark operates on
one synthetic shard in one process, with warmed pages and repeated IDs. Default
tables are small; increase `--rows-per-head` within the node's memory budget
and report the recorded byte count. Neither `table_bytes` nor this warmed
microbenchmark proves realistic capacity, cold-memory latency, NUMA sharing,
multi-rank collective performance, CPU core-seconds or model throughput.

## Remaining model acceptance

The implemented CPU producer/model path still needs target verification of checkpoint rows, transfers,
TP/DP/EP/SP and legal PCP layouts, accepted-token history, buffer lifetime,
logits/generation, graph refresh and quality before performance acceptance.
The CPU bridge preserves NVIDIA's legacy E8M0 byte 0/255 interpretation and
Ascend's native interpretation; verify both with actual checkpoints/device rows.
Keep the existing device UVA backend available throughout these steps.
