# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Synchronous host microbenchmark; requires a prebuilt engram_cpu_C library.

No device, serving, distributed, or cold-memory result is implied. Numerical
checks precede timing. Output directories are exclusive and failures retained.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback
from functools import partial
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]


def positive(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def ints(value):
    try:
        return [positive(item) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected positive comma-separated integers"
        ) from exc


def command_output(argv):
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=10)
        return {
            "argv": argv,
            "exit_code": result.returncode,
            "output": result.stdout + result.stderr,
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"argv": argv, "error": str(exc)}


def sample(function, warmup, repeats):
    for _ in range(warmup):
        function()
    elapsed = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        function()
        elapsed.append(time.perf_counter_ns() - start)
    ordered = sorted(elapsed)
    return {
        "samples_ns": elapsed,
        "median_ns": statistics.median(elapsed),
        "p95_ns": ordered[(95 * len(ordered) + 99) // 100 - 1],
    }


def gather_reference(codes, scales, ids, start, count, vocab_start, block):
    columns = ids[:, start : start + count].long()
    owned = (columns >= vocab_start) & (columns < vocab_start + len(codes))
    local = (columns - vocab_start).clamp(0, len(codes) - 1).flatten()
    if codes.dtype == torch.float8_e4m3fn:
        values = (
            codes.view(torch.uint8).index_select(0, local).view(codes.dtype).float()
        )
    else:
        values = codes.index_select(0, local).float()
    if scales.numel():
        sf = scales.index_select(0, local)
        if sf.dtype == torch.uint8:
            sf = sf.view(torch.float8_e8m0fnu).float()
        values = (values.unflatten(-1, (-1, block)) * sf[..., None]).flatten(-2)
    return (
        values.bfloat16()
        .view(len(ids), count, codes.size(1))
        .masked_fill(~owned[..., None], 0)
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--library", type=Path, default=ROOT / "vllm/engram_cpu_C.abi3.so"
    )
    parser.add_argument("--tokens", type=ints, default=[1, 8, 128, 1024])
    parser.add_argument("--threads", type=ints, default=[1, 8])
    parser.add_argument("--tp", type=int, choices=[1, 2, 4, 8], default=8)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--rows-per-head", type=positive, default=4096)
    parser.add_argument("--dim", type=positive, default=256)
    parser.add_argument("--block", type=positive, default=32)
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=["mxfp8", "int8", "bf16"],
        default=["mxfp8", "int8", "bf16"],
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=["scalar", "auto", "neon", "sve2"],
        default=["scalar", "neon", "sve2"],
    )
    parser.add_argument("--prefetch", type=int, nargs="+", default=[0, 4])
    parser.add_argument("--warmup", type=positive, default=5)
    parser.add_argument("--repeats", type=positive, default=20)
    args = parser.parse_args()
    if not 0 <= args.rank < args.tp or args.dim % args.block:
        parser.error("rank must be within TP, and dim must be divisible by block")
    if min(args.prefetch) < 0 or args.rows_per_head > (2**31 - 1) // 24:
        parser.error("invalid prefetch distance or global vocabulary size")
    return args


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest_path = args.output / "manifest.json"
    manifest = {
        "status": "running",
        "scope": "single-process synthetic shard; warmed synchronous CPU calls",
        "includes_python_dispatch": True,
        "arguments": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "python": sys.version,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "topology": command_output(["lscpu"]),
        "numa": command_output(["numactl", "--show"]),
        "git": command_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
        "skipped_backends": [],
    }

    def save():
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    save()
    try:
        library = args.library.resolve()
        manifest["library_sha256"] = hashlib.sha256(library.read_bytes()).hexdigest()
        source_paths = [
            ROOT / "csrc" / name
            for name in (
                "engram_cpu.cpp",
                "engram_cpu_kernels.h",
                "engram_cpu_neon.cpp",
                "engram_cpu_sve.cpp",
            )
        ]
        manifest["source_sha256"] = {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source_paths
            if p.exists()
        }
        spec = importlib.util.spec_from_file_location("engram_cpu_C", library)
        spec.loader.exec_module(importlib.util.module_from_spec(spec))
        ops = torch.ops._engram_cpu
        available = {
            "scalar": True,
            "auto": True,
            "neon": ops.has_neon(),
            "sve2": ops.has_sve2(),
        }
        manifest["capabilities"] = available | {
            "sve_vector_bytes_calling_thread": ops.sve_vector_bytes()
        }
        manifest["skipped_backends"] = [b for b in args.backends if not available[b]]
        backends = [b for b in args.backends if available[b]]
        if not backends:
            raise RuntimeError("no requested gather backend is usable")
        save()
        generator = torch.Generator().manual_seed(731)
        count = 24 // args.tp
        start = args.rank * count
        rows = count * args.rows_per_head
        vocab_start = start * args.rows_per_head
        with (args.output / "samples.jsonl").open("w") as receipt:

            def record(row):
                receipt.write(json.dumps(row) + "\n")
                receipt.flush()

            for threads in args.threads:
                torch.set_num_threads(threads)
                generator.manual_seed(731)
                # Hash is measured separately. No device or full-pipeline timing.
                for tokens in args.tokens:
                    sources = torch.randint(
                        -1, 99092, (tokens, 4), generator=generator, dtype=torch.int32
                    )
                    mult = torch.randint(
                        -(1 << 62), (1 << 62), (2, 4), generator=generator
                    )
                    primes = torch.full((2, 24), 16000057, dtype=torch.int64)
                    offsets = (primes.cumsum(1) - primes).contiguous()
                    out_hash = torch.empty((tokens, 2, 24), dtype=torch.int32)
                    ops.hash_out(sources, mult, primes, offsets, 0, out_hash, False)
                    reference = out_hash.clone()
                    for sve in [False, True] if available["sve2"] else [False]:
                        hash_call = partial(
                            ops.hash_out,
                            sources,
                            mult,
                            primes,
                            offsets,
                            0,
                            out_hash,
                            sve,
                        )
                        hash_call()
                        torch.testing.assert_close(out_hash, reference, rtol=0, atol=0)
                        record(
                            {
                                "op": "hash",
                                "threads": threads,
                                "tokens": tokens,
                                "backend": "sve2" if sve else "scalar",
                                "includes_reciprocal_setup": True,
                                **sample(hash_call, args.warmup, args.repeats),
                            }
                        )
                for fmt in args.formats:
                    shape = (rows, args.dim)
                    if fmt == "mxfp8":
                        codes = torch.randint(
                            0, 256, shape, generator=generator, dtype=torch.uint8
                        ).view(torch.float8_e4m3fn)
                        scales = torch.randint(
                            120,
                            131,
                            (rows, args.dim // args.block),
                            generator=generator,
                            dtype=torch.uint8,
                        )
                    elif fmt == "int8":
                        codes = torch.randint(
                            -127, 128, shape, generator=generator, dtype=torch.int8
                        )
                        scales = torch.rand(
                            (rows, args.dim // args.block), generator=generator
                        )
                    else:
                        codes = torch.empty(shape, dtype=torch.bfloat16).uniform_(
                            -2, 2, generator=generator
                        )
                        scales = torch.empty(0)
                    table_bytes = (
                        codes.numel() * codes.element_size()
                        + scales.numel() * scales.element_size()
                    )
                    for tokens in args.tokens:
                        ids = (
                            torch.randint(
                                args.rows_per_head, (tokens, 24), generator=generator
                            )
                            + torch.arange(24) * args.rows_per_head
                        ).int()
                        expected = gather_reference(
                            codes, scales, ids, start, count, vocab_start, args.block
                        )
                        out = torch.empty(
                            (tokens, count, args.dim), dtype=torch.bfloat16
                        )
                        for backend in backends:
                            for prefetch in args.prefetch:
                                gather_call = partial(
                                    ops.gather_out,
                                    codes,
                                    scales,
                                    ids,
                                    start,
                                    count,
                                    vocab_start,
                                    args.block,
                                    out,
                                    backend,
                                    prefetch,
                                )
                                gather_call()
                                torch.testing.assert_close(
                                    out, expected, rtol=0, atol=0, equal_nan=True
                                )
                                record(
                                    {
                                        "op": "gather",
                                        "threads": threads,
                                        "tokens": tokens,
                                        "format": fmt,
                                        "backend": backend,
                                        "prefetch": prefetch,
                                        "tp": args.tp,
                                        "rank": args.rank,
                                        "table_bytes": table_bytes,
                                        "output_bytes": out.numel()
                                        * out.element_size(),
                                        **sample(
                                            gather_call, args.warmup, args.repeats
                                        ),
                                    }
                                )
        manifest["status"] = "completed"
    except BaseException:
        manifest["status"] = "failed"
        manifest["failure"] = traceback.format_exc()
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
