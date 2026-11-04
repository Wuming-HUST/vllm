# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare actual model generations in separate device/CPU processes.

Requires a complete target runtime and a local DeepSeek V4.1 checkpoint.
This is a deterministic generation smoke gate, not a dataset quality score.
"""

import argparse
import hashlib
import json
import math
import subprocess
import sys
import traceback
from pathlib import Path


def worker(run_dir, backend):
    from vllm import LLM, SamplingParams

    config = json.loads((run_dir / "inputs.json").read_text())
    engine = config["engine_args"]
    llm = LLM(
        model=config["model"],
        enforce_eager=config["execution"] == "eager",
        seed=17,
        engram_config={
            "cpu_offload": True,
            "lookup_backend": backend,
            "cpu_kernel": config["cpu_kernel"],
            "cpu_prefetch_distance": 0,
        },
        **engine,
    )
    try:
        requests = []
        prompts = config["prompts"]
        # Reuse capture buckets with new row ordering and then a smaller batch.
        for step_prompts in (prompts, prompts[::-1], prompts[:1]):
            requests.extend(
                llm.generate(
                    step_prompts,
                    SamplingParams(
                        temperature=0,
                        seed=17,
                        max_tokens=config["max_tokens"],
                        logprobs=5,
                    ),
                    use_tqdm=False,
                )
            )
    finally:
        llm.llm_engine.engine_core.shutdown()
    (run_dir / f"{backend}-cleanup.json").write_text(
        json.dumps({"engine_core_shutdown_completed": True}) + "\n"
    )
    rows = []
    for request in requests:
        generated = request.outputs[0]
        token_ids = list(generated.token_ids)
        rows.append(
            {
                "prompt_token_ids": list(request.prompt_token_ids),
                "token_ids": token_ids,
                "text": generated.text,
                "finish_reason": generated.finish_reason,
                "chosen_token_logprobs": [
                    float(step[token].logprob)
                    for token, step in zip(token_ids, generated.logprobs, strict=True)
                ],
            }
        )
    (run_dir / f"{backend}.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--prompts", type=Path, help="JSON list of nonempty prompt strings"
    )
    parser.add_argument("--engine-args", type=Path, help="common LLM kwargs as JSON")
    parser.add_argument("--tp", type=int, choices=[1, 2, 4, 8], default=1)
    parser.add_argument("--execution", choices=["eager", "graph"], default="graph")
    parser.add_argument(
        "--cpu-kernel", choices=["auto", "scalar", "neon", "sve2"], default="auto"
    )
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--logprob-atol", type=float, default=0.01)
    parser.add_argument("--worker", choices=["device", "cpu"], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args.output, args.worker)
        return
    if args.model is None or not args.model.is_dir():
        parser.error("--model must be an existing local checkpoint directory")
    if (
        args.max_tokens <= 0
        or not math.isfinite(args.logprob_atol)
        or args.logprob_atol < 0
    ):
        parser.error("invalid max-tokens or logprob-atol")
    engine = json.loads(args.engine_args.read_text()) if args.engine_args else {}
    reserved = {
        "model",
        "enforce_eager",
        "seed",
        "engram_config",
        "tensor_parallel_size",
    }
    if not isinstance(engine, dict) or reserved.intersection(engine):
        parser.error("engine-args must omit reserved model/backend/seed/TP settings")
    engine = {
        "tensor_parallel_size": args.tp,
        "max_model_len": 2048,
        "max_num_batched_tokens": 512,
        "max_num_seqs": 8,
        **engine,
    }
    prompts = (
        json.loads(args.prompts.read_text())
        if args.prompts
        else [
            "Explain CPU memory bandwidth in one sentence.",
            "计算 17 加 25，并给出答案。",
            "Continue the pattern: 2, 4, 8, 16,",
            "A system processes several independent requests. " * 64
            + "Summarize the system.",
        ]
    )
    if (
        not isinstance(prompts, list)
        or not prompts
        or any(not isinstance(p, str) or not p for p in prompts)
    ):
        parser.error("prompts must be a nonempty JSON list of nonempty strings")
    args.output.mkdir(parents=True, exist_ok=False)
    run_dir = args.output.resolve()
    model = args.model.resolve()
    inputs = {
        "model": str(model),
        "execution": args.execution,
        "engine_args": engine,
        "prompts": prompts,
        "cpu_kernel": args.cpu_kernel,
        "max_tokens": args.max_tokens,
    }
    (run_dir / "inputs.json").write_text(
        json.dumps(inputs, ensure_ascii=False, indent=2) + "\n"
    )
    manifest = {
        "status": "running",
        "commands": [],
        "scope": f"{args.execution} greedy generation smoke with repeated steps",
        "checkpoint_metadata_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(model.glob("*.json"))
            if p.name == "config.json" or p.name.endswith(".index.json")
        },
        "logprob_atol": args.logprob_atol,
    }

    def save():
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    save()
    try:
        for backend in ("device", "cpu"):
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                backend,
                "--output",
                str(run_dir),
            ]
            with (run_dir / f"{backend}.txt").open("w") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            manifest["commands"].append(
                {"argv": command, "exit_code": result.returncode}
            )
            save()
            if result.returncode:
                raise RuntimeError(f"{backend} model process failed; see {backend}.txt")
        baseline = json.loads((run_dir / "device.json").read_text())
        candidate = json.loads((run_dir / "cpu.json").read_text())
        differences = []
        max_error = 0.0
        if len(baseline) != len(candidate):
            differences.append("request count mismatch")
        for index, (before, after) in enumerate(zip(baseline, candidate)):
            for key in ("prompt_token_ids", "token_ids", "finish_reason"):
                if before[key] != after[key]:
                    differences.append(f"request {index}: {key} mismatch")
            a, b = before["chosen_token_logprobs"], after["chosen_token_logprobs"]
            if len(a) != len(b):
                differences.append(f"request {index}: logprob count mismatch")
            for left, right in zip(a, b):
                if not math.isfinite(left) or not math.isfinite(right):
                    differences.append(f"request {index}: nonfinite logprob")
                else:
                    max_error = max(max_error, abs(left - right))
        manifest["max_chosen_logprob_error"] = max_error
        if max_error > args.logprob_atol:
            differences.append("chosen-token logprob tolerance exceeded")
        manifest["differences"] = differences
        if differences:
            raise RuntimeError(
                "model parity failed; see manifest and per-backend results"
            )
        manifest["status"] = "completed"
        print(f"Model generation smoke parity passed: {run_dir}")
    except BaseException:
        manifest["status"] = "failed"
        manifest["failure"] = traceback.format_exc()
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
