# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run source-isolated checks only; no Ascend runtime validation is implied."""

from __future__ import annotations

import ast
import enum
import functools
import hashlib
import json
import subprocess
import sys
import types
from pathlib import Path
from unittest import mock

import pytest
import torch

V = Path(__file__).resolve().parents[3]
A = V.parent / "vllm-ascend"


class CUDAGraphMode(enum.Enum):
    NONE = 0
    FULL = 1


def method(path, cls, name, baseline=False):
    repo = A if path.is_relative_to(A) else V
    revision = (
        "59a979ef168500ba12682cdb6f669f96be450e7b"
        if repo == A
        else "4a5d453d8bc86e09fd51ab8dff4156bb772592c6"
    )
    source = (
        subprocess.check_output(
            ["git", "show", f"{revision}:{path.relative_to(repo)}"],
            cwd=repo,
            text=True,
        )
        if baseline
        else path.read_text()
    )
    root = ast.parse(source)
    owner = next(n for n in root.body if isinstance(n, ast.ClassDef) and n.name == cls)
    node = next(
        n
        for n in owner.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
    )
    node.decorator_list = []
    env = dict(
        globals(),
        IntermediateTensors=object,
        get_forward_context=lambda: None,
        partial=functools.partial,
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
            str(path),
            "exec",
            flags=__future_annotations,
        ),
        env,
    )
    return env[name], env


__future_annotations = __import__("__future__").annotations.compiler_flag
hash_path = A / "vllm_ascend/models/deepseek_v41/engram/hash_state.py"
base_path = V / "vllm/models/deepseek_v41/common/engram.py"
runner_path = A / "vllm_ascend/worker/model_runner_v1.py"
test_path = A / "tests/ut/models/test_engram.py"


def module(name, **kwargs):
    m = types.ModuleType(name)
    m.__dict__.update(kwargs)
    sys.modules[name] = m
    return m


def setup(baseline=False):
    cpu_forward, _ = method(base_path, "NgramHashState", "_forward_cpu", baseline)
    forward, _ = method(hash_path, "AscendNgramHashState", "forward", baseline)

    class AscendNgramHashState(torch.nn.Module):
        _forward_cpu = cpu_forward

    AscendNgramHashState.forward = forward
    runner_forward, runner_env = method(
        runner_path, "NPUModelRunner", "_model_forward", baseline
    )

    class NPUModelRunner:
        pass

    NPUModelRunner._model_forward = runner_forward
    # Module setattr must mutate the globals used by the extracted function.
    runner_mod = module("vllm_ascend.worker.model_runner_v1")
    runner_mod.__dict__.update(runner_env)
    runner_forward.__globals__.update(runner_mod.__dict__)

    class ProxyModule(types.ModuleType):
        def __setattr__(self, name, value):
            super().__setattr__(name, value)
            runner_forward.__globals__[name] = value

    runner_mod.__class__ = ProxyModule
    module("vllm", config=module("vllm.config", CUDAGraphMode=CUDAGraphMode))
    module(
        "vllm_ascend", worker=module("vllm_ascend.worker", model_runner_v1=runner_mod)
    )
    module(
        "vllm_ascend.models.deepseek_v41.engram.hash_state",
        AscendNgramHashState=AscendNgramHashState,
    )
    torch.npu = types.SimpleNamespace(is_current_stream_capturing=lambda: False)
    env = dict(
        globals(),
        SimpleNamespace=types.SimpleNamespace,
        NPUModelRunner=NPUModelRunner,
        Mock=mock.Mock,
    )
    nodes = [
        n
        for n in ast.parse(test_path.read_text()).body
        if isinstance(n, ast.FunctionDef)
        and n.name
        in (
            "test_cpu_hash_keeps_local_ids_on_host_but_publishes_for_device_exchange",
            "test_v1_engram_failure_retires_events_and_releases_metadata",
        )
    ]
    for n in nodes:
        n.decorator_list = []
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
            str(test_path),
            "exec",
        ),
        env,
    )
    return env


receipts = []
for baseline in [False, True]:
    env = setup(baseline)
    cases = []
    for host in [False, True]:
        try:
            env[
                "test_cpu_hash_keeps_local_ids_on_host_but_publishes_for_device_exchange"
            ](host)
            cases.append(dict(case=f"host_hash host={host}", status="passed"))
        except Exception as e:
            cases.append(
                dict(
                    case=f"host_hash host={host}",
                    status="failed",
                    error=f"{type(e).__name__}: {e}",
                )
            )
    for failure in [None, "prepare", "forward", "callback"]:
        for enpu in [False, True]:
            for capture in [False, True]:
                with pytest.MonkeyPatch.context() as mp:
                    name = (
                        f"runner failure={failure} enable_enpu={enpu} capture={capture}"
                    )
                    try:
                        env[
                            "test_v1_engram_failure_retires_events_and_releases_metadata"
                        ](mp, failure, enpu, capture)
                        cases.append(dict(case=name, status="passed"))
                    except Exception as e:
                        cases.append(
                            dict(
                                case=name,
                                status="failed",
                                error=f"{type(e).__name__}: {e}",
                            )
                        )
    receipts.append(
        dict(
            version="original-integration" if baseline else "current-code",
            passed=sum(c["status"] == "passed" for c in cases),
            failed=sum(c["status"] == "failed" for c in cases),
            cases=cases,
        )
    )
receipt = dict(
    python=sys.executable,
    torch=torch.__version__,
    source_sha256={
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [hash_path, base_path, runner_path, test_path]
    },
    runs=receipts,
    limitations=(
        "AST-isolated Python control flow and dispatch only. "
        "Uses actual source methods and test functions, real CPU torch, "
        "mocked package imports and NPU API/model/callback/event/metadata. "
        "Does not import full Ascend runtime or exercise NPU, CANN, HCCL, "
        "real graph capture/replay, native callback, or device synchronization."
    ),
)
Path(__file__).with_name("receipt.json").write_text(json.dumps(receipt, indent=2))
for r in receipts:
    print(r["version"], r["passed"], "passed", r["failed"], "failed")
if receipts[0]["failed"]:
    print(json.dumps(receipts[0]["cases"], indent=2))
    sys.exit(1)
