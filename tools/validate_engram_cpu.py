# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline host build and numerical checks; never downloads dependencies."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def tool(name):
    sibling = Path(sys.executable).parent / name
    return str(sibling) if sibling.exists() else shutil.which(name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build/engram-cpu")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument(
        "--require-isa", nargs="+", choices=["neon", "sve2"], default=[]
    )
    args = parser.parse_args()
    if args.jobs <= 0:
        parser.error("jobs must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output.resolve()
    manifest = {"status": "running", "python": sys.version, "commands": []}

    def save():
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    def run(name, argv):
        start = time.time()
        with (output / f"{name}.txt").open("w") as log:
            result = subprocess.run(
                argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT
            )
        manifest["commands"].append(
            {
                "name": name,
                "argv": argv,
                "cwd": str(ROOT),
                "exit_code": result.returncode,
                "elapsed_s": time.time() - start,
            }
        )
        save()
        if result.returncode:
            raise RuntimeError(
                f"{name} failed ({result.returncode}); see {output / (name + '.txt')}"
            )

    save()
    try:
        cmake, ninja = tool("cmake"), tool("ninja")
        if not cmake or not ninja:
            raise RuntimeError(
                "cmake and ninja must already be installed in the environment"
            )
        build = str(args.build_dir.resolve())
        run(
            "configure",
            [
                cmake,
                "-S",
                str(ROOT),
                "-B",
                build,
                "-G",
                "Ninja",
                "-DVLLM_TARGET_DEVICE=empty",
                f"-DVLLM_PYTHON_EXECUTABLE={sys.executable}",
                f"-DCMAKE_MAKE_PROGRAM={ninja}",
                "-DCMAKE_BUILD_TYPE=Release",
                f"-DCMAKE_INSTALL_PREFIX={ROOT}",
            ],
        )
        run(
            "build",
            [cmake, "--build", build, "--target", "engram_cpu_C", "-j", str(args.jobs)],
        )
        run("install", [cmake, "--install", build, "--component", "engram_cpu_C"])
        library = ROOT / "vllm/engram_cpu_C.abi3.so"
        probe = (
            "import importlib.util,json,torch; "
            "s=importlib.util.spec_from_file_location("
            f"'engram_cpu_C',{str(library)!r}); "
            "s.loader.exec_module(importlib.util.module_from_spec(s)); "
            "c={'neon':torch.ops._engram_cpu.has_neon(),'sve2':torch.ops._engram_cpu.has_sve2(),"
            "'sve_vector_bytes':torch.ops._engram_cpu.sve_vector_bytes()}; "
            "print(json.dumps(c)); "
            f"assert all(c[k] for k in {args.require_isa!r}), "
            "'required ISA is unavailable'"
        )
        run("capabilities", [sys.executable, "-c", probe])
        run(
            "numerical",
            [
                sys.executable,
                "-m",
                "pytest",
                "--noconftest",
                "tests/kernels/test_engram_cpu.py",
                "-q",
                "-ra",
            ],
        )
        manifest["library_sha256"] = hashlib.sha256(library.read_bytes()).hexdigest()
        run("source-state", ["git", "status", "--short"])
        run("source-diff", ["git", "diff", "--no-ext-diff", "--no-color"])
        run("base-commit", ["git", "rev-parse", "HEAD"])
        manifest["source_sha256"] = {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                ROOT / "csrc" / name
                for name in (
                    "engram_cpu.cpp",
                    "engram_cpu_kernels.h",
                    "engram_cpu_neon.cpp",
                    "engram_cpu_sve.cpp",
                )
            ]
            + [ROOT / "CMakeLists.txt", ROOT / "tests/kernels/test_engram_cpu.py"]
        }
        manifest["status"] = "completed"
        print(f"Host build and numerical checks completed: {output}")
    except BaseException:
        manifest["status"] = "failed"
        manifest["failure"] = traceback.format_exc()
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
