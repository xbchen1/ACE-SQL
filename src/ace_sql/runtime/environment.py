#!/usr/bin/env python3
"""Inspect an H20 slime environment without installing or changing packages.

A successful check only establishes the recorded preflight checks; a complete
rollout, weight synchronization, and optimizer-step smoke run is still required.
"""

from __future__ import annotations

import argparse
import ctypes
import importlib
import importlib.metadata
import importlib.util
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


SLIME_COMMIT = "6195417d45f272b72d04f619ea334613135d1c1f"
SLIME_VERSION = "0.2.3"
UPSTREAM_VERSIONS = {
    "sglang": "0.5.9",
    "torch": "2.9.1",
    "sgl-kernel": "0.3.21",
    "flashinfer-python": "0.6.3",
    "transformers": "4.57.1",
}
PACKAGES = (
    "ace-sql", "slime", "torch", "sglang", "sglang-router", "sgl-kernel",
    "flashinfer-python", "transformers", "ray", "megatron-core",
    "transformer-engine", "flash-attn", "torch-memory-saver", "triton",
    "nvidia-cuda-runtime-cu12", "nvidia-cuda-nvrtc-cu12",
    "nvidia-cudnn-cu12", "nvidia-nccl-cu12",
)

# Deliberate overrides in the immutable upstream v0.2.3 image Docker history.
# Do not suppress any other pip-check conflict.
UPSTREAM_OVERRIDES = {
    "nvidia-cudnn-cu12": ("9.10.2.21", "9.16.0.29"),
    "nvidia-nccl-cu12": ("2.27.5", "2.28.3"),
}


def pip_check_conflicts(inventory: dict) -> tuple[list[str], list[str]]:
    result = inventory["pip_check"]
    if result["returncode"] == 0:
        return [], []
    known, unknown = [], []
    lines = (result["stdout"] or result["stderr"]).splitlines()
    for line in lines:
        allowed = False
        if inventory.get("torch_cuda") == "12.9" and line.startswith("torch 2.9.1"):
            for package, (required, overridden) in UPSTREAM_OVERRIDES.items():
                if (f"has requirement {package}=={required}" in line
                        and f"but you have {package} {overridden}" in line
                        and inventory["packages"].get(package) == overridden):
                    allowed = True
        (known if allowed else unknown).append(line)
    if not lines:
        unknown.append(f"pip check exited {result['returncode']} without output")
    return known, unknown


def command(argv: list[str], timeout: int = 30) -> dict:
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
        return {"returncode": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"returncode": -1, "stdout": "", "stderr": str(exc)}


def collect_environment(slime_dir: Path | None, gpu_smoke: bool) -> dict:
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None

    inventory = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.system(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "packages": versions,
        "nvidia_smi": command([
            "nvidia-smi", "--query-gpu=name,driver_version,compute_cap", "--format=csv,noheader"
        ]),
        "nvcc": command(["nvcc", "--version"]),
        "pip_check": command([sys.executable, "-m", "pip", "check"], timeout=60),
        "slime_commit": None,
        "slime_import_root": None,
        "slime_checkout_matches_import": False,
        "imports": {},
        "gpus": [],
        "torch_cuda": None,
        "cuda_driver_api": None,
        "gpu_smoke_requested": gpu_smoke,
    }
    try:
        driver = ctypes.CDLL("libcuda.so.1")
        version = ctypes.c_int()
        result = driver.cuDriverGetVersion(ctypes.byref(version))
        if result != 0:
            raise RuntimeError(f"cuDriverGetVersion returned CUDA error {result}")
        inventory["cuda_driver_api"] = f"{version.value // 1000}.{(version.value % 1000) // 10}"
    except (OSError, AttributeError, RuntimeError) as exc:
        inventory["cuda_driver_error"] = str(exc)
    try:
        spec = importlib.util.find_spec("slime")
        if spec is not None and spec.origin:
            imported_root = Path(spec.origin).resolve().parent.parent
            inventory["slime_import_root"] = str(imported_root)
            root = slime_dir.resolve() if slime_dir is not None else imported_root
            inventory["slime_checkout_root"] = str(root)
            inventory["slime_checkout_matches_import"] = root == imported_root
            git_head = command(["git", "-C", str(root), "rev-parse", "HEAD"])
            if git_head["returncode"] == 0:
                inventory["slime_commit"] = git_head["stdout"]
    except (ImportError, ValueError, OSError) as exc:
        inventory["slime_import_error"] = str(exc)

    for module in (
        "torch", "sglang.srt.server_args", "sgl_kernel", "flashinfer",
        "megatron.core", "transformer_engine.pytorch",
        "slime.utils.types", "slime.rollout.base_types",
    ):
        try:
            importlib.import_module(module)
            inventory["imports"][module] = {"ok": True}
        except Exception as exc:
            inventory["imports"][module] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    if inventory["imports"]["torch"]["ok"]:
        import torch

        inventory["torch_cuda"] = torch.version.cuda
        try:
            inventory["torch_cuda_available"] = torch.cuda.is_available()
            for index in range(torch.cuda.device_count()):
                gpu = {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "capability": list(torch.cuda.get_device_capability(index)),
                }
                if gpu_smoke:
                    try:
                        with torch.cuda.device(index):
                            a = torch.ones((16, 16), device=f"cuda:{index}", dtype=torch.bfloat16)
                            result = a @ a
                            torch.cuda.synchronize(index)
                            if not torch.all(result == 16).item():
                                raise RuntimeError("BF16 matrix multiplication returned incorrect values")
                            del a, result
                        gpu["bf16_smoke"] = {"ok": True}
                    except Exception as exc:
                        gpu["bf16_smoke"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                inventory["gpus"].append(gpu)
        except Exception as exc:
            inventory["torch_cuda_error"] = f"{type(exc).__name__}: {exc}"
    return inventory


def validate_environment(inventory: dict, expected_cuda: str = "12.9", expected_gpus: int = 8) -> list[dict]:
    checks = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    check("linux_x86_64", inventory["platform"] == "Linux" and inventory["machine"] == "x86_64",
          f"found {inventory['platform']} {inventory['machine']}")
    python_parts = tuple(int(x) for x in inventory["python"].split(".")[:2])
    check("python", (3, 10) <= python_parts <= (3, 12), f"expected Python 3.10–3.12; found {inventory['python']}")
    check("slime_revision", inventory["slime_commit"] == SLIME_COMMIT,
          f"expected {SLIME_COMMIT}; found {inventory['slime_commit']}")
    check("slime_import_root", inventory["slime_checkout_matches_import"],
          "the imported slime package must resolve to the checked checkout")
    check("slime_version", inventory["packages"].get("slime") == SLIME_VERSION,
          f"expected slime {SLIME_VERSION}; found {inventory['packages'].get('slime')}")
    for package, expected in UPSTREAM_VERSIONS.items():
        actual = inventory["packages"].get(package)
        # Local version suffixes identify builds (e.g. +cu124); CUDA is checked separately.
        check(f"version:{package}", actual is not None and actual.split("+", 1)[0] == expected,
              f"expected upstream API version {expected}; found {actual}")
    check("torch_cuda", inventory["torch_cuda"] == expected_cuda,
          f"expected torch.version.cuda == {expected_cuda}; found {inventory['torch_cuda']}. "
          "nvidia-smi's CUDA version does not establish the installed runtime version.")
    driver_version = inventory.get("cuda_driver_api")
    minimum = tuple(int(part) for part in expected_cuda.split("."))
    driver_ok = driver_version is not None and tuple(int(part) for part in driver_version.split(".")) >= minimum
    check("cuda_driver_api", driver_ok,
          f"require driver API >= {expected_cuda}; found {driver_version}. "
          "The pinned image uses CUDA 12.9 JIT kernels; an older driver needs separate compatibility validation.")
    nvcc_match = re.search(r"release\s+(\d+\.\d+)", inventory["nvcc"]["stdout"])
    nvcc_version = nvcc_match.group(1) if nvcc_match else None
    check("cuda_toolkit", nvcc_version == expected_cuda,
          f"expected nvcc {expected_cuda} for source/JIT kernels; found {nvcc_version}")
    for package in ("nvidia-cuda-runtime-cu12", "nvidia-cuda-nvrtc-cu12"):
        version = inventory["packages"].get(package)
        if version is not None:
            check(f"runtime:{package}", version.startswith(expected_cuda + "."),
                  f"expected {expected_cuda}.x; found {version}")
    check("cuda_available", inventory.get("torch_cuda_available", False),
          inventory.get("torch_cuda_error", "torch.cuda.is_available() must be true"))
    check("gpu_count", len(inventory["gpus"]) == expected_gpus,
          f"expected {expected_gpus} visible GPUs; found {len(inventory['gpus'])}")
    for gpu in inventory["gpus"]:
        check(f"gpu:{gpu['index']}", "H20" in gpu["name"] and gpu["capability"] == [9, 0],
              f"expected NVIDIA H20, SM90; found {gpu['name']}, capability {gpu['capability']}")
        if inventory["gpu_smoke_requested"]:
            smoke = gpu.get("bf16_smoke", {})
            check(f"bf16:{gpu['index']}", smoke.get("ok", False), smoke.get("error", "BF16 matrix multiplication"))
    known, unknown = pip_check_conflicts(inventory)
    check("pip_check", not unknown,
          "\n".join(unknown) if unknown else (
              "Only upstream-pinned cuDNN/NCCL overrides: " + " | ".join(known)
              if known else "No broken requirements found"))
    for module, result in inventory["imports"].items():
        check(f"import:{module}", result["ok"], result.get("error", "import succeeded"))
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slime-dir", "--slime-root", type=Path, help="Pinned editable slime checkout")
    parser.add_argument("--expected-cuda", default="12.9", help="Required runtime/toolkit, driver API minimum (default: 12.9)")
    parser.add_argument("--expected-gpus", type=int, default=8)
    parser.add_argument("--gpu-smoke", action="store_true", help="Run a tiny BF16 matmul on each visible GPU")
    parser.add_argument("--report", type=Path, help="Optional JSON evidence file")
    args = parser.parse_args()
    if not re.fullmatch(r"\d+\.\d+", args.expected_cuda):
        parser.error("--expected-cuda must be a major.minor version")
    if args.expected_gpus < 1:
        parser.error("--expected-gpus must be positive")
    inventory = collect_environment(args.slime_dir, args.gpu_smoke)
    checks = validate_environment(inventory, args.expected_cuda, args.expected_gpus)
    passed = all(check["ok"] for check in checks)
    report = {
        "status": "preflight_passed" if passed else "blocked",
        "expected_cuda": args.expected_cuda,
        "expected_gpus": args.expected_gpus,
        "hardware_training_validated": False,
        "limitation": "The pinned upstream v0.2.3 image supplies CUDA 12.9.1; this recipe has not been H20-tested. "
                      "Passing preflight is not proof that SGLang kernels, weight synchronization, "
                      "or an optimizer step work; record a complete GPU training smoke test separately.",
        "checks": checks,
        "environment": inventory,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
