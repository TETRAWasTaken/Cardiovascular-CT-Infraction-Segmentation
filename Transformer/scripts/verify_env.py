#!/usr/bin/env python3
"""
Phase 1: Environment & CUDA Verification Script
==============================================
Validates host architecture (targeting x86_64), PyTorch CUDA acceleration,
NVIDIA compute capability, cuDNN integration, and spatial processing libraries.
"""

from __future__ import annotations

import platform
import sys


def verify_system_architecture() -> dict[str, str]:
    arch = platform.machine()
    system = platform.system()
    python_ver = sys.version.split()[0]
    print("=" * 60)
    print("SYSTEM ARCHITECTURE & RUNTIME AUDIT")
    print("=" * 60)
    print(f"OS / Kernel      : {system} ({platform.release()})")
    print(f"CPU Architecture : {arch}")
    print(f"Python Version   : {python_ver}")

    is_x86 = arch.lower() in ("x86_64", "amd64")
    if is_x86:
        print("[STATUS] CPU Architecture matches targeted: x86_64")
    else:
        print(f"[NOTE] Host is '{arch}'. For production training, x86_64 Linux node is targeted.")

    return {"system": system, "arch": arch, "python": python_ver, "is_x86": str(is_x86)}


def verify_cuda() -> dict[str, str | bool | int]:
    print("\n" + "=" * 60)
    print("NVIDIA CUDA & PYTORCH ACCELERATION AUDIT")
    print("=" * 60)

    try:
        import torch
        torch_ver = torch.__version__
        cuda_available = torch.cuda.is_available()

        print(f"PyTorch Version  : {torch_ver}")
        print(f"CUDA Available   : {cuda_available}")

        report: dict[str, str | bool | int] = {
            "torch_version": torch_ver,
            "cuda_available": cuda_available,
        }

        if cuda_available:
            dev_count = torch.cuda.device_count()
            dev_name = torch.cuda.get_device_name(0)
            capability = torch.cuda.get_device_capability(0)
            cuda_ver = torch.version.cuda or "N/A"
            cudnn_ver = torch.backends.cudnn.version() or "N/A"

            print(f"CUDA Device Count: {dev_count}")
            print(f"Primary Device   : {dev_name}")
            print(f"Compute Cap.     : {capability[0]}.{capability[1]} (SM {capability[0]}{capability[1]})")
            print(f"CUDA Runtime     : {cuda_ver}")
            print(f"cuDNN Version    : {cudnn_ver}")

            # Verify CUDA tensor allocation
            x = torch.zeros((1, 1, 96, 96, 96), device="cuda", dtype=torch.float32)
            print(f"[STATUS] CUDA Allocation Successful: {x.shape} on {x.device}")
            report.update({
                "device_name": dev_name,
                "compute_capability": f"{capability[0]}.{capability[1]}",
                "cuda_runtime": cuda_ver,
                "cudnn_version": str(cudnn_ver),
            })
        else:
            print("[NOTICE] CUDA not available on this host. Ready for deployment to x86_64 CUDA node.")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                print("[INFO] Apple Silicon MPS is detected on local machine.")

        return report

    except ImportError as e:
        print(f"[ERROR] Failed to import PyTorch: {e}")
        return {"error": str(e), "cuda_available": False}


def verify_spatial_libraries() -> dict[str, str]:
    print("\n" + "=" * 60)
    print("SPATIAL & MEDICAL IMAGING LIBRARIES AUDIT")
    print("=" * 60)

    libs = [
        ("monai", "MONAI"),
        ("SimpleITK", "SimpleITK"),
        ("nibabel", "Nibabel"),
        ("scipy", "SciPy"),
        ("skimage", "scikit-image"),
        ("cc3d", "connected-components-3d (cc3d)"),
    ]

    report: dict[str, str] = {}
    all_ok = True

    for mod_name, display_name in libs:
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "Available")
            print(f"  [OK] {display_name:<30} : v{ver}")
            report[mod_name] = str(ver)
        except ImportError as e:
            print(f"  [FAIL] {display_name:<30} : NOT INSTALLED ({e})")
            report[mod_name] = "MISSING"
            all_ok = False

    if all_ok:
        print("\n[STATUS] All spatial processing and medical imaging dependencies are verified.")
    else:
        print("\n[WARNING] Some packages are missing. Install with 'pip install -r requirements.txt'")

    return report


def main() -> int:
    verify_system_architecture()
    cuda_report = verify_cuda()
    verify_spatial_libraries()
    print("\n" + "=" * 60)
    print("Verification Script Finished.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
