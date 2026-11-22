"""Check PyTorch, GPU kernels, packages and data before training.

RTX 50-series GPUs need the CUDA 12.8 PyTorch build (sm_120).
"""

import sys

import config as C

OK, WARN, BAD = "  ok  ", " warn ", " FAIL "


def line(status, msg):
    print(f"[{status}] {msg}")


def check_torch() -> bool:
    try:
        import torch
    except ImportError:
        line(BAD, "PyTorch is not installed. See README.md for the install command.")
        return False

    line(OK, f"PyTorch {torch.__version__}")

    if not torch.cuda.is_available():
        line(WARN, "CUDA not available. Training will fall back to CPU and be very slow.")
        return False

    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    line(OK, f"GPU: {name}  compute capability {cap[0]}.{cap[1]}")
    line(OK, f"CUDA runtime: {torch.version.cuda}")

    arch_list = torch.cuda.get_arch_list()
    needed = f"sm_{cap[0]}{cap[1]}"
    if needed not in arch_list:
        line(BAD, f"This PyTorch build has no {needed} kernels. Found: {arch_list}")
        line(BAD, "Reinstall with the cu128 wheels. See README.md.")
        return False
    line(OK, f"{needed} kernels present in this build")

    try:
        x = torch.randn(2048, 2048, device="cuda")
        (x @ x).sum().item()
        torch.cuda.synchronize()
        line(OK, "GPU matmul succeeded")
    except Exception as exc:
        line(BAD, f"GPU compute failed: {exc}")
        return False

    free, total = torch.cuda.mem_get_info()
    line(OK, f"GPU memory: {free / 1e9:.1f} GB free of {total / 1e9:.1f} GB")
    return True


def check_data() -> bool:
    good = True

    if C.FITZ_CSV.exists():
        import pandas as pd
        n = len(pd.read_csv(C.FITZ_CSV))
        line(OK, f"Fitzpatrick17k metadata: {n} rows")
    else:
        line(BAD, f"Missing {C.FITZ_CSV}")
        good = False

    if C.FITZ_IMAGES.is_dir():
        n = sum(1 for p in C.FITZ_IMAGES.glob("*.jpg") if not p.name.startswith("._"))
        line(OK
             if n > 16000 else WARN, f"Fitzpatrick17k images: {n}")
        if n < 16000:
            good = False
    else:
        line(BAD, f"Missing {C.FITZ_IMAGES}")
        good = False

    if C.DDI_CSV.exists() and C.DDI_IMAGES.is_dir():
        import pandas as pd
        n_meta = len(pd.read_csv(C.DDI_CSV))
        n_img = sum(1 for p in C.DDI_IMAGES.glob("*") if not p.name.startswith("._"))
        line(OK, f"DDI: {n_meta} metadata rows, {n_img} images")
    else:
        line(BAD, "DDI data missing. Expected data/ddi/ddi_metadata.csv and data/ddi/images/")
        good = False

    return good


def check_packages() -> bool:
    good = True
    for mod, label in [("torchvision", "torchvision"), ("pandas", "pandas"),
                       ("numpy", "numpy"), ("sklearn", "scikit-learn"),
                       ("matplotlib", "matplotlib"), ("PIL", "pillow")]:
        try:
            __import__(mod)
            line(OK, label)
        except ImportError:
            line(BAD, f"{label} is not installed")
            good = False
    return good


def main() -> None:
    print("Environment\n" + "-" * 60)
    print(f"Python {sys.version.split()[0]}")
    torch_ok = check_torch()

    print("\nPackages\n" + "-" * 60)
    pkg_ok = check_packages()

    print("\nData\n" + "-" * 60)
    data_ok = check_data()

    print("\n" + "=" * 60)
    if torch_ok and pkg_ok and data_ok:
        print("All checks passed. Run:  python run_all.py")
    else:
        print("Some checks failed. Fix the items marked FAIL, then run this again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
