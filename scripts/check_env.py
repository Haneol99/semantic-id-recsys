"""Print Python/PyTorch versions, MPS availability, and run a small matmul on the chosen device."""

import platform
import sys
import time

import torch


def main() -> None:
    print(f"Python        : {sys.version.split()[0]} ({platform.machine()})")
    print(f"PyTorch       : {torch.__version__}")
    print(f"MPS built     : {torch.backends.mps.is_built()}")
    print(f"MPS available : {torch.backends.mps.is_available()}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device        : {device}")

    torch.manual_seed(0)
    a = torch.randn(1024, 1024, device=device)
    b = torch.randn(1024, 1024, device=device)
    start = time.perf_counter()
    c = a @ b
    if device.type == "mps":
        torch.mps.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000

    ref = a.cpu() @ b.cpu()
    max_err = (c.cpu() - ref).abs().max().item()
    print(f"Matmul        : 1024x1024 @ 1024x1024 in {elapsed_ms:.2f} ms, max |diff| vs CPU = {max_err:.2e}")


if __name__ == "__main__":
    main()
