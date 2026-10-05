"""Entry point; for now a hello world that reports where it runs."""

import os
import platform
import shutil
import subprocess

import matplotlib
import numpy as np


def nvidia_gpus():
    """Names of the NVIDIA GPUs this process can see (only the allocated ones under Slurm)."""
    if shutil.which("nvidia-smi") is None:
        return []
    out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout
    return [line.split(":", 1)[1].split("(UUID")[0].strip()
            for line in out.splitlines() if line.startswith("GPU ")]


print(
    f"Hello from {platform.node()}: {os.process_cpu_count()} CPUs, "
    f"GPUs: {', '.join(nvidia_gpus()) or 'none'}, "
    f"numpy {np.__version__}, matplotlib {matplotlib.__version__}"
)
