"""
Kernel and Orchestration Configuration

Defines the kernels and orchestration function used by the example.
Supports both hardware (a2a3) and simulation (a2a3sim) platforms.
"""

from pathlib import Path

_KERNELS_ROOT = Path(__file__).parent

# Orchestration config - separate sources for host and device modes
# Host orchestration runs on host CPU and writes to a host mirror buffer
# Device orchestration runs on AICPU thread 3 (works on both a2a3 and a2a3sim)
ORCHESTRATION = {
    "source": str(_KERNELS_ROOT / "orchestration" / "example_aicpu_orchestration_entry_skeleton.cpp"),
    "function_name": "aicpu_orchestration_entry",
    # Host orchestration uses different file
    "host_source": str(_KERNELS_ROOT.parent.parent.parent / "src" / "runtime" / "rt2" / "host" / "example_host_orchestration_entry.cpp"),
    "host_function_name": "host_orchestration_entry",
}

# Kernel configs
KERNELS = [
    {"func_id": 0, "source": str(_KERNELS_ROOT / "aiv" / "kernel_add.cpp"),        "core_type": "aiv"},
    {"func_id": 1, "source": str(_KERNELS_ROOT / "aiv" / "kernel_add_scalar.cpp"), "core_type": "aiv"},
    {"func_id": 2, "source": str(_KERNELS_ROOT / "aiv" / "kernel_mul.cpp"),        "core_type": "aiv"},
]
