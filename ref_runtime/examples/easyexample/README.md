# PTO Runtime Example - Task Dependency Graph

This example demonstrates how to build and execute task dependency graphs on both Ascend hardware (a2a3) and simulation platform (a2a3sim).

## Overview

The example implements the formula `(a + b + 1)(a + b + 2)` using a task dependency graph:

- Task 0: `c = a + b`
- Task 1: `d = c + 1`
- Task 2: `e = c + 2`
- Task 3: `f = d * e`

With input values `a=2.0` and `b=3.0`, the expected result is `f = (2+3+1)*(2+3+2) = 42.0`.

## Platform Support

This example supports two platforms:

| Platform | Description | Flag | Requirements |
|----------|-------------|------|--------------|
| **a2a3** | Ascend hardware | `-p a2a3` | CANN toolkit, Ascend device |
| **a2a3sim** | Thread-based simulation | `-p a2a3sim` | gcc/g++ only |

### Key Differences

| Aspect | Hardware (a2a3) | Simulation (a2a3sim) |
|--------|-----------------|----------------------|
| Kernel compilation | ccec (Bisheng) compiler | g++ compiler |
| Execution | AICPU/AICore on device | Host threads |
| Kernel format | PTO ISA | Plain C++ loops |
| Device required | Yes | No |

## Dependencies

### Hardware Platform (a2a3)
- Python 3
- NumPy
- CANN Runtime (Ascend) with ASCEND_HOME_PATH set
- gcc/g++ compiler
- PTO-ISA headers (PTO_ISA_ROOT environment variable)

### Simulation Platform (a2a3sim)
- Python 3
- NumPy
- gcc/g++ compiler

## Quick Start

### Run on Simulation Platform (No Hardware Required)

```bash
# From repository root (ref_runtime)
python examples/scripts/run_example.py \
  -k examples/easyexample/kernels \
  -g examples/easyexample/golden.py \
  -p a2a3sim

# With verbose output
python examples/scripts/run_example.py \
  -k examples/easyexample/kernels \
  -g examples/easyexample/golden.py \
  -p a2a3sim \
  -v
```

### Run on Ascend Hardware

```bash
# From repository root (ref_runtime)
python examples/scripts/run_example.py \
  -k examples/easyexample/kernels \
  -g examples/easyexample/golden.py \
  -p a2a3

# With specific device ID
python examples/scripts/run_example.py \
  -k examples/easyexample/kernels \
  -g examples/easyexample/golden.py \
  -p a2a3 \
  -d 0

# With verbose output
python examples/scripts/run_example.py \
  -k examples/easyexample/kernels \
  -g examples/easyexample/golden.py \
  -p a2a3 \
  -v
```

## Directory Structure

```
easyexample/
├── README.md                    # This file
├── golden.py                    # Input generation and expected output
└── kernels/
    ├── kernel_config.py         # Kernel configuration
    ├── aiv/                      # AIV kernel implementations
    │   ├── kernel_add.cpp        # Element-wise tensor addition
    │   ├── kernel_add_scalar.cpp # Add scalar to tensor elements
    │   └── kernel_mul.cpp        # Element-wise tensor multiplication
    └── orchestration/
        ├── example_orch.cpp      # Task graph building function
        └── example_aicpu_orchestration_entry_skeleton.cpp
```

## Files

### `golden.py`

Defines input tensors and expected output computation:

```python
__outputs__ = ["f"]           # Output tensor names
TENSOR_ORDER = ["a", "b", "f"]  # Order passed to orchestration function

def generate_inputs(params: dict) -> dict:
    # Returns: {"a": ..., "b": ..., "f": ...}

def compute_golden(tensors: dict, params: dict) -> None:
    # Computes expected output in-place
```

### `kernels/kernel_config.py`

Defines kernel sources and orchestration function:

```python
KERNELS = [
    {"func_id": 0, "core_type": "aiv", "source": ".../kernel_add.cpp"},
    {"func_id": 1, "core_type": "aiv", "source": ".../kernel_add_scalar.cpp"},
    {"func_id": 2, "core_type": "aiv", "source": ".../kernel_mul.cpp"},
]

ORCHESTRATION = {
    "source": ".../example_aicpu_orchestration_entry_skeleton.cpp",
    "function_name": "aicpu_orchestration_entry"
}
```

## Expected Output

```
=== Building Runtime: rt2 (platform: a2a3/a2a3sim) ===
...
=== Compiling and Registering Kernels ===
Compiling kernel: kernels/aiv/kernel_add.cpp (func_id=0)
...
=== Generating Input Tensors ===
Inputs: ['a', 'b']
Outputs: ['f']
...
=== Launching Runtime ===
...
=== Comparing Results ===
Comparing f: shape=(16384,), dtype=float32
  First 10 actual:   [42. 42. 42. 42. 42. 42. 42. 42. 42. 42.]
  First 10 expected: [42. 42. 42. 42. 42. 42. 42. 42. 42. 42.]
  f: PASS (16384/16384 elements matched)

============================================================
TEST PASSED
============================================================
```

## Environment Setup

### For Hardware Platform (a2a3)

```bash
# Required environment variables
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest

# PTO_ISA_ROOT is auto-detected (cloned to examples/scripts/_deps/pto-isa on first run)
# Or you can set it manually if you have it elsewhere:
# export PTO_ISA_ROOT=/path/to/pto-isa

# Optional
export PTO_DEVICE_ID=0
```

### For Simulation Platform (a2a3sim)

No special environment setup required. Just ensure gcc/g++ is in PATH.

## Kernels

The same kernel source files work for both platforms:

- **kernel_add.cpp** - Element-wise tensor addition
- **kernel_add_scalar.cpp** - Add scalar to each tensor element
- **kernel_mul.cpp** - Element-wise tensor multiplication

On a2a3, kernels are compiled with PTO ISA using ccec. On a2a3sim, they are compiled as plain C++ with g++.

## Simulation Architecture

The simulation platform (a2a3sim) emulates the AICPU/AICore execution model:

- **Kernel loading**: Kernel `.text` sections are mmap'd into executable memory
- **Thread execution**: Host threads emulate AICPU scheduling and AICore computation
- **Memory**: All allocations use host memory (malloc/free)
- **Same API**: Uses identical C API as the real a2a3 platform

## Troubleshooting

### Kernel Compilation Failed (a2a3)

The test framework auto-clones pto-isa on first run. If this fails, clone it manually:
```bash
mkdir -p examples/scripts/_deps
git clone --branch ci_simpler https://gitcode.com/zhangqi-chen/pto-isa.git examples/scripts/_deps/pto-isa
```
Or set PTO_ISA_ROOT to an existing installation:
```bash
export PTO_ISA_ROOT=/path/to/pto-isa
```

### Device Initialization Failed (a2a3)

- Verify CANN runtime is installed and ASCEND_HOME_PATH is set
- Check that the specified device ID is valid (0-15)
- Ensure you have permission to access the device

### AICPU Error 507018 / 0x2a (a2a3)

When running on real Ascend hardware, the runtime loads the AICPU SO by name **libaicpu_kernel.so**. The test framework builds this SO and writes it to the **current working directory** (`getcwd()/libaicpu_kernel.so`) so CANN can load it. If you see `rtStreamSynchronize (AICPU) failed: 507018` or `after Init failed: 507018`:

1. **Kernel name length**: The AICPU init/main kernel names (`DynTileFwkBackendKernelServerInit`, `DynTileFwkBackendKernelServer`) must not be truncated; the host passes them in a buffer of 64 bytes. If the name were truncated, CANN would fail to resolve the symbol (errcode 11006 / inner error).
2. **SO path**: The test script switches to `ref_runtime` (project root) during launch so `libaicpu_kernel.so` is always written to `ref_runtime/libaicpu_kernel.so`. Run the example from either the repo root or `ref_runtime`; use absolute path for `so_name` so the host can open the SO.
3. **Enable CANN debug logs** to get SO load / kernel name in the error report:
   ```bash
   export ASCEND_GLOBAL_LOG_LEVEL=0
   export ASCEND_SLOG_PRINT_TO_STDOUT=1
   ```
4. **Check plog** under `/root/ascend/log/` or `$ASCEND_HOME/log/` (e.g. `plog/`) for detailed AICPU exception info (fault kernel name, SO path, errcode 11006).
5. Ensure no other `libaicpu_kernel.so` in `LD_LIBRARY_PATH` or CANN search path shadows the built one.
6. **Debug launch**: set `PTO2_DEBUG_AICPU_LAUNCH=1` to print so_name and kernel_name on each AICPU launch.

### "binary_data cannot be empty" Error

- Verify correct platform flag (`-p a2a3` or `-p a2a3sim`) is used
- Check if kernel source files exist
- Use `-v` to view detailed compilation logs

### Compilation Errors (a2a3sim)

- Ensure gcc/g++ is installed and available in PATH
- Check kernel source syntax for C++ errors

## See Also

- [Test Framework Documentation](../scripts/README.md)
- [Main Project README](../../README.md)
