# PTO Runtime - Graph Execution Framework

ARM64 runtime for executing task dependency graphs on Ascend devices with AICPU and AICore coordination.

## Overview

This runtime enables building and executing computational task graphs on Ascend devices. It provides:
- Task dependency graph management
- Device memory allocation and management
- AICPU and AICore kernel coordination
- Host-device data transfer
- **Python bindings for easy integration**

## Directory Structure

```
runtime/
├── src/
│   ├── graph/              # Task dependency graph (merged from graph + common)
│   │   ├── graph.h/cpp         # Graph class implementation
│   │   ├── handshake.h         # AICPU-AICore handshake protocol
│   │   └── kernel_args.h       # Kernel argument structures
│   ├── host/               # Host-side runtime
│   │   ├── devicerunner.h/cpp      # Device execution interface
│   │   ├── memoryallocator.h/cpp   # Memory management
│   │   ├── kernel_compiler.h/cpp   # Runtime kernel compilation
│   │   └── binary_loader.h/cpp     # Binary loading utilities
│   ├── aicpu/              # AICPU kernel implementation
│   │   ├── kernel.cpp         # Task scheduler for AICPU
│   │   └── device_log.h/cpp    # Device logging utilities
│   └── aicore/             # AICore kernel implementation
│       └── kernel.cpp          # Task execution kernels (add, mul, etc.)
├── python/             # Python bindings
│   ├── runtime_bindings.py # Python bindings wrapper
│   ├── toolchain.py        # Build toolchain utilities
│   ├── binary_compiler.py  # Binary compilation utilities
│   └── build/              # Python module build directory
├── examples/           # Example applications
│   └── basic/              # Basic example (C++ and Python)
│       ├── graphbuilder.cpp    # C++ implementation
│       ├── main.py             # Python entry point
│       ├── kernels/            # Runtime kernels
│       └── graph/              # Graph utilities
└── CMakeLists.txt      # Main build configuration
```

## Key Components

### Graph Class ([src/graph/](src/graph/))
- Manages task dependency graphs with fixed-size arrays
- Tracks task arguments, dependencies (fanin/fanout), and execution state
- Provides topological ordering for execution

### DeviceRunner ([src/host/devicerunner.h](src/host/devicerunner.h))
- Singleton interface for device operations
- Manages AICPU and AICore kernel launching
- Handles memory allocation and data transfer
- Coordinates graph execution workflow

### MemoryAllocator ([src/host/memoryallocator.h](src/host/memoryallocator.h))
- Centralized device memory management
- Automatic tracking of allocations
- Prevents memory leaks with automatic cleanup

### AICPU Graph Executor ([src/aicpu/kernel.cpp](src/aicpu/kernel.cpp))
- Task scheduler running on AICPU
- Manages handshake protocol with AICore
- Dispatches ready tasks to AICore cores
- Implements task dependency resolution

### AICore Kernels ([src/aicore/kernel.cpp](src/aicore/kernel.cpp))
- Task execution on AICore using PTO ISA
- Implements arithmetic operations (add, mul, etc.)
- Polls handshake buffer for task assignments

## Building

### Prerequisites

- CMake 3.15+
- CANN toolkit (Ascend runtime)
- GCC/G++ with C++17 support
- **Python 3 with development headers (for Python bindings)**
- **pybind11 (automatically fetched if not available)**

### Environment Setup

```bash
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
```

### Build C++ Components

```bash
cd runtime
mkdir -p build && cd build
cmake ..
make
```

This produces:
- `graph_executor` - C++ example executable
- `aicpu/libaicpu_graph_kernel.so` - AICPU kernel library
- `aicore/kernel.o` - AICore kernel binary

### Build Python Bindings

```bash
cd runtime
mkdir -p build && cd build
cmake .. -DBUILD_PYTHON_BINDINGS=ON
make
```

This additionally builds:
- `python/pto_runtime.so` - Python module

To disable Python bindings:
```bash
cmake .. -DBUILD_PYTHON_BINDINGS=OFF
```

## Usage

### C++ Example

```bash
cd runtime/build
./graph_executor 9  # Run on device 9
```

Expected output:
```
=== Graph Builder Example ===
...
✓ SUCCESS: All 16384 elements are correct (42.0)
Formula verified: (a + b + 1)(a + b + 2) = (2+3+1)*(2+3+2) = 42
```

### Python Example

```bash
cd runtime/examples/basic
python3 main.py 9
```

Quick Python example:
```python
# See examples/basic/main.py for complete working example
import numpy as np
from runtime_bindings import load_runtime

# Load runtime library
DeviceRunner, Graph = load_runtime("path/to/libpto_runtime.so")

# Initialize device
runner = DeviceRunner()
runner.init(device_id, num_cores, aicpu_binary, aicore_binary, pto_isa_root)

# Create and run graph
graph = Graph()
# Add tasks, allocate memory, copy data...
runner.run(graph)

# Cleanup
runner.finalize()
```

## Testing

For detailed testing procedures and validation of the refactoring changes, see [TESTING.md](../TESTING.md).

Quick test:
```bash
cd runtime/examples/basic && python main.py 9
```

## Runtime Kernel Compilation

The runtime supports compiling AICore kernel source files at runtime using the `ccec` compiler, allowing dynamic kernel loading without build-time compilation.

### Basic Usage

```cpp
#include "host/devicerunner.h"

// Initialize DeviceRunner
DeviceRunner& runner = DeviceRunner::Get();
std::string ptoIsaRoot = "/path/to/pto-isa";
runner.Init(deviceId, numCores, "./aicpu/lib.so", "./aicore/kernel.o", ptoIsaRoot);

// Compile and load kernels at runtime
runner.CompileAndLoadKernel(0, "./aicore/kernels/kernel_add.cpp", 1);
runner.CompileAndLoadKernel(1, "./aicore/kernels/kernel_mul.cpp", 1);

// Use the kernels in your graph
Graph graph;
graph.add_task(args, 4, 0);  // Uses func_id=0 (kernel_add)
runner.Run(graph);
```


### Python API

```python
from runtime_bindings import load_runtime

# Load the runtime library
DeviceRunner, Graph = load_runtime("/path/to/libpto_runtime.so")

# Initialize DeviceRunner with PTO-ISA root
runner = DeviceRunner()
pto_isa_root = "/path/to/pto-isa"
runner.init(device_id=9, num_cores=3,
           aicpu_binary=aicpu_bytes,
           aicore_binary=aicore_bytes,
           pto_isa_root=pto_isa_root)

# See examples/basic/main.py for complete working example with kernel compilation
```

### Compiler Requirements

- `ASCEND_HOME_PATH` environment variable must be set
- `ccec` compiler available at `${ASCEND_HOME_PATH}/bin/ccec`
- PTO-ISA headers (automatically fetched at build time)

Compiled `.o` files are stored in `/tmp` with unique names. See [src/host/kernel_compiler.h](src/host/kernel_compiler.h) for implementation details.

## Architecture

### Execution Flow

```
1. Host (graphbuilder.cpp or Python):
   ├─ Build task graph (Graph class)
   ├─ Allocate device memory (DeviceRunner)
   ├─ Copy input data to device
   └─ Call DeviceRunner::Run(graph)

2. DeviceRunner::Run():
   ├─ Copy graph to device memory
   ├─ Launch AICPU init kernel (handshake)
   ├─ Launch AICPU main kernel (scheduler)
   ├─ Launch AICore kernel (workers)
   └─ Synchronize streams

3. AICPU (graph_executor.cpp):
   ├─ Handshake with AICore cores
   ├─ Find initially ready tasks (fanin=0)
   ├─ Dispatch tasks to idle AICore cores
   ├─ Wait for task completion
   ├─ Update fanin counters
   └─ Repeat until all tasks done

4. AICore (kernel.cpp):
   ├─ Wait for task assignment
   ├─ Read task arguments
   ├─ Execute kernel (add/mul/etc.)
   ├─ Signal completion
   └─ Repeat until quit signal
```

### Handshake Protocol

Each AICore core has a dedicated handshake buffer:

```c
struct Handshake {
    volatile uint32_t aicpu_ready;  // AICPU→AICore: scheduler ready
    volatile uint32_t aicore_done;  // AICore→AICPU: core ready
    volatile uint64_t task;         // AICPU→AICore: task pointer
    volatile int32_t task_status;   // Task state: 1=busy, 0=done
    volatile int32_t control;       // AICPU→AICore: 1=quit
};
```

## Recent Changes

### Reorganization (Current)

The runtime has been reorganized for better structure and Python integration:

1. **Merged `graph/` and `common/` into `src/graph/`**
   - Moved `handshake.h` and `kernel_args.h` from `common/` to `src/graph/`
   - All shared structures now in one location
   - Simplified include paths

2. **Created `src/host/` directory**
   - Moved `devicerunner.cpp/h` and `memoryallocator.cpp/h` to `src/host/`
   - Separates host-side runtime from device kernels
   - Clearer organization

3. **Reorganized to `src/` structure**
   - All core runtime components under `src/` directory
   - Components: `src/graph/`, `src/host/`, `src/aicpu/`, `src/aicore/`
   - Better separation of source code from examples and bindings

4. **Added Python bindings**
   - New `python/` directory with runtime bindings utilities
   - Python API matches C++ interface
   - NumPy integration for efficient data transfer
   - Example Python application in `examples/basic/main.py`

5. **Added runtime kernel compilation**
   - New `src/host/kernel_compiler.cpp/h` for dynamic kernel loading
   - Compile AICore kernels at runtime using `ccec` compiler
   - Flexible kernel development without full rebuilds

See git history for detailed changes.

## Logging

Device logs are written to `~/ascend/log/debug/device-<id>/`

Kernel uses `DEV_INFO`, `DEV_DEBUG`, `DEV_WARN`, `DEV_ERROR` macros.

## Notes

- Device ID range: 0-15
- Default device: 9
- Graph supports up to 1024 tasks (configurable via `GRAPH_MAX_TASKS`)
- Memory allocator automatically tracks and frees allocations
- Python bindings require NumPy for array operations

## References

- See [src/graph/](src/graph/) for Graph class implementation
- See [examples/basic/](examples/basic/) for complete working examples
- See [examples/basic/main.py](examples/basic/main.py) for Python example usage
