# AICPU Kernel Example

This example demonstrates how to create and launch an AICPU kernel using CANN runtime APIs. It shows the complete workflow from loading the kernel binary to executing it on the device.

## Project Structure

```
03-aicpu-kernel/
├── kernel/                          # AICPU kernel code
│   ├── graph_executor.cpp           # Kernel implementation with entry points
│   └── CMakeLists.txt               # Build configuration for kernel
├── launcher.cpp                     # Host-side launcher
├── CMakeLists.txt                   # Main build configuration
└── README.md                        # This file
```

## Components

### 1. Kernel Code (`kernel/graph_executor.cpp`)

The AICPU kernel implements three entry points that are called by the system's `libaicpu_extend_kernels.so`:

- **`StaticTileFwkBackendKernelServer`**: Static kernel server entry point
- **`DynTileFwkBackendKernelServerInit`**: Dynamic kernel initialization entry point
  - Called when launching `DynTileFwkKernelServerInit` kernel
  - Saves the backend server SO file to device filesystem
  - Loads and binds function names from the SO file
- **`DynTileFwkBackendKernelServer`**: Dynamic kernel execution entry point
  - Called when launching `DynTileFwkKernelServer` kernel
  - Executes the main kernel logic

### 2. Launcher (`launcher.cpp`)

The launcher demonstrates the complete workflow:

#### Data Structures

- **`DeviceArgs`**: Contains device-side arguments including:
  - `aicpuSoBin`: Device memory address of the backend server SO binary
  - `aicpuSoLen`: Size of the SO binary

- **`KernelArgs`**: Contains kernel arguments including:
  - `deviceArgs`: Pointer to device memory containing `DeviceArgs`
  - Methods:
    - `InitDeviceArgs()`: Allocates device memory and copies `DeviceArgs` to device
    - `FinalizeDeviceArgs()`: Frees device memory

- **`AicpuSoInfo`**: Manages the backend server SO file:
  - `Init()`: Loads SO file from disk and copies to device memory
  - `Finalize()`: Frees device memory

#### Workflow

1. **Parse Arguments**: Parse device ID from command line (default: 0)
2. **Initialize Device**: Set device and create stream
3. **Load SO File**: Use `AicpuSoInfo::Init()` to load `libtilefwk_backend_server.so` to device memory
4. **Prepare Arguments**: Create `DeviceArgs` with SO binary address and size
5. **Initialize Kernel Args**: Use `KernelArgs::InitDeviceArgs()` to copy arguments to device
6. **Launch Kernels**: Call `DeviceRunner::Run()` which:
   - Launches `DynTileFwkKernelServerInit` kernel (initialization)
   - Launches `DynTileFwkKernelServer` kernel (execution)
   - Synchronizes the stream
7. **Cleanup**: Free all allocated resources

## Building

```bash
mkdir build
cd build
cmake ..
make
```

This will generate:
- `kernel/libtilefwk_backend_server.so`: The AICPU backend server binary (kernel implementation)
- `launcher`: The host-side launcher executable

## Usage

```bash
./launcher [device_id]
```

Where `device_id` is an optional device ID (0-15). If not provided, defaults to 0.

Examples:
```bash
./launcher        # Uses device 0 (default)
./launcher 0      # Uses device 0
./launcher 6      # Uses device 6
```

The launcher will:
1. Parse device ID from command line (default: 0)
2. Set the device
3. Create a stream
4. Load the backend server SO file
5. Initialize kernel arguments
6. Launch the init and main kernels
7. Clean up resources

## Logging

The kernel uses PLOG (Platform Log) for device-side logging. Log files are written to:

```
~/ascend/log/debug/device-<device_id>/
```

Where `<device_id>` is the device ID used when launching the kernel. For example:
- Device 0: `~/ascend/log/debug/device-0/`
- Device 6: `~/ascend/log/debug/device-6/`

The kernel logs messages using `DEV_INFO`, `DEV_DEBUG`, `DEV_WARN`, and `DEV_ERROR` macros defined in `kernel/device_log.h`. These logs are written to the device log files and can be viewed to debug kernel execution.

## Key Concepts

### Offset Requirements

The structure layouts are critical because offsets are hardcoded in the AICPU kernel (`libaicpu_extend_kernels.so`):

1. **Offset from `KernelArgs` to `deviceArgs` pointer**: Must match the expected offset
2. **Offset from `DeviceArgs` to `aicpuSoBin`**: Must match the expected offset
3. **Offset from `DeviceArgs` to `aicpuSoLen`**: Must match the expected offset

### Function Name Mapping

The system kernel (`libaicpu_extend_kernels.so`) calls functions in our backend server SO:

| System Kernel Name | Backend Function Name |
|-------------------|----------------------|
| `DynTileFwkKernelServerInit` | `DynTileFwkBackendKernelServerInit` |
| `DynTileFwkKernelServer` | `DynTileFwkBackendKernelServer` |
| `StaticTileFwkKernelServer` | `StaticTileFwkBackendKernelServer` |

### Execution Flow

```
Host Side:
  1. Load libtilefwk_backend_server.so to device memory
  2. Create DeviceArgs with SO binary address
  3. Launch DynTileFwkKernelServerInit kernel

Device Side (libaicpu_extend_kernels.so):
  1. Receives KernelArgs containing DeviceArgs pointer
  2. Reads DeviceArgs from device memory
  3. Saves SO binary to device filesystem
  4. Loads SO file using dlopen()
  5. Calls DynTileFwkBackendKernelServerInit()

Host Side:
  1. Launch DynTileFwkKernelServer kernel

Device Side:
  1. Calls DynTileFwkBackendKernelServer() from loaded SO
  2. Executes kernel logic
```

## Error Handling

The code includes comprehensive error handling:
- Device initialization failures
- Memory allocation failures
- SO file loading failures
- Kernel launch failures
- Stream synchronization failures

All errors are reported with descriptive messages and proper cleanup is performed.

## Notes

- The module ID for `rtMalloc` is set to 0 (works but reason unknown)
- Device ID can be specified via command-line argument (default: 0, valid range: 0-15)
- The `launchAicpuNum` parameter controls how many AICPU cores are used for the main kernel (default: 1)
- The init kernel always uses 1 AICPU core
