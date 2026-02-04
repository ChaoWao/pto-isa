"""

PTO Runtime ctypes Bindings

Provides a Pythonic interface to the PTO runtime via ctypes.
Users must provide a pre-compiled libpto_runtime.so (built via binary_compiler.py).

Usage:
    from bindings import bind_host_binary, register_kernel, launch_runtime

    Runtime = bind_host_binary("/path/to/libpto_runtime.so")

    runtime = Runtime()
    runtime.initialize(orch_so_binary, "aicpu_orchestration_entry", func_args)

    register_kernel(0, kernel_add)
    register_kernel(1, kernel_add_scalar)
    register_kernel(2, kernel_mul)

    launch_runtime(runtime, aicpu_thread_num=1, block_dim=1,
                 device_id=0, aicpu_binary=aicpu_bytes,
                 aicore_binary=aicore_bytes)

    runtime.finalize()
"""


from ctypes import (
    CDLL,
    POINTER,
    c_char_p,
    c_int,
    c_int32,
    c_void_p,
    c_uint8,
    c_uint64,
    c_size_t,
)
from pathlib import Path
from typing import Union, List, Optional
import ctypes
import tempfile


# Module-level library reference
_lib = None


# ============================================================================
# Runtime Library Loader
# ============================================================================

class RuntimeLibraryLoader:
    """Loads and manages the PTO runtime C API library."""


    def __init__(self, lib_path: Union[str, Path]):
        """

        Load the PTO runtime library.

        Args:
            lib_path: Path to libpto_runtime.so

        Raises:
            FileNotFoundError: If library file not found
            OSError: If library cannot be loaded
        """

        lib_path = Path(lib_path)
        if not lib_path.exists():
            raise FileNotFoundError(f"Library not found: {lib_path}")

        self.lib_path = lib_path
        self.lib = CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
        self._setup_functions()

    def _setup_functions(self):
        """Set up ctypes function signatures."""

        # get_runtime_size - returns sizeof(Runtime) for user allocation
        self.lib.get_runtime_size.argtypes = []
        self.lib.get_runtime_size.restype = c_size_t

        # init_runtime - placement new + load SO + build runtime with orchestration
        self.lib.init_runtime.argtypes = [
            c_void_p,               # runtime
            POINTER(c_uint8),       # orch_so_binary
            c_size_t,               # orch_so_size
            c_char_p,               # orch_func_name
            POINTER(c_uint64),      # func_args
            c_int,                  # func_args_count
            c_int,                  # use_device_orchestration
            c_int,                  # run_orchestrator_on_host
        ]
        self.lib.init_runtime.restype = c_int

        # launch_runtime - device init + execute runtime
        self.lib.launch_runtime.argtypes = [
            c_void_p,           # runtime
            c_int,              # aicpu_thread_num
            c_int,              # block_dim
            c_int,              # device_id
            POINTER(c_uint8),   # aicpu_binary
            c_size_t,           # aicpu_size
            POINTER(c_uint8),   # aicore_binary
            c_size_t,           # aicore_size
        ]
        self.lib.launch_runtime.restype = c_int

        # finalize_runtime - validate + cleanup
        self.lib.finalize_runtime.argtypes = [c_void_p]
        self.lib.finalize_runtime.restype = c_int

        # register_kernel - register kernel binary for func_id
        self.lib.register_kernel.argtypes = [c_int, POINTER(c_uint8), c_size_t]
        self.lib.register_kernel.restype = c_int

        # set_device - set device and create streams
        self.lib.set_device.argtypes = [c_int]
        self.lib.set_device.restype = c_int

        # device_malloc / device_free / copy_to_device (for device orchestration)
        self.lib.device_malloc.argtypes = [c_size_t]
        self.lib.device_malloc.restype = c_void_p

        self.lib.device_free.argtypes = [c_void_p]
        self.lib.device_free.restype = None

        self.lib.copy_to_device.argtypes = [c_void_p, c_void_p, c_size_t]
        self.lib.copy_to_device.restype = c_int

        self.lib.copy_from_device.argtypes = [c_void_p, c_void_p, c_size_t]
        self.lib.copy_from_device.restype = c_int

        # record_tensor_pair / set_pto2_gm_sm_ptr (for device orchestration copy-back)
        self.lib.record_tensor_pair.argtypes = [c_void_p, c_void_p, c_void_p, c_size_t]
        self.lib.record_tensor_pair.restype = None

        self.lib.set_pto2_gm_sm_ptr.argtypes = [c_void_p, c_void_p]
        self.lib.set_pto2_gm_sm_ptr.restype = None

        # PTO2 host orchestration (rt2 run_orchestrator_on_host)
        self.lib.get_pto2_sm_size.argtypes = [c_void_p]
        self.lib.get_pto2_sm_size.restype = c_int32

        self.lib.allocate_pto2_shared_memory.argtypes = [c_void_p]
        self.lib.allocate_pto2_shared_memory.restype = c_int

        self.lib.run_host_orchestration.argtypes = [c_void_p, c_void_p]
        self.lib.run_host_orchestration.restype = c_int


# ============================================================================
# Python Wrapper Classes
# ============================================================================

class Runtime:
    """

    Task dependency runtime.

    Python wrapper around the C Runtime API.
    User allocates memory via ctypes buffer, C++ uses placement new.
    """


    def __init__(self, lib: CDLL):
        """

        Create a new runtime handle.

        Args:
            lib: Loaded ctypes library (RuntimeLibraryLoader.lib)
        """

        self.lib = lib
        # Allocate buffer of size get_runtime_size() for placement new
        size = lib.get_runtime_size()
        self._buffer = ctypes.create_string_buffer(size)
        self._handle = ctypes.cast(self._buffer, c_void_p)

    def initialize(
        self,
        orch_so_binary: bytes,
        orch_func_name: str,
        func_args: Optional[List[int]] = None,
        use_device_orchestration: bool = False,
        run_orchestrator_on_host: bool = False,
    ) -> None:
        """

        Initialize the runtime structure with dynamic orchestration.

        Calls init_runtime() in C++ which loads the orchestration SO,
        resolves the function, and calls it to build the task graph.
        The orchestration function is responsible for:
        1. Allocating device memory
        2. Copying data to device
        3. Building the task graph
        4. Recording tensor pairs for copy-back

        Args:
            orch_so_binary: Orchestration shared library binary data
            orch_func_name: Name of the orchestration function to call
            func_args: Arguments for orchestration (host pointers, sizes, etc.)
            use_device_orchestration: If True (rt2 only), orchestration runs on AICPU thread 3
            run_orchestrator_on_host: If True (rt2 only), orchestration runs on host CPU; use allocate_pto2_shared_memory and run_host_orchestration after init

        Raises:
            RuntimeError: If initialization fails
        """

        func_args = func_args or []
        func_args_count = len(func_args)

        # Convert func_args to ctypes array
        if func_args_count > 0:
            func_args_array = (c_uint64 * func_args_count)(*func_args)
        else:
            func_args_array = None

        # Convert orch_so_binary to ctypes array (can be empty when use_device_orchestration or run_orchestrator_on_host)
        orch_so_size = len(orch_so_binary) if orch_so_binary else 0
        orch_so_array = (c_uint8 * orch_so_size).from_buffer_copy(orch_so_binary) if orch_so_size else None

        # Keep reference to prevent GC - C++ stores pointer to this data for later use by AICPU
        self._orch_so_array = orch_so_array

        rc = self.lib.init_runtime(
            self._handle,
            orch_so_array,
            orch_so_size,
            orch_func_name.encode('utf-8'),
            func_args_array,
            func_args_count,
            1 if use_device_orchestration else 0,
            1 if run_orchestrator_on_host else 0,
        )
        if rc != 0:
            raise RuntimeError(f"init_runtime failed: {rc}")

    def record_tensor_pair(self, host_ptr: int, dev_ptr: int, size: int) -> None:
        """
        Record a host-device tensor pair for copy-back during finalize.

        When using device orchestration, host allocates device memory and must
        record output tensors so finalize can copy results back to host.

        Args:
            host_ptr: Host buffer address (e.g. numpy array .ctypes.data)
            dev_ptr: Device buffer address (from device_malloc)
            size: Size in bytes to copy
        """
        self.lib.record_tensor_pair(
            self._handle,
            ctypes.c_void_p(host_ptr),
            ctypes.c_void_p(dev_ptr),
            c_size_t(size),
        )

    def set_pto2_gm_sm_ptr(self, dev_ptr: int) -> None:
        """
        Set device pointer to PTO2 shared memory (GM buffer).

        When using device orchestration, host allocates GM memory and passes
        its device address so AICPU thread 3 can use it in aicpu_orchestration_entry.

        Args:
            dev_ptr: Device pointer to shared memory buffer (from device_malloc)
        """
        self.lib.set_pto2_gm_sm_ptr(self._handle, ctypes.c_void_p(dev_ptr))

    def get_pto2_sm_size(self) -> int:
        """
        Get required PTO2 shared memory size in bytes (same for host mirror).

        Used when run_orchestrator_on_host: allocate a host buffer of this size
        and pass it to run_host_orchestration().

        Returns:
            Size in bytes, or 0 if not available
        """
        return int(self.lib.get_pto2_sm_size(self._handle))

    def allocate_pto2_shared_memory(self) -> None:
        """
        Allocate PTO2 shared memory and heap via runtime host_api; set pto2_gm_sm_ptr.

        Call when run_orchestrator_on_host after initialize() and before run_host_orchestration().

        Raises:
            RuntimeError: If allocation fails
        """
        rc = self.lib.allocate_pto2_shared_memory(self._handle)
        if rc != 0:
            raise RuntimeError(f"allocate_pto2_shared_memory failed: {rc}")

    def run_host_orchestration(self, host_mirror_ptr: int) -> None:
        """
        Run host orchestration into host_mirror then copy to device SM.

        Requires allocate_pto2_shared_memory() and set_orch_args (via initialize with func_args) done.
        host_mirror_ptr must point to a buffer of size get_pto2_sm_size().

        Args:
            host_mirror_ptr: Host buffer address (e.g. ctypes buffer or numpy array .ctypes.data)

        Raises:
            RuntimeError: If orchestration or copy fails
        """
        rc = self.lib.run_host_orchestration(self._handle, ctypes.c_void_p(host_mirror_ptr))
        if rc != 0:
            raise RuntimeError(f"run_host_orchestration failed: {rc}")

    def finalize(self) -> None:
        """

        Finalize and cleanup the runtime.

        Calls finalize_runtime() in C++ which validates computation results,
        frees device tensors, and calls the Runtime destructor.

        Raises:
            RuntimeError: If finalization fails
        """

        rc = self.lib.finalize_runtime(self._handle)
        if rc != 0:
            raise RuntimeError(f"finalize_runtime failed: {rc}")

    def __del__(self):
        """Clean up runtime resources."""

        # Runtime destructor is called by finalize(), buffer freed by Python GC
        pass


# ============================================================================
# Module-level Functions
# ============================================================================

def register_kernel(func_id: int, binary_data: bytes) -> None:
    """

    Register a kernel binary for a func_id.

    Receives pre-extracted .text section binary data,
    allocates device GM memory, copies the binary to device,
    and stores the GM address for later use by launch_runtime().

    Args:
        func_id: Function identifier (0, 1, 2, ...)
        binary_data: Kernel .text section binary data

    Raises:
        RuntimeError: If not initialized or registration fails
        ValueError: If binary_data is empty
    """

    global _lib
    if _lib is None:
        raise RuntimeError("Runtime not loaded. Call bind_host_binary() first.")

    if not binary_data:
        raise ValueError("binary_data cannot be empty")

    # Convert bytes to ctypes array
    bin_array = (c_uint8 * len(binary_data)).from_buffer_copy(binary_data)
    rc = _lib.register_kernel(func_id, bin_array, len(binary_data))
    if rc != 0:
        raise RuntimeError(f"register_kernel failed: {rc}")


def set_device(device_id: int) -> None:
    """

    Set device and create streams for memory operations.

    Must be called before runtime.initialize() to enable device tensor allocation.
    Only performs minimal initialization:
    - rtSetDevice(device_id)
    - Create AICPU and AICore streams

    Binary loading happens later in launch_runtime().

    Args:
        device_id: Device ID (0-15)

    Raises:
        RuntimeError: If not loaded or device setup fails
    """

    global _lib
    if _lib is None:
        raise RuntimeError("Runtime not loaded. Call bind_host_binary() first.")

    rc = _lib.set_device(device_id)
    if rc != 0:
        raise RuntimeError(f"set_device failed: {rc}")


def device_malloc(size: int) -> Optional[int]:
    """
    Allocate device memory (GM). Use for device orchestration tensors and shared memory.

    Args:
        size: Size in bytes

    Returns:
        Device pointer as integer, or None on failure
    """
    global _lib
    if _lib is None:
        raise RuntimeError("Runtime not loaded. Call bind_host_binary() first.")
    ptr = _lib.device_malloc(c_size_t(size))
    return ptr if ptr is not None else None


def device_free(dev_ptr: int) -> None:
    """Free device memory. Caller is responsible for not double-freeing."""
    global _lib
    if _lib is None:
        raise RuntimeError("Runtime not loaded. Call bind_host_binary() first.")
    _lib.device_free(ctypes.c_void_p(dev_ptr))


def copy_to_device(dev_ptr: int, host_ptr: int, size: int) -> None:
    """
    Copy data from host to device.

    Args:
        dev_ptr: Device destination (from device_malloc)
        host_ptr: Host source (e.g. numpy array .ctypes.data)
        size: Size in bytes
    """
    global _lib
    if _lib is None:
        raise RuntimeError("Runtime not loaded. Call bind_host_binary() first.")
    rc = _lib.copy_to_device(
        ctypes.c_void_p(dev_ptr),
        ctypes.c_void_p(host_ptr),
        c_size_t(size),
    )
    if rc != 0:
        raise RuntimeError(f"copy_to_device failed: {rc}")


def copy_from_device(host_ptr: int, dev_ptr: int, size: int) -> None:
    """
    Copy data from device to host (for debug or copy-back).

    Args:
        host_ptr: Host destination (e.g. buffer address)
        dev_ptr: Device source (from device_malloc)
        size: Size in bytes
    """
    global _lib
    if _lib is None:
        raise RuntimeError("Runtime not loaded. Call bind_host_binary() first.")
    rc = _lib.copy_from_device(
        ctypes.c_void_p(host_ptr),
        ctypes.c_void_p(dev_ptr),
        c_size_t(size),
    )
    if rc != 0:
        raise RuntimeError(f"copy_from_device failed: {rc}")


def launch_runtime(
    runtime: "Runtime",
    aicpu_thread_num: int,
    block_dim: int,
    device_id: int,
    aicpu_binary: bytes,
    aicore_binary: bytes,
) -> None:
    """

    Execute a runtime on the device.

    Initializes DeviceRunner singleton (if first call), copies runtime to device,
    launches kernels, synchronizes, and copies runtime back from device.

    Args:
        runtime: Runtime to execute (must have been initialized via runtime.initialize())
        aicpu_thread_num: Number of AICPU scheduler threads
        block_dim: Number of blocks (1 block = 1 AIC + 2 AIV)
        device_id: Device ID (0-15)
        aicpu_binary: Binary data of AICPU shared object
        aicore_binary: Binary data of AICore kernel

    Raises:
        RuntimeError: If not initialized or execution fails
    """

    global _lib
    if _lib is None:
        raise RuntimeError("Runtime not loaded. Call bind_host_binary() first.")

    # Convert bytes to ctypes arrays
    aicpu_array = (c_uint8 * len(aicpu_binary)).from_buffer_copy(aicpu_binary)
    aicore_array = (c_uint8 * len(aicore_binary)).from_buffer_copy(aicore_binary)

    rc = _lib.launch_runtime(
        runtime._handle,
        aicpu_thread_num,
        block_dim,
        device_id,
        aicpu_array,
        len(aicpu_binary),
        aicore_array,
        len(aicore_binary),
    )
    if rc != 0:
        raise RuntimeError(f"launch_runtime failed: {rc}")


# ============================================================================
# Public API
# ============================================================================

def bind_host_binary(lib_path: Union[str, Path, bytes]) -> type:
    """

    Load the PTO runtime library and return Runtime class.

    Args:
        lib_path: Path to libpto_runtime.so (str/Path), or compiled binary data (bytes)

    Returns:
        Runtime class initialized with the library

    Example:
        from bindings import bind_host_binary, register_kernel, launch_runtime

        Runtime = bind_host_binary("/path/to/libpto_runtime.so")

        runtime = Runtime()
        runtime.initialize(orch_so_binary, "aicpu_orchestration_entry", func_args)

        register_kernel(0, kernel_add)
        register_kernel(1, kernel_add_scalar)
        register_kernel(2, kernel_mul)

        launch_runtime(runtime, aicpu_thread_num=1, block_dim=1,
                     device_id=0, aicpu_binary=aicpu_bytes,
                     aicore_binary=aicore_bytes)

        runtime.finalize()
    """

    global _lib

    # If bytes are provided, write to temporary file
    if isinstance(lib_path, bytes):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.so') as f:
            f.write(lib_path)
            lib_path = f.name

    loader = RuntimeLibraryLoader(lib_path)
    _lib = loader.lib

    # Create wrapper class with the loaded library
    class _Runtime(Runtime):
        def __init__(self):
            super().__init__(_lib)

    return _Runtime
