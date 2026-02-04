"""
CodeRunner - Simplified test framework for PTO runtime tests.

This module provides a simplified interface for writing runtime tests.
Users only need to provide:
1. A kernels directory with kernel_config.py
2. A golden.py script with generate_inputs() and compute_golden()

Usage:
    # Command line
    python examples/scripts/run_example.py --kernels ./my_test/kernels --golden ./my_test/golden.py

    # In Python
    from code_runner import CodeRunner
    runner = CodeRunner("./kernels", "./golden.py")
    runner.run()

Golden.py interface:
    # Required functions
    def generate_inputs(params: dict) -> dict:
        '''Return dict of numpy arrays (inputs + outputs)'''
        return {"a": np.array(...), "b": np.array(...), "out_f": np.zeros(...)}

    def compute_golden(tensors: dict, params: dict) -> None:
        '''Compute expected outputs in-place'''
        tensors["out_f"][:] = tensors["a"] + tensors["b"]

    # Optional configuration
    PARAMS_LIST = [{"size": 1024}, {"size": 2048}]  # Multiple test cases
    RTOL = 1e-5  # Relative tolerance
    ATOL = 1e-5  # Absolute tolerance
    __outputs__ = ["out_f"]  # Explicit output names (or use 'out_' prefix)
"""

import ctypes
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.testing import assert_allclose


def _has_torch() -> bool:
    """Check if PyTorch is available."""
    try:
        import torch
        return True
    except ImportError:
        return False


def _to_numpy(tensor) -> np.ndarray:
    """Convert tensor to numpy array, handling PyTorch tensors."""
    if hasattr(tensor, 'detach'):
        # PyTorch tensor
        return tensor.detach().cpu().numpy()
    elif hasattr(tensor, '__array__'):
        return np.asarray(tensor)
    return tensor


def _load_module_from_path(module_path: Path, module_name: str):
    """Dynamically load a Python module from file path."""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent  # examples/scripts/ -> examples/ -> simpler/


def _check_ascend_env() -> bool:
    """Check if ASCEND_HOME_PATH environment is set."""
    return bool(os.environ.get("ASCEND_HOME_PATH"))


def _check_pto_isa_root() -> bool:
    """Check if PTO_ISA_ROOT environment is set."""
    return bool(os.environ.get("PTO_ISA_ROOT"))


def _get_device_id() -> int:
    """Get device ID from environment variables."""
    device_id = os.environ.get("PTO_DEVICE_ID")
    if device_id is None:
        device_id = os.environ.get("TILE_FWK_DEVICE_ID", "0")
    return int(device_id)


def _get_pto_isa_clone_path() -> Path:
    """Get the expected path to pto-isa clone."""
    return _get_project_root() / "examples" / "scripts" / "_deps" / "pto-isa"


def _is_pto_isa_cloned() -> bool:
    """
    Check if pto-isa is cloned.

    A clone is considered valid if:
    1. The directory exists
    2. It contains the include directory (essential content)
    """
    clone_path = _get_pto_isa_clone_path()
    if not clone_path.exists():
        return False

    # Check for essential content
    include_dir = clone_path / "include"
    return include_dir.exists() and include_dir.is_dir()


def _is_git_available() -> bool:
    """Check if git command is available."""
    try:
        import subprocess
        result = subprocess.run(
            ["git", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _clone_pto_isa(verbose: bool = False) -> bool:
    """
    Clone pto-isa repository.

    Args:
        verbose: Print detailed progress information

    Returns:
        True if successful, False otherwise
    """
    import subprocess

    if not _is_git_available():
        if verbose:
            print("Warning: git command not available, cannot clone pto-isa")
        return False

    clone_path = _get_pto_isa_clone_path()

    # Create parent deps directory if it doesn't exist
    deps_dir = clone_path.parent
    try:
        deps_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        if verbose:
            print(f"Warning: Failed to create deps directory: {e}")
        return False

    try:
        if verbose:
            print(f"\nCloning pto-isa to {clone_path}...")
            print("This may take a few moments on first run...")

        # Clone with shallow depth for faster download
        result = subprocess.run(
            [
                "git", "clone",
                "--branch", "ci_simpler",
                "--depth", "1",
                "https://gitcode.com/zhangqi-chen/pto-isa.git",
                str(clone_path)
            ],
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes timeout
        )

        if result.returncode != 0:
            if verbose:
                print(f"Warning: Failed to clone pto-isa:\n{result.stderr}")
            return False

        if verbose:
            if result.stdout:
                print(result.stdout)
            print(f"pto-isa cloned successfully to: {clone_path}")

        return True

    except subprocess.TimeoutExpired:
        if verbose:
            print("Warning: Clone operation timed out")
        return False
    except Exception as e:
        if verbose:
            print(f"Warning: Failed to clone pto-isa: {e}")
        return False


def _ensure_pto_isa_root(verbose: bool = False) -> Optional[str]:
    """
    Ensure PTO_ISA_ROOT is set, either from environment or cloned repo.

    This function:
    1. Checks if PTO_ISA_ROOT is already set
    2. If not, tries to clone pto-isa repository
    3. Sets PTO_ISA_ROOT to the clone path

    Args:
        verbose: Print detailed progress information

    Returns:
        PTO_ISA_ROOT path if successful, None otherwise
    """
    # Check if already set in environment
    existing_root = os.environ.get("PTO_ISA_ROOT")
    if existing_root:
        if verbose:
            print(f"Using existing PTO_ISA_ROOT: {existing_root}")
        return existing_root

    # Try to use cloned repository
    clone_path = _get_pto_isa_clone_path()

    # Clone if needed
    if not _is_pto_isa_cloned():
        if verbose:
            print("PTO_ISA_ROOT not set, cloning pto-isa repository...")
        if not _clone_pto_isa(verbose=verbose):
            if verbose:
                print("\nFailed to automatically clone pto-isa.")
                print("You can manually clone it with:")
                print(f"  mkdir -p {clone_path.parent}")
                print(f"  git clone --branch ci_simpler https://gitcode.com/zhangqi-chen/pto-isa.git {clone_path}")
                print(f"Or set PTO_ISA_ROOT to an existing pto-isa installation:")
                print(f"  export PTO_ISA_ROOT=/path/to/pto-isa")
            return None

    # Verify clone has expected content
    include_dir = clone_path / "include"
    if not include_dir.exists():
        if verbose:
            print(f"Warning: pto-isa cloned but missing include directory: {include_dir}")
        return None

    # Set environment variable
    pto_isa_root = str(clone_path.resolve())
    os.environ["PTO_ISA_ROOT"] = pto_isa_root

    if verbose:
        print(f"Set PTO_ISA_ROOT to: {pto_isa_root}")

    return pto_isa_root


class CodeRunner:
    """
    Simplified test runner that loads kernel config and golden script.

    This class automates:
    - Loading kernel_config.py and golden.py dynamically
    - Building func_args automatically from numpy arrays
    - Converting PyTorch tensors to numpy
    - Separating inputs and outputs based on naming convention
    - Running the full test flow

    Args:
        kernels_dir: Path to kernels directory containing kernel_config.py
        golden_path: Path to golden.py script
        runtime_name: Runtime implementation name (default: "rt2")
        device_id: Device ID (defaults to PTO_DEVICE_ID env var or 0)
        platform: Platform name ("a2a3" for hardware, "a2a3sim" for simulation, default: "a2a3")
        use_device_orchestration: If True (rt2 only), orchestration runs on AICPU thread 3; host does not build graph. Deprecated in favor of orchestrator_location.
        orchestrator_location: For rt2, "host_cpu" (orchestration on host, 3 AICPU threads) or "device_aicpu" (orchestration on AICPU thread 3, 4 threads). Ignored for non-rt2.
        keep_artifacts_dir: If set, compiled .o and .so are written to this directory and not deleted (for debug).
    """

    def __init__(
        self,
        kernels_dir: str,
        golden_path: str,
        runtime_name: str = "rt2",
        device_id: Optional[int] = None,
        platform: str = "a2a3",
        use_device_orchestration: bool = False,
        orchestrator_location: Optional[str] = None,
        keep_artifacts_dir: Optional[str] = None,
    ):
        self.kernels_dir = Path(kernels_dir).resolve()
        self.golden_path = Path(golden_path).resolve()
        self.runtime_name = runtime_name
        self.platform = platform
        self.project_root = _get_project_root()
        if keep_artifacts_dir:
            p = Path(keep_artifacts_dir)
            self.keep_artifacts_dir = (self.project_root / p).resolve() if not p.is_absolute() else p.resolve()
        else:
            self.keep_artifacts_dir = None

        # Orchestrator: explicit choice or legacy use_device_orchestration
        if orchestrator_location is not None:
            if orchestrator_location not in ("host_cpu", "device_aicpu"):
                raise ValueError(f"orchestrator_location must be 'host_cpu' or 'device_aicpu', got {orchestrator_location!r}")
            self.orchestrator_location = orchestrator_location
        else:
            self.orchestrator_location = "device_aicpu" if use_device_orchestration else "host_cpu"
        if runtime_name == "rt2":
            self.use_device_orchestration = self.orchestrator_location == "device_aicpu"
            self.run_orchestrator_on_host = self.orchestrator_location == "host_cpu"
        else:
            self.use_device_orchestration = False
            self.run_orchestrator_on_host = False

        # Resolve device ID
        if device_id is None:
            device_id = _get_device_id()
        self.device_id = device_id

        # Load configurations
        self._kernel_config = self._load_kernel_config()
        self._golden_module = self._load_golden_module()

        # Extract kernel configuration
        self.kernels = self._kernel_config.KERNELS
        self.orchestration = self._kernel_config.ORCHESTRATION

        # Extract golden configuration
        self.params_list = getattr(self._golden_module, 'PARAMS_LIST', [{}])
        self.rtol = getattr(self._golden_module, 'RTOL', 1e-5)
        self.atol = getattr(self._golden_module, 'ATOL', 1e-5)
        self.output_names = getattr(self._golden_module, '__outputs__', None)
        self.tensor_order = getattr(self._golden_module, 'TENSOR_ORDER', None)

        # Runtime configuration: rt2 host_cpu -> 3 AICPU threads; rt2 device_aicpu -> 4; non-rt2 -> 3
        if runtime_name == "rt2":
            self.aicpu_thread_num = 3 if self.run_orchestrator_on_host else 4
        else:
            self.aicpu_thread_num = 3
        self.block_dim = 3

    def _load_kernel_config(self):
        """Load kernel_config.py from kernels directory."""
        config_path = self.kernels_dir / "kernel_config.py"
        if not config_path.exists():
            raise FileNotFoundError(
                f"kernel_config.py not found in {self.kernels_dir}\n"
                f"Expected: {config_path}"
            )
        return _load_module_from_path(config_path, f"kernel_config_{id(self)}")

    def _load_golden_module(self):
        """Load golden.py script."""
        if not self.golden_path.exists():
            raise FileNotFoundError(f"Golden script not found: {self.golden_path}")

        module = _load_module_from_path(self.golden_path, f"golden_{id(self)}")

        # Validate required functions
        if not hasattr(module, 'generate_inputs'):
            raise AttributeError(
                f"golden.py must define generate_inputs(params) function\n"
                f"File: {self.golden_path}"
            )
        if not hasattr(module, 'compute_golden'):
            raise AttributeError(
                f"golden.py must define compute_golden(tensors, params) function\n"
                f"File: {self.golden_path}"
            )

        return module

    def _identify_outputs(self, tensors: Dict[str, np.ndarray]) -> Tuple[Dict, Dict]:
        """
        Separate inputs and outputs from tensor dict.

        Uses either explicit __outputs__ list or 'out_' prefix convention.

        Returns:
            Tuple of (inputs_dict, outputs_dict)
        """
        if self.output_names:
            # Use explicit output names
            outputs = {k: v for k, v in tensors.items() if k in self.output_names}
            inputs = {k: v for k, v in tensors.items() if k not in self.output_names}
        else:
            # Use 'out_' prefix convention
            outputs = {k: v for k, v in tensors.items() if k.startswith('out_')}
            inputs = {k: v for k, v in tensors.items() if not k.startswith('out_')}

        if not outputs:
            raise ValueError(
                "No output tensors identified. Either:\n"
                "1. Define __outputs__ = ['tensor_name'] in golden.py, or\n"
                "2. Use 'out_' prefix for output tensor names (e.g., 'out_result')"
            )

        return inputs, outputs

    def _build_func_args(self, tensors: Dict[str, np.ndarray]) -> List[int]:
        """
        Build func_args from tensors automatically.

        Convention for orchestration function signature:
            int BuildGraph(Runtime* runtime, uint64_t* args, int arg_count)

        Where args layout is:
            [ptr_0, ptr_1, ..., ptr_n, nbytes_0, nbytes_1, ..., nbytes_n, count]

        Args:
            tensors: Dict of numpy arrays

        Returns:
            List of func_args values (pointers, sizes, count)
        """
        # Determine tensor order
        if self.tensor_order:
            order = self.tensor_order
        else:
            order = list(tensors.keys())

        ptrs = []
        sizes = []

        for name in order:
            if name not in tensors:
                raise KeyError(
                    f"Tensor '{name}' from TENSOR_ORDER not found in generate_inputs() result.\n"
                    f"Available tensors: {list(tensors.keys())}"
                )
            arr = tensors[name]
            ptrs.append(arr.ctypes.data)
            sizes.append(arr.nbytes)

        # Get element count from first tensor
        count = tensors[order[0]].size

        return ptrs + sizes + [count]

    # Shared memory size for PTO2 device orchestration (skeleton uses 16384 task window, 65536 dep pool)
    PTO2_SM_SIZE = 4 * 1024 * 1024  # 4 MiB safe upper bound

    def _build_func_args_device_orchestration(
        self, tensors: Dict[str, np.ndarray]
    ) -> Tuple[List[int], List[int]]:
        """
        Build func_args for device orchestration: host allocates device memory and
        passes device pointers. Layout: [dev_a, dev_b, dev_f, size_a, size_b, size_f,
        SIZE, dev_c, dev_d, dev_e]. Also allocates PTO2 shared memory.

        Returns:
            (func_args list, list of device pointers to free after finalize)

        Note: Device memory allocation and copy-to-device are now handled by
        init_runtime_impl() in C++. Python only passes host pointers and sizes.
        """
        import ctypes
        from bindings import device_malloc, device_free

        order = self.tensor_order if self.tensor_order else list(tensors.keys())
        if len(order) < 3:
            raise ValueError(
                "Device orchestration expects at least 3 tensors (e.g. a, b, f). "
                f"Got TENSOR_ORDER: {order}"
            )
        a_name, b_name, f_name = order[0], order[1], order[2]
        for name in (a_name, b_name, f_name):
            if name not in tensors:
                raise KeyError(f"Tensor '{name}' not in tensors: {list(tensors.keys())}")
        a_arr = tensors[a_name]
        b_arr = tensors[b_name]
        f_arr = tensors[f_name]
        size_a = a_arr.nbytes
        size_b = b_arr.nbytes
        size_f = f_arr.nbytes
        SIZE = int(a_arr.size)

        # Get host pointers
        host_a = int(a_arr.ctypes.data)
        host_b = int(b_arr.ctypes.data)
        host_f = int(f_arr.ctypes.data)

        # Allocate shared memory buffer (still needed for orchestrator state)
        sm_ptr = device_malloc(self.PTO2_SM_SIZE)
        if sm_ptr is None:
            raise RuntimeError("device_malloc failed for PTO2 shared memory")
        sm_ptr_i = int(sm_ptr)

        # Pass host pointers and sizes - C++ will handle device allocation
        # Layout: [host_a, host_b, host_f, size_a, size_b, size_f, SIZE]
        func_args = [
            host_a,
            host_b,
            host_f,
            size_a,
            size_b,
            size_f,
            SIZE,
        ]
        # Only sm_ptr needs to be freed by Python; device tensors are managed by C++
        to_free = [sm_ptr_i]
        return func_args, to_free

    def _build_func_args_host_orchestration(
        self, tensors: Dict[str, np.ndarray]
    ) -> Tuple[List[int], List[int]]:
        """
        Build func_args for host orchestration (rt2 run_orchestrator_on_host).

        Note: host_orchestration_entry expects device pointers, so we allocate
        device memory and copy data here (same as before).

        Returns:
            (func_args list, list of device pointers to free: dev_a, dev_b, dev_c, dev_d, dev_e)
        """
        import ctypes
        from bindings import device_malloc, device_free, copy_to_device, copy_from_device

        order = self.tensor_order if self.tensor_order else list(tensors.keys())
        if len(order) < 3:
            raise ValueError(
                "Host orchestration expects at least 3 tensors (e.g. a, b, f). "
                f"Got TENSOR_ORDER: {order}"
            )
        a_name, b_name, f_name = order[0], order[1], order[2]
        for name in (a_name, b_name, f_name):
            if name not in tensors:
                raise KeyError(f"Tensor '{name}' not in tensors: {list(tensors.keys())}")
        a_arr = tensors[a_name]
        b_arr = tensors[b_name]
        f_arr = tensors[f_name]
        size_a = a_arr.nbytes
        size_b = b_arr.nbytes
        size_f = f_arr.nbytes
        SIZE = int(a_arr.size)
        BYTES = SIZE * 4  # float32

        dev_a = device_malloc(size_a)
        dev_b = device_malloc(size_b)
        dev_f = device_malloc(size_f)
        dev_c = device_malloc(BYTES)
        dev_d = device_malloc(BYTES)
        dev_e = device_malloc(BYTES)
        if any(p is None for p in (dev_a, dev_b, dev_f, dev_c, dev_d, dev_e)):
            for p in (dev_a, dev_b, dev_f, dev_c, dev_d, dev_e):
                if p is not None:
                    device_free(int(p))
            raise RuntimeError("device_malloc failed for host orchestration")

        dev_a_i = int(dev_a)
        dev_b_i = int(dev_b)
        dev_f_i = int(dev_f)
        dev_c_i = int(dev_c)
        dev_d_i = int(dev_d)
        dev_e_i = int(dev_e)

        copy_to_device(dev_a_i, a_arr.ctypes.data, size_a)
        copy_to_device(dev_b_i, b_arr.ctypes.data, size_b)

        func_args = [
            dev_a_i,
            dev_b_i,
            dev_f_i,
            size_a,
            size_b,
            size_f,
            SIZE,
            dev_c_i,
            dev_d_i,
            dev_e_i,
        ]
        to_free = [dev_a_i, dev_b_i, dev_c_i, dev_d_i, dev_e_i]
        return func_args, to_free

    def skip_if_no_env(self) -> None:
        """Raise error if required environment is not available."""
        if not _check_ascend_env():
            raise EnvironmentError("ASCEND_HOME_PATH not set")
        if not _check_pto_isa_root():
            raise EnvironmentError(
                "PTO_ISA_ROOT environment variable is not set.\n"
                "Please set it to the PTO-ISA root directory, e.g.:\n"
                "  export PTO_ISA_ROOT=$(pwd)/examples/scripts/_deps/pto-isa"
            )

    def run(self) -> None:
        """
        Execute the full test flow:
        1. Check environment
        2. Build runtime
        3. Load runtime and set device
        4. Compile orchestration
        5. Compile and register kernels
        6. For each params in params_list:
           - Generate inputs using golden.py
           - Initialize and launch runtime
           - Finalize and compare with golden
        """
        # Import runtime modules (deferred to allow skip_if_no_env to work)
        from runtime_builder import RuntimeBuilder
        from bindings import (
            bind_host_binary,
            register_kernel,
            set_device,
            launch_runtime,
            device_malloc,
            device_free,
            copy_to_device,
            copy_from_device,
        )
        from elf_parser import extract_text_section

        # Auto-setup PTO_ISA_ROOT if needed (for all platforms, since kernels may use PTO ISA headers)
        pto_isa_root = _ensure_pto_isa_root(verbose=True)
        if pto_isa_root is None:
            print("Warning: Could not auto-setup PTO_ISA_ROOT")
            print("         If kernels use PTO ISA headers, they may fail to compile")

        # Check platform-specific environment (only for a2a3 hardware platform)
        if self.platform == "a2a3":
            self.skip_if_no_env()

        # Optional: directory to keep compiled artifacts for debug
        if self.keep_artifacts_dir is not None:
            self.keep_artifacts_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n=== Keeping build artifacts in: {self.keep_artifacts_dir} ===")

        # Step 1: Build runtime
        print(f"\n=== Building Runtime: {self.runtime_name} (platform: {self.platform}) ===")
        builder = RuntimeBuilder(runtime_root=self.project_root, platform=self.platform)
        pto_compiler = builder.get_pto_compiler()
        try:
            host_binary, aicpu_binary, aicore_binary = builder.build(self.runtime_name)
        except Exception as e:
            raise RuntimeError(
                f"Failed to build runtime '{self.runtime_name}' for platform '{self.platform}'.\n"
                f"Error: {e}"
            ) from e

        if self.keep_artifacts_dir is not None:
            (self.keep_artifacts_dir / "libhost_runtime.so").write_bytes(host_binary)
            (self.keep_artifacts_dir / "libaicpu_kernel.so").write_bytes(aicpu_binary)
            (self.keep_artifacts_dir / "aicore_kernel.bin").write_bytes(aicore_binary)
            print(f"  Saved runtime binaries to {self.keep_artifacts_dir}")

        # Step 2: Load runtime and set device
        print(f"\n=== Loading Runtime ({len(host_binary)} bytes) ===")
        Runtime = bind_host_binary(host_binary)

        print(f"\n=== Setting Device {self.device_id} ===")
        set_device(self.device_id)

        # Step 3: Compile orchestration
        print("\n=== Compiling Orchestration ===")

        # Build include directories for orchestration
        orch_include_dirs = [
            str(self.project_root / "src" / "runtime" / self.runtime_name / "runtime"),
        ] + pto_compiler.get_platform_include_dirs()

        if self.use_device_orchestration:
            # Device orchestration: compile for AICPU
            orch_so_binary = pto_compiler.compile_orchestration(
                self.orchestration["source"],
                extra_include_dirs=orch_include_dirs,
                target="device",
            )
            self._orch_func_name = self.orchestration["function_name"]
            print(f"Compiled device orchestration: {len(orch_so_binary)} bytes")
            if self.keep_artifacts_dir is not None:
                (self.keep_artifacts_dir / "device_orchestration.so").write_bytes(orch_so_binary)
                print(f"  Saved device_orchestration.so to {self.keep_artifacts_dir}")
        else:
            # Host orchestration: compile for host CPU
            # Use host-specific source/function if available
            host_source = self.orchestration.get("host_source", self.orchestration["source"])
            orch_so_binary = pto_compiler.compile_orchestration(
                host_source,
                extra_include_dirs=orch_include_dirs,
                target="host",
            )
            self._orch_func_name = self.orchestration.get("host_function_name", self.orchestration["function_name"])
            print(f"Compiled host orchestration: {len(orch_so_binary)} bytes")
            if self.keep_artifacts_dir is not None:
                (self.keep_artifacts_dir / "orchestration.so").write_bytes(orch_so_binary)
                print(f"  Saved orchestration.so to {self.keep_artifacts_dir}")

        # Step 4: Compile and register kernels
        print("\n=== Compiling and Registering Kernels ===")

        # Get PTO_ISA_ROOT (use default for sim platform)
        pto_isa_root = os.environ.get("PTO_ISA_ROOT", "/tmp/unused")

        for kernel in self.kernels:
            print(f"Compiling kernel: {kernel['source']} (func_id={kernel['func_id']})")
            incore_o = pto_compiler.compile_incore(
                kernel["source"],
                core_type=kernel["core_type"],
                pto_isa_root=pto_isa_root,
            )
            if self.keep_artifacts_dir is not None:
                fid = kernel["func_id"]
                (self.keep_artifacts_dir / f"kernel_func_id_{fid}.o").write_bytes(incore_o)
            kernel_bin = extract_text_section(incore_o)
            register_kernel(kernel["func_id"], kernel_bin)

        if self.keep_artifacts_dir is not None:
            print(f"  Saved incore .o files to {self.keep_artifacts_dir}")
        print("All kernels compiled and registered")

        # Step 5: Run each parameter set
        total_cases = len(self.params_list)
        for case_idx, params in enumerate(self.params_list):
            print(f"\n{'='*60}")
            print(f"=== Case {case_idx + 1}/{total_cases}: {params} ===")
            print(f"{'='*60}")

            # Generate tensors using golden.py
            print("\n=== Generating Inputs ===")
            tensors = self._golden_module.generate_inputs(params)

            # Convert any PyTorch tensors to numpy
            tensors = {k: _to_numpy(v) for k, v in tensors.items()}

            # Identify inputs and outputs
            inputs, outputs = self._identify_outputs(tensors)
            print(f"Inputs: {list(inputs.keys())}")
            print(f"Outputs: {list(outputs.keys())}")

            # Build func_args and optionally allocate device memory (device or host orchestration)
            device_ptrs_to_free: List[int] = []
            if self.run_orchestrator_on_host:
                func_args, device_ptrs_to_free = self._build_func_args_host_orchestration(
                    tensors
                )
            elif self.use_device_orchestration:
                func_args, device_ptrs_to_free = self._build_func_args_device_orchestration(
                    tensors
                )
            else:
                func_args = self._build_func_args(tensors)

            # Determine actual tensor order for debugging
            order = self.tensor_order if self.tensor_order else list(tensors.keys())
            print(f"Tensor order: {order}")
            print(f"func_args count: {len(func_args)}")

            # Create and initialize runtime
            print("\n=== Initializing Runtime ===")
            if self.run_orchestrator_on_host:
                print("Mode: host orchestration (graph built on host CPU, 3 AICPU threads)")
            elif self.use_device_orchestration:
                print("Mode: device orchestration (AICPU thread 3 will build graph)")
            runtime = Runtime()

            # Use orchestration function name (set during compilation)
            orch_func_name = self._orch_func_name

            runtime.initialize(
                orch_so_binary,
                orch_func_name,
                func_args,
                use_device_orchestration=self.use_device_orchestration,
                run_orchestrator_on_host=self.run_orchestrator_on_host,
            )
            if self.run_orchestrator_on_host:
                # Allocate PTO2 SM via runtime; run host orchestration into host mirror then copy to device
                runtime.allocate_pto2_shared_memory()
                sm_size = runtime.get_pto2_sm_size()
                if sm_size <= 0:
                    raise RuntimeError("get_pto2_sm_size returned invalid size")
                host_mirror = (ctypes.c_uint8 * sm_size)()
                runtime.run_host_orchestration(ctypes.addressof(host_mirror))
                # Note: record_tensor_pair is called inside example_orch.cpp
            elif self.use_device_orchestration:
                # Set GM shared memory for AICPU thread 3
                # Note: record_tensor_pair and device memory allocation are handled by init_runtime_impl
                sm_ptr = device_ptrs_to_free[0]  # First (and only) item is sm_ptr
                runtime.set_pto2_gm_sm_ptr(sm_ptr)

            # Launch runtime
            print("\n=== Launching Runtime ===")
            print(f"Device ID: {self.device_id}")
            print(f"AICPU threads: {self.aicpu_thread_num}, Block dim: {self.block_dim}")
            import sys
            sys.stdout.flush()  # Ensure output is visible before potential hang

            # On a2a3, C++ writes libaicpu_kernel.so to getcwd(); use project_root so CANN can find it.
            prev_cwd = None
            if self.platform == "a2a3":
                prev_cwd = os.getcwd()
                os.chdir(self.project_root)

            try:
                launch_runtime(
                    runtime,
                    aicpu_thread_num=self.aicpu_thread_num,
                    block_dim=self.block_dim,
                    device_id=self.device_id,
                    aicpu_binary=aicpu_binary,
                    aicore_binary=aicore_binary,
                )
            except Exception as e:
                print("try launch runtime failed")
            finally:
                if prev_cwd is not None:
                    os.chdir(prev_cwd)

            print("Launch completed successfully")  # Will only print if not hung

            # Finalize
            print("\n=== Finalizing Runtime ===")
            runtime.finalize()

            if os.environ.get("PTO2_DEBUG_TENSOR") and (self.use_device_orchestration or self.run_orchestrator_on_host) and outputs:
                out_name = list(outputs.keys())[0]
                arr = tensors[out_name]
                print("[Host output after copy-back] %s first16=%s" % (out_name, arr.tobytes()[:16].hex()))

            # Free device buffers allocated for device orchestration (dev_f already freed in finalize)
            # for ptr in device_ptrs_to_free:
            #     device_free(ptr)

            # Compute golden and compare
            print("\n=== Comparing Results ===")
            self._compare_with_golden(tensors, inputs, outputs, params)

            print(f"\n=== Case {case_idx + 1}/{total_cases} Passed ===")

        print(f"\n{'='*60}")
        print(f"=== All {total_cases} cases passed ===")
        print(f"{'='*60}")

    def _compare_with_golden(
        self,
        tensors: Dict[str, np.ndarray],
        inputs: Dict[str, np.ndarray],
        outputs: Dict[str, np.ndarray],
        params: Dict[str, Any],
    ) -> None:
        """Compare outputs with golden values."""
        # Create copies for golden computation
        golden_outputs = {k: v.copy() for k, v in outputs.items()}
        golden_tensors = {**inputs, **golden_outputs}

        # Compute golden
        self._golden_module.compute_golden(golden_tensors, params)

        # Compare each output
        for name in outputs:
            actual = outputs[name]
            expected = golden_outputs[name]
            print(f"Comparing {name}: shape={actual.shape}, dtype={actual.dtype}")

            # Show first 10 values
            if actual.size > 0:
                flat_actual = actual.flatten()
                flat_expected = expected.flatten()
                n_show = min(10, flat_actual.size)
                print(f"  First {n_show} actual:   {flat_actual[:n_show]}")
                print(f"  First {n_show} expected: {flat_expected[:n_show]}")

            assert_allclose(
                actual,
                expected,
                rtol=self.rtol,
                atol=self.atol,
                err_msg=f"Output '{name}' does not match golden",
            )
            matched = np.sum(np.isclose(actual, expected, rtol=self.rtol, atol=self.atol))
            print(f"  {name}: PASS ({matched}/{actual.size} elements matched)")
