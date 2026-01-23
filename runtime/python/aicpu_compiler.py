#!/usr/bin/env python3
"""
AICPU Compiler Module

This module provides functionality to compile AICPU code (C++ sources) into
a shared object (.so) file. It passes the following paths as CMake variables:
- Base platform code path: Defines where the base AICPU platform code is located
- Custom code path: Defines where the customized AICPU code is located
- Toolchain path: Defines where the ASCEND compilation toolchain is located

The CMakeLists.txt decides how to use these paths (e.g., include directories,
source files, how to merge/override code, etc.)
"""

import os
import sys
import shutil
import tempfile
import subprocess
import logging
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class CompileConfig:
    """Configuration for AICPU compilation"""
    base_platform_path: str
    custom_code_path: str
    toolchain_path: str
    output_dir: str
    cmake_file: str
    build_dir: Optional[str] = None
    verbose: bool = False


class AICPUCompiler:
    """Compiler for AICPU code to generate shared object files"""

    # Expected environment variables
    ASCEND_HOME_ENV = "ASCEND_HOME_PATH"
    ASCEND_CANN_ENV = "ASCEND_CANN_PACKAGE_PATH"

    def __init__(self, config: CompileConfig):
        """
        Initialize AICPU compiler with configuration

        Args:
            config: CompileConfig object with all necessary paths
        """
        self.config = config
        self.work_dir = None
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate all configuration paths exist"""
        errors = []

        # Validate base platform path
        if not Path(self.config.base_platform_path).is_dir():
            errors.append(f"Base platform path does not exist: {self.config.base_platform_path}")

        # Validate custom code path
        if not Path(self.config.custom_code_path).is_dir():
            errors.append(f"Custom code path does not exist: {self.config.custom_code_path}")

        # Validate toolchain path
        if not Path(self.config.toolchain_path).is_dir():
            errors.append(f"Toolchain path does not exist: {self.config.toolchain_path}")

        # Validate output directory can be created/accessed
        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        if not output_path.is_dir():
            errors.append(f"Cannot create/access output directory: {self.config.output_dir}")

        # Validate CMakeLists.txt exists
        if not Path(self.config.cmake_file).is_file():
            errors.append(f"CMakeLists.txt not found: {self.config.cmake_file}")

        if errors:
            error_msg = "\n".join(errors)
            logger.error(f"Configuration validation failed:\n{error_msg}")
            raise ValueError(f"Invalid configuration:\n{error_msg}")

        logger.info("Configuration validated successfully")


    def _setup_cmake_environment(self) -> dict:
        """
        Setup environment variables for CMake compilation

        Returns:
            Environment dictionary with required variables set
        """
        env = os.environ.copy()

        # Set CMake toolchain variables
        # The toolchain path should point to the ASCEND_CANN_PACKAGE_PATH directory
        env[self.ASCEND_CANN_ENV] = self.config.toolchain_path
        env[self.ASCEND_HOME_ENV] = self.config.toolchain_path

        if self.config.verbose:
            logger.info(f"CMake environment: {self.ASCEND_CANN_ENV}={self.config.toolchain_path}")

        return env

    def _run_cmake(self, source_dir: str, build_dir: str, env: dict) -> bool:
        """
        Run CMake configuration

        Args:
            source_dir: Directory containing CMakeLists.txt
            build_dir: Build output directory
            env: Environment variables

        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Running CMake configuration...")

        # Create build directory
        Path(build_dir).mkdir(parents=True, exist_ok=True)

        # Convert paths to absolute paths
        base_platform_path = os.path.abspath(self.config.base_platform_path)
        custom_code_path = os.path.abspath(self.config.custom_code_path)
        toolchain_path = os.path.abspath(self.config.toolchain_path)

        # Run cmake configure with path variables
        cmake_cmd = [
            "cmake",
            "-B", build_dir,
            "-S", source_dir,
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DBASE_PLATFORM_PATH={base_platform_path}",
            f"-DCUSTOM_CODE_PATH={custom_code_path}",
            f"-DTOOLCHAIN_PATH={toolchain_path}"
        ]

        if self.config.verbose:
            cmake_cmd.append("-DCMAKE_VERBOSE_MAKEFILE=ON")

        logger.info(f"CMake configuration parameters:")
        logger.info(f"  BASE_PLATFORM_PATH={base_platform_path}")
        logger.info(f"  CUSTOM_CODE_PATH={custom_code_path}")
        logger.info(f"  TOOLCHAIN_PATH={toolchain_path}")
        logger.info(f"CMake command: {' '.join(cmake_cmd)}")

        try:
            result = subprocess.run(
                cmake_cmd,
                env=env,
                cwd=source_dir,
                capture_output=False,
                text=True,
                check=False
            )

            if result.returncode != 0:
                logger.error(f"CMake configuration failed with return code {result.returncode}")
                return False

            logger.info("CMake configuration completed successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to run CMake: {e}")
            return False

    def _run_cmake_build(self, build_dir: str, env: dict) -> bool:
        """
        Run CMake build to compile the code

        Args:
            build_dir: Build directory
            env: Environment variables

        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Building AICPU code in {build_dir}...")

        build_cmd = [
            "cmake",
            "--build", build_dir,
            "--config", "Release",
            "-j", str(os.cpu_count() or 4)
        ]

        if self.config.verbose:
            build_cmd.append("--verbose")

        logger.info(f"Build command: {' '.join(build_cmd)}")

        try:
            result = subprocess.run(
                build_cmd,
                env=env,
                capture_output=False,
                text=True,
                check=False
            )

            if result.returncode != 0:
                logger.error(f"Build failed with return code {result.returncode}")
                return False

            logger.info("Build completed successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to run build: {e}")
            return False

    def _find_so_files(self, build_dir: str) -> list:
        """
        Find compiled .so files in build directory

        Args:
            build_dir: Build directory to search

        Returns:
            List of paths to .so files
        """
        so_files = []
        for root, dirs, files in os.walk(build_dir):
            for file in files:
                if file.endswith('.so') or '.so.' in file:
                    so_files.append(os.path.join(root, file))

        return so_files

    def _copy_so_to_output(self, build_dir: str) -> list:
        """
        Copy compiled .so files to output directory

        Args:
            build_dir: Build directory containing compiled binaries

        Returns:
            List of output .so file paths
        """
        so_files = self._find_so_files(build_dir)

        if not so_files:
            logger.warning("No .so files found in build directory")
            return []

        output_files = []
        for so_file in so_files:
            filename = os.path.basename(so_file)
            output_path = os.path.join(self.config.output_dir, filename)

            try:
                shutil.copy2(so_file, output_path)
                logger.info(f"Copied {filename} to output directory")
                output_files.append(output_path)
            except Exception as e:
                logger.error(f"Failed to copy {so_file}: {e}")

        return output_files

    def compile(self) -> Tuple[bool, Optional[list]]:
        """
        Compile AICPU code to generate .so file

        Process:
        1. Setup CMake environment with toolchain
        2. Pass base platform path, custom code path, and toolchain path as CMake variables
        3. Run CMake configuration (CMakeLists.txt decides how to use these paths)
        4. Run CMake build
        5. Copy output .so files to output directory

        Returns:
            Tuple of (success: bool, output_files: list or None)
        """
        try:
            # Get directory containing CMakeLists.txt
            cmake_dir = os.path.dirname(os.path.abspath(self.config.cmake_file))
            logger.info(f"Using CMakeLists.txt from: {cmake_dir}")

            # Step 1: Setup environment
            env = self._setup_cmake_environment()

            # Step 2: Create build directory
            if self.config.build_dir:
                build_dir = self.config.build_dir
            else:
                # Create a temporary build directory
                build_dir = tempfile.mkdtemp(prefix="aicpu_build_")
                logger.info(f"Created build directory: {build_dir}")

            try:
                # Step 3: Run CMake configuration
                # The base_platform_path, custom_code_path, and toolchain_path
                # are passed as CMake variables for the CMakeLists.txt to use
                if not self._run_cmake(cmake_dir, build_dir, env):
                    return False, None

                # Step 4: Run CMake build
                if not self._run_cmake_build(build_dir, env):
                    return False, None

                # Step 5: Copy output .so files
                output_files = self._copy_so_to_output(build_dir)

                if output_files:
                    logger.info(f"Compilation successful. Output files: {output_files}")
                    return True, output_files
                else:
                    logger.warning("Compilation completed but no .so files were generated")
                    return False, None

            finally:
                # Clean up temporary build directory if we created one
                if not self.config.build_dir and os.path.exists(build_dir):
                    try:
                        shutil.rmtree(build_dir)
                        logger.info(f"Cleaned up build directory: {build_dir}")
                    except Exception as e:
                        logger.warning(f"Failed to clean up build directory: {e}")

        except Exception as e:
            logger.error(f"Compilation failed: {e}")
            return False, None


def compile_aicpu(
    base_platform_path: str,
    custom_code_path: str,
    toolchain_path: str,
    output_dir: str,
    cmake_file: str,
    verbose: bool = False
) -> Tuple[bool, Optional[list]]:
    """
    Convenience function to compile AICPU code

    This function passes the base platform path, custom code path, and toolchain path
    as CMake variables to the CMakeLists.txt. The CMakeLists.txt is responsible for
    deciding how to use these paths (e.g., which source files to compile, include
    directories, how custom code overrides base code, etc.)

    CMake variables passed:
        - BASE_PLATFORM_PATH: Path to base AICPU platform code
        - CUSTOM_CODE_PATH: Path to custom AICPU code
        - TOOLCHAIN_PATH: Path to ASCEND toolchain

    Environment variables set:
        - ASCEND_HOME_PATH: Set to toolchain_path
        - ASCEND_CANN_PACKAGE_PATH: Set to toolchain_path

    Args:
        base_platform_path: Path to base AICPU platform code
        custom_code_path: Path to custom AICPU code
        toolchain_path: Path to ASCEND toolchain (ASCEND_CANN_PACKAGE_PATH)
        output_dir: Output directory for .so files
        cmake_file: Path to CMakeLists.txt
        verbose: Enable verbose output

    Returns:
        Tuple of (success: bool, output_files: list or None)

    Example:
        success, output_files = compile_aicpu(
            base_platform_path="/path/to/base_code",
            custom_code_path="/path/to/custom_code",
            toolchain_path="/opt/ascend/toolkit",
            output_dir="/output",
            cmake_file="/path/to/CMakeLists.txt"
        )
        if success:
            print(f"Generated: {output_files}")
        else:
            print("Compilation failed")
    """
    config = CompileConfig(
        base_platform_path=base_platform_path,
        custom_code_path=custom_code_path,
        toolchain_path=toolchain_path,
        output_dir=output_dir,
        cmake_file=cmake_file,
        verbose=verbose
    )

    compiler = AICPUCompiler(config)
    return compiler.compile()


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(
        description="Compile AICPU code to generate shared object (.so) files"
    )
    parser.add_argument(
        "--base-platform-path",
        required=True,
        help="Path to base AICPU platform code"
    )
    parser.add_argument(
        "--custom-code-path",
        required=True,
        help="Path to custom AICPU code (will override base platform code)"
    )
    parser.add_argument(
        "--toolchain-path",
        required=True,
        help="Path to ASCEND toolchain (ASCEND_CANN_PACKAGE_PATH)"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for generated .so files"
    )
    parser.add_argument(
        "--cmake-file",
        required=True,
        help="Path to CMakeLists.txt"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    success, output_files = compile_aicpu(
        base_platform_path=args.base_platform_path,
        custom_code_path=args.custom_code_path,
        toolchain_path=args.toolchain_path,
        output_dir=args.output_dir,
        cmake_file=args.cmake_file,
        verbose=args.verbose
    )

    if success:
        print(f"\nCompilation successful!")
        print(f"Output files:")
        for f in output_files:
            print(f"  - {f}")
        sys.exit(0)
    else:
        print(f"\nCompilation failed. Check logs above for details.")
        sys.exit(1)
