cmake_minimum_required(VERSION 3.16.3)

project(aicpu_graph_kernel LANGUAGES C CXX)

# ==============================================================================
# CMake variables passed by aicpu_compiler.py:
#   - BASE_PLATFORM_PATH: Path to base AICPU platform code (runtime/aicpu)
#   - CUSTOM_CODE_PATH: Path to custom AICPU code (e.g., runtime/example or runtime/graph)
#   - TOOLCHAIN_PATH: Path to ASCEND toolchain (optional, defaults to ASCEND_HOME_PATH env var)
# ==============================================================================

# Validate required CMake variables
if(NOT DEFINED BASE_PLATFORM_PATH)
    message(FATAL_ERROR "BASE_PLATFORM_PATH must be set by aicpu_compiler.py")
endif()

if(NOT DEFINED CUSTOM_CODE_PATH)
    message(FATAL_ERROR "CUSTOM_CODE_PATH must be set by aicpu_compiler.py")
endif()

# Handle TOOLCHAIN_PATH - use provided value or fall back to ASCEND_HOME_PATH environment variable
if(NOT DEFINED TOOLCHAIN_PATH)
    if(DEFINED ENV{ASCEND_HOME_PATH})
        set(TOOLCHAIN_PATH $ENV{ASCEND_HOME_PATH})
        message(STATUS "TOOLCHAIN_PATH not set, using ASCEND_HOME_PATH: ${TOOLCHAIN_PATH}")
    else()
        message(FATAL_ERROR "TOOLCHAIN_PATH not set and ASCEND_HOME_PATH environment variable not found")
    endif()
endif()

# Set ASCEND_CANN_PACKAGE_PATH from TOOLCHAIN_PATH
set(ASCEND_CANN_PACKAGE_PATH ${TOOLCHAIN_PATH})

# Set compilers from toolchain
set(CMAKE_C_COMPILER "${ASCEND_CANN_PACKAGE_PATH}/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-gcc")
set(CMAKE_CXX_COMPILER "${ASCEND_CANN_PACKAGE_PATH}/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-g++")

message(STATUS "========== AICPU Compilation Configuration ==========")
message(STATUS "BASE_PLATFORM_PATH: ${BASE_PLATFORM_PATH}")
message(STATUS "CUSTOM_CODE_PATH: ${CUSTOM_CODE_PATH}")
message(STATUS "TOOLCHAIN_PATH: ${TOOLCHAIN_PATH}")
message(STATUS "=====================================================")

# ==============================================================================
# Collect source files from base platform and custom code paths
# ==============================================================================

# Collect base platform sources
file(GLOB_RECURSE BASE_PLATFORM_SOURCES
    "${BASE_PLATFORM_PATH}/*.cpp"
)

# Collect custom code sources (includes all custom implementation and graph if provided)
file(GLOB_RECURSE CUSTOM_CODE_SOURCES
    "${CUSTOM_CODE_PATH}/*.cpp"
)

message(STATUS "Base platform sources: ${BASE_PLATFORM_SOURCES}")
message(STATUS "Custom code sources: ${CUSTOM_CODE_SOURCES}")

# Combine all sources (custom code + base platform)
# Custom code is added first so it can override base implementations if needed
set(ALL_SOURCES ${CUSTOM_CODE_SOURCES} ${BASE_PLATFORM_SOURCES})

# Create shared library
add_library(aicpu_graph_kernel SHARED)
target_sources(aicpu_graph_kernel PRIVATE ${ALL_SOURCES})

# Compile options
target_compile_options(aicpu_graph_kernel
        PRIVATE
            -Wall
            -Wextra
            -std=gnu++17
            -rdynamic
            -O3
            -shared
            -fPIC
            -Werror
            -Wno-error=variadic-macros
            -fno-common
            -g
)

# Include directories - include both base platform and custom code paths
target_include_directories(aicpu_graph_kernel
        PRIVATE
            ${BASE_PLATFORM_PATH}
        PRIVATE
            ${CUSTOM_CODE_PATH}
        PRIVATE
            ${ASCEND_CANN_PACKAGE_PATH}/include
            ${ASCEND_CANN_PACKAGE_PATH}/include/toolchain
            ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/base
)

# Link directories
target_link_directories(aicpu_graph_kernel
        PRIVATE
            ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/devlib/linux/aarch64/
            ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/devlib/device/
            ${ASCEND_CANN_PACKAGE_PATH}/lib64
)

# Set output library name
set_target_properties(aicpu_graph_kernel PROPERTIES OUTPUT_NAME aicpu_graph_kernel)
