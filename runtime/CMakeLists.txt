cmake_minimum_required(VERSION 3.15)
project(runtime_gm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(-Wall -Wextra -Wpedantic)

# Ascend path
if(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
else()
    message(FATAL_ERROR "ASCEND_HOME_PATH is not set, CANN runtime is required")
endif()

# Compile kernels
add_subdirectory(src/aicpu)
add_subdirectory(src/aicore)

# Example programs
add_subdirectory(example)

# Python bindings (optional - requires Python and pybind11)
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
if(BUILD_PYTHON_BINDINGS)
    add_subdirectory(python)
endif()

message(STATUS "ASCEND_HOME_PATH = ${ASCEND_HOME_PATH}")
