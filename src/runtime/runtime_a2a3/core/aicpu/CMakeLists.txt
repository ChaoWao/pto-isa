# Build AICPU kernel using ARM64 gcc
#
# CMAKE_C_COMPILER and CMAKE_CXX_COMPILER should be ARM64 gcc
# (or native gcc when building on ARM64 host)
#
# Required variables (set when calling cmake):
#   ASCEND_HOME_PATH    - Path to CANN SDK
#   CUSTOM_INCLUDE_DIRS - Additional include directories
#   CUSTOM_SOURCE_DIRS  - Additional source directories
#
# Output: libaicpu_kernel.so

cmake_minimum_required(VERSION 3.16.3)
project(aicpu_kernel LANGUAGES C CXX)

# Build include list
set(CMAKE_CUSTOM_INCLUDE_DIRS "")
list(APPEND CMAKE_CUSTOM_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../common")
if(DEFINED CUSTOM_INCLUDE_DIRS)
    foreach(INC_DIR ${CUSTOM_INCLUDE_DIRS})
        list(APPEND CMAKE_CUSTOM_INCLUDE_DIRS "${INC_DIR}")
    endforeach()
endif()

# Build source list
set(AICPU_KERNEL_SOURCES "")
file(GLOB AICPU_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.c")
list(APPEND AICPU_KERNEL_SOURCES ${AICPU_SOURCES})
file(GLOB COMMON_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/../common/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/../common/*.c")
list(APPEND AICPU_KERNEL_SOURCES ${COMMON_SOURCES})
if(DEFINED CUSTOM_SOURCE_DIRS)
    foreach(SRC_DIR ${CUSTOM_SOURCE_DIRS})
        file(GLOB DIR_SOURCES "${SRC_DIR}/*.cpp" "${SRC_DIR}/*.c")
        list(APPEND AICPU_KERNEL_SOURCES ${DIR_SOURCES})
    endforeach()
endif()
list(LENGTH AICPU_KERNEL_SOURCES NUM_SOURCES)
message(STATUS "AICPU sources: ${NUM_SOURCES} files")

# Create shared library
add_library(aicpu_kernel SHARED ${AICPU_KERNEL_SOURCES})

# Compile options
target_compile_options(aicpu_kernel
    PRIVATE
        -Wall
        -Wextra
        -std=gnu++17
        -rdynamic
        -O3
        -fPIC
        -Werror
        -Wno-error=variadic-macros
        -fno-common
        -g
        -DENABLE_AICPU_LOG
)

# Include directories
target_include_directories(aicpu_kernel
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CUSTOM_INCLUDE_DIRS}
)

# Add CANN SDK includes if available
if(DEFINED ASCEND_HOME_PATH)
    target_include_directories(aicpu_kernel
        PRIVATE
            ${ASCEND_HOME_PATH}/include
            ${ASCEND_HOME_PATH}/include/toolchain
            ${ASCEND_HOME_PATH}/pkg_inc/base
    )
    
    target_link_directories(aicpu_kernel
        PRIVATE
            ${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/devlib/linux/aarch64/
            ${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/devlib/device/
            ${ASCEND_HOME_PATH}/lib64
    )
endif()

# Output name
set_target_properties(aicpu_kernel PROPERTIES OUTPUT_NAME aicpu_kernel)
