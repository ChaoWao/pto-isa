cmake_minimum_required(VERSION 3.15)

# Header-only library for shared structures
# This library provides common headers that can be used by host, AICPU, and AICore
add_library(runtime_common INTERFACE)

# Set include directory to parent directory so that includes work as:
# #include "common/handshake.h"
target_include_directories(runtime_common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/..)
