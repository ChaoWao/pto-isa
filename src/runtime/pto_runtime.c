/**
 * PTO Runtime System - Unified Implementation
 * 
 * This file provides the unified implementation by including:
 * 1. Common platform-independent code
 * 2. Platform-specific code based on compile-time selection
 * 
 * Build Configuration:
 * ====================
 * 
 * For ARM64 (default):
 *   gcc -c pto_runtime.c -o pto_runtime.o
 *   
 * For A2A3/Ascend:
 *   gcc -DPTO_PLATFORM_A2A3 -c pto_runtime.c -o pto_runtime.o
 * 
 * Alternatively, compile the individual source files:
 *   gcc -c pto_runtime_common.c -o pto_runtime_common.o
 *   gcc -c pto_runtime_arm64.c -o pto_runtime_arm64.o  # or pto_runtime_a2a3.c
 *   
 * Module Structure:
 * =================
 * 
 * pto_runtime_common.c:
 *   - Runtime initialization/shutdown
 *   - Task allocation and argument management
 *   - TensorMap operations
 *   - Record & Replay core logic
 *   - Cycle trace recording
 *   - Debug dump functions
 * 
 * pto_runtime_arm64.c:
 *   - Single ready queue management
 *   - Distributed dependency tracking (in task_complete)
 *   - Multi-threaded worker execution
 *   - ARM64 runtime entry point
 * 
 * pto_runtime_a2a3.c:
 *   - Dual ready queue (vector/cube)
 *   - Dedicated dependency management module
 *   - Heterogeneous worker execution
 *   - A2A3 runtime entry point
 */

#include "pto_runtime.h"

// =============================================================================
// Include Common Implementation
// =============================================================================

// Note: When compiling as a single unit, we include the .c files directly.
// When compiling separately, link the individual object files instead.

#ifndef PTO_RUNTIME_SEPARATE_COMPILE

#include "pto_runtime_common.c"

// =============================================================================
// Include Platform-Specific Implementation
// =============================================================================

#ifdef PTO_PLATFORM_ARM64
#include "pto_runtime_arm64.c"
#endif

#ifdef PTO_PLATFORM_A2A3
#include "runtime_a2a3/pto_runtime_a2a3.c"
#endif

#endif // PTO_RUNTIME_SEPARATE_COMPILE
