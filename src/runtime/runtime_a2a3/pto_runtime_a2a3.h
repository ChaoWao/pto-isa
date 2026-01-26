/**
 * PTO Runtime System - A2A3 (Ascend) Platform Header
 * 
 * A2A3-specific runtime components:
 * - Dual ready queues: vector (is_cube=0) and cube (is_cube=1) 
 * - Dedicated dependency management module for heterogeneous execution
 * - Simulation support for cube and vector workers
 * 
 * A2A3 Architecture Notes:
 * - Vector workers handle element-wise and reduction operations (is_cube=false)
 * - Cube workers handle matrix multiplication operations (is_cube=true)
 * - Typical configuration: 48 vector workers, 24 cube workers
 */

#ifndef PTO_RUNTIME_A2A3_H
#define PTO_RUNTIME_A2A3_H

#include "../pto_runtime_common.h"

// =============================================================================
// A2A3 Platform Configuration
// =============================================================================

// Default worker configuration for A2A3
#define A2A3_DEFAULT_VECTOR_WORKERS  48
#define A2A3_DEFAULT_CUBE_WORKERS    24

// =============================================================================
// A2A3-Specific API
// =============================================================================

/**
 * Enable A2A3 simulation mode with dual queues
 * 
 * In A2A3 mode:
 * - Creates separate ready queues for vector and cube tasks
 * - Vector workers (is_cube=false) pull from vector queue
 * - Cube workers (is_cube=true) pull from cube queue
 * - Enables accurate simulation of heterogeneous NPU execution
 * 
 * @param rt                 Runtime context
 * @param num_vector_workers Number of vector workers (is_cube=0 tasks)
 * @param num_cube_workers   Number of cube workers (is_cube=1 tasks)
 */
void pto_runtime_enable_a2a3_sim(PTORuntime* rt, int32_t num_vector_workers, int32_t num_cube_workers);

/**
 * A2A3 Runtime Entry Point - Dual-queue heterogeneous execution
 * 
 * @param orch_func               Orchestration function that builds the task graph
 * @param user_data               User data passed to orchestration function
 * @param num_vector_workers      Number of vector worker threads
 * @param num_cube_workers        Number of cube worker threads
 * @param execution_task_threshold Task threshold for pipelined execution
 * @return 0 on success, -1 on failure
 */
int runtime_entry_a2a3(PTOOrchFunc orch_func, void* user_data, 
                       int num_vector_workers, int num_cube_workers,
                       int execution_task_threshold);

// =============================================================================
// A2A3 Dual Queue API
// =============================================================================

/**
 * Get next ready task for a vector worker (is_cube=0 tasks only)
 * @return task_id or -1 if no tasks ready
 */
int32_t pto_get_ready_task_vector(PTORuntime* rt);

/**
 * Get next ready task for a cube worker (is_cube=1 tasks only)
 * @return task_id or -1 if no tasks ready
 */
int32_t pto_get_ready_task_cube(PTORuntime* rt);

/**
 * Thread-safe get ready task for vector worker (blocking)
 */
int32_t pto_get_ready_task_vector_blocking(PTORuntime* rt);

/**
 * Thread-safe get ready task for cube worker (blocking)
 */
int32_t pto_get_ready_task_cube_blocking(PTORuntime* rt);

// =============================================================================
// A2A3 Platform Implementation Notes
// =============================================================================

// The A2A3 platform implements the following common API functions with 
// dual-queue behavior:
//
// - pto_task_submit_a2a3()           : Routes to vector or cube queue based on is_cube
// - pto_task_complete_a2a3()         : Dedicated dependency module, routes dependents
// - pto_task_complete_a2a3_threadsafe(): Thread-safe version with dual queue support
// - pto_loop_replay_a2a3()           : Replay with dual queue routing
//
// When PTO_PLATFORM_A2A3 is defined, these replace the default implementations.

#endif // PTO_RUNTIME_A2A3_H
