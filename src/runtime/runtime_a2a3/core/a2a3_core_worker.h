/**
 * PTO Runtime - Ascend A2/A3 Core Worker Interface
 * 
 * This header defines the worker functions that execute InCore functions
 * on Cube and Vector cores. These are compiled as part of the core layer.
 * 
 * The implementation differs between hardware (CANN SDK) and simulator.
 */

#ifndef A2A3_CORE_WORKER_H
#define A2A3_CORE_WORKER_H

#include "../../pto_runtime_common.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Worker Context
// =============================================================================

/**
 * Context passed to each worker thread.
 */
typedef struct {
    PTORuntime* rt;
    int worker_id;
    bool is_cube_worker;
} A2A3WorkerContext;

// =============================================================================
// Task Execution (called by workers)
// =============================================================================

/**
 * Execute an InCore function task.
 * This is called by workers after dequeuing a task.
 * 
 * @param rt        Runtime context
 * @param task_id   Task to execute
 * @param worker_id Worker executing the task
 */
void a2a3_core_execute_task(PTORuntime* rt, int32_t task_id, int32_t worker_id);

// =============================================================================
// Task Completion (called by workers after execution)
// =============================================================================

/**
 * Mark a task as complete and propagate to dependents (thread-safe).
 * This is called by workers after executing an InCore function.
 * 
 * Implementation differs between hardware and simulator:
 * - Hardware: Uses CANN SDK synchronization primitives
 * - Simulator: Uses pthread mutexes with cycle-accurate tracking
 * 
 * @param rt        Runtime context
 * @param task_id   Completed task ID
 */
void a2a3_core_complete_task(PTORuntime* rt, int32_t task_id);

// =============================================================================
// Worker Thread Functions
// =============================================================================

/**
 * Vector core worker thread main function.
 * Loops: get task from vector queue -> execute -> complete -> repeat
 * 
 * @param arg   Pointer to A2A3WorkerContext
 * @return      NULL (thread exit)
 */
void* a2a3_vector_worker_func(void* arg);

/**
 * Cube core worker thread main function.
 * Loops: get task from cube queue -> execute -> complete -> repeat
 * 
 * @param arg   Pointer to A2A3WorkerContext
 * @return      NULL (thread exit)
 */
void* a2a3_cube_worker_func(void* arg);

#ifdef __cplusplus
}
#endif

#endif /* A2A3_CORE_WORKER_H */
