/**
 * PTO Runtime2 - Worker Thread Interface
 * 
 * Implements worker threads that execute tasks dispatched by the scheduler.
 * Each worker:
 * 1. Waits for tasks in its ready queue
 * 2. Executes the task (calls InCore function or simulates cycles)
 * 3. Signals completion to the scheduler
 * 
 * Based on: docs/runtime_buffer_manager_methods.md
 */

#ifndef PTO_WORKER_H
#define PTO_WORKER_H

#include "pto_runtime2_types.h"
#include "pto_shared_memory.h"
#include "pto_scheduler.h"

// Forward declaration (PTO2ThreadContext is defined in pto_runtime2_types.h)
struct PTO2Runtime;

// =============================================================================
// Worker Initialization
// =============================================================================

/**
 * Initialize worker context
 * 
 * @param worker      Worker context to initialize
 * @param worker_id   Unique worker ID
 * @param worker_type Worker type (CUBE, VECTOR, etc.)
 * @param runtime     Runtime reference
 * @return true on success
 */
bool pto2_worker_init(PTO2WorkerContext* worker, int32_t worker_id,
                       PTO2WorkerType worker_type, struct PTO2Runtime* runtime);

/**
 * Destroy worker context
 */
void pto2_worker_destroy(PTO2WorkerContext* worker);

/**
 * Reset worker statistics
 */
void pto2_worker_reset(PTO2WorkerContext* worker);

// =============================================================================
// Worker Thread Entry Points
// =============================================================================

/**
 * Worker thread main function
 * 
 * This is the entry point for worker threads. It:
 * 1. Waits for tasks in the appropriate ready queue
 * 2. Executes tasks (or simulates execution)
 * 3. Signals completion
 * 4. Loops until shutdown
 * 
 * @param arg Pointer to PTO2WorkerContext
 * @return NULL
 */
void* pto2_worker_thread_func(void* arg);

/**
 * Worker thread main function for simulation mode
 * Uses cycle estimation instead of actual execution
 * 
 * @param arg Pointer to PTO2WorkerContext
 * @return NULL
 */
void* pto2_worker_thread_func_sim(void* arg);

// =============================================================================
// Task Execution
// =============================================================================

/**
 * Get next task for this worker (blocks if queue empty)
 * 
 * @param worker Worker context
 * @return task_id, or -1 if shutdown
 */
int32_t pto2_worker_get_task(PTO2WorkerContext* worker);

/**
 * Try to get next task without blocking
 * 
 * @param worker Worker context
 * @return task_id, or -1 if no task available
 */
int32_t pto2_worker_try_get_task(PTO2WorkerContext* worker);

/**
 * Execute a task (call the InCore function)
 * 
 * @param worker  Worker context
 * @param task_id Task ID to execute
 */
void pto2_worker_execute_task(PTO2WorkerContext* worker, int32_t task_id);

/**
 * Simulate task execution (estimate cycles)
 * 
 * @param worker  Worker context
 * @param task_id Task ID to simulate
 * @return Estimated cycle count
 */
int64_t pto2_worker_simulate_task(PTO2WorkerContext* worker, int32_t task_id);

/**
 * Signal task completion to scheduler
 * 
 * @param worker      Worker context
 * @param task_id     Completed task ID
 * @param start_cycle When task started (for tracing)
 * @param end_cycle   When task completed (for tracing)
 */
void pto2_worker_task_complete(PTO2WorkerContext* worker, int32_t task_id, 
                                int64_t start_cycle, int64_t end_cycle);

// =============================================================================
// Completion Queue Operations
// =============================================================================

/**
 * Initialize completion queue
 * 
 * @param queue    Completion queue to initialize
 * @param capacity Queue capacity
 * @return true on success
 */
bool pto2_completion_queue_init(PTO2CompletionQueue* queue, int32_t capacity);

/**
 * Destroy completion queue
 */
void pto2_completion_queue_destroy(PTO2CompletionQueue* queue);

/**
 * Push completion entry (called by worker)
 * 
 * @param queue       Completion queue
 * @param task_id     Completed task ID
 * @param worker_id   Worker that completed the task
 * @param start_cycle When task started
 * @param end_cycle   When task completed
 * @return true if successful
 */
bool pto2_completion_queue_push(PTO2CompletionQueue* queue,
                                 int32_t task_id, int32_t worker_id,
                                 int64_t start_cycle, int64_t end_cycle);

/**
 * Pop completion entry (called by scheduler)
 * 
 * @param queue Completion queue
 * @param entry Output entry (must not be NULL)
 * @return true if entry was retrieved, false if queue empty
 */
bool pto2_completion_queue_pop(PTO2CompletionQueue* queue, PTO2CompletionEntry* entry);

/**
 * Check if completion queue is empty
 */
bool pto2_completion_queue_empty(PTO2CompletionQueue* queue);

// =============================================================================
// Worker Statistics
// =============================================================================

/**
 * Print worker statistics
 */
void pto2_worker_print_stats(PTO2WorkerContext* worker);

/**
 * Get worker type name
 */
const char* pto2_worker_type_name(PTO2WorkerType type);

#endif // PTO_WORKER_H
