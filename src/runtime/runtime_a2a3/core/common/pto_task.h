/**
 * PTO Runtime - Task Structure for NPU Execution
 * 
 * This header defines the task structure shared between:
 * - Host (for building task graph)
 * - AICPU (for task scheduling)
 * - AICore (for task execution)
 * 
 * Must be compiled with both gcc (host/AICPU) and ccec (AICore).
 */

#ifndef PTO_TASK_H
#define PTO_TASK_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Memory Attributes for AICore/AICPU
// =============================================================================

#ifndef __gm__
#define __gm__
#endif

#ifndef __global__
#define __global__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

// =============================================================================
// Constants
// =============================================================================

#define PTO_MAX_TASK_ARGS    16
#define PTO_MAX_TASKS        4096
#define PTO_MAX_WORKERS      128
#define PTO_FUNC_NAME_LEN    64

// =============================================================================
// Task Argument Structure
// =============================================================================

/**
 * Task argument - points to a region in global memory.
 */
typedef struct {
    uint64_t base_addr;      // Base address in GM
    uint64_t offset;         // Offset from base
    uint64_t size;           // Size in bytes
} PTOTaskArg;

// =============================================================================
// Task Structure
// =============================================================================

/**
 * Task structure for NPU execution.
 * 
 * The task contains:
 * - Function identification (func_id, func_name)
 * - Function binary address (functionBinAddr) for runtime dispatch
 * - Arguments (args array)
 * - Dependency tracking (dep_count, dependents)
 */
typedef struct {
    // Task identification
    int32_t task_id;                      // Unique task ID
    int32_t func_id;                      // Function ID for lookup
    char func_name[PTO_FUNC_NAME_LEN];    // Function name
    
    // Function binary address (for runtime dispatch)
    // This points to the compiled kernel binary in GM
    uint64_t functionBinAddr;
    
    // Arguments
    int32_t num_args;
    PTOTaskArg args[PTO_MAX_TASK_ARGS];
    
    // Dependency tracking
    int32_t dep_count;                    // Number of dependencies
    int32_t deps_remaining;               // Remaining dependencies (atomic)
    int32_t num_dependents;               // Number of tasks depending on this
    int32_t dependents[PTO_MAX_TASK_ARGS]; // Task IDs of dependents
    
    // Execution state
    int32_t status;                       // 0=pending, 1=ready, 2=running, 3=complete
    int32_t core_type;                    // 0=AIC (Cube), 1=AIV (Vector)
} PTOTask;

// Task status values
#define PTO_TASK_PENDING   0
#define PTO_TASK_READY     1
#define PTO_TASK_RUNNING   2
#define PTO_TASK_COMPLETE  3

// =============================================================================
// Task Graph Structure
// =============================================================================

/**
 * Task graph - contains all tasks to be executed.
 */
typedef struct {
    int32_t num_tasks;                    // Total number of tasks
    int32_t tasks_completed;              // Number of completed tasks
    PTOTask tasks[PTO_MAX_TASKS];         // Task array
} PTOTaskGraph;

// =============================================================================
// Handshake Structure (AICore <-> AICPU Communication)
// =============================================================================

/**
 * Handshake buffer for AICore-AICPU communication.
 * 
 * Protocol:
 * 1. AICPU sets aicpu_ready = 1
 * 2. AICore polls until aicpu_ready != 0
 * 3. AICore sets aicore_done = core_id + 1
 * 4. Execution loop:
 *    - AICPU sets task pointer
 *    - AICore polls task != 0, executes, sets task_status = 0
 * 5. AICPU sets control = 1 to shutdown
 */
typedef struct {
    volatile uint32_t aicpu_ready;    // AICPU ready signal
    volatile uint32_t aicore_done;    // AICore ready signal (core_id + 1)
    volatile uint32_t control;        // Control: 0=run, 1=quit
    volatile uint64_t task;           // Task pointer (0 = no task)
    volatile uint32_t task_status;    // 0=idle/done, 1=busy
    volatile uint32_t core_type;      // 0=AIC, 1=AIV
    uint32_t padding[2];              // Align to cache line
} PTOHandshake;

// =============================================================================
// Kernel Arguments (passed from Host to AICPU)
// =============================================================================

/**
 * Kernel arguments structure passed from host to AICPU.
 */
typedef struct {
    int64_t* deviceArgs;              // Device-specific arguments
    PTOHandshake* hankArgs;           // Handshake buffer array
    PTOTaskGraph* graphArgs;          // Task graph
    int32_t core_num;                 // Total number of cores
    int32_t aic_num;                  // Number of AIC (Cube) cores
    int32_t aiv_num;                  // Number of AIV (Vector) cores
} PTOKernelArgs;

#ifdef __cplusplus
}
#endif

#endif // PTO_TASK_H
