/**
 * PTO Runtime - AICPU Kernel
 * 
 * This kernel runs on AICPU (ARM64 CPU on Ascend device).
 * Compiled with ARM64 gcc into libaicpu_kernel.so
 * 
 * Responsibilities:
 * - Handshake with AICore instances
 * - Task scheduling and dispatch to AICore
 * - Dependency resolution
 * - Shutdown coordination
 */

#include <cstdint>
#include <cstdio>
#include <atomic>
#include "pto_task.h"

// Device logging (AICPU)
#ifdef ENABLE_AICPU_LOG
#define AICPU_LOG(fmt, ...) printf("[AICPU] " fmt "\n", ##__VA_ARGS__)
#else
#define AICPU_LOG(fmt, ...) ((void)0)
#endif

// Static atomic counter for thread indexing (multiple AICPU threads)
static std::atomic<int> s_thread_id(0);

// =============================================================================
// AICore Handshake
// =============================================================================

/**
 * Initialize and synchronize with all AICore instances.
 * 
 * Protocol:
 * 1. Set aicpu_ready flag for each core
 * 2. Wait for each core to respond with aicore_done signal
 * 
 * @param kargs Kernel arguments containing handshake buffers
 * @return 0 on success
 */
static int handshake_aicore(PTOKernelArgs* kargs) {
    int32_t core_num = kargs->core_num;
    
    AICPU_LOG("Starting handshake with %d AICore instances", core_num);
    
    // Phase 1: Signal all cores that AICPU is ready
    for (int32_t i = 0; i < core_num; i++) {
        PTOHandshake* hank = &kargs->hankArgs[i];
        hank->aicpu_ready = 1;
    }
    
    // Phase 2: Wait for all cores to acknowledge (busy-wait polling)
    for (int32_t i = 0; i < core_num; i++) {
        PTOHandshake* hank = &kargs->hankArgs[i];
        while (hank->aicore_done == 0) {
            // Busy-wait - no sleep to minimize latency
        }
        AICPU_LOG("Core %d ready (aicore_done=%u)", i, hank->aicore_done);
    }
    
    AICPU_LOG("Handshake complete - all %d cores ready", core_num);
    return 0;
}

/**
 * Send quit signal to all AICore instances.
 * 
 * @param kargs Kernel arguments containing handshake buffers
 * @return 0 on success
 */
static int shutdown_aicore(PTOKernelArgs* kargs) {
    int32_t core_num = kargs->core_num;
    
    AICPU_LOG("Sending shutdown signal to %d cores", core_num);
    
    for (int32_t i = 0; i < core_num; i++) {
        PTOHandshake* hank = &kargs->hankArgs[i];
        hank->control = 1;  // Set quit signal
    }
    
    return 0;
}

// =============================================================================
// Task Scheduling
// =============================================================================

/**
 * Find and assign ready tasks to idle cores.
 * 
 * Simple round-robin scheduling:
 * - For each idle core, find a ready task of matching type
 * - Assign task to core and mark as running
 * 
 * @param graph Task graph
 * @param hank Handshake buffer array
 * @param core_num Number of cores
 * @param aic_num Number of AIC (Cube) cores
 * @return Number of tasks assigned
 */
static int schedule_tasks(PTOTaskGraph* graph, PTOHandshake* hank, 
                          int32_t core_num, int32_t aic_num) {
    int assigned = 0;
    
    for (int32_t core_id = 0; core_id < core_num; core_id++) {
        PTOHandshake* core_hank = &hank[core_id];
        
        // Skip busy cores
        if (core_hank->task != 0 && core_hank->task_status != 0) {
            continue;
        }
        
        // Determine core type: first aic_num are Cube, rest are Vector
        int32_t core_type = (core_id < aic_num) ? 0 : 1;
        
        // Find a ready task matching this core type
        for (int32_t task_id = 0; task_id < graph->num_tasks; task_id++) {
            PTOTask* task = &graph->tasks[task_id];
            
            // Skip non-ready tasks
            if (task->status != PTO_TASK_READY) {
                continue;
            }
            
            // Skip tasks with wrong core type
            if (task->core_type != core_type) {
                continue;
            }
            
            // Assign task to this core
            task->status = PTO_TASK_RUNNING;
            core_hank->task = (uint64_t)task;
            core_hank->task_status = 1;  // Mark busy
            
            AICPU_LOG("Assigned task %d (%s) to core %d", 
                     task_id, task->func_name, core_id);
            assigned++;
            break;
        }
    }
    
    return assigned;
}

/**
 * Check for completed tasks and update dependencies.
 * 
 * @param graph Task graph
 * @param hank Handshake buffer array
 * @param core_num Number of cores
 * @return Number of tasks completed
 */
static int check_completions(PTOTaskGraph* graph, PTOHandshake* hank, int32_t core_num) {
    int completed = 0;
    
    for (int32_t core_id = 0; core_id < core_num; core_id++) {
        PTOHandshake* core_hank = &hank[core_id];
        
        // Check if core has completed a task
        if (core_hank->task != 0 && core_hank->task_status == 0) {
            PTOTask* task = (PTOTask*)core_hank->task;
            
            // Mark task complete
            task->status = PTO_TASK_COMPLETE;
            graph->tasks_completed++;
            
            AICPU_LOG("Task %d (%s) completed on core %d", 
                     task->task_id, task->func_name, core_id);
            
            // Update dependents - decrement their deps_remaining
            for (int32_t i = 0; i < task->num_dependents; i++) {
                int32_t dep_id = task->dependents[i];
                PTOTask* dep_task = &graph->tasks[dep_id];
                
                // Atomic decrement (in AICPU we can use atomic)
                int32_t remaining = __atomic_sub_fetch(&dep_task->deps_remaining, 1, __ATOMIC_SEQ_CST);
                
                // If all dependencies satisfied, mark as ready
                if (remaining == 0 && dep_task->status == PTO_TASK_PENDING) {
                    dep_task->status = PTO_TASK_READY;
                    AICPU_LOG("Task %d is now ready", dep_id);
                }
            }
            
            // Clear task assignment
            core_hank->task = 0;
            completed++;
        }
    }
    
    return completed;
}

/**
 * Execute the task graph on AICore instances.
 * 
 * Main scheduling loop:
 * 1. Schedule ready tasks to idle cores
 * 2. Check for completed tasks
 * 3. Update dependencies
 * 4. Repeat until all tasks complete
 * 
 * @param graph Task graph
 * @param hank Handshake buffer array
 * @param core_num Number of cores
 * @param aic_num Number of AIC cores
 * @return Number of tasks executed
 */
static int execute_graph(PTOTaskGraph* graph, PTOHandshake* hank, 
                         int32_t core_num, int32_t aic_num) {
    if (graph == nullptr || graph->num_tasks == 0) {
        return 0;
    }
    
    AICPU_LOG("Executing graph with %d tasks on %d cores (%d AIC, %d AIV)",
             graph->num_tasks, core_num, aic_num, core_num - aic_num);
    
    // Initialize: mark tasks with no dependencies as ready
    for (int32_t i = 0; i < graph->num_tasks; i++) {
        PTOTask* task = &graph->tasks[i];
        if (task->dep_count == 0) {
            task->status = PTO_TASK_READY;
            AICPU_LOG("Task %d (%s) initially ready", i, task->func_name);
        }
    }
    
    // Main scheduling loop
    while (graph->tasks_completed < graph->num_tasks) {
        schedule_tasks(graph, hank, core_num, aic_num);
        check_completions(graph, hank, core_num);
    }
    
    AICPU_LOG("Graph execution complete: %d tasks", graph->tasks_completed);
    return graph->tasks_completed;
}

// =============================================================================
// AICPU Kernel Entry Points
// =============================================================================

/**
 * AICPU kernel initialization entry point.
 * 
 * Called once during kernel initialization by CANN runtime.
 * Function name is hardcoded in libaicpu_extend_kernels.so
 * 
 * @param arg Pointer to PTOKernelArgs
 * @return 0 on success, -1 on error
 */
extern "C" __attribute__((visibility("default"))) 
int DynTileFwkBackendKernelServerInit(void* arg) {
    if (arg == nullptr) {
        AICPU_LOG("ERROR: Invalid kernel arguments (null)");
        return -1;
    }
    
    AICPU_LOG("Kernel initialization complete");
    return 0;
}

/**
 * AICPU kernel main execution entry point.
 * 
 * Called by CANN runtime to execute the task graph.
 * Function name is hardcoded in libaicpu_extend_kernels.so
 * 
 * Flow:
 * 1. Handshake with all AICore instances
 * 2. Execute task graph
 * 3. Shutdown all AICore instances
 * 
 * @param arg Pointer to PTOKernelArgs
 * @return 0 on success, non-zero on error
 */
extern "C" __attribute__((visibility("default"))) 
int DynTileFwkBackendKernelServer(void* arg) {
    if (arg == nullptr) {
        AICPU_LOG("ERROR: Invalid kernel arguments (null)");
        return -1;
    }
    
    int thread_id = s_thread_id++;
    AICPU_LOG("Starting kernel execution (thread %d)", thread_id);
    
    PTOKernelArgs* kargs = (PTOKernelArgs*)arg;
    
    // Step 1: Handshake with all AICore instances
    int rc = handshake_aicore(kargs);
    if (rc != 0) {
        AICPU_LOG("ERROR: Handshake failed");
        return rc;
    }
    
    // Step 2: Execute task graph if provided
    if (kargs->graphArgs != nullptr) {
        int completed = execute_graph(kargs->graphArgs, kargs->hankArgs,
                                      kargs->core_num, kargs->aic_num);
        AICPU_LOG("Executed %d tasks", completed);
    }
    
    // Step 3: Shutdown all AICore instances
    rc = shutdown_aicore(kargs);
    if (rc != 0) {
        AICPU_LOG("ERROR: Shutdown failed");
        return rc;
    }
    
    AICPU_LOG("Kernel execution complete (thread %d)", thread_id);
    return 0;
}

/**
 * Static kernel entry point (for compatibility).
 */
extern "C" __attribute__((visibility("default"))) 
int StaticTileFwkBackendKernelServer(void* arg) {
    (void)arg;
    return 0;
}
