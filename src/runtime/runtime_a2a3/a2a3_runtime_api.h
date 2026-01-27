/**
 * PTO Runtime - Ascend A2/A3 Runtime API
 * 
 * Execution Architecture (all computation on Device):
 * 
 * Device (NPU):
 *   - AICore workers: Poll handshake registers, execute InCore functions
 *   - AICPU scheduler: Distribute tasks to AICore workers
 *   - AICPU orchestration: Dynamically generate tasks, submit to scheduler
 * 
 * Host (CPU):
 *   - Initialize device and load kernels
 *   - Launch AICPU + AICore kernels
 *   - Wait for completion
 *   - Copy results back
 *   - Shutdown
 * 
 * This is STREAMING execution - orchestration generates tasks on-the-fly,
 * not batch execution with pre-built task graph.
 */

#ifndef A2A3_RUNTIME_API_H
#define A2A3_RUNTIME_API_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Default Configuration
// =============================================================================

#define A2A3_DEFAULT_ORCH_THREADS    1
#define A2A3_DEFAULT_DEP_THREADS     3
#define A2A3_DEFAULT_AIV_WORKERS     48
#define A2A3_DEFAULT_AIC_WORKERS     24

#define A2A3_MAX_INCORE_FUNCS        256

// =============================================================================
// Runtime Configuration
// =============================================================================

/**
 * Runtime initialization configuration.
 * 
 * Paths to kernel binaries:
 * - orchestration_so_path: Orchestration function (.so, runs on AICPU)
 * - aicpu_kernel_path: AICPU scheduler kernel (.so)
 * - aicore_kernel_path: AICore worker kernel (.o)
 * - incore_*_dir: InCore function binaries (.o files)
 */
typedef struct {
    // Orchestration function (runs on AICPU, generates tasks dynamically)
    const char* orchestration_so_path;
    const char* orchestration_func_name;
    
    // InCore function directories (contain .o files for AICore)
    const char* incore_aiv_dir;
    const char* incore_aic_dir;
    
    // NPU kernel paths
    const char* aicore_kernel_path;   // AICore worker kernel (.o)
    const char* aicpu_kernel_path;    // AICPU scheduler kernel (.so)
    
    // Core configuration
    int num_orch_threads;    // AICPU orchestration threads
    int num_dep_threads;     // AICPU scheduler threads  
    int num_aiv_workers;     // AIV (Vector) cores
    int num_aic_workers;     // AIC (Cube) cores
    
    // User data passed to orchestration function
    void* user_data;
    
    // Debug options
    bool debug_enabled;
    bool debug_orchestration_only;  // Only run orchestration, skip task execution
} A2A3RuntimeConfig;

/**
 * Initialize configuration with default values.
 */
static inline void a2a3_config_init_defaults(A2A3RuntimeConfig* config) {
    if (!config) return;
    config->orchestration_so_path = NULL;
    config->orchestration_func_name = NULL;
    config->incore_aiv_dir = NULL;
    config->incore_aic_dir = NULL;
    config->aicore_kernel_path = NULL;
    config->aicpu_kernel_path = NULL;
    config->num_orch_threads = A2A3_DEFAULT_ORCH_THREADS;
    config->num_dep_threads = A2A3_DEFAULT_DEP_THREADS;
    config->num_aiv_workers = A2A3_DEFAULT_AIV_WORKERS;
    config->num_aic_workers = A2A3_DEFAULT_AIC_WORKERS;
    config->user_data = NULL;
    config->debug_enabled = false;
    config->debug_orchestration_only = false;
}

// =============================================================================
// Runtime Lifecycle API
// =============================================================================

/**
 * Initialize the A2A3 runtime.
 * 
 * This function:
 * 1. Sets the device and creates streams
 * 2. Loads AICPU kernel (scheduler + orchestration)
 * 3. Loads AICore kernel (workers)
 * 4. Loads InCore function binaries to device GM
 * 5. Allocates handshake buffers
 */
int a2a3_runtime_init(A2A3RuntimeConfig* config);

/**
 * Execute on device.
 * 
 * This function:
 * 1. Launches AICore kernel (workers enter polling loop)
 * 2. Launches AICPU kernel (scheduler + orchestration)
 * 3. Waits for AICPU orchestration to complete
 * 4. Waits for all tasks to be executed
 * 5. Signals AICore workers to shutdown
 */
int a2a3_runtime_execute(void* user_data);

/**
 * Finalize the runtime.
 */
void a2a3_runtime_finalize(void);

// =============================================================================
// Memory Management API
// =============================================================================

void* a2a3_runtime_malloc(size_t size_bytes);
void a2a3_runtime_free(void* ptr);
int a2a3_runtime_copy_to_device(void* dst_device, const void* src_host, size_t size_bytes);
int a2a3_runtime_copy_from_device(void* dst_host, const void* src_device, size_t size_bytes);

// =============================================================================
// InCore Function Registry
// =============================================================================

typedef void (*A2A3InCoreFunc)(void** args, int32_t num_args);

int a2a3_runtime_register_incore(const char* func_name, A2A3InCoreFunc func_ptr, bool is_cube);
A2A3InCoreFunc a2a3_runtime_lookup_incore(const char* func_name);

// =============================================================================
// Kernel Compilation and Loading API (NEW)
// =============================================================================

/**
 * Compile and load a kernel from CCE C++ source to device GM.
 *
 * This function compiles a CCE C++ kernel and loads it to device global memory,
 * making it available for runtime function pointer dispatch.
 *
 * @param func_id Function identifier (0-255) to associate with this kernel
 * @param cpp_path Path to CCE C++ source file (generated by ptoas)
 * @param pto_isa_root Path to PTO-ISA include directory (for compilation)
 * @param core_type Target core type: 0=any, 1=AIC(cube), 2=AIV(vector)
 * @return A2A3_SUCCESS on success, error code otherwise
 *
 * Workflow:
 * 1. Python: ptoas kernel.pto -o kernel.cpp
 * 2. C API: a2a3_compile_and_load_kernel(0, "kernel.cpp", pto_isa_root, 1)
 * 3. Runtime: ccec kernel.cpp -o kernel.o
 * 4. Runtime: Load kernel.o to device GM
 * 5. Runtime: Store address in g_kernel_func_table[func_id]
 *
 * Note: Current implementation expects pre-compiled .o files.
 * Full ccec compilation will be added in a2a3_kernel_compiler module.
 */
int a2a3_compile_and_load_kernel(int func_id, const char* cpp_path,
                                   const char* pto_isa_root, int core_type);

/**
 * Get the device GM address of a loaded kernel.
 *
 * Returns the function pointer address in device global memory for the
 * specified func_id. This address is set by a2a3_compile_and_load_kernel().
 *
 * @param func_id Function identifier (0-255)
 * @return Device GM address (void*), or NULL if func_id invalid or not loaded
 *
 * Usage:
 * - Task->functionBinAddr = (uint64_t)a2a3_get_function_bin_addr(task->func_id)
 * - AICore casts this to function pointer: (KernelFunc)functionBinAddr
 */
void* a2a3_get_function_bin_addr(int func_id);

// =============================================================================
// Status and Statistics
// =============================================================================

typedef struct {
    int64_t total_tasks_scheduled;
    int64_t total_tasks_completed;
    int64_t aiv_tasks_executed;
    int64_t aic_tasks_executed;
    double  total_execution_time_ms;
    int     num_incore_funcs_loaded;
} A2A3RuntimeStats;

void a2a3_runtime_get_stats(A2A3RuntimeStats* stats);
void a2a3_runtime_print_stats(void);
bool a2a3_runtime_is_initialized(void);

// =============================================================================
// Error Codes
// =============================================================================

#define A2A3_SUCCESS                 0
#define A2A3_ERROR_INVALID_CONFIG     -1
#define A2A3_ERROR_SO_LOAD_FAILED     -2
#define A2A3_ERROR_FUNC_NOT_FOUND     -3
#define A2A3_ERROR_MEMORY_ALLOC       -4
#define A2A3_ERROR_THREAD_CREATE      -5
#define A2A3_ERROR_NOT_INITIALIZED    -6
#define A2A3_ERROR_ALREADY_INIT       -7
#define A2A3_ERROR_BINARY_LOAD_FAILED -8
#define A2A3_ERROR_DEVICE_LAUNCH      -9

const char* a2a3_runtime_error_string(int error_code);

#ifdef __cplusplus
}
#endif

#endif // A2A3_RUNTIME_API_H
