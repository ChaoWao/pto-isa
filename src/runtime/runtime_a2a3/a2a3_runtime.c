/**
 * PTO Runtime - Ascend A2/A3 Runtime Implementation
 * 
 * Execution Architecture:
 * 
 * Device (NPU) - all computation here:
 *   1. AICore workers: Load aicore_kernel.o, enter polling loop
 *   2. AICPU scheduler: Load scheduler code, distribute tasks
 *   3. AICPU orchestration: Run orchestration function, generate tasks dynamically
 * 
 * Host (CPU) - control only:
 *   1. Initialize device, load kernels
 *   2. Launch AICPU + AICore kernels
 *   3. Wait for completion
 *   4. Copy results, shutdown
 * 
 * This is STREAMING execution - tasks generated on-the-fly by AICPU orchestration.
 */

#define _POSIX_C_SOURCE 199309L

#include "a2a3_runtime_api.h"
#include "host/a2a3_so_loader.h"
#include "host/a2a3_host.h"
#include "host/a2a3_binary_loader.h"
#include "../pto_runtime_common.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// =============================================================================
// CANN SDK Headers
// =============================================================================

#ifdef CANN_SDK_AVAILABLE
#include <acl/acl.h>
#include <runtime/rt.h>  // For rtKernelLaunchWithHandleV2, rtAicpuKernelLaunchExWithArgs
#endif

// =============================================================================
// Internal State
// =============================================================================

static A2A3RuntimeConfig g_config;
static bool g_initialized = false;
static A2A3RuntimeStats g_stats;

#ifdef CANN_SDK_AVAILABLE
static aclrtStream g_stream_aicpu = NULL;
static aclrtStream g_stream_aicore = NULL;
static void* g_aicore_bin_handle = NULL;
static void* g_device_handshake = NULL;  // Handshake buffer in device GM
#endif

// =============================================================================
// Handshake Structure - 64-byte cache-aligned for AICPU-AICore communication
// =============================================================================

/**
 * Handshake buffer for AICPU-AICore communication
 *
 * Each AICore has its own handshake buffer for synchronization with AICPU.
 * The structure is cache-line aligned (64 bytes) to prevent false sharing
 * between cores and optimize cache coherency operations.
 *
 * Protocol:
 * 1. AICPU sets aicpu_ready=1
 * 2. AICore sets aicore_done=core_id+1
 * 3. AICPU assigns task and sets task_status=1
 * 4. AICore executes and sets task_status=0
 * 5. AICPU sets control=1 to signal shutdown
 */
typedef struct __attribute__((aligned(64))) {
    volatile uint32_t aicpu_ready;    // AICPU ready signal: 0=not ready, 1=ready
    volatile uint32_t aicore_done;    // AICore ready signal: 0=not ready, core_id+1=ready
    volatile uint64_t task;           // Task pointer: 0=no task, non-zero=Task* address
    volatile int32_t task_status;     // Task execution status: 0=idle, 1=busy
    volatile int32_t control;         // Control signal: 0=execute, 1=quit
    volatile int32_t core_type;       // Core type: 0=AIC, 1=AIV
    volatile uint32_t profile_enable; // 0=disable per-task profiling, 1=enable
    volatile uint32_t reserved;       // Reserved for alignment
} Handshake;

// =============================================================================
// TileFwk ABI Structures - Must match ref_runtime/src/platform/a2a3/common/kernel_args.h
// =============================================================================

// Minimal profiling config (kept for layout compatibility)
typedef struct {
    uint32_t profConfig;
} ToSubMachineConfig;

// Device-side args blob (cfgdata in DeviceKernelArgs)
// This matches the layout used by TileFwk's dynamic launchers (PyPTO)
typedef struct {
    uint32_t nrAic;              // Number of AIC cores
    uint32_t nrAiv;              // Number of AIV cores
    uint32_t nrAicpu;            // Number of AICPU
    uint32_t nrValidAic;         // Valid AIC count
    uint64_t opaque;             // Pointer to PtoRuntimeArgs
    uint64_t devQueueAddr;
    uint64_t sharedBuffer;
    uint64_t coreRegAddr;
    uint64_t corePmuRegAddr;
    uint64_t corePmuAddr;
    uint64_t pmuEventAddr;
    uint64_t taskType_machineConfig_taskId;  // Bit fields packed
    uint64_t taskData;
    uint64_t taskWastTime;
    uint64_t aicpuSoBin;         // Device address of AICPU SO binary
    uint64_t aicpuSoLen;         // Size of AICPU SO binary
    uint64_t deviceId;
    uint64_t startArgsAddr;
    uint64_t taskQueue;
    uint64_t taskCtrl;
    uint32_t scheCpuNum;
    uint32_t enableCtrl_validGetPgMask_disableSync;  // Bit fields packed
    uint64_t generalAddr;
    uint64_t stitchPoolAddr;
    uint64_t aicpuPerfAddr;
    uint32_t archInfo;           // ArchInfo enum value
    ToSubMachineConfig toSubMachineConfig;
} DeviceArgs;

// Host->AICPU launch args (first field of the struct passed to rtAicpuKernelLaunchExWithArgs)
typedef struct {
    uint64_t generalAddr;
    uint64_t stitchPoolAddr;
} OpMetaAddrs;

typedef struct {
    int64_t *ctrlFlowCache;
    int64_t *inputs;
    int64_t *outputs;
    int64_t *workspace;
    int64_t *tilingdata;
    int64_t *cfgdata;            // Points to DeviceArgs
    void *costmodeldata;
    void *aicoreModel;
    uint64_t taskWastTime;
    uint8_t machineConfig;
    ToSubMachineConfig toSubMachineConfig;
    OpMetaAddrs opMetaAddrs;
} DeviceKernelArgs;

// PTO-ISA runtime mailbox (owned by this repo; stored in DeviceArgs.opaque)
typedef struct {
    Handshake *hankArgs;         // Device handshake array
    void *graphArgs;             // Device Graph pointer (if using Graph)
    int64_t core_num;            // Total cores
} PtoRuntimeArgs;

// =============================================================================
// Internal State Variables
// =============================================================================

static Handshake* g_host_handshake = NULL;
static int g_total_cores = 0;

// Kernel function table - maps func_id to device GM address of compiled kernel
#define A2A3_MAX_KERNELS 256
static void* g_kernel_func_table[A2A3_MAX_KERNELS] = {NULL};

// AICPU SO info
static void* g_aicpu_so_device = NULL;      // Device GM address of AICPU SO
static size_t g_aicpu_so_size = 0;

// Device-side kernel args
static DeviceArgs g_device_args;
static DeviceKernelArgs g_kernel_args;
static PtoRuntimeArgs g_runtime_args;
static void* g_runtime_args_device = NULL;   // Device copy of PtoRuntimeArgs
static void* g_device_args_device = NULL;    // Device copy of DeviceArgs

// AICore kernel binary
static uint8_t* g_aicore_kernel_data = NULL;
static size_t g_aicore_kernel_size = 0;

// PTO-ISA root path (for kernel compilation)
static char g_pto_isa_root[1024] = {0};

// Error messages
static const char* g_error_messages[] = {
    "Success",
    "Invalid configuration",
    "Failed to load .so file",
    "Function not found",
    "Memory allocation failed",
    "Thread creation failed",
    "Runtime not initialized",
    "Runtime already initialized",
    "Binary load failed",
    "Device launch failed",
};

// =============================================================================
// Error Handling
// =============================================================================

const char* a2a3_runtime_error_string(int error_code) {
    if (error_code > 0 || error_code < -9) {
        return "Unknown error";
    }
    return g_error_messages[-error_code];
}

// =============================================================================
// Helper Functions for Device Execution
// =============================================================================

#ifdef CANN_SDK_AVAILABLE

/**
 * Load AICPU SO binary to device GM memory
 * Reference: devicerunner.cpp:22-48 AicpuSoInfo::Init
 */
static int load_aicpu_so_to_device(const char* so_path) {
    if (!so_path) {
        fprintf(stderr, "[A2A3] ERROR: AICPU SO path is NULL\n");
        return A2A3_ERROR_INVALID_CONFIG;
    }

    // 1. Read .so file into memory
    FILE* f = fopen(so_path, "rb");
    if (!f) {
        fprintf(stderr, "[A2A3] ERROR: Failed to open AICPU SO: %s\n", so_path);
        return A2A3_ERROR_SO_LOAD_FAILED;
    }

    fseek(f, 0, SEEK_END);
    size_t file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    uint8_t* buffer = (uint8_t*)malloc(file_size);
    if (!buffer) {
        fclose(f);
        fprintf(stderr, "[A2A3] ERROR: Failed to allocate buffer for AICPU SO\n");
        return A2A3_ERROR_MEMORY_ALLOC;
    }

    fread(buffer, 1, file_size, f);
    fclose(f);

    // 2. Allocate device memory
    aclError rc = aclrtMalloc(&g_aicpu_so_device, file_size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (rc != ACL_SUCCESS) {
        fprintf(stderr, "[A2A3] ERROR: Failed to allocate device memory for AICPU SO: %d\n", rc);
        free(buffer);
        return A2A3_ERROR_MEMORY_ALLOC;
    }

    // 3. Copy to device
    rc = aclrtMemcpy(g_aicpu_so_device, file_size, buffer, file_size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (rc != ACL_SUCCESS) {
        fprintf(stderr, "[A2A3] ERROR: Failed to copy AICPU SO to device: %d\n", rc);
        aclrtFree(g_aicpu_so_device);
        free(buffer);
        return A2A3_ERROR_DEVICE_LAUNCH;
    }

    g_aicpu_so_size = file_size;
    free(buffer);

    // 4. Store in DeviceArgs
    g_device_args.aicpuSoBin = (uint64_t)g_aicpu_so_device;
    g_device_args.aicpuSoLen = file_size;

    printf("[A2A3] Loaded AICPU SO to device: %s (%zu bytes)\n", so_path, file_size);
    return A2A3_SUCCESS;
}

/**
 * Launch AICPU kernel
 * Reference: devicerunner.cpp:531-552 LaunchAiCpuKernel
 */
static int launch_aicpu_kernel(const char* kernel_name, int aicpu_num) {
    struct Args {
        DeviceKernelArgs kArgs;
        char kernelName[32];
        char soName[32];
        char opName[32];
    } args;

    memset(&args, 0, sizeof(args));
    args.kArgs = g_kernel_args;
    strncpy(args.kernelName, kernel_name, sizeof(args.kernelName) - 1);
    strncpy(args.soName, "libaicpu_extend_kernels.so", sizeof(args.soName) - 1);

    // DEBUG: Print struct sizes and offsets
    printf("[A2A3] DEBUG AICPU launch args:\n");
    printf("[A2A3]   sizeof(DeviceKernelArgs) = %zu\n", sizeof(DeviceKernelArgs));
    printf("[A2A3]   sizeof(Args) = %zu\n", sizeof(args));
    printf("[A2A3]   kernelName offset = %zu\n", offsetof(struct Args, kernelName));
    printf("[A2A3]   soName offset = %zu\n", offsetof(struct Args, soName));
    printf("[A2A3]   opName offset = %zu\n", offsetof(struct Args, opName));
    printf("[A2A3]   args.kernelName = \"%s\"\n", args.kernelName);
    printf("[A2A3]   args.soName = \"%s\"\n", args.soName);
    printf("[A2A3]   args.opName = \"%s\"\n", args.opName);
    printf("[A2A3]   args.kArgs.cfgdata = %p\n", (void*)args.kArgs.cfgdata);

    rtAicpuArgsEx_t rtArgs;
    memset(&rtArgs, 0, sizeof(rtArgs));
    rtArgs.args = &args;
    rtArgs.argsSize = sizeof(args);
    rtArgs.kernelNameAddrOffset = offsetof(struct Args, kernelName);
    rtArgs.soNameAddrOffset = offsetof(struct Args, soName);

    int rc = rtAicpuKernelLaunchExWithArgs(
        KERNEL_TYPE_AICPU_KFC,
        "AST_DYN_AICPU",
        aicpu_num,
        &rtArgs,
        NULL,
        g_stream_aicpu,
        0
    );

    if (rc != RT_ERROR_NONE) {
        fprintf(stderr, "[A2A3] ERROR: rtAicpuKernelLaunchExWithArgs(%s) failed: %d\n", kernel_name, rc);
        return A2A3_ERROR_DEVICE_LAUNCH;
    }

    printf("[A2A3] Launched AICPU kernel: %s\n", kernel_name);
    return A2A3_SUCCESS;
}

/**
 * Launch AICore kernel
 * Reference: devicerunner.cpp:554-599 LauncherAicoreKernel
 */
static int launch_aicore_kernel(void) {
    if (!g_aicore_kernel_data || g_aicore_kernel_size == 0) {
        fprintf(stderr, "[A2A3] ERROR: AICore kernel binary not loaded\n");
        return A2A3_ERROR_BINARY_LOAD_FAILED;
    }

    // 1. Register kernel binary
    rtDevBinary_t binary;
    memset(&binary, 0, sizeof(binary));
    binary.magic = RT_DEV_BINARY_MAGIC_ELF;
    binary.version = 0;
    binary.data = g_aicore_kernel_data;
    binary.length = g_aicore_kernel_size;

    void* binHandle = NULL;
    int rc = rtRegisterAllKernel(&binary, &binHandle);
    if (rc != RT_ERROR_NONE) {
        fprintf(stderr, "[A2A3] ERROR: rtRegisterAllKernel failed: %d\n", rc);
        return A2A3_ERROR_DEVICE_LAUNCH;
    }

    g_aicore_bin_handle = binHandle;

    // 2. Prepare kernel args (handshake pointer)
    struct {
        Handshake* hankArgs;
    } args = { (Handshake*)g_device_handshake };

    rtArgsEx_t rtArgs;
    memset(&rtArgs, 0, sizeof(rtArgs));
    rtArgs.args = &args;
    rtArgs.argsSize = sizeof(args);

    // 3. Launch kernel
    rtTaskCfgInfo_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.schemMode = RT_SCHEM_MODE_BATCH;

    uint32_t blockDim = (g_config.num_aic_workers > 0) ? g_config.num_aic_workers : 1;

    rc = rtKernelLaunchWithHandleV2(binHandle, 0, blockDim, &rtArgs, NULL, g_stream_aicore, &cfg);
    if (rc != RT_ERROR_NONE) {
        fprintf(stderr, "[A2A3] ERROR: rtKernelLaunchWithHandleV2 failed: %d\n", rc);
        return A2A3_ERROR_DEVICE_LAUNCH;
    }

    printf("[A2A3] Launched AICore kernel with %d blocks\n", blockDim);
    return A2A3_SUCCESS;
}

#endif // CANN_SDK_AVAILABLE

// =============================================================================
// Runtime Lifecycle
// =============================================================================

int a2a3_runtime_init(A2A3RuntimeConfig* config) {
    if (g_initialized) {
        fprintf(stderr, "[A2A3 Runtime] ERROR: Runtime already initialized\n");
        return A2A3_ERROR_ALREADY_INIT;
    }
    
    if (!config) {
        fprintf(stderr, "[A2A3 Runtime] ERROR: NULL config\n");
        return A2A3_ERROR_INVALID_CONFIG;
    }
    
    memcpy(&g_config, config, sizeof(A2A3RuntimeConfig));
    
    // Apply defaults
    if (g_config.num_aiv_workers < 1) g_config.num_aiv_workers = A2A3_DEFAULT_AIV_WORKERS;
    if (g_config.num_aic_workers < 1) g_config.num_aic_workers = A2A3_DEFAULT_AIC_WORKERS;
    
    g_total_cores = g_config.num_aic_workers + g_config.num_aiv_workers;
    
    printf("[A2A3 Runtime] ================================================\n");
    printf("[A2A3 Runtime] Initializing Ascend A2/A3 Runtime\n");
    printf("[A2A3 Runtime]   Device execution model:\n");
    printf("[A2A3 Runtime]     - AICore workers (polling loop): %d AIC + %d AIV\n",
           g_config.num_aic_workers, g_config.num_aiv_workers);
    printf("[A2A3 Runtime]     - AICPU scheduler: task distribution\n");
    printf("[A2A3 Runtime]     - AICPU orchestration: dynamic task generation\n");
    printf("[A2A3 Runtime] ================================================\n");
    
#ifdef CANN_SDK_AVAILABLE
    // Initialize ACL
    aclError rc = aclInit(NULL);
    if (rc != ACL_SUCCESS && rc != ACL_ERROR_REPEAT_INITIALIZE) {
        fprintf(stderr, "[A2A3 Runtime] ERROR: aclInit failed: %d\n", rc);
        return A2A3_ERROR_DEVICE_LAUNCH;
    }
    
    // Set device
    rc = aclrtSetDevice(0);
    if (rc != ACL_SUCCESS) {
        fprintf(stderr, "[A2A3 Runtime] ERROR: aclrtSetDevice failed: %d\n", rc);
        return A2A3_ERROR_DEVICE_LAUNCH;
    }
    
    // Create streams
    rc = aclrtCreateStream(&g_stream_aicpu);
    if (rc != ACL_SUCCESS) {
        fprintf(stderr, "[A2A3 Runtime] ERROR: Failed to create AICPU stream: %d\n", rc);
        return A2A3_ERROR_DEVICE_LAUNCH;
    }
    
    rc = aclrtCreateStream(&g_stream_aicore);
    if (rc != ACL_SUCCESS) {
        fprintf(stderr, "[A2A3 Runtime] ERROR: Failed to create AICore stream: %d\n", rc);
        aclrtDestroyStream(g_stream_aicpu);
        return A2A3_ERROR_DEVICE_LAUNCH;
    }
    
    printf("[A2A3 Runtime] ACL initialized, streams created\n");
#else
    printf("[A2A3 Runtime] WARNING: CANN SDK not available, stub mode\n");
#endif
    
    // Initialize SO loader (for orchestration .so)
    a2a3_so_loader_init();
    
    // Load InCore binaries (.o files) for AICore execution
    if (g_config.incore_aiv_dir) {
        int count = a2a3_load_incore_dir(g_config.incore_aiv_dir, false);
        printf("[A2A3 Runtime] Loaded %d AIV InCore binaries\n", count);
    }
    
    if (g_config.incore_aic_dir) {
        int count = a2a3_load_incore_dir(g_config.incore_aic_dir, true);
        printf("[A2A3 Runtime] Loaded %d AIC InCore binaries\n", count);
    }
    
    // Allocate host handshake buffer
    g_host_handshake = (Handshake*)calloc(g_total_cores, sizeof(Handshake));
    if (!g_host_handshake) {
        fprintf(stderr, "[A2A3 Runtime] ERROR: Failed to allocate handshake buffer\n");
        return A2A3_ERROR_MEMORY_ALLOC;
    }
    
    // Initialize handshake (AIC cores first, then AIV)
    for (int i = 0; i < g_total_cores; i++) {
        g_host_handshake[i].aicpu_ready = 0;
        g_host_handshake[i].aicore_done = 0;
        g_host_handshake[i].control = 0;
        g_host_handshake[i].task = 0;
        g_host_handshake[i].task_status = 0;
        g_host_handshake[i].core_type = (i < g_config.num_aic_workers) ? 0 : 1;
    }
    
#ifdef CANN_SDK_AVAILABLE
    // Allocate device handshake buffer
    size_t handshake_size = sizeof(Handshake) * g_total_cores;
    rc = aclrtMalloc(&g_device_handshake, handshake_size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (rc != ACL_SUCCESS) {
        fprintf(stderr, "[A2A3 Runtime] ERROR: Failed to allocate device handshake: %d\n", rc);
        free(g_host_handshake);
        return A2A3_ERROR_MEMORY_ALLOC;
    }
    
    // Copy handshake to device
    rc = aclrtMemcpy(g_device_handshake, handshake_size, 
                     g_host_handshake, handshake_size,
                     ACL_MEMCPY_HOST_TO_DEVICE);
    if (rc != ACL_SUCCESS) {
        fprintf(stderr, "[A2A3 Runtime] ERROR: Failed to copy handshake to device: %d\n", rc);
        aclrtFree(g_device_handshake);
        free(g_host_handshake);
        return A2A3_ERROR_DEVICE_LAUNCH;
    }
    
    printf("[A2A3 Runtime] Handshake buffer allocated: %zu bytes for %d cores\n",
           handshake_size, g_total_cores);

    // =========================================================================
    // Device Args Setup for Kernel Launches
    // =========================================================================

    // Load AICPU SO to device (if path provided)
    if (g_config.aicpu_kernel_path) {
        int so_rc = load_aicpu_so_to_device(g_config.aicpu_kernel_path);
        if (so_rc != A2A3_SUCCESS) {
            fprintf(stderr, "[A2A3 Runtime] ERROR: Failed to load AICPU SO\n");
            aclrtFree(g_device_handshake);
            free(g_host_handshake);
            return so_rc;
        }
    } else {
        printf("[A2A3 Runtime] WARNING: No AICPU kernel path provided\n");
    }

    // Allocate and initialize PtoRuntimeArgs on device
    rc = aclrtMalloc(&g_runtime_args_device, sizeof(PtoRuntimeArgs), ACL_MEM_MALLOC_HUGE_FIRST);
    if (rc != ACL_SUCCESS) {
        fprintf(stderr, "[A2A3 Runtime] ERROR: Failed to allocate PtoRuntimeArgs: %d\n", rc);
        aclrtFree(g_device_handshake);
        free(g_host_handshake);
        return A2A3_ERROR_MEMORY_ALLOC;
    }

    g_runtime_args.hankArgs = (Handshake*)g_device_handshake;
    g_runtime_args.graphArgs = NULL;  // Will be set during execute
    g_runtime_args.core_num = g_total_cores;

    rc = aclrtMemcpy(g_runtime_args_device, sizeof(PtoRuntimeArgs),
                     &g_runtime_args, sizeof(PtoRuntimeArgs),
                     ACL_MEMCPY_HOST_TO_DEVICE);
    if (rc != ACL_SUCCESS) {
        fprintf(stderr, "[A2A3 Runtime] ERROR: Failed to copy PtoRuntimeArgs: %d\n", rc);
        aclrtFree(g_runtime_args_device);
        aclrtFree(g_device_handshake);
        free(g_host_handshake);
        return A2A3_ERROR_DEVICE_LAUNCH;
    }

    printf("[A2A3 Runtime] PtoRuntimeArgs allocated and initialized\n");

    // Initialize DeviceArgs
    memset(&g_device_args, 0, sizeof(DeviceArgs));
    g_device_args.nrAic = g_config.num_aic_workers;
    g_device_args.nrAiv = g_config.num_aiv_workers;
    g_device_args.nrAicpu = 1;
    g_device_args.nrValidAic = g_config.num_aic_workers;
    g_device_args.opaque = (uint64_t)g_runtime_args_device;
    g_device_args.deviceId = 0;
    // aicpuSoBin and aicpuSoLen are set by load_aicpu_so_to_device

    // Copy DeviceArgs to device
    rc = aclrtMalloc(&g_device_args_device, sizeof(DeviceArgs), ACL_MEM_MALLOC_HUGE_FIRST);
    if (rc != ACL_SUCCESS) {
        fprintf(stderr, "[A2A3 Runtime] ERROR: Failed to allocate DeviceArgs: %d\n", rc);
        aclrtFree(g_runtime_args_device);
        aclrtFree(g_device_handshake);
        free(g_host_handshake);
        return A2A3_ERROR_MEMORY_ALLOC;
    }

    rc = aclrtMemcpy(g_device_args_device, sizeof(DeviceArgs),
                     &g_device_args, sizeof(DeviceArgs),
                     ACL_MEMCPY_HOST_TO_DEVICE);
    if (rc != ACL_SUCCESS) {
        fprintf(stderr, "[A2A3 Runtime] ERROR: Failed to copy DeviceArgs: %d\n", rc);
        aclrtFree(g_device_args_device);
        aclrtFree(g_runtime_args_device);
        aclrtFree(g_device_handshake);
        free(g_host_handshake);
        return A2A3_ERROR_DEVICE_LAUNCH;
    }

    // Initialize DeviceKernelArgs
    memset(&g_kernel_args, 0, sizeof(DeviceKernelArgs));
    g_kernel_args.cfgdata = (int64_t*)g_device_args_device;

    printf("[A2A3 Runtime] DeviceArgs and DeviceKernelArgs initialized\n");

    // Load AICore kernel binary (ENTIRE ELF, not just .text section)
    // Note: rtRegisterAllKernel requires the complete ELF binary
    if (g_config.aicore_kernel_path) {
        FILE* f = fopen(g_config.aicore_kernel_path, "rb");
        if (f) {
            fseek(f, 0, SEEK_END);
            g_aicore_kernel_size = ftell(f);
            fseek(f, 0, SEEK_SET);
            g_aicore_kernel_data = (uint8_t*)malloc(g_aicore_kernel_size);
            if (g_aicore_kernel_data) {
                size_t bytes_read = fread(g_aicore_kernel_data, 1, g_aicore_kernel_size, f);
                if (bytes_read == g_aicore_kernel_size) {
                    printf("[A2A3 Runtime] Loaded AICore kernel: %s (%zu bytes)\n",
                           g_config.aicore_kernel_path, g_aicore_kernel_size);
                } else {
                    printf("[A2A3 Runtime] WARNING: Failed to read AICore kernel fully\n");
                    free(g_aicore_kernel_data);
                    g_aicore_kernel_data = NULL;
                    g_aicore_kernel_size = 0;
                }
            }
            fclose(f);
        } else {
            printf("[A2A3 Runtime] WARNING: Could not open AICore kernel file\n");
        }
    } else {
        printf("[A2A3 Runtime] WARNING: No AICore kernel path provided\n");
    }

    // Copy InCore binaries to device GM
    int incore_rc = a2a3_copy_incore_binaries_to_device();
    if (incore_rc < 0) {
        fprintf(stderr, "[A2A3 Runtime] WARNING: Failed to copy InCore binaries to device\n");
    } else {
        printf("[A2A3 Runtime] Copied %d InCore binaries to device GM\n", incore_rc);
    }
#endif
    
    memset(&g_stats, 0, sizeof(g_stats));
    g_stats.num_incore_funcs_loaded = a2a3_get_incore_count();
    
    g_initialized = true;
    
    printf("[A2A3 Runtime] Initialization complete\n");
    printf("[A2A3 Runtime]   Loaded %d InCore functions\n", g_stats.num_incore_funcs_loaded);
    
    return A2A3_SUCCESS;
}

int a2a3_runtime_execute(void* user_data) {
    if (!g_initialized) {
        fprintf(stderr, "[A2A3 Runtime] ERROR: Runtime not initialized\n");
        return A2A3_ERROR_NOT_INITIALIZED;
    }

    printf("[A2A3 Runtime] ================================================\n");
    printf("[A2A3 Runtime] Starting device execution\n");
    printf("[A2A3 Runtime] ================================================\n");

    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);

#ifdef CANN_SDK_AVAILABLE
    aclError rc;
    int launch_rc;

    // =========================================================================
    // Step 1: Reset handshake buffers
    // =========================================================================
    printf("[A2A3 Runtime] Step 1: Resetting handshake buffers...\n");

    for (int i = 0; i < g_total_cores; i++) {
        g_host_handshake[i].aicpu_ready = 0;
        g_host_handshake[i].aicore_done = 0;
        g_host_handshake[i].control = 0;
        g_host_handshake[i].task = 0;
        g_host_handshake[i].task_status = 0;
        g_host_handshake[i].profile_enable = 0;
        g_host_handshake[i].core_type = (i < g_config.num_aic_workers) ? 0 : 1;  // 0=AIC, 1=AIV
    }

    size_t hank_size = sizeof(Handshake) * g_total_cores;
    rc = aclrtMemcpy(g_device_handshake, hank_size,
                     g_host_handshake, hank_size,
                     ACL_MEMCPY_HOST_TO_DEVICE);
    if (rc != ACL_SUCCESS) {
        fprintf(stderr, "[A2A3 Runtime] ERROR: Failed to reset handshakes: %d\n", rc);
        return A2A3_ERROR_DEVICE_LAUNCH;
    }

    printf("[A2A3 Runtime]   Reset %d handshake buffers\n", g_total_cores);

    // =========================================================================
    // Step 2: Update DeviceArgs on device
    // =========================================================================
    printf("[A2A3 Runtime] Step 2: Updating DeviceArgs...\n");

    g_device_args.nrAic = g_config.num_aic_workers;
    g_device_args.nrAiv = g_config.num_aiv_workers;
    g_device_args.nrValidAic = g_config.num_aic_workers;

    rc = aclrtMemcpy(g_device_args_device, sizeof(DeviceArgs),
                     &g_device_args, sizeof(DeviceArgs),
                     ACL_MEMCPY_HOST_TO_DEVICE);
    if (rc != ACL_SUCCESS) {
        fprintf(stderr, "[A2A3 Runtime] ERROR: Failed to update DeviceArgs: %d\n", rc);
        return A2A3_ERROR_DEVICE_LAUNCH;
    }

    printf("[A2A3 Runtime]   DeviceArgs updated (nrAic=%d, nrAiv=%d)\n",
           g_config.num_aic_workers, g_config.num_aiv_workers);

    // =========================================================================
    // Step 3: Launch AICPU init kernel
    // =========================================================================
    printf("[A2A3 Runtime] Step 3: Launching AICPU init kernel...\n");

    launch_rc = launch_aicpu_kernel("DynTileFwkKernelServerInit", 1);
    if (launch_rc != A2A3_SUCCESS) {
        fprintf(stderr, "[A2A3 Runtime] ERROR: Failed to launch AICPU init kernel\n");
        return launch_rc;
    }

    // =========================================================================
    // Step 4: Launch AICPU main kernel (scheduler + orchestration)
    // =========================================================================
    printf("[A2A3 Runtime] Step 4: Launching AICPU main kernel...\n");
    printf("[A2A3 Runtime]   AICPU will run: HankAiCore() -> execute_graph() -> ShutdownAiCore()\n");

    launch_rc = launch_aicpu_kernel("DynTileFwkKernelServer", 1);
    if (launch_rc != A2A3_SUCCESS) {
        fprintf(stderr, "[A2A3 Runtime] ERROR: Failed to launch AICPU main kernel\n");
        return launch_rc;
    }

    // =========================================================================
    // Step 5: Launch AICore kernel (workers enter polling loop)
    // =========================================================================
    printf("[A2A3 Runtime] Step 5: Launching AICore kernel...\n");
    printf("[A2A3 Runtime]   %d AIC + %d AIV cores will enter polling loop\n",
           g_config.num_aic_workers, g_config.num_aiv_workers);

    if (g_aicore_kernel_data && g_aicore_kernel_size > 0) {
        launch_rc = launch_aicore_kernel();
        if (launch_rc != A2A3_SUCCESS) {
            fprintf(stderr, "[A2A3 Runtime] ERROR: Failed to launch AICore kernel\n");
            return launch_rc;
        }
    } else {
        printf("[A2A3 Runtime]   WARNING: AICore kernel not loaded, skipping launch\n");
    }

    // =========================================================================
    // Step 6: Synchronize streams
    // =========================================================================
    printf("[A2A3 Runtime] Step 6: Waiting for device execution...\n");

    // Synchronize AICPU stream (wait for orchestration + scheduler to complete)
    rc = aclrtSynchronizeStream(g_stream_aicpu);
    if (rc != ACL_SUCCESS) {
        fprintf(stderr, "[A2A3 Runtime] ERROR: AICPU sync failed: %d\n", rc);
        return A2A3_ERROR_DEVICE_LAUNCH;
    }
    printf("[A2A3 Runtime]   AICPU stream synchronized\n");

    // Synchronize AICore stream
    rc = aclrtSynchronizeStream(g_stream_aicore);
    if (rc != ACL_SUCCESS) {
        fprintf(stderr, "[A2A3 Runtime] ERROR: AICore sync failed: %d\n", rc);
        return A2A3_ERROR_DEVICE_LAUNCH;
    }
    printf("[A2A3 Runtime]   AICore stream synchronized\n");

    printf("[A2A3 Runtime]   Device execution complete\n");

#else
    // Stub mode - no actual device execution
    (void)user_data;  // Suppress unused warning
    printf("[A2A3 Runtime] ================================================\n");
    printf("[A2A3 Runtime] STUB MODE - No actual device execution\n");
    printf("[A2A3 Runtime] \n");
    printf("[A2A3 Runtime] Real execution requires:\n");
    printf("[A2A3 Runtime]   1. CANN SDK with runtime API (runtime/rt.h)\n");
    printf("[A2A3 Runtime]   2. Ascend NPU device\n");
    printf("[A2A3 Runtime]   3. AICore kernel binary (aicore_kernel.o)\n");
    printf("[A2A3 Runtime]   4. AICPU kernel binary (libaicpu_kernel.so)\n");
    printf("[A2A3 Runtime] \n");
    printf("[A2A3 Runtime] In stub mode, output will be zeros.\n");
    printf("[A2A3 Runtime] ================================================\n");
#endif

    clock_gettime(CLOCK_MONOTONIC, &end_time);

    g_stats.total_execution_time_ms =
        (end_time.tv_sec - start_time.tv_sec) * 1000.0 +
        (end_time.tv_nsec - start_time.tv_nsec) / 1000000.0;

    printf("[A2A3 Runtime] ================================================\n");
    printf("[A2A3 Runtime] Execution Complete\n");
    printf("[A2A3 Runtime]   Execution time: %.2f ms\n", g_stats.total_execution_time_ms);
    printf("[A2A3 Runtime] ================================================\n");

    return A2A3_SUCCESS;
}

void a2a3_runtime_finalize(void) {
    if (!g_initialized) return;

    printf("[A2A3 Runtime] Finalizing...\n");

#ifdef CANN_SDK_AVAILABLE
    // Free kernel function table
    for (int i = 0; i < A2A3_MAX_KERNELS; i++) {
        if (g_kernel_func_table[i]) {
            aclrtFree(g_kernel_func_table[i]);
            g_kernel_func_table[i] = NULL;
        }
    }

    // Free device handshake
    if (g_device_handshake) {
        aclrtFree(g_device_handshake);
        g_device_handshake = NULL;
    }

    // Free AICPU SO from device
    if (g_aicpu_so_device) {
        aclrtFree(g_aicpu_so_device);
        g_aicpu_so_device = NULL;
        g_aicpu_so_size = 0;
    }

    // Free PtoRuntimeArgs from device
    if (g_runtime_args_device) {
        aclrtFree(g_runtime_args_device);
        g_runtime_args_device = NULL;
    }

    // Free DeviceArgs from device
    if (g_device_args_device) {
        aclrtFree(g_device_args_device);
        g_device_args_device = NULL;
    }

    // Free AICore kernel binary (host)
    if (g_aicore_kernel_data) {
        free(g_aicore_kernel_data);
        g_aicore_kernel_data = NULL;
        g_aicore_kernel_size = 0;
    }

    // Destroy streams
    if (g_stream_aicpu) {
        aclrtDestroyStream(g_stream_aicpu);
        g_stream_aicpu = NULL;
    }
    if (g_stream_aicore) {
        aclrtDestroyStream(g_stream_aicore);
        g_stream_aicore = NULL;
    }

    // Reset device
    aclrtResetDevice(0);
    aclFinalize();
#endif

    // Free host handshake
    if (g_host_handshake) {
        free(g_host_handshake);
        g_host_handshake = NULL;
    }

    // Cleanup binary loader
    a2a3_unload_all_incore_binaries();

    // Cleanup SO loader
    a2a3_so_loader_cleanup();

    g_initialized = false;

    printf("[A2A3 Runtime] Finalized\n");
}

// =============================================================================
// Memory Management
// =============================================================================

void* a2a3_runtime_malloc(size_t size_bytes) {
    if (!g_initialized || size_bytes == 0) return NULL;
    
#ifdef CANN_SDK_AVAILABLE
    void* ptr = NULL;
    aclError rc = aclrtMalloc(&ptr, size_bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (rc != ACL_SUCCESS) {
        fprintf(stderr, "[A2A3 Runtime] ERROR: aclrtMalloc failed: %d\n", rc);
        return NULL;
    }
    return ptr;
#else
    return malloc(size_bytes);
#endif
}

void a2a3_runtime_free(void* ptr) {
    if (!ptr) return;
    
#ifdef CANN_SDK_AVAILABLE
    aclrtFree(ptr);
#else
    free(ptr);
#endif
}

int a2a3_runtime_copy_to_device(void* dst_device, const void* src_host, size_t size_bytes) {
    if (!g_initialized || !dst_device || !src_host || size_bytes == 0) {
        return A2A3_ERROR_INVALID_CONFIG;
    }
    
#ifdef CANN_SDK_AVAILABLE
    aclError rc = aclrtMemcpy(dst_device, size_bytes, src_host, size_bytes, 
                              ACL_MEMCPY_HOST_TO_DEVICE);
    return (rc == ACL_SUCCESS) ? A2A3_SUCCESS : A2A3_ERROR_DEVICE_LAUNCH;
#else
    memcpy(dst_device, src_host, size_bytes);
    return A2A3_SUCCESS;
#endif
}

int a2a3_runtime_copy_from_device(void* dst_host, const void* src_device, size_t size_bytes) {
    if (!g_initialized || !dst_host || !src_device || size_bytes == 0) {
        return A2A3_ERROR_INVALID_CONFIG;
    }
    
#ifdef CANN_SDK_AVAILABLE
    aclError rc = aclrtMemcpy(dst_host, size_bytes, src_device, size_bytes,
                              ACL_MEMCPY_DEVICE_TO_HOST);
    return (rc == ACL_SUCCESS) ? A2A3_SUCCESS : A2A3_ERROR_DEVICE_LAUNCH;
#else
    memcpy(dst_host, src_device, size_bytes);
    return A2A3_SUCCESS;
#endif
}

// =============================================================================
// InCore Function Registry
// =============================================================================

int a2a3_runtime_register_incore(const char* func_name, A2A3InCoreFunc func_ptr, bool is_cube) {
    return a2a3_register_incore(func_name, func_ptr, is_cube);
}

A2A3InCoreFunc a2a3_runtime_lookup_incore(const char* func_name) {
    return a2a3_lookup_incore(func_name);
}

// =============================================================================
// Kernel Compilation and Loading API
// =============================================================================

/**
 * Get function binary address for a given func_id
 * Returns the device GM address where the kernel binary resides
 */
void* a2a3_get_function_bin_addr(int func_id) {
    if (func_id < 0 || func_id >= A2A3_MAX_KERNELS) {
        return NULL;
    }
    return g_kernel_func_table[func_id];
}

#ifdef CANN_SDK_AVAILABLE

/**
 * Load a compiled kernel binary (.o file) to device GM
 * Reference: devicerunner.cpp:780-840 LoadSingleKernelToDevice
 */
static int a2a3_load_kernel_to_device(int func_id, const char* bin_path) {
    if (func_id < 0 || func_id >= A2A3_MAX_KERNELS) {
        fprintf(stderr, "[A2A3] ERROR: Invalid func_id: %d\n", func_id);
        return A2A3_ERROR_INVALID_CONFIG;
    }

    if (!bin_path) {
        fprintf(stderr, "[A2A3] ERROR: Binary path is NULL\n");
        return A2A3_ERROR_INVALID_CONFIG;
    }

    // 1. Read .o file
    FILE* f = fopen(bin_path, "rb");
    if (!f) {
        fprintf(stderr, "[A2A3] ERROR: Failed to open kernel binary: %s\n", bin_path);
        return A2A3_ERROR_BINARY_LOAD_FAILED;
    }

    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    rewind(f);

    uint8_t* buf = (uint8_t*)malloc(size);
    if (!buf) {
        fclose(f);
        fprintf(stderr, "[A2A3] ERROR: Failed to allocate buffer for kernel\n");
        return A2A3_ERROR_MEMORY_ALLOC;
    }

    size_t read_bytes = fread(buf, 1, size, f);
    fclose(f);

    if (read_bytes != size) {
        free(buf);
        fprintf(stderr, "[A2A3] ERROR: Failed to read kernel binary\n");
        return A2A3_ERROR_BINARY_LOAD_FAILED;
    }

    // 2. Allocate device GM
    void* dev_ptr = NULL;
    aclError rc = aclrtMalloc(&dev_ptr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (rc != ACL_SUCCESS) {
        fprintf(stderr, "[A2A3] ERROR: Failed to allocate device GM for kernel: %d\n", rc);
        free(buf);
        return A2A3_ERROR_MEMORY_ALLOC;
    }

    // 3. Copy to device
    rc = aclrtMemcpy(dev_ptr, size, buf, size, ACL_MEMCPY_HOST_TO_DEVICE);
    free(buf);

    if (rc != ACL_SUCCESS) {
        fprintf(stderr, "[A2A3] ERROR: Failed to copy kernel to device: %d\n", rc);
        aclrtFree(dev_ptr);
        return A2A3_ERROR_DEVICE_LAUNCH;
    }

    // 4. Free old kernel if exists
    if (g_kernel_func_table[func_id]) {
        aclrtFree(g_kernel_func_table[func_id]);
    }

    // 5. Store in table
    g_kernel_func_table[func_id] = dev_ptr;

    printf("[A2A3] Loaded kernel func_id=%d -> 0x%lx (%zu bytes)\n",
           func_id, (uint64_t)dev_ptr, size);

    return A2A3_SUCCESS;
}

/**
 * Compile and load kernel from CCE C++ source
 * Reference: pto_runtime_c_api.cpp:268-280 DeviceRunner_CompileAndLoadKernel
 *
 * Note: This is a simplified version that expects pre-compiled .o files.
 * Full compilation (via ccec) would be added in a2a3_kernel_compiler.c
 */
int a2a3_compile_and_load_kernel(int func_id, const char* cpp_path,
                                   const char* pto_isa_root, int core_type) {
    if (!g_initialized) {
        fprintf(stderr, "[A2A3] ERROR: Runtime not initialized\n");
        return A2A3_ERROR_NOT_INITIALIZED;
    }

    // For now, expect cpp_path to already be a compiled .o file
    // Full implementation would:
    // 1. Check if cpp_path ends with .cpp
    // 2. Call a2a3_compile_kernel(cpp_path, output_o, pto_isa_root, core_type)
    // 3. Load the output .o file

    printf("[A2A3] Loading kernel func_id=%d from %s (core_type=%d)\n",
           func_id, cpp_path, core_type);

    // Store PTO-ISA root for future compilations
    if (pto_isa_root && strlen(pto_isa_root) < sizeof(g_pto_isa_root)) {
        strncpy(g_pto_isa_root, pto_isa_root, sizeof(g_pto_isa_root) - 1);
        g_pto_isa_root[sizeof(g_pto_isa_root) - 1] = '\0';
    }

    // Load the binary to device
    int rc = a2a3_load_kernel_to_device(func_id, cpp_path);
    if (rc != A2A3_SUCCESS) {
        fprintf(stderr, "[A2A3] ERROR: Failed to load kernel to device\n");
        return rc;
    }

    return A2A3_SUCCESS;
}

#else  // !CANN_SDK_AVAILABLE

// Stub implementations for non-CANN builds
int a2a3_compile_and_load_kernel(int func_id, const char* cpp_path,
                                   const char* pto_isa_root, int core_type) {
    (void)func_id;
    (void)cpp_path;
    (void)pto_isa_root;
    (void)core_type;
    fprintf(stderr, "[A2A3] ERROR: Kernel compilation requires CANN SDK\n");
    return A2A3_ERROR_NOT_INITIALIZED;
}

#endif  // CANN_SDK_AVAILABLE

// =============================================================================
// Statistics
// =============================================================================

void a2a3_runtime_get_stats(A2A3RuntimeStats* stats) {
    if (stats) {
        memcpy(stats, &g_stats, sizeof(A2A3RuntimeStats));
    }
}

void a2a3_runtime_print_stats(void) {
    printf("\n=== A2A3 Runtime Statistics ===\n");
    printf("Tasks Scheduled:     %lld\n", (long long)g_stats.total_tasks_scheduled);
    printf("Tasks Completed:     %lld\n", (long long)g_stats.total_tasks_completed);
    printf("AIV Tasks:           %lld\n", (long long)g_stats.aiv_tasks_executed);
    printf("AIC Tasks:           %lld\n", (long long)g_stats.aic_tasks_executed);
    printf("Execution Time:      %.2f ms\n", g_stats.total_execution_time_ms);
    printf("InCore Functions:    %d\n", g_stats.num_incore_funcs_loaded);
    printf("================================\n\n");
}

bool a2a3_runtime_is_initialized(void) {
    return g_initialized;
}
