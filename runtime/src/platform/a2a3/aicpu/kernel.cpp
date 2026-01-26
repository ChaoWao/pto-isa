#include <cstdint>
#include <cstdio>
#include <atomic>
#include <mutex>
#include "device_log.h"
#include "graph.h"
#include "kernel_args.h"

// Forward declaration of execute function (defined in execute.cpp)
extern int execute(Graph& g, Handshake* hank, int thread_num, int thread_idx,
                   const int* cur_thread_cores, int core_num);

constexpr int MAX_AICPU_THREADS = 4;
constexpr int MAX_AIC_PER_THREAD = 24;
constexpr int MAX_AIV_PER_THREAD = 48;
constexpr int MAX_CORES_PER_THREAD = MAX_AIC_PER_THREAD + MAX_AIV_PER_THREAD;

struct MultiThreadManager {
    std::atomic<int> thread_idx_{0};
    std::atomic<bool> initialized_{false};
    std::atomic<bool> init_done_{false};
    std::atomic<bool> init_failed_{false};
    std::atomic<int> finished_{0};

    int thread_num_{0};
    int total_cores_{0};
    int cores_per_thread_{0};
    int core_assignments_[MAX_AICPU_THREADS][MAX_CORES_PER_THREAD];

    int Init(KernelArgs* kargs) {
        bool expected = false;
        if (!initialized_.compare_exchange_strong(expected, true, std::memory_order_seq_cst)) {
            return 0;
        }

        DEV_INFO("MultiThreadManager: Initializing");

        DeviceArgs* devArgs = (DeviceArgs*)kargs->deviceArgs;
        thread_num_ = devArgs->scheCpuNum;
        if (thread_num_ == 0) thread_num_ = 1;

        total_cores_ = kargs->core_num;
        cores_per_thread_ = total_cores_ / thread_num_;

        DEV_INFO("Config: threads=%d, cores=%d, cores_per_thread=%d",
                 thread_num_, total_cores_, cores_per_thread_);

        if (thread_num_ < 1 || thread_num_ > MAX_AICPU_THREADS) {
            DEV_ERROR("Invalid thread_num: %d", thread_num_);
            init_failed_.store(true, std::memory_order_release);
            return -1;
        }

        if (cores_per_thread_ > MAX_CORES_PER_THREAD) {
            DEV_ERROR("Cores per thread %d exceeds maximum %d", cores_per_thread_, MAX_CORES_PER_THREAD);
            init_failed_.store(true, std::memory_order_release);
            return -1;
        }

        // Pre-compute core assignments for each thread
        int num_aic = devArgs->nrAic;
        for (int t = 0; t < thread_num_; t++) {
            // Each thread manages: 1 AIC + 2 AIV
            int aic_idx = t;
            int aiv_base = num_aic;
            int aiv_idx0 = aiv_base + t * 2;
            int aiv_idx1 = aiv_idx0 + 1;

            core_assignments_[t][0] = aic_idx;
            core_assignments_[t][1] = aiv_idx0;
            core_assignments_[t][2] = aiv_idx1;

            DEV_INFO("Thread %d: AIC[%d] AIV[%d,%d]", t, aic_idx, aiv_idx0, aiv_idx1);
        }

        init_done_.store(true, std::memory_order_release);
        DEV_INFO("MultiThreadManager: Init complete");
        return 0;
    }

    /**
     * Handshake AICore - Initialize and synchronize with AICore kernels
     *
     * This function performs the initial handshake protocol with all AICore instances:
     * 1. Set aicpu_ready flag for each core
     * 2. Wait for each core to respond with aicore_done signal
     *
     * This ensures all cores are running and ready to receive tasks before
     * graph execution begins.
     *
     * @param arg Pointer to KernelArgs structure containing handshake buffers
     * @param thread_idx Thread index
     * @param cur_thread_cores Array of core IDs assigned to this thread
     * @return 0 on success
     */
    int HankAiCore(void *arg, int thread_idx, const int* cur_thread_cores) {
        auto kargs = (KernelArgs *)arg;
        Handshake* all_hanks = (Handshake*)kargs->hankArgs;

        DEV_INFO("Thread %d: Handshaking with %d cores", thread_idx, cores_per_thread_);

        for (int i = 0; i < cores_per_thread_; i++) {
            int core_id = cur_thread_cores[i];
            Handshake* hank = &all_hanks[core_id];
            DEV_INFO("Thread %d: AICPU hank addr = 0x%lx", thread_idx, (uint64_t)hank);
            hank->aicpu_ready = 1;
        }

        for (int i = 0; i < cores_per_thread_; i++) {
            int core_id = cur_thread_cores[i];
            Handshake* hank = &all_hanks[core_id];
            while (hank->aicore_done == 0) { }
            DEV_INFO("Thread %d: success hank->aicore_done = %u", thread_idx, (uint64_t)hank->aicore_done);
        }
        return 0;
    }

    /**
     * Shutdown AICore - Send quit signal to all AICore kernels
     *
     * Sets the control flag to 1 for all cores, signaling them to exit
     * their execution loops and terminate gracefully.
     *
     * @param arg Pointer to KernelArgs structure containing handshake buffers
     * @param thread_idx Thread index
     * @param cur_thread_cores Array of core IDs assigned to this thread
     * @return 0 on success
     */
    int ShutdownAiCore(void *arg, int thread_idx, const int* cur_thread_cores) {
        auto kargs = (KernelArgs *)arg;
        Handshake* all_hanks = (Handshake*)kargs->hankArgs;

        DEV_INFO("Thread %d: Shutting down %d cores", thread_idx, cores_per_thread_);

        for (int i = 0; i < cores_per_thread_; i++) {
            int core_id = cur_thread_cores[i];
            Handshake* hank = &all_hanks[core_id];
            DEV_INFO("Thread %d: AICPU hank addr = 0x%lx", thread_idx, (uint64_t)hank);
            hank->control = 1;
        }
        DEV_INFO("Thread %d: Shutdown complete", thread_idx);
        return 0;
    }

    int Run(void *arg) {
        int thread_idx = thread_idx_++;

        auto kargs = (KernelArgs *)arg;

        DEV_INFO("Thread %d: Start", thread_idx);

        const int* cur_thread_cores = core_assignments_[thread_idx];

        auto rc = HankAiCore(arg, thread_idx, cur_thread_cores);
        if (rc != 0) {
            return rc;
        }

        if (kargs->graphArgs != nullptr) {
            Graph* g = kargs->graphArgs;
            Handshake* hank = (Handshake*)kargs->hankArgs;
            DEV_INFO("Thread %d: Graph has %d tasks", thread_idx, g->get_task_count());
            int completed = execute(*g, hank, thread_num_, thread_idx, cur_thread_cores, cores_per_thread_);
            DEV_INFO("Thread %d: Executed %d tasks from graph", thread_idx, completed);
        }

        rc = ShutdownAiCore(arg, thread_idx, cur_thread_cores);
        if (rc != 0) {
            return rc;
        }

        DEV_INFO("Thread %d: Completed", thread_idx);

        return 0;
    }

    void DeInit() {
        initialized_.store(false, std::memory_order_release);
        init_done_.store(false, std::memory_order_release);
        init_failed_.store(false, std::memory_order_release);
        thread_idx_.store(0, std::memory_order_release);
        finished_.store(0, std::memory_order_release);
    }
};

static MultiThreadManager g_mt_mgr;

extern "C" __attribute__((visibility("default"))) int StaticTileFwkBackendKernelServer(void *arg) {
    if (arg == nullptr) {
        DEV_ERROR("%s", "Invalid kernel arguments: null pointer");
        return -1;
    }

    return 0;
}

/**
 * AICPU kernel initialization entry point
 *
 * This function is called once during kernel initialization by the CANN runtime.
 * It initializes logging and validates kernel arguments.
 *
 * Note: Function name is hardcoded in libaicpu_extend_kernels.so
 *
 * @param arg Pointer to KernelArgs structure
 * @return 0 on success, -1 on error
 */
extern "C" __attribute__((visibility("default"))) int DynTileFwkBackendKernelServerInit(void *arg) {
    InitLogSwitch();
    if (arg == nullptr) {
        DEV_ERROR("%s", "Invalid kernel arguments: null pointer");
        return -1;
    }

    DEV_INFO("%s", "Graph Executor Init: Initializing AICPU kernel");
    return 0;
}

/**
 * AICPU kernel main execution entry point
 *
 * This is the main entry point for the AICPU graph executor kernel.
 * It orchestrates the complete task graph execution:
 * 1. Handshake with all AICore instances
 * 2. Execute task graph using polling-based dispatch
 * 3. Shutdown all AICore instances
 *
 * Note: Function name is hardcoded in libaicpu_extend_kernels.so
 *
 * @param arg Pointer to KernelArgs structure containing:
 *            - deviceArgs: device-specific arguments
 *            - hankArgs: handshake buffer array
 *            - core_num: number of cores
 *            - graphArgs: task graph to execute
 * @return 0 on success, non-zero on error
 */
extern "C" __attribute__((visibility("default"))) int DynTileFwkBackendKernelServer(void *arg) {
    if (arg == nullptr) {
        DEV_ERROR("%s", "Invalid kernel arguments: null pointer");
        return -1;
    }

    auto kargs = (KernelArgs *)arg;

    DEV_INFO("%s", "Graph Executor: Starting AICPU kernel execution");

    // Step 1: Initialize manager (first thread only)
    g_mt_mgr.Init(kargs);

    // Step 2: Wait for initialization to complete
    while (!g_mt_mgr.init_done_.load(std::memory_order_acquire)) {
        // Spin wait for init completion

        // Step 3: Check if initialization failed
        if (g_mt_mgr.init_failed_.load(std::memory_order_acquire)) {
            DEV_ERROR("%s", "Graph Executor: Initialization failed, aborting execution");
            return -1;
        }
    }

    // Step 4: Execute graph (all threads in parallel)
    int rc = g_mt_mgr.Run(arg);
    if (rc != 0) {
        DEV_ERROR("Graph Executor: Thread execution failed with rc=%d", rc);
        return rc;
    }

    // Step 5: Check if all threads have finished
    int prev_finished = g_mt_mgr.finished_.fetch_add(1, std::memory_order_acq_rel);
    if (prev_finished + 1 == g_mt_mgr.thread_num_) {
        DEV_INFO("Graph Executor: Last thread finished");
        g_mt_mgr.DeInit();
    }

    DEV_INFO("%s", "Graph Executor: Kernel execution completed successfully");
    return 0;
}
