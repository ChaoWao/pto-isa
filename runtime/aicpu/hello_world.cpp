#include "device_log.h"
#include "../graph/graph.h"
#include <cstdint>
#include <cstdio>
#include <sched.h>

struct Handshake {
    volatile uint32_t aicpu_ready;
    volatile uint32_t aicore_done;
    volatile int32_t control;  // 0=execute, 1=quit
    volatile int32_t task;     // task ID: -1=none, 0=TADD, etc.
} __attribute__((aligned(64)));

struct KernelArgs {
    uint64_t unused[5] = {0};
    int64_t *deviceArgs{nullptr};
    int64_t *hankArgs{nullptr};
    int64_t core_num;
    Graph *graphArgs{nullptr};
};

int HankAiCore(void *arg) {
    auto kargs = (KernelArgs *)arg;
    uint64_t core_num = kargs->core_num;
    for (uint64_t i = 0; i < core_num; i++) {
        Handshake* hank = (Handshake*)kargs->hankArgs + i;
        DEV_INFO("AICPU: hank addr = 0x%lx", (uint64_t)hank);
        hank->aicpu_ready = 1;
    }
    for (uint64_t i = 0; i < core_num; i++) {
        Handshake* hank = (Handshake*)kargs->hankArgs + i;
        while (hank->aicore_done == 0) {

        };
        DEV_INFO("success hank->aicore_done = %u", (uint64_t)hank->aicore_done);
    }
    return 0;
}

int ShutdownAiCore(void *arg) {
    auto kargs = (KernelArgs *)arg;
    uint64_t core_num = kargs->core_num;
    for (uint64_t i = 0; i < core_num; i++) {
        Handshake* hank = (Handshake*)kargs->hankArgs + i;
        hank->control = 1;
    }
    return 0;
}

// Runtime helper function to execute all tasks in the graph
int execute_graph(Graph& g) {
    // Get initially ready tasks from graph
    int ready_queue[GRAPH_MAX_TASKS];
    int ready_count = g.get_initial_ready_tasks(ready_queue);

    int completed = 0;

    // Execute tasks
    while (ready_count > 0) {
        // Pop task from queue (using stack order for simplicity)
        int task_id = ready_queue[--ready_count];
        Task* task = g.get_task(task_id);

        DEV_INFO("  Executing task %d", task_id);
        completed++;

        // Update successors' fanin
        for (int i = 0; i < task->fanout_count; i++) {
            int dep_id = task->fanout[i];
            Task* dep = g.get_task(dep_id);
            dep->fanin--;

            // Add to ready queue if ready
            if (dep->fanin == 0) {
                ready_queue[ready_count++] = dep_id;
            }
        }
    }

    return completed;
}

extern "C" __attribute__((visibility("default"))) int StaticTileFwkBackendKernelServer(void *arg) {
    if (arg == nullptr) {
        DEV_ERROR("%s", "Invalid kernel arguments: null pointer");
        return -1;
    }

    return 0;
}

extern "C" __attribute__((visibility("default"))) int DynTileFwkBackendKernelServerInit(void *arg) {
    InitLogSwitch();
    if (arg == nullptr) {
        DEV_ERROR("%s", "Invalid kernel arguments: null pointer");
        return -1;
    }

    DEV_INFO("%s", "Hello World Kernel Init: Initializing AICPU kernel");
    return 0;
}

extern "C" __attribute__((visibility("default"))) int DynTileFwkBackendKernelServer(void *arg) {
    if (arg == nullptr) {
        DEV_ERROR("%s", "Invalid kernel arguments: null pointer");
        return -1;
    }
    DEV_INFO("%s", "Hello World from AICPU Kernel!");

    auto kargs = (KernelArgs *)arg;


    auto rc = HankAiCore(arg);
    if (rc != 0) {
        return rc;
    }

    if (kargs->graphArgs != nullptr) {
        Graph* g = kargs->graphArgs;
        DEV_INFO("Graph has %d tasks", g->get_task_count());
        int completed = execute_graph(*g);
        DEV_INFO("Executed %d tasks from graph", completed);
    }

    rc = ShutdownAiCore(arg);
    if (rc != 0) {
        return rc;
    }

    DEV_INFO("%s", "Kernel execution completed successfully");
    return 0;
}
