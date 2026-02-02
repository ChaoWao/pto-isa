#include <stddef.h>
#include <stdint.h>
#include "runtime/mem.h"
#include "runtime/base.h"
#include "ascend_hal.h"


#define dmb(opt) asm volatile("dmb " #opt : : : "memory")
#define dsb(opt) asm volatile("dsb " #opt : : : "memory")
#define rmb() dsb(ld)
#define wmb() dsb(st)


#define TASK_QUEUE_DEPTH 256
typedef struct TaskContext{
    int32_t task_queue[TASK_QUEUE_DEPTH];
    int32_t queue_head;
    int32_t queue_tail;
    int32_t queue_count;
} TaskContext;


/*申请一块内存host、device可以同时访问*/
// 方法一：申请host内存注册到device
void mallocHostDeviceShareMem(uint32_t deviceId, uint64_t size, void **hostPtr, void **devPtr)
{
    // 安全问题，当前runtime开放API只支持host内存注册到device。若需要device内存映射到host需要runtime新开API。

    // rtSetDevice(deviceId);  提前调用过SetDevice即可。
    rtMallocHost(hostPtr, size, 0);
    rtsHostRegister(*hostPtr, size, RT_HOST_REGISTER_MAPPED, devPtr);
    return;
}

// 方法二：申请device内存注册到host
void mallocHostDeviceShareMemV2(uint32_t deviceId, uint64_t size, void **hostPtr, void **devPtr)
{
    // 注册到host可以直接使用驱动底层接口halHostRegister。需要传入deviceId
    rtMalloc(devPtr, size, 0);
    halHostRegister(*devPtr, size, DEV_SVM_MAP_HOST, deviceId, hostPtr);
    return;
}

void init_context(TaskContext *taskCtx) {
    rtMemset(taskCtx, sizeof(TaskContext), 0, sizeof(TaskContext));
    return;
}

void submit_task(TaskContext *taskCtx, uint32_t task) {
    // step1: write data
    taskCtx->task_queue[taskCtx->queue_tail] = task;
    taskCtx->queue_count++;

    // step2: barrier
    wmb();

    // step3: write doorbell to notify device
    taskCtx->queue_tail = (taskCtx->queue_tail + 1) % TASK_QUEUE_DEPTH;
    return;
}

void read_task(TaskContext *taskCtx, uint32_t *task) {
    // step1: read data
    *task = taskCtx->task_queue[taskCtx->queue_head];
    taskCtx->queue_count--;

    // step2: barrier
    rmb();

    // step3: modify head
    taskCtx->queue_head = (taskCtx->queue_head + 1) % TASK_QUEUE_DEPTH;
    return;
}


// Host侧入口函数：申请内存，把devPtr通过kernellaunch发送给device侧，往task_queue提交任务，等待执行结束。
void host_main_entry()
{
    void *hostPtr = NULL;
    void *devPtr = NULL;
    uint64_t size = 1024;

    rtSetDevice(0);
    mallocHostDeviceShareMem(0, size, &hostPtr, &devPtr);

    TaskContext *taskCtx = (TaskContext *)hostPtr;
    init_context(taskCtx);

    // TODO： send the devPtr to device
    //rtCpuKernelLaunch(...);

    submit_task(taskCtx, 0x11);

    // wait finished..
    while (taskCtx->queue_head != taskCtx->queue_tail) {
        rmb();
    }

    rtFreeHost(hostPtr);
}

// Device侧入口函数： 从队列中读取任务，执行。
void device_main_entry(void *devPtr) {
    TaskContext *taskCtx = (TaskContext *)devPtr;
    uint32_t task;
    while (taskCtx->queue_head != taskCtx->queue_tail) {
        read_task(taskCtx, &task);

        // excute task;
    }

    return;
}
