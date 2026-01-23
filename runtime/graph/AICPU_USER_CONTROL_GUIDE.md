# AICPU 用户自定义控制和调度

## 概述

AICPU 的控制逻辑和调度逻辑已经交由用户自定义实现。用户需要在 `runtime/graph/aicpu_func.cpp` 中实现 `Control` 和 `Schdule` 函数来定义自己的任务图执行策略。

## 架构变更

### 之前的实现
- AICPU 使用内置的 `execute_graph` 函数执行任务图
- 调度策略固定，无法自定义

### 现在的实现
- 控制流逻辑由用户在 `Control` 函数中实现
- 调度逻辑由用户在 `Schdule` 函数中实现
- 用户拥有完全的控制权

## 函数接口

### Control 函数

```cpp
void Control(Graph& g, Handshake* hank, int core_num);
```

**用途**: 处理任务图的控制流逻辑

**参数**:
- `g`: 任务图引用
- `hank`: Handshake 缓冲区数组，用于与 AICore 通信
- `core_num`: 可用的 AICore 核心数量

**实现建议**:
- 分析图的拓扑结构
- 解析任务依赖关系
- 设置控制流模式
- 准备执行策略

**示例实现**:
```cpp
void Control(Graph& g, Handshake* hank, int core_num) {
    // 1. 分析图结构
    int task_count = g.get_task_count();

    // 2. 识别关键路径
    // ... 用户代码 ...

    // 3. 设置控制策略
    // ... 用户代码 ...
}
```

### Schdule 函数

```cpp
void Schdule(Graph& g, Handshake* hank, int core_num);
```

**用途**: 实现任务图的调度逻辑

**参数**:
- `g`: 任务图引用
- `hank`: Handshake 缓冲区数组，用于与 AICore 通信
- `core_num`: 可用的 AICore 核心数量

**实现建议**:
- 实现任务调度算法
- 管理就绪队列
- 分配任务到核心
- 处理任务完成和依赖更新
- 管理核心利用率

**示例实现**:
```cpp
void Schdule(Graph& g, Handshake* hank, int core_num) {
    // 1. 获取初始就绪任务
    int ready_tasks[GRAPH_MAX_TASKS];
    int ready_count = g.get_initial_ready_tasks(ready_tasks);

    // 2. 调度循环
    while (有任务需要执行) {
        // 检查核心状态
        for (int i = 0; i < core_num; i++) {
            if (hank[i].task_status == 0) {  // 核心空闲
                // 分配任务
                // ... 用户代码 ...
            }
        }

        // 处理完成的任务
        // ... 用户代码 ...
    }
}
```

## 执行流程

AICPU 内核执行流程如下：

```
1. 初始化和握手
   └─> HankAiCore()
       └─> 与所有 AICore 实例握手

2. 用户定义的控制和调度
   ├─> Control(graph, hank, core_num)
   │   └─> 用户实现的控制流逻辑
   │
   └─> Schdule(graph, hank, core_num)
       └─> 用户实现的调度逻辑

3. 清理和关闭
   └─> ShutdownAiCore()
       └─> 关闭所有 AICore 实例
```

## Handshake 结构

用户可以通过 Handshake 结构与 AICore 核心通信：

```cpp
struct Handshake {
    uint32_t core_type;      // 核心类型: 0=AIC, 1=AIV
    uint32_t control;        // 控制信号
    uint32_t aicpu_ready;    // AICPU 就绪标志
    uint32_t aicore_done;    // AICore 完成标志
    uint32_t task_status;    // 任务状态: 0=空闲, 1=忙碌
    uint64_t task;           // 当前任务指针
};
```

**使用方式**:
```cpp
// 检查核心是否空闲
if (hank[core_id].task_status == 0 && hank[core_id].task == 0) {
    // 核心空闲，可以分配任务
}

// 分配任务到核心
hank[core_id].task = reinterpret_cast<uint64_t>(task_ptr);
hank[core_id].task_status = 1;  // 标记为忙碌

// 检查任务是否完成
if (hank[core_id].task_status == 0 && hank[core_id].task != 0) {
    // 任务完成
    Task* completed_task = reinterpret_cast<Task*>(hank[core_id].task);
    hank[core_id].task = 0;  // 清除任务指针
}
```

## Graph API

任务图对象提供以下 API：

```cpp
// 获取任务数量
int get_task_count() const;

// 获取特定任务
Task* get_task(int task_id);

// 获取初始就绪任务
int get_initial_ready_tasks(int* ready_tasks);

// 打印图结构（调试用）
void print_graph() const;
```

## Task 结构

```cpp
struct Task {
    int task_id;             // 任务 ID
    int func_id;             // 函数 ID
    int core_type;           // 核心类型: 0=AIC, 1=AIV
    int fanin;               // 输入依赖数（剩余未完成的前驱任务数）
    int fanout_count;        // 输出依赖数（后继任务数）
    int fanout[MAX_FANOUT];  // 后继任务 ID 数组
    uint64_t args[MAX_ARGS]; // 任务参数
    int arg_count;           // 参数数量
};
```

## 完整示例：简单的轮询调度器

```cpp
#include "aicpu_func.h"

void Control(Graph& g, Handshake* hank, int core_num) {
    // 本示例中，控制逻辑为空
    // 实际应用中可以在这里做图分析、优化等
}

void Schdule(Graph& g, Handshake* hank, int core_num) {
    // 获取初始就绪任务
    int ready_queue[GRAPH_MAX_TASKS];
    int ready_count = g.get_initial_ready_tasks(ready_queue);

    int tasks_in_flight = 0;
    int completed = 0;
    int total_tasks = g.get_task_count();

    // 主调度循环
    while (completed < total_tasks) {
        // 遍历所有核心
        for (int core_id = 0; core_id < core_num; core_id++) {
            Handshake* h = &hank[core_id];

            // 检查是否有任务完成
            if (h->task_status == 0 && h->task != 0) {
                Task* task = reinterpret_cast<Task*>(h->task);

                // 更新后继任务的依赖计数
                for (int i = 0; i < task->fanout_count; i++) {
                    int dep_id = task->fanout[i];
                    Task* dep_task = g.get_task(dep_id);
                    dep_task->fanin--;

                    // 如果依赖满足，加入就绪队列
                    if (dep_task->fanin == 0) {
                        ready_queue[ready_count++] = dep_id;
                    }
                }

                h->task = 0;
                completed++;
                tasks_in_flight--;
            }

            // 如果核心空闲且有就绪任务，分配任务
            if (h->task_status == 0 && h->task == 0 && ready_count > 0) {
                int task_id = ready_queue[--ready_count];
                Task* task = g.get_task(task_id);

                h->task = reinterpret_cast<uint64_t>(task);
                h->task_status = 1;
                tasks_in_flight++;
            }
        }
    }
}
```

## 调试建议

### 使用设备日志

在 AICPU 上可以使用设备日志功能：

```cpp
#include "device_log.h"

void Schdule(Graph& g, Handshake* hank, int core_num) {
    DEV_INFO("Starting scheduling with %d cores", core_num);
    DEV_INFO("Graph has %d tasks", g.get_task_count());

    // ... 调度逻辑 ...

    DEV_INFO("Scheduling completed");
}
```

### 打印图结构

```cpp
void Control(Graph& g, Handshake* hank, int core_num) {
    g.print_graph();  // 打印任务图结构
}
```

## 高级主题

### 1. 按核心类型分离队列

```cpp
void Schdule(Graph& g, Handshake* hank, int core_num) {
    int aic_queue[GRAPH_MAX_TASKS];
    int aiv_queue[GRAPH_MAX_TASKS];
    int aic_count = 0, aiv_count = 0;

    // 根据任务类型分离就绪任务
    int ready[GRAPH_MAX_TASKS];
    int ready_count = g.get_initial_ready_tasks(ready);

    for (int i = 0; i < ready_count; i++) {
        Task* task = g.get_task(ready[i]);
        if (task->core_type == 0) {
            aic_queue[aic_count++] = ready[i];
        } else {
            aiv_queue[aiv_count++] = ready[i];
        }
    }

    // 调度时根据核心类型匹配队列
    // ...
}
```

### 2. 优先级调度

```cpp
void Schdule(Graph& g, Handshake* hank, int core_num) {
    // 可以基于任务属性（如关键路径）实现优先级调度
    // ...
}
```

### 3. 负载均衡

```cpp
void Schdule(Graph& g, Handshake* hank, int core_num) {
    // 跟踪每个核心的工作负载
    // 优先分配任务给负载较轻的核心
    // ...
}
```

## 文件位置

- **函数声明**: `runtime/graph/aicpu_func.h`
- **函数实现**: `runtime/graph/aicpu_func.cpp`
- **调用位置**: `runtime/aicpu/graph_executor.cpp`

## 编译和使用

用户在 `aicpu_func.cpp` 中实现自己的逻辑后，重新编译 AICPU 共享库即可：

```bash
# 进入 runtime/aicpu 目录
cd runtime/aicpu

# 使用 CMake 编译
mkdir -p build
cd build
cmake ..
make

# 生成 libaicpu_graph_kernel.so
```

编译好的库可以传递给 DeviceRunner 使用：

```python
import pto_runtime

runner = pto_runtime.DeviceRunner.get()
runner.init(
    device_id=0,
    num_cores=3,
    aicpu_so_path="./libaicpu_graph_kernel.so"
)
```

## 总结

通过将控制和调度逻辑交给用户实现，AICPU 提供了最大的灵活性。用户可以：

- ✅ 实现自定义的调度算法
- ✅ 优化特定工作负载的性能
- ✅ 实验不同的调度策略
- ✅ 完全控制任务执行流程

用户需要在 `Control` 和 `Schdule` 函数中实现自己的逻辑，这两个函数会在 AICPU 内核执行期间被调用。
