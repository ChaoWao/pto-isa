# Runtime2 互斥锁使用与性能影响分析

本文档分析 `src/runtime2` 中**所有**互斥与锁的用途、调用路径及潜在性能影响，包括：
- **pthread_mutex_t**：所有在 runtime2 中定义或使用的 mutex；
- **自旋锁 / spinlock**：如 per-task 的 `fanout_lock`。
文档力求无遗漏，便于与代码交叉核对。

---

## 1. 概览

| 锁/同步 | 实现 | 锁类型 | 所在类型/文件 | 用途 | 热路径 | 竞争方 |
|--------|------|--------|----------------|------|--------|--------|
| `fanout_lock` (per-task) | **spinlock** | **局部锁**（per-task） | PTO2TaskDescriptor | Orchestrator 与 Scheduler 同步 fanout 链表 | ✅ 高 | 同一 producer 的提交与完成 |
| `PTO2CompletionQueue.mutex` | mutex | **全局锁** | 完成队列 | MPSC 队列同步 | ✅ 高 | 所有 Worker + Scheduler |
| `ready_mutex[type]` | mutex | **局部锁**（按 worker 类型） | PTO2ThreadContext | 就绪队列 (per type) | ✅ 高 | 同类型 Worker + Scheduler |
| `done_mutex` | mutex | **全局锁** | PTO2ThreadContext | 完成信号 / all_done | ✅ 高 | 所有 Worker + Scheduler |
| `task_end_mutex` | mutex | **全局锁** | PTO2ThreadContext | 仿真 task_end_cycles | ✅ 仿真时 | 所有 Worker（仿真） |
| `startup_mutex` | mutex | **全局锁** | PTO2ThreadContext | 启动阶段同步 | ❌ 仅启动 | 主线程 + Scheduler + Workers |
| `trace_mutex` | mutex | **全局锁** | PTO2RuntimeThreaded | 写 trace 缓冲区 | ✅ 开 trace 时 | 所有 Worker |
| `task_mutex` (A2A3 sim) | mutex | **全局锁** | PTORuntime | A2A3 任务完成与依赖更新 | ✅ 高 | A2A3 Workers |
| `queue_mutex` (A2A3 sim) | mutex | **全局锁** | PTORuntime | all_done/队列条件变量 | 中 | A2A3 Workers |

**锁类型约定**：**全局锁** = 整个 runtime/上下文仅此一把，所有相关线程竞争同一把锁；**局部锁** = 按实体或类型分片，每类/每个实体一把锁，竞争范围缩小。

### 1.1 锁与同步原语类型说明

Runtime2 中出现的**锁与同步手段**分为以下几类，便于与代码对照：

| 类型 | 说明 | 在 Runtime2 中的使用 |
|------|------|----------------------|
| **pthread_mutex_t（互斥锁）** | 内核参与、可睡眠的锁；适合持锁时间较长或需与条件变量配合的场景。 | 完成队列、就绪队列、done/startup/task_end/trace、A2A3 task_mutex/queue_mutex（见上表）。 |
| **Spinlock（自旋锁）** | 基于原子操作的忙等锁，不睡眠；适合临界区极短、争用低的场景。Runtime2 中**仅有一处**：per-task 的 `fanout_lock`。 | **实现**：`PTO2TaskDescriptor.fanout_lock`（`volatile int32_t`，0=未锁，1=已锁）；**加锁**：`while (PTO2_EXCHANGE(&task->fanout_lock, 1) != 0) { PTO2_SPIN_PAUSE(); }`；**解锁**：`PTO2_STORE_RELEASE(&task->fanout_lock, 0)`。封装在 `pto_orchestrator.c` 的 `task_fanout_lock` / `task_fanout_unlock`，Scheduler 在 `on_task_complete_threadsafe` 中直接使用 `PTO2_EXCHANGE` / `PTO2_STORE_RELEASE`。 |
| **自旋锁用到的原语**（定义在 `pto_runtime2_types.h`） | 用于实现自旋锁或自旋等待的原子/指令。 | **PTO2_EXCHANGE(ptr, val)**：`__atomic_exchange_n(ptr, val, __ATOMIC_ACQ_REL)`，用于 spinlock 加锁。**PTO2_STORE_RELEASE(ptr, val)**：用于 spinlock 解锁。**PTO2_SPIN_PAUSE()**：自旋等待时让出 CPU 以减轻总线压力；按架构：aarch64 用 `yield`，x86_64 用 `pause`，否则 `sched_yield()`。 |
| **原子操作（atomics）** | 无锁同步，非“锁”但属于同步手段。 | **task_state**：`__atomic_compare_exchange_n` 做 PENDING→READY、COMPLETED→CONSUMED 等状态迁移。**fanin_refcount / fanout_refcount**：`__atomic_load_n`、`__atomic_fetch_add`、`__atomic_store_n` 等，供 Orchestrator（scope_end）与 Scheduler（任务完成、release_producer）并发访问。**current_task_index、orchestrator_done、last_task_alive** 等：LOAD_ACQUIRE/STORE_RELEASE 保证可见性。 |
| **自旋等待（spin-wait）** | 非锁；在“等待条件成立”时忙等并调用 `PTO2_SPIN_PAUSE()`，避免纯空转。 | **HeapRing 分配**（`pto_ring_buffer.c`）：`pto2_heap_ring_alloc` 在无空间时自旋等待并 `PTO2_SPIN_PAUSE()`。**TaskRing 分配**（`pto_ring_buffer.c`）：`pto2_task_ring_alloc` 在窗口满时自旋等待，带流控上限与死锁检测，内部 `PTO2_SPIN_PAUSE()`。**Completion queue 满重试**（`pto_worker.c`）：`pto2_completion_queue_push` 失败时重试循环内 `PTO2_SPIN_PAUSE()`。**Orchestrator wait_all**（`pto_orchestrator.c`）：`pto2_orchestrator_wait_all` 自旋等待 scheduler 完成，`PTO2_SPIN_PAUSE()`。**单线程/仿真等待**（`pto_runtime2.c`、`pto_runtime2_sim.c`）：部分等待循环使用 `PTO2_SPIN_PAUSE()`。 |

**说明**：Orchestrator 在调用 runtime API（如 `pto2_rt_submit_task`、`pto2_rt_scope_end`）时**不持有上述任何 pthread_mutex**。Orchestrator 与 Scheduler 对同一任务 fanout 的同步通过 **per-task spinlock (fanout_lock)** 实现；主 runtime2 中**没有**名为 `task_mutex` 的 pthread_mutex，详见第 2 节。

---

## 2. Orchestrator 调用 Runtime API 时的锁

Orchestrator 线程只执行用户编排函数，期间会调用 `pto2_rt_scope_begin`、`pto2_rt_submit_task`、`pto2_rt_scope_end`、`pto2_rt_orchestration_done` 等 Runtime API。这些路径**不使用任何 pthread_mutex**。

### 2.0.1 Orchestrator 侧使用的同步：per-task spinlock (fanout_lock)

- **锁类型**: **局部锁（per-task）** — 每个任务描述符自带一把 `fanout_lock`，仅保护该任务的 fanout 链表；不同任务之间无锁竞争，只有对**同一 producer 任务**同时“Orchestrator 添加 consumer”与“Scheduler 处理该 producer 完成”时才会争用。
- **定义**: `pto_runtime2_types.h` — `PTO2TaskDescriptor.fanout_lock`（`volatile int32_t`，0=未锁，1=已锁），注释为 “Per-task spinlock”。
- **用途**: 保护同一任务的 `fanout_head`、`fanout_count`，避免 Orchestrator 在提交时往 producer 的 fanout 链表追加 consumer 的同时，Scheduler 在任务完成时遍历该 producer 的 fanout 列表，产生竞态。

**调用路径**:

| 调用方 | 文件 | 说明 |
|--------|------|------|
| **Orchestrator** | pto_orchestrator.c | `pto2_submit_task` → 对每个输入 producer 调用 `pto2_add_consumer_to_producer` → 内部 `task_fanout_lock(producer)` / `task_fanout_unlock(producer)`，在持锁下 prepend consumer、更新 `fanout_count`，必要时直接更新 consumer 的 `fanin_refcount`（若 producer 已完成）。 |
| **Scheduler** | pto_scheduler.c | 任务完成时 `on_task_complete_threadsafe` → `while (PTO2_EXCHANGE(&task->fanout_lock, 1))` 自旋获取锁，遍历 fanout 更新各 consumer 的 `fanin_refcount` 并可能入队 READY，最后 `PTO2_STORE_RELEASE(&task->fanout_lock, 0)`。 |

**性能影响**:

- **频率**: 每个“边”（producer→consumer）在提交时 Orchestrator 持一次 fanout_lock；每个任务完成时 Scheduler 持一次该任务的 fanout_lock。与任务图边数、任务数同量级。
- **竞争**: 仅当 **同一 producer 任务** 上“Orchestrator 正在添加 consumer”与“Scheduler 正在处理该 producer 完成”重叠时才会自旋等待。per-task 锁粒度细，冲突概率相对低，但若同一 producer 的 fanout 很多或完成很频繁，该 producer 的 fanout_lock 仍可能被频繁争用。
- **说明**: 这是 **spinlock**，不是 `pthread_mutex`；文档仍在此说明，因为用户关心“Orchestrator 调用 runtime API 中使用的锁”，尤其是与“task”相关的锁。

### 2.0.2 关于 “task_mutex” 的说明

- **主 runtime2 (pto_runtime2_*)** 中**没有**名为 `task_mutex` 的 **pthread_mutex**。与“任务”相关的**互斥**是上述 **per-task spinlock `fanout_lock`**（Orchestrator + Scheduler 都会用）。
- **A2A3 仿真 (runtime_a2a3_sim)** 中，`PTORuntime` 有一个 **`task_mutex`**（pthread_mutex），仅用于 **Worker 完成任务** 的路径（`a2a3_core_complete_task`）：保护任务完成状态、窗口推进、fanout 的 fanin 更新等。**Orchestrator 不持有 A2A3 的 task_mutex**；A2A3 的编排/提交若在其他模块，也不在该 task_mutex 的临界区内。

---

## 3. 主运行时 (pto_runtime2_*) 中的互斥锁

### 3.1 完成队列 `PTO2CompletionQueue.mutex`

- **锁类型**: **全局锁** — 整个运行时仅有一个完成队列，对应一把 mutex；所有 Worker（任意类型）push 与 Scheduler pop 都竞争同一把锁。
- **定义**: `pto_runtime2_types.h` — `PTO2CompletionQueue.mutex`
- **用途**: 多生产者单消费者 (MPSC) 完成队列的互斥访问；Worker 完成任务后 push，Scheduler 循环 pop。

**调用路径**:

| 操作 | 文件 | 说明 |
|------|------|------|
| `pto2_completion_queue_push` | pto_worker.c | 每次任务完成时调用，持锁：检查满、写 entry、移动 tail |
| `pto2_completion_queue_pop` | pto_scheduler.c → `pto2_scheduler_process_completions` | Scheduler 循环 pop，持锁：检查空、读 entry、移动 head |
| `pto2_completion_queue_empty` | pto_worker.c | 仅在 completion queue 满重试时使用 |

**性能影响**:

- **频率**: 与任务数线性相关，每个任务完成一次 push，Scheduler 侧多次 pop（每次持锁 pop 一个）。
- **竞争**: 所有 Worker 竞争同一把锁做 push；Scheduler 与 Worker 竞争同一把锁（pop vs push）。
- **持锁时间**: 较短（几次赋值 + 取模），但调用频率极高，总锁开销显著。
- **建议**: 考虑无锁 MPSC 队列（如基于 atomic + CAS 的 ring buffer），或批量 push/pop 减少锁次数。

---

### 3.2 就绪队列 `ready_mutex[PTO2_NUM_WORKER_TYPES]`

- **锁类型**: **局部锁（按 worker 类型）** — 共 `PTO2_NUM_WORKER_TYPES` 把（如 4：CUBE、VECTOR、AI_CPU、ACCELERATOR），每种 worker 类型一把；仅同类型的 Worker 与 Scheduler 竞争同一把，不同类型之间无锁竞争。
- **定义**: `pto_runtime2_types.h` — `PTO2ThreadContext.ready_mutex[4]`
- **用途**: 按 worker 类型保护就绪队列 `ready_queues[type]`；同一类型内多 Worker 与 Scheduler 共享。

**调用路径**:

| 操作 | 文件 | 说明 |
|------|------|------|
| `pto2_ready_queue_push_wake_min_clock` | pto_scheduler.c | Scheduler 将就绪任务入队并唤醒最小 clock 的 Worker，持锁 push + signal |
| `pto2_ready_queue_push_threadsafe` | pto_scheduler.h 声明 | 通用线程安全 push（若被使用） |
| `pto2_worker_get_task` | pto_worker.c | Worker 持锁：等条件/空则 wait、非空则 pop、必要时 broadcast/signal 后再 unlock |
| `pto2_worker_try_get_task` → `pto2_ready_queue_try_pop_threadsafe` | pto_worker.c, pto_scheduler.c | 非阻塞取任务，持锁 pop |
| `pto2_runtime_stop_threads` | pto_runtime2_threaded.c | 停止时对每个 type 持锁 broadcast |

**性能影响**:

- **频率**: 每个任务调度一次 push（Scheduler），每个被执行的任务对应一次 pop（Worker）；与任务数同量级。
- **竞争**: 按类型分离——CUBE 与 VECTOR 不互相竞争；但同一类型下所有 CUBE Worker 与 Scheduler 竞争 `ready_mutex[CUBE]`，VECTOR 同理。
- **持锁时间**: 在 `pto2_worker_get_task` 中持锁时间较长：可能包含 `worker_has_min_clock` 遍历、多次 `pthread_cond_wait`/`pthread_cond_timedwait`、以及取完任务后“若队列非空则对同类型 Worker 做一轮 signal”。长持锁会放大竞争。
- **建议**:  
  - 尽量缩短“持锁下做的逻辑”（例如把 min_clock 计算或 signal 策略移到锁外或拆成更小临界区）。  
  - 若类型内 Worker 数较多，可考虑按 worker 或按子队列进一步分片，减少对同一 mutex 的竞争。

---

### 3.3 完成与全部结束 `done_mutex`

- **锁类型**: **全局锁** — 整个 `PTO2ThreadContext` 仅此一把；所有 Worker（任意类型）、Scheduler 以及主线程（wait_completion）都竞争同一把锁。
- **定义**: `pto_runtime2_types.h` — `PTO2ThreadContext.done_mutex`
- **用途**: 保护 `completion_cond`、`all_done_cond`、`all_done`；用于“有完成事件”和“全部结束”两种等待。

**调用路径**:

| 操作 | 文件 | 说明 |
|------|------|------|
| `pto2_worker_task_complete` | pto_worker.c | 每次任务完成：持锁 `pthread_cond_signal(&completion_cond)`，可能两次（push 失败重试时也 signal） |
| `pto2_runtime_wait_completion` | pto_runtime2_threaded.c | 主线程：持锁 `pthread_cond_wait(all_done_cond)` 直到 `all_done` |
| `pto2_scheduler_thread_func` | pto_scheduler.c | 无工作时：持锁 `pthread_cond_timedwait(completion_cond)`（约 1ms）；全部完成时：持锁置 `all_done` 并 `pthread_cond_broadcast(all_done_cond)` |
| `pto2_runtime_stop_threads` | pto_runtime2_threaded.c | 持锁 broadcast `completion_cond` 和 `all_done_cond` |

**性能影响**:

- **频率**: 每个任务完成至少一次 lock/signal/unlock；Scheduler 在空闲时约每 1ms 一次 timedwait；结束时各一次。
- **竞争**: 所有 Worker（任意类型）+ 1 个 Scheduler（以及主线程在 wait_completion 时）共用同一把锁，高并发时竞争集中。
- **持锁时间**: 单次通常很短（置位 + signal/broadcast），但调用次数多，总系统调用与锁竞争明显。
- **建议**:  
  - 若仅需“有事件”提示，可考虑用原子变量 + eventfd 或 pipe 等减少对 `done_mutex` 的依赖。  
  - 保留 `done_mutex` 时，尽量保证临界区内仅做必要的状态更新和 signal，不做额外逻辑。

---

### 3.4 仿真任务结束时间 `task_end_mutex`

- **锁类型**: **全局锁** — 整个 `PTO2ThreadContext` 仅此一把，保护整块 `task_end_cycles[]` 数组；所有仿真 Worker 读/写不同 slot 时仍共用同一把锁，粒度粗。
- **定义**: `pto_runtime2_types.h` — `PTO2ThreadContext.task_end_mutex`
- **用途**: 保护 `task_end_cycles[]` 的读写；仿真模式下用于依赖的“最早开始时间”计算。

**调用路径**:

| 操作 | 文件 | 说明 |
|------|------|------|
| `pto2_worker_thread_func_sim` | pto_worker.c | 取任务后：持锁遍历 fanin，读 `task_end_cycles[dep_slot]` 算 `earliest_start`；执行完：持锁写 `task_end_cycles[slot] = end_cycle` |

**性能影响**:

- **频率**: 仅仿真模式；每个任务 2 次持锁（一次读依赖，一次写自己的 end_cycle）。
- **竞争**: 所有 Worker 共用同一把锁，且读写的是同一数组的不同槽位，锁粒度粗。
- **持锁时间**: 读侧可能较长（fanin 链表遍历）；写侧很短。读多写多时容易形成排队。
- **建议**:  
  - 将 `task_end_cycles[]` 改为 per-slot 或 per-bucket 的细粒度锁或原子变量（例如每槽一个 atomic），减少全局锁。  
  - 或对“读依赖”的区间使用 RCU/读锁，仅“写本任务 end_cycle”用短临界区。

---

### 3.5 启动同步 `startup_mutex`

- **锁类型**: **全局锁** — 整个 `PTO2ThreadContext` 仅此一把；主线程、Scheduler 线程、所有 Worker 线程在启动阶段共用，运行期不再使用。
- **定义**: `pto_runtime2_types.h` — `PTO2ThreadContext.startup_mutex`
- **用途**: 保证启动顺序：Workers 先就绪 → Scheduler 就绪 → 主线程再开始执行编排。

**调用路径**:

| 操作 | 文件 | 说明 |
|------|------|------|
| `pto2_runtime_start_threads` | pto_runtime2_threaded.c | 主线程：持锁重置 `workers_ready`/`scheduler_ready`；然后持锁 wait 直到 `scheduler_ready` |
| `pto2_scheduler_thread_func` | pto_scheduler.c | 持锁 wait 直到 `workers_ready == num_workers`，然后置 `scheduler_ready` 并 broadcast |
| `pto2_worker_thread_func` / `pto2_worker_thread_func_sim` | pto_worker.c | 持锁递增 `workers_ready` 并 broadcast |

**性能影响**:

- 仅在线程启动阶段使用，运行期不再使用，对稳态性能无影响；无需优化。

---

### 3.6 Trace 记录 `trace_mutex`

- **锁类型**: **全局锁** — 整个 `PTO2RuntimeThreaded` 实例仅此一把，保护单一 `trace_events[]` 与 `trace_count`；所有 Worker 写 trace 时竞争同一把锁。
- **定义**: `pto_runtime2_threaded.h` — `PTO2RuntimeThreaded.trace_mutex`
- **用途**: 保护 `trace_events[]` 与 `trace_count`；多 Worker 并发写 trace 时保证一致性。

**调用路径**:

| 操作 | 文件 | 说明 |
|------|------|------|
| `pto2_runtime_record_trace` | pto_runtime2_threaded.c | 若开启 trace：持锁检查 `trace_count`、写一条 event、`trace_count++`、unlock |

**性能影响**:

- **频率**: 开启 trace 时，每个任务完成调用一次，与任务数线性相关。
- **竞争**: 所有 Worker 写同一缓冲区、同一计数器，全局单锁，竞争明显。
- **持锁时间**: 短（一次赋值和递增），但高任务量时锁调用次数大。
- **建议**:  
  - 使用 per-worker 的 trace 缓冲区，Worker 只写本线程 buffer，最后合并或写文件时再单线程处理。  
  - 或使用无锁 ring buffer（SPSC/MPMC）按 worker 或按 core 分片，减少对单一 `trace_mutex` 的竞争。

---

## 4. A2A3 仿真 (runtime_a2a3_sim) 中的互斥锁

### 4.1 `task_mutex` (PTORuntime)

- **锁类型**: **全局锁** — 整个 `PTORuntime` 实例仅此一把；所有 A2A3 Worker 在任务完成时都竞争同一把锁，临界区大，易成瓶颈。
- **文件**: `runtime_a2a3_sim/core/a2a3_core_worker.c`
- **用途**: 保护任务完成逻辑：更新 `is_complete`、`active_task_count`、滑动窗口、fanout 的 fanin 递减与 `earliest_start_cycle`、以及 `window_not_full` 的 broadcast。

**调用路径**:

| 操作 | 说明 |
|------|------|
| `a2a3_core_complete_task` | 持锁完成上述全部更新后 unlock；然后在外层对 `newly_ready` 做路由，并在 `all_done` 时使用 `queue_mutex` |

**性能影响**:

- 每次任务完成都会进入一次大临界区，持锁时间较长（循环 fanout、可能多次窗口推进），所有 A2A3 Worker 竞争同一把锁，容易成为 A2A3 仿真的瓶颈。
- **建议**: 将“更新本任务状态 + 本任务窗口”“与“更新依赖任务 fanin/earliest_start”拆开，或对依赖更新使用更细粒度结构（如 per-task 或 per-bucket 锁），缩短 `task_mutex` 持锁时间。

---

### 4.2 `queue_mutex` (PTORuntime)

- **锁类型**: **全局锁** — 整个 `PTORuntime` 实例仅此一把，保护上述条件变量；仅在“全部任务完成”时由完成该任务的 Worker 持锁 broadcast，调用频率低。
- **文件**: `runtime_a2a3_sim/core/a2a3_core_worker.c`
- **用途**: 在“全部完成”时保护对 `all_done`、`vector_queue_not_empty`、`cube_queue_not_empty` 等条件变量的 broadcast。

**调用路径**:

| 操作 | 说明 |
|------|------|
| `a2a3_core_complete_task`（当 `all_done`） | 持锁 `pthread_cond_broadcast` 三个 cond，然后 unlock |

**性能影响**:

- 仅在“全部任务完成”时调用，频率低，对整体性能影响小。

---

## 5. 性能影响汇总与优化优先级

| 优先级 | 锁/同步 | 实现 | 锁类型 | 原因 | 建议方向 |
|--------|--------|--------|--------|------|----------|
| 高 | 完成队列 mutex | mutex | 全局锁 | 每任务 push/pop，全局 MPSC 单锁 | 无锁 MPSC 或批量操作 |
| 高 | ready_mutex[type] | mutex | 局部锁（按 type） | 每任务入队/出队，且 Worker 持锁时间偏长 | 缩短临界区；必要时按 worker 分片 |
| 高 | done_mutex | mutex | 全局锁 | 每任务完成都 signal，多线程竞争 | 原子 + 轻量通知机制 |
| 中 | fanout_lock (per-task) | **spinlock** | 局部锁（per-task） | 每边/每任务持锁，同一 producer 上 Orchestrator 与 Scheduler 可能争用 | 持锁时间短，一般可保持；若热点可考虑无锁 fanout 结构 |
| 中 | task_end_mutex | mutex | 全局锁 | 仿真下每任务 2 次、粗粒度 | 细粒度锁或 per-slot 原子 |
| 中 | trace_mutex | mutex | 全局锁 | 开 trace 时每任务一次、单缓冲 | per-worker 缓冲或无锁 ring |
| 中 | A2A3 task_mutex | mutex | 全局锁 | 任务完成大临界区 | 拆分子临界区或细粒度锁 |
| 低 | startup_mutex | mutex | 全局锁 | 仅启动阶段 | 保持现状 |
| 低 | A2A3 queue_mutex | mutex | 全局锁 | 仅结束时刻 | 保持现状 |

---

## 6. 附录：锁的初始化与销毁

- **主运行时**: `pto_runtime2_threaded.c` 中 `thread_ctx_init` 初始化 `ready_mutex`/`ready_cond`、`done_mutex`/`all_done_cond`/`completion_cond`、`task_end_mutex`、`startup_mutex`/`startup_cond`；`pto2_completion_queue_init`（在需要时）初始化完成队列 mutex；`pto2_runtime_create_threaded` 初始化 `trace_mutex`。销毁在 `thread_ctx_destroy` 与 `pto2_runtime_destroy_threaded` 中对称进行。
- **A2A3 仿真**: `task_mutex`、`queue_mutex` 在 PTORuntime 的创建/销毁中初始化与销毁（未在本次列出的片段中展示，但由 a2a3 仿真层使用）。

### 6.1 锁清单（按定义/所在文件）

| 文件 | 锁名 | 类型 | 说明 |
|------|------|------|------|
| pto_runtime2_types.h | `PTO2TaskDescriptor.fanout_lock` | 局部（per-task）spinlock | Orchestrator + Scheduler 同步 fanout |
| pto_runtime2_types.h | `PTO2CompletionQueue.mutex` | 全局 mutex | 完成队列 MPSC |
| pto_runtime2_types.h | `PTO2ThreadContext.ready_mutex[4]` | 局部（per type）mutex | 就绪队列 |
| pto_runtime2_types.h | `PTO2ThreadContext.done_mutex` | 全局 mutex | 完成 / all_done 条件变量 |
| pto_runtime2_types.h | `PTO2ThreadContext.task_end_mutex` | 全局 mutex | 仿真 task_end_cycles |
| pto_runtime2_types.h | `PTO2ThreadContext.startup_mutex` | 全局 mutex | 启动阶段 |
| pto_runtime2_threaded.h | `PTO2RuntimeThreaded.trace_mutex` | 全局 mutex | trace 缓冲区 |
| runtime_a2a3_sim (PTORuntime) | `task_mutex` | 全局 mutex | A2A3 任务完成 |
| runtime_a2a3_sim (PTORuntime) | `queue_mutex` | 全局 mutex | A2A3 all_done/队列条件变量 |

以上分析基于当前代码结构；实际热点应以 profiling（如 mutex 争用、锁持有时间）为准进行验证和调整。
