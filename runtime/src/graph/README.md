# Graph Class - Task Dependency Management

A lightweight, standalone task dependency graph implementation with compile-time configurable arrays.

## Features

- **Fixed-size arrays** - No dynamic allocation, suitable for embedded systems
- **Compile-time configuration** - Customize limits via macros
- **Simple API** - Easy to use for task scheduling
- **Circular queue** - Efficient O(1) ready task management
- **Manual dependencies** - Explicit control over task relationships

## Files

- `graph.h` - Graph class declaration
- `graph.cpp` - Graph class implementation
- `queue.h` - Circular queue declaration
- `queue.c` - Circular queue implementation

## API Overview

### Task Management

```cpp
// Create a new task with arguments
int task_id = g.add_task(args, num_args);

// Add dependency: from_task must complete before to_task can start
g.add_successor(from_task, to_task);

// Finalize graph (call after adding all tasks/dependencies)
g.finalize();
```

### Execution

```cpp
// Get next ready task
int task_id = g.get_ready_task();  // Returns -1 if none ready

// Mark task as complete (updates dependencies)
g.complete_task(task_id);
```

### Queries

```cpp
int ready_count = g.get_ready_count();
int total_tasks = g.get_total_tasks();
bool active = g.is_task_active(task_id);
```

### Utility

```cpp
g.print_graph();  // Debug output
g.reset();        // Clear all data
```

## Configuration Macros

Define these before including `graph.h` to customize limits:

```cpp
#define GRAPH_MAX_TASKS 2048     // Maximum tasks
#define GRAPH_MAX_ARGS 32        // Arguments per task
#define GRAPH_MAX_FANOUT 1024    // Successors per task
#define QUEUE_MAX_SIZE 2048      // Ready queue size
```

## Usage Example

```cpp
#include "runtime/graph/graph.h"

int main() {
    Graph g;

    // Create tasks
    uint64_t load_args[] = {0x1000, 256};
    uint64_t process_args[] = {0x1000, 0x2000};
    uint64_t store_args[] = {0x2000, 256};

    int load = g.add_task(load_args, 2);
    int process = g.add_task(process_args, 2);
    int store = g.add_task(store_args, 2);

    // Create pipeline: load -> process -> store
    g.add_successor(load, process);
    g.add_successor(process, store);

    // Finalize and execute
    g.finalize();

    while (g.get_ready_count() > 0) {
        int task_id = g.get_ready_task();

        // Execute task...
        printf("Executing task %d\n", task_id);

        g.complete_task(task_id);
    }

    return 0;
}
```

## Compilation

```bash
g++ -std=c++11 -I. your_program.cpp \
    runtime/graph/graph.cpp \
    runtime/graph/queue.c \
    -o your_program
```

## Important Notes

1. **Call finalize()** after adding all tasks and dependencies, before execution
2. **Task IDs** are array indices (0, 1, 2, ...) and never reused within a graph instance
3. **No thread safety** - Single-threaded use only
4. **Manual dependencies** - Unlike PTO Runtime, dependencies must be specified explicitly

## Comparison with PTO Runtime

| Feature | PTO Runtime | Graph Class |
|---------|-------------|-------------|
| Dependency Tracking | Automatic via TensorMap | Manual via add_successor() |
| Thread Safety | Full support | None |
| Task Arguments | TensorRegion structs | Simple uint64_t arrays |
| Use Case | Production scheduler | Educational/lightweight |

## Testing

Run the test suite:

```bash
g++ -std=c++11 -I. test_graph.cpp \
    runtime/graph/graph.cpp \
    runtime/graph/queue.c \
    -o test_graph
./test_graph
```

Tests include:
- Linear pipeline
- Diamond dependencies
- Independent tasks
- Empty graph
- Multiple pipelines
- Reset functionality
