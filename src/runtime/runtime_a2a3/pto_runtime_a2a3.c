/**
 * PTO Runtime System - A2A3 (Ascend) Platform Implementation
 * 
 * This file includes the modular A2A3 implementation from the
 * host/, orchestration/, and core/ subdirectories.
 * 
 * The implementation is split into:
 * - host/a2a3_host.c: Host CPU interface, memory management
 * - orchestration/a2a3_orchestration.c: Task queues, dependency management
 * - core/a2a3_core_worker.c: Worker threads and task execution
 * - core/: InCore intrinsics (header-only, inline implementations)
 */

#include "pto_runtime_a2a3.h"

// Include layer implementations
#include "orchestration/a2a3_orchestration.c"
#include "host/a2a3_host.c"

// Include core worker implementation (platform-specific)
#if defined(A2A3_TARGET_SIMULATOR)
// Simulator: use cycle-accurate simulation
#include "../runtime_a2a3_sim/core/a2a3_core_worker.c"
#else
// Hardware or default: use hardware implementation (with CANN SDK check)
#include "core/a2a3_core_worker.c"
#endif
