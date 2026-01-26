/**
 * PTO Runtime - Ascend A2/A3 Simulator Core Worker Interface
 * 
 * This header wraps the common core worker interface for the simulator platform.
 * It defines A2A3_TARGET_SIMULATOR before including the common header.
 */

#ifndef A2A3_CORE_WORKER_SIM_H
#define A2A3_CORE_WORKER_SIM_H

// Force simulator mode for this platform
#define A2A3_TARGET_SIMULATOR

// Include the common core worker interface
#include "../../runtime_a2a3/core/a2a3_core_worker.h"

#endif /* A2A3_CORE_WORKER_SIM_H */
