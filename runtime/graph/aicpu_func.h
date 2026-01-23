#ifndef GRAPH_AICPU_FUNC_H
#define GRAPH_AICPU_FUNC_H

#include "graph.h"
#include "handshake.h"
#include "kernel_args.h"

/**
 * Control - Process control flow for task graph
 *
 * This function handles control flow operations on the task graph.
 * User should implement their custom control flow logic here.
 *
 * @param g Reference to the task graph
 * @param hank Pointer to handshake buffer array
 * @param core_num Number of AICore instances available
 */
void Control(Graph& g, Handshake* hank, int core_num);

/**
 * Schdule - Schedule task execution on available cores
 *
 * This function handles task scheduling for the graph.
 * User should implement their custom scheduling logic here,
 * including task dispatch and execution control.
 *
 * @param g Reference to the task graph
 * @param hank Pointer to handshake buffer array
 * @param core_num Number of AICore instances available
 */
void Schdule(Graph& g, Handshake* hank, int core_num);

#endif  // GRAPH_AICPU_FUNC_H
