cmake_minimum_required(VERSION 3.16.3)

set(BISHENG_COMPILER "${ASCEND_HOME_PATH}/bin/ccec")
set(BISHENG_LD "${ASCEND_HOME_PATH}/bin/ld.lld")
set(GRAPH_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../graph")

# Source files - dispatcher only
set(AICORE_DISPATCHER ${CMAKE_CURRENT_SOURCE_DIR}/kernel.cpp)

# Object files for dispatcher only
set(AICORE_OBJ_AIV ${CMAKE_CURRENT_BINARY_DIR}/kernel_aiv.o)
set(AICORE_OBJ_AIC ${CMAKE_CURRENT_BINARY_DIR}/kernel_aic.o)

# Final linked kernel binary
set(AICORE_O ${CMAKE_CURRENT_BINARY_DIR}/kernel.o)

set(AI_CORE_FLAGS
    "-c -O3 -g -x cce -Wall -std=c++17 \
    --cce-aicore-only \
    -mllvm -cce-aicore-stack-size=0x8000 \
    -mllvm -cce-aicore-function-stack-size=0x8000 \
    -mllvm -cce-aicore-record-overflow=false \
    -mllvm -cce-aicore-addr-transform \
    -mllvm -cce-aicore-dcci-insert-for-scalar=false \
    -I${GRAPH_DIR}"
)
separate_arguments(AI_CORE_FLAGS)

# Compile dispatcher only (AIC + AIV)
add_custom_target(kernel_obj ALL
    DEPENDS ${AICORE_DISPATCHER}

    # Compile dispatcher (main kernel entry point with execute_task)
    COMMAND ${BISHENG_COMPILER} ${AI_CORE_FLAGS} -D__AIC__ --cce-aicore-arch=dav-c220-cube
            -o ${AICORE_OBJ_AIC} ${AICORE_DISPATCHER}
    COMMAND ${BISHENG_COMPILER} ${AI_CORE_FLAGS} -D__AIV__ --cce-aicore-arch=dav-c220-vec
            -o ${AICORE_OBJ_AIV} ${AICORE_DISPATCHER}

    # Link dispatcher binary (entry point)
    COMMAND ${BISHENG_LD} -m aicorelinux -Ttext=0 -static -n -o ${AICORE_O}
            ${AICORE_OBJ_AIC} ${AICORE_OBJ_AIV}

    BYPRODUCTS ${AICORE_OBJ_AIC} ${AICORE_OBJ_AIV} ${AICORE_O}
    COMMENT "Build dispatcher kernel binary"
)
