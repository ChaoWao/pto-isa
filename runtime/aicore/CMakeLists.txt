cmake_minimum_required(VERSION 3.16.3)

set(BISHENG_COMPILER "${ASCEND_HOME_PATH}/bin/ccec")
set(BISHENG_LD "${ASCEND_HOME_PATH}/bin/ld.lld")

# Fetch pto-isa from gitcode.com using CMake FetchContent
include(FetchContent)

# Fetch pto-isa from gitcode.com
FetchContent_Declare(
    pto-isa
    GIT_REPOSITORY https://gitcode.com/cann/pto-isa.git
    GIT_TAG        master  # default branch is master
    GIT_SHALLOW    TRUE    # only fetch latest commit for faster download
)

# Populate (download) pto-isa but don't execute its CMakeLists.txt
# We only need the header files, not the full build
FetchContent_GetProperties(pto-isa)
if(NOT pto-isa_POPULATED)
    FetchContent_Populate(pto-isa)
endif()

# Set PTO_ISA_ROOT to the fetched source directory
set(PTO_ISA_ROOT ${pto-isa_SOURCE_DIR})
message(STATUS "Fetched pto-isa to: ${PTO_ISA_ROOT}")

set(AICORE_SRC ${CMAKE_CURRENT_SOURCE_DIR}/kernel.cpp)
set(AICORE_OBJ_AIV ${CMAKE_CURRENT_BINARY_DIR}/kernel_aiv.o)
set(AICORE_OBJ_AIC ${CMAKE_CURRENT_BINARY_DIR}/kernel_aic.o)
set(AICORE_O ${CMAKE_CURRENT_BINARY_DIR}/kernel.o)

set(AI_CORE_FLAGS
    "-c -O3 -g -x cce -Wall -std=c++17 \
    --cce-aicore-only \
    -mllvm -cce-aicore-stack-size=0x8000 \
    -mllvm -cce-aicore-function-stack-size=0x8000 \
    -mllvm -cce-aicore-record-overflow=false \
    -mllvm -cce-aicore-addr-transform \
    -mllvm -cce-aicore-dcci-insert-for-scalar=false \
    -DMEMORY_BASE \
    -I${PTO_ISA_ROOT}/include \
    -I${PTO_ISA_ROOT}/include/pto \
    -I${CMAKE_CURRENT_SOURCE_DIR}/.."
)
separate_arguments(AI_CORE_FLAGS)

add_custom_target(kernel_obj ALL DEPENDS ${AICORE_SRC}
    COMMAND ${BISHENG_COMPILER} ${AI_CORE_FLAGS} -D__AIC__ --cce-aicore-arch=dav-c220-cube -o ${AICORE_OBJ_AIC} ${AICORE_SRC}
    COMMAND ${BISHENG_COMPILER} ${AI_CORE_FLAGS} -D__AIV__ --cce-aicore-arch=dav-c220-vec -o ${AICORE_OBJ_AIV} ${AICORE_SRC}
    COMMAND ${BISHENG_LD} -m aicorelinux -Ttext=0 -static -n -o ${AICORE_O} ${AICORE_OBJ_AIC} ${AICORE_OBJ_AIV}
    BYPRODUCTS ${AICORE_OBJ_AIC} ${AICORE_OBJ_AIV} ${AICORE_O}
    COMMENT "Build kernel (AIC + AIV) with bisheng/ld.lld"
)
