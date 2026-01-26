# Build AICore kernel using ccec compiler
#
# Required variables (set when calling cmake):
#   BISHENG_CC  - Path to ccec compiler
#   BISHENG_LD  - Path to ld.lld linker
#   CUSTOM_INCLUDE_DIRS - Additional include directories (for InCore functions)
#   CUSTOM_SOURCE_DIRS  - Additional source directories (for InCore functions)
#
# Output: aicore_kernel.o (combined AIC + AIV binary)

cmake_minimum_required(VERSION 3.16.3)

# Check required variables
if(NOT DEFINED BISHENG_CC)
    message(FATAL_ERROR "BISHENG_CC must be set to ccec compiler path")
endif()
if(NOT DEFINED BISHENG_LD)
    message(FATAL_ERROR "BISHENG_LD must be set to ld.lld linker path")
endif()

# Build include directory flags
set(CMAKE_CUSTOM_INCLUDE_DIR_FLAGS "")
list(APPEND CMAKE_CUSTOM_INCLUDE_DIR_FLAGS "-I${CMAKE_CURRENT_SOURCE_DIR}/../common")
if(DEFINED CUSTOM_INCLUDE_DIRS)
    foreach(INC_DIR ${CUSTOM_INCLUDE_DIRS})
        list(APPEND CMAKE_CUSTOM_INCLUDE_DIR_FLAGS "-I${INC_DIR}")
    endforeach()
endif()
string(REPLACE ";" " " CMAKE_CUSTOM_INCLUDE_DIR_FLAGS "${CMAKE_CUSTOM_INCLUDE_DIR_FLAGS}")

# Build source list: aicore_kernel.cpp + InCore function sources
set(AICORE_KERNEL_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/aicore_kernel.cpp")
if(DEFINED CUSTOM_SOURCE_DIRS)
    foreach(SRC_DIR ${CUSTOM_SOURCE_DIRS})
        file(GLOB DIR_SOURCES "${SRC_DIR}/*.cpp" "${SRC_DIR}/*.c")
        list(APPEND AICORE_KERNEL_SOURCES ${DIR_SOURCES})
    endforeach()
endif()
list(LENGTH AICORE_KERNEL_SOURCES NUM_SOURCES)
message(STATUS "AICore sources: ${NUM_SOURCES} files")

# Output: combined kernel binary
set(AICORE_KERNEL_OBJ ${CMAKE_CURRENT_BINARY_DIR}/aicore_kernel.o)

# Generate object file paths for each source
set(ALL_AIC_OBJECTS "")
set(ALL_AIV_OBJECTS "")

foreach(SRC_FILE ${AICORE_KERNEL_SOURCES})
    get_filename_component(SRC_NAME ${SRC_FILE} NAME_WE)
    list(APPEND ALL_AIC_OBJECTS "${CMAKE_CURRENT_BINARY_DIR}/${SRC_NAME}_aic.o")
    list(APPEND ALL_AIV_OBJECTS "${CMAKE_CURRENT_BINARY_DIR}/${SRC_NAME}_aiv.o")
endforeach()

# Compilation flags for AICore
set(AICORE_FLAGS
    "-c -O3 -g -x cce -Wall -std=c++17 \
    --cce-aicore-only \
    -mllvm -cce-aicore-stack-size=0x8000 \
    -mllvm -cce-aicore-function-stack-size=0x8000 \
    -mllvm -cce-aicore-record-overflow=false \
    -mllvm -cce-aicore-addr-transform \
    -mllvm -cce-aicore-dcci-insert-for-scalar=false \
    ${CMAKE_CUSTOM_INCLUDE_DIR_FLAGS}"
)
separate_arguments(AICORE_FLAGS)

# Generate compilation commands for each source file
foreach(SRC_FILE ${AICORE_KERNEL_SOURCES})
    get_filename_component(SRC_NAME ${SRC_FILE} NAME_WE)
    set(OBJ_AIC "${CMAKE_CURRENT_BINARY_DIR}/${SRC_NAME}_aic.o")
    set(OBJ_AIV "${CMAKE_CURRENT_BINARY_DIR}/${SRC_NAME}_aiv.o")

    # Compile for AIC (Cube) architecture
    add_custom_command(
        OUTPUT ${OBJ_AIC}
        COMMAND ${BISHENG_CC} ${AICORE_FLAGS} ${CMAKE_CUSTOM_INCLUDE_DIR_FLAGS} 
                -D__AIC__ --cce-aicore-arch=dav-c220-cube
                -o ${OBJ_AIC} ${SRC_FILE}
        DEPENDS ${SRC_FILE}
        COMMENT "Compiling ${SRC_NAME} for AIC (Cube)"
    )

    # Compile for AIV (Vector) architecture
    add_custom_command(
        OUTPUT ${OBJ_AIV}
        COMMAND ${BISHENG_CC} ${AICORE_FLAGS} ${CMAKE_CUSTOM_INCLUDE_DIR_FLAGS}
                -D__AIV__ --cce-aicore-arch=dav-c220-vec
                -o ${OBJ_AIV} ${SRC_FILE}
        DEPENDS ${SRC_FILE}
        COMMENT "Compiling ${SRC_NAME} for AIV (Vector)"
    )
endforeach()

# Link all AIC and AIV objects together
add_custom_target(aicore_kernel ALL
    DEPENDS ${ALL_AIC_OBJECTS} ${ALL_AIV_OBJECTS}

    COMMAND ${BISHENG_LD} -m aicorelinux -Ttext=0 -static -n
            -o ${AICORE_KERNEL_OBJ} ${ALL_AIC_OBJECTS} ${ALL_AIV_OBJECTS}

    BYPRODUCTS ${AICORE_KERNEL_OBJ}
    COMMENT "Linking AICore kernel binary"
)
