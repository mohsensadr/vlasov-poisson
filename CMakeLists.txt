cmake_minimum_required(VERSION 3.18)
project(vlasov_poisson LANGUAGES CXX CUDA)

# Default to Debug if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Debug flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CUDA_FLAGS_DEBUG "-G -O0")

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA compute capability (adjust for your GPU)
set(CMAKE_CUDA_ARCHITECTURES 70)

# Set the source directory
set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)

# Include headers from src/
include_directories(${SRC_DIR})

# Define source files explicitly
set(SOURCES
    ${SRC_DIR}/main.cpp
    ${SRC_DIR}/constants.cpp
    ${SRC_DIR}/solver.cu
    ${SRC_DIR}/moments.cu
    ${SRC_DIR}/vlasov_poisson.cu
    ${SRC_DIR}/sorting.cu
    ${SRC_DIR}/IO.cpp
    ${SRC_DIR}/MxE.cu
    ${SRC_DIR}/Depositors/BruteDepositor.cu
    ${SRC_DIR}/Depositors/TiledDepositor.cu
    ${SRC_DIR}/Depositors/SortedDepositor.cu
)

# Create the executable
add_executable(main ${SOURCES})

# Ensure proper compilation and linking
target_include_directories(main PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}     # for local headers like "IO.hpp"
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}  # for cuda_runtime.h
)

# Enable separable compilation for CUDA
#set_target_properties(main PROPERTIES
#    CUDA_SEPARABLE_COMPILATION ON
#)

# Optional: Make sure we're linking statically to cudart
# target_link_libraries(main PRIVATE cudart_static)

# Enable CUDA extended lambdas (for device lambdas in thrust::for_each)
target_compile_options(main PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
)

# Optional: output directory
set_target_properties(main PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin
)
