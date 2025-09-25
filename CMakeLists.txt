cmake_minimum_required(VERSION 3.18)
project(vlasov_poisson LANGUAGES CXX CUDA)

# Default to Release if not specified
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

# Precision type (default = float)
# build with -DFLOAT_TYPE=float or double
set(FLOAT_TYPE "float" CACHE STRING "Floating point type (float or double)")
message(STATUS "Using FLOAT_TYPE=${FLOAT_TYPE}")

# Set the source directory
set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)

# Define source files explicitly
set(SOURCES
    ${SRC_DIR}/main.cpp
    ${SRC_DIR}/Constants/constants.cpp
    ${SRC_DIR}/Containers/particle_container.cu
    ${SRC_DIR}/Solvers/solver.cu
    ${SRC_DIR}/Depositors/moments.cu
    ${SRC_DIR}/Models/vlasov_poisson.cu
    ${SRC_DIR}/Sorters/sorting.cu
    ${SRC_DIR}/IOs/IO.cpp
    ${SRC_DIR}/VRs/MxE.cu
    ${SRC_DIR}/Depositors/BruteDepositor.cu
    ${SRC_DIR}/Depositors/TiledDepositor.cu
    ${SRC_DIR}/Depositors/SortedDepositor.cu
)

# Create the executable with the sources
add_executable(main ${SOURCES})

# Ensure proper compilation and linking
target_include_directories(main PRIVATE
    ${SRC_DIR}
    ${SRC_DIR}/Depositors
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}  # for cuda_runtime.h
)

target_compile_definitions(main PRIVATE FLOAT_TYPE=${FLOAT_TYPE})

# Link against cuFFT
target_link_libraries(main PRIVATE cufft)

# Optional: output directory
set_target_properties(main PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin
)

# Enable CUDA extended lambdas (for thrust::for_each etc.)
target_compile_options(main PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
)

# Optional: output directory
set_target_properties(main PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin
)
