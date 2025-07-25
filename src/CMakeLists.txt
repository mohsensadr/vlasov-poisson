cmake_minimum_required(VERSION 3.18)
project(MyCudaApp LANGUAGES CXX CUDA)

# Set standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Set CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 70)

# Find all source files
file(GLOB CPP_SOURCES "*.cpp")
file(GLOB CU_SOURCES "*.cu")

# Create executable
add_executable(main ${CPP_SOURCES} ${CU_SOURCES})

# Optional: enable device+host builds
set_target_properties(main PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# ✅ Add CUDA include path for host code (e.g. <cuda_runtime.h>)
# This ensures g++ can find cuda_runtime.h
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
