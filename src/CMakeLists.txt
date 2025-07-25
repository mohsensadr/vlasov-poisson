cmake_minimum_required(VERSION 3.18)
project(MyCudaApp LANGUAGES CXX CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA architecture (adjust if needed)
set(CMAKE_CUDA_ARCHITECTURES 70)

# Enable verbose build output (optional)
#set(CMAKE_VERBOSE_MAKEFILE ON)

# Include all source files
file(GLOB_RECURSE CPP_SOURCES CONFIGURE_DEPENDS *.cpp)
file(GLOB_RECURSE CU_SOURCES CONFIGURE_DEPENDS *.cu)
file(GLOB_RECURSE HEADERS CONFIGURE_DEPENDS *.hpp *.h *.cuh)

# Create the main executable
add_executable(main ${CPP_SOURCES} ${CU_SOURCES} ${HEADERS})

# Ensure proper compilation and linking
target_include_directories(main PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}     # for local headers like "IO.hpp"
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}  # for cuda_runtime.h
)

# Optional: enable device+host compilation support
set_target_properties(main PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
)

# Optional: warnings
# target_compile_options(main PRIVATE -Wall -Wextra)

# Optional: link CUDA runtime if needed explicitly
# find_package(CUDA REQUIRED)
# target_link_libraries(main PRIVATE CUDA::cudart)
