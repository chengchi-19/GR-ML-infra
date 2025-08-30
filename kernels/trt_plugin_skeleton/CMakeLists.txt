cmake_minimum_required(VERSION 3.10)
project(simple_plugin)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(CUDA REQUIRED)
find_package(TensorRT REQUIRED)

# CUTLASS support (optional)
option(USE_CUTLASS "Enable CUTLASS support" OFF)
if(USE_CUTLASS)
    find_package(CUTLASS REQUIRED)
    add_definitions(-DUSE_CUTLASS)
    message(STATUS "CUTLASS support enabled")
endif()

# Set CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_80")  # For A100
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

# Include directories
include_directories(
    ${CUDA_TOOLKIT_ROOT_DIR}/include
    ${TensorRT_INCLUDE_DIRS}
)

if(USE_CUTLASS)
    include_directories(${CUTLASS_INCLUDE_DIRS})
endif()

# Create shared library
add_library(simple_plugin SHARED 
    simple_plugin.cpp
    cutlass_prototype/cutlass_stub.cpp
)

# Set target properties
set_target_properties(simple_plugin PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Link libraries
target_link_libraries(simple_plugin
    ${CUDA_LIBRARIES}
    ${TensorRT_LIBRARIES}
    cublas
    cudart
)

if(USE_CUTLASS)
    target_link_libraries(simple_plugin ${CUTLASS_LIBRARIES})
endif()

# Set RPATH for runtime library loading
set_target_properties(simple_plugin PROPERTIES
    INSTALL_RPATH "${TensorRT_LIBRARY_DIRS}"
    BUILD_WITH_INSTALL_RPATH TRUE
)

# Installation
install(TARGETS simple_plugin
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

# Copy TensorRT headers if needed
if(NOT TARGET tensorrt_headers)
    add_custom_target(tensorrt_headers
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${TensorRT_INCLUDE_DIRS} ${CMAKE_BINARY_DIR}/include
        COMMENT "Copying TensorRT headers"
    )
endif()

# Add test target
enable_testing()
add_executable(plugin_test test_plugin.cpp)
target_link_libraries(plugin_test simple_plugin)
add_test(NAME PluginTest COMMAND plugin_test)

# Print configuration summary
message(STATUS "Build configuration:")
message(STATUS "  CUDA version: ${CUDA_VERSION}")
message(STATUS "  TensorRT version: ${TensorRT_VERSION}")
message(STATUS "  CUTLASS support: ${USE_CUTLASS}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
