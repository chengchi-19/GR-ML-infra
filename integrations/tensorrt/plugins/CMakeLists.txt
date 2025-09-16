cmake_minimum_required(VERSION 3.15)
project(GRMLInfraTensorRTPlugins LANGUAGES CXX CUDA)

# C++17 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA架构
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)  # 支持V100, A100, RTX30xx, RTX40xx

# 查找依赖包
find_package(CUDA REQUIRED)
find_package(PkgConfig REQUIRED)

# 查找TensorRT
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES include)

find_library(TENSORRT_LIBRARY nvinfer
    HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 lib/x64)

find_library(TENSORRT_PLUGIN_LIBRARY nvinfer_plugin
    HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 lib/x64)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
    message(FATAL_ERROR "TensorRT not found! Set TENSORRT_ROOT environment variable.")
endif()

# 查找CUDA相关库
find_library(CUDA_RUNTIME_LIBRARY cudart ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
find_library(CUDA_CUBLAS_LIBRARY cublas ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
find_library(CUDA_CURAND_LIBRARY curand ${CUDA_TOOLKIT_ROOT_DIR}/lib64)

# 包含目录
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${TENSORRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
)

# 编译器标志
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -Xcompiler -fPIC")

# 添加调试信息（Debug模式）
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")
endif()

# 源文件
set(CPP_SOURCES
    cpp/tensorrt_plugin_base.cpp
    cpp/fused_attention_layernorm_plugin.cpp
    cpp/sequence_recommendation_interaction_plugin.cpp
    cpp/hierarchical_sequence_fusion_plugin.cpp
    cpp/interaction_triton_fast_plugin.cpp
    cpp/hstu_hierarchical_attention_plugin.cpp
    cpp/plugin_registry.cpp
)

set(CUDA_SOURCES
    cpp/fused_attention_layernorm_kernel.cu
    cpp/sequence_recommendation_interaction_kernel.cu
    cpp/hierarchical_sequence_fusion_kernel.cu
    cpp/interaction_triton_fast_kernel.cu
    cpp/hstu_hierarchical_attention_kernel.cu
)

# 创建共享库
add_library(gr_ml_infra_tensorrt_plugins SHARED
    ${CPP_SOURCES}
    ${CUDA_SOURCES}
)

# 链接库
target_link_libraries(gr_ml_infra_tensorrt_plugins
    ${TENSORRT_LIBRARY}
    ${TENSORRT_PLUGIN_LIBRARY}
    ${CUDA_RUNTIME_LIBRARY}
    ${CUDA_CUBLAS_LIBRARY}
    ${CUDA_CURAND_LIBRARY}
)

# 设置库属性
set_target_properties(gr_ml_infra_tensorrt_plugins PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
)

# 安装
install(TARGETS gr_ml_infra_tensorrt_plugins
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/
    DESTINATION include/gr_ml_infra_tensorrt_plugins
    FILES_MATCHING PATTERN "*.h"
)

# 创建Python绑定（可选）
find_package(pybind11 QUIET)
if(pybind11_FOUND)
    pybind11_add_module(gr_ml_infra_tensorrt_plugins_py
        python/tensorrt_plugins_binding.cpp
    )

    target_link_libraries(gr_ml_infra_tensorrt_plugins_py PRIVATE
        gr_ml_infra_tensorrt_plugins
    )

    target_compile_definitions(gr_ml_infra_tensorrt_plugins_py PRIVATE
        VERSION_INFO=${EXAMPLE_VERSION_INFO}
    )
endif()

# 打印配置信息
message(STATUS "TensorRT Include: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "TensorRT Library: ${TENSORRT_LIBRARY}")
message(STATUS "CUDA Include: ${CUDA_INCLUDE_DIRS}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")