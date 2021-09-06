cmake_minimum_required(VERSION 2.6)

project(yolov3-tiny-int8)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)
if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    message("embed_platform on")
    include_directories(/usr/local/cuda/targets/aarch64-linux/include)
    link_directories(/usr/local/cuda/targets/aarch64-linux/lib)
else()
    message("embed_platform off")
    include_directories(/usr/local/cuda/include)
    link_directories(/usr/local/cuda/lib64)
endif()


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

#cuda_add_library(leaky ${PROJECT_SOURCE_DIR}/leaky.cu)
cuda_add_library(yololayer SHARED ${PROJECT_SOURCE_DIR}/src/yololayer.cu)
target_link_libraries(yololayer nvinfer cudart)

find_package(OpenCV)
include_directories(OpenCV_INCLUDE_DIRS)

add_executable(build_engine ${PROJECT_SOURCE_DIR}/build_engine.cpp)
add_executable(detect ${PROJECT_SOURCE_DIR}/detect.cpp ${PROJECT_SOURCE_DIR}/src/uart.cpp)

target_link_libraries(build_engine nvinfer cudart yololayer ${OpenCV_LIBS})
target_link_libraries(detect nvinfer cudart yololayer ${OpenCV_LIBS})

add_definitions(-pthread -g)
