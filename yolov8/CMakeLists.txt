cmake_minimum_required(VERSION 3.10)
project(yolov8 LANGUAGES CXX CUDA)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(TensorRT REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(build
    build.cu
)
target_link_libraries(build PUBLIC ${TensorRT_LIBRARIES})
target_include_directories(build PUBLIC ${TensorRT_INCLUDE_DIRS})

add_executable(runtime
    runtime.cu
    yolov8_utils.cpp
)
target_link_libraries(runtime PUBLIC ${PROJECT_BINARY_DIR} ${TensorRT_LIBRARIES} ${OpenCV_LIBS})
target_include_directories(runtime PUBLIC ${PROJECT_BINARY_DIR} ${TensorRT_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})