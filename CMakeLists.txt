cmake_minimum_required(VERSION 2.8.3)
project(test_SIFTGPU)
add_compile_options(-std=c++11)

find_package( OpenCV REQUIRED )

# OpenGL
find_package(OpenGL REQUIRED)

# GLUT
find_package(GLUT REQUIRED)

# Glew
find_package(GLEW REQUIRED)

find_package(CUDA REQUIRED)

# SiftGPU：手动设置其头文件与库文件所在位置
include_directories("/home/yao/Environment/SiftGPU/src/SiftGPU/" ${OpenGL_INCLUDE_DIR} ${CUDA_INCLUDE_DIR})
set(SIFTGPU_LIBS "/home/yao/Environment/SiftGPU/bin/libsiftgpu.so")

add_executable(test_SIFTGPU main.cpp )

target_link_libraries(test_SIFTGPU
    ${OpenCV_LIBS}
    ${CUDA_LIBS}
    ${SIFTGPU_LIBS}
    ${GLEW_LIBRARIES} ${GLUT_LIBRARIES} ${OPENGL_LIBRARIES}
)
