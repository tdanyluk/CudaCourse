cmake_minimum_required(VERSION 3.10)

project(hw LANGUAGES C CXX CUDA)

find_package(CUDA  REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")
find_package( OpenCV REQUIRED )

file(GLOB files *.h *.hpp *.c *.cpp *.cu)

ADD_EXECUTABLE(hw ${files})
target_link_libraries(hw ${OpenCV_LIBS} )
