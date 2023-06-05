cmake_minimum_required(VERSION 3.22)
project(ImageProcessing)

set(CMAKE_CXX_STANDARD 14)
add_executable(main main.cpp)
add_subdirectory(util)
add_subdirectory(process)

target_link_libraries(main util processor)
