cmake_minimum_required(VERSION 3.22)
project(reconstruction)

add_executable(cpp_recon main.cpp)
target_include_directories(cpp_recon PUBLIC ../util)


