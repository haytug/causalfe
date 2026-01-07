cmake_minimum_required(VERSION 3.14)
project(cffe_core)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 REQUIRED)

pybind11_add_module(cffe_core cffe_core.cpp)

target_compile_options(cffe_core PRIVATE -O3 -march=native)
