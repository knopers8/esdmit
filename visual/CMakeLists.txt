cmake_minimum_required(VERSION 2.8)
project(esdmit_svm)

set(CMAKE_CXX_FLAGS "-O3 -pedantic -Wall -Wextra -std=c++11 -fopenmp -DNDEBUG")

set(EIGEN_INCLUDE "Eigen/")
if (NOT EIGEN_INCLUDE)
    message("Eigen include dir not specified via EIGEN3_INCLUDE_DIR. Assuming submodule is being used.")
    set(EIGEN_INCLUDE "deps/eigen")
endif()

include_directories(SYSTEM "${EIGEN_INCLUDE}")

set(SOURCE_FILES
    esdmit_svm.cpp
    BinarySVM.cpp
    BinarySVM.h
    MultiSVM.cpp
    MultiSVM.h
    FileLoader.hpp
    util.hpp)

add_library(aaa SHARED ${SOURCE_FILES})

add_executable(esdmit_svm esdmit_svm.cpp)
target_link_libraries(esdmit_svm aaa)
