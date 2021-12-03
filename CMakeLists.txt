cmake_minimum_required(VERSION 3.19)
project(Zadanie2)

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})
find_package(PNG REQUIRED)
include_directories(${PNG_INCLUDE_PATH})
find_package(OpenMP)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lpng16 -fopenmp")
SET(CMAKE_C_COMPILER mpicc)
SET(CMAKE_CXX_COMPILER mpicxx)

set(SOURCE_FILES main.cpp)
add_executable(Zadanie2 ${SOURCE_FILES})
