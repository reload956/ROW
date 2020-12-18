cmake_minimum_required(VERSION 3.16)
project(RowCalculation)

set(CMAKE_CXX_STANDARD 20)

add_executable(RowCalculation src/config.h src/cpp_accumulator.cpp src/experiments.h src/interface.h
        src/num_threads.cpp src/num_threads.h src/omp_accumulator.c src/run_experiments.cpp main.cpp)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -fopenmp")