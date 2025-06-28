cmake_minimum_required(VERSION 3.20)

# Set compilers BEFORE project() call

project(SYCkLe LANGUAGES CXX)

# Find Intel SYCL package
find_package(IntelSYCL REQUIRED)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Collect source files
file(GLOB_RECURSE SRC_FILES src/*.cpp)

# Create executable
add_executable(syckle ${SRC_FILES})

# Add SYCL support using Intel's modern CMake approach
add_sycl_to_target(TARGET syckle SOURCES ${SRC_FILES})

