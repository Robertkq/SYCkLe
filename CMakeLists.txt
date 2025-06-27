cmake_minimum_required(VERSION 3.2)
project(SYCkLe)

set(APPLICATION_NAME "syckle")

find_program(ICPX_COMPILER icpx)

if(NOT ICPX_COMPILER)
    message(FATAL_ERROR "icpx compiler not found. Please install Intel oneAPI DPC++/C++ Compiler.")
endif()

set(CMAKE_CXX_COMPILER ${ICPX_COMPILER})
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE SRC_FILES src/*.cpp)


add_executable(${APPLICATION_NAME} ${SRC_FILES})

target_compile_options(${APPLICATION_NAME} PRIVATE
    -fsycl
)

target_link_options(${APPLICATION_NAME} PRIVATE
    -fsycl
)

