cmake_minimum_required(VERSION 3.20)

# Set compilers BEFORE project() call

project(SYCkLe LANGUAGES CXX)

# Find Intel SYCL package
find_package(IntelSYCL REQUIRED)
find_package(CLI11 CONFIG REQUIRED)
find_package(OpenCV REQUIRED)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Collect source files
file(GLOB_RECURSE SRC_FILES src/*.cpp)


# Create executable
add_executable(syckle ${SRC_FILES})

# Add SYCL support using Intel's modern CMake approach
add_sycl_to_target(TARGET syckle SOURCES ${SRC_FILES})

target_include_directories(syckle PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(syckle PRIVATE ${OpenCV_INCLUDE_DIRS})

message("Linked libraries:")
message("CLI11: Using CLI11::CLI11 target")
message("OpenCV: ${OpenCV_LIBS}")

target_compile_options(syckle PRIVATE -fsycl -nocudalib)

target_link_libraries(syckle PRIVATE CLI11::CLI11)
target_link_libraries(syckle PRIVATE ${OpenCV_LIBS})

