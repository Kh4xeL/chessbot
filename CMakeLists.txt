cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(AlphaZeroEngine)

set(CMAKE_CXX_STANDARD 17)

# Point CMake to your downloaded libtorch folder
set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/libtorch")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(engine src/main.cpp)
target_link_libraries(engine "${TORCH_LIBRARIES}")
set_property(TARGET engine PROPERTY CXX_STANDARD 17)