cmake_minimum_required (VERSION 2.8)
set (CMAKE_CXX_STANDARD 11)

project (Assignment2 CXX)

set (CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set (EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set (LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

find_package (OpenCL REQUIRED)
include_directories (${OpenCL_INCLUDE_DIRS})
link_directories (${OpenCL_LIBRARY})

include_directories (${CMAKE_SOURCE_DIR}/lib)
link_directories (${CMAKE_SOURCE_DIR}/lib)

add_executable (program main.cpp)
target_link_libraries (program jpeg)
target_link_libraries (program ${OpenCL_LIBRARY})

file (COPY ${CMAKE_SOURCE_DIR}/main.cl DESTINATION ${CMAKE_BINARY_DIR})
