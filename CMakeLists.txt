cmake_minimum_required(VERSION 3.10)
project(HPC_FEM_3D)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)

include_directories(src)
include_directories(src/grid)
include_directories(src/basis)
include_directories(src/quadrule)
include_directories(src/problem)

set(SOURCES
    src/main.cpp
    src/grid/grid.cpp
    src/basis/basis.cpp
    src/quadrule/quadrule.cpp
    src/problem/problem.cpp
)

add_executable(main ${SOURCES})

target_link_libraries(main Eigen3::Eigen)

set_target_properties(main PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/output
)

target_compile_options(main PRIVATE -O2 -Wall -Wextra)