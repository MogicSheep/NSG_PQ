cmake_minimum_required(VERSION 2.8)

project(efanna2e)
set(BOOST_ROOT "E:/software/boost/include/boost-1_78")

#添加头文件搜索路径
include_directories(E:/software/boost/include/boost-1_78)

#/添加库文件搜索路径
link_directories(E:/software/boost/lib)

include_directories(${PROJECT_SOURCE_DIR}/include)
#OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(FATAL_ERROR "no OpenMP supprot")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free")
add_definitions (-std=c++11 -O2 -lboost -march=native -Wall -DINFO)

add_subdirectory(src)
add_subdirectory(tests)
