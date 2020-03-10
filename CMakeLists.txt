cmake_minimum_required(VERSION 3.4)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


project(opencv-transfer-learning CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type (Debug, Release, RelWithDebInfo, MinSizeRel)" FORCE)
endif()


if(MSVC)
  add_definitions(-D_WIN32_WINNT=0x600)
endif()

#if(UNIX)
#  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
#endif()

# Requires OpenCV
FIND_PACKAGE( OpenCV 4 REQUIRED )



add_executable(opencv-transfer-learning
               opencv-transfer-learning.cpp)
target_include_directories(opencv-transfer-learning PRIVATE )

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  target_link_libraries(opencv-transfer-learning ${OpenCV_LIBRARIES} c++experimental)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  target_link_libraries(opencv-transfer-learning ${OpenCV_LIBRARIES} stdc++fs)
endif()

