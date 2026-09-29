cmake_minimum_required(VERSION 3.16)
project(butterfly_rf CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(ENABLE_COVERAGE "Build with code coverage instrumentation" OFF)
if(ENABLE_COVERAGE)
  add_compile_options(--coverage -O0)
  add_link_options(--coverage)
endif()

find_package(OpenCV REQUIRED)
include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.4.0
)
FetchContent_MakeAvailable(Catch2)

add_executable(tests src/pretreatment.cc src/ml.cc tests/test_pretreatment.cc)
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain ${OpenCV_LIBS})

add_executable(test_ml src/ml.cc tests/test_ml.cc)

add_executable(butterfly_rf
  main.cc
  src/pretreatment.cc
  src/ml.cc
)
target_link_libraries(butterfly_rf ${OpenCV_LIBS})
target_include_directories(butterfly_rf PRIVATE include)
