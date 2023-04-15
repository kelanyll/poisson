cmake_minimum_required(VERSION 3.18.4)
project(poisson)

set(BOOM_SOURCE_DIR ${CMAKE_SOURCE_DIR}/external/BOOM)

function(find_boom)
    find_library(BOOM_LIB libboom.a PATHS ${BOOM_SOURCE_DIR}/bazel-bin)
endfunction()

find_boom()

if(NOT BOOM_LIB)
    execute_process(COMMAND bazelisk build boom WORKING_DIRECTORY ${BOOM_SOURCE_DIR})
    find_boom()
    if(NOT BOOM_LIB)
        message(FATAL_ERROR "Unable to build BOOM library")
    endif()
endif()

add_executable(poisson main.cpp)

target_include_directories(poisson PRIVATE ${CMAKE_SOURCE_DIR}/external/BOOM/cpputil)
target_include_directories(poisson PRIVATE ${CMAKE_SOURCE_DIR}/external/BOOM)

target_link_libraries(poisson ${BOOM_LIB})

install(TARGETS poisson DESTINATION ${CMAKE_BINARY_DIR}/bin)