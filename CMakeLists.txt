cmake_minimum_required(VERSION 3.18.4)

include(ExternalProject)

project(poisson)

add_executable(poisson main.cpp)

set(BOOM_SRC_DIR "${CMAKE_BINARY_DIR}/boom")

ExternalProject_Add(
    boom
    GIT_REPOSITORY https://github.com/steve-the-bayesian/BOOM.git
    GIT_TAG 2ff87325fbd2e3178b4c54d6508131a2fbfef90e
    UPDATE_DISCONNECTED true
    SOURCE_DIR ${BOOM_SRC_DIR}
    BUILD_IN_SOURCE true
    CONFIGURE_COMMAND ""
    BUILD_COMMAND bazelisk build boom
    INSTALL_COMMAND ""
)

add_dependencies(poisson boom)

target_include_directories(poisson PRIVATE ${BOOM_SRC_DIR})
target_include_directories(poisson PRIVATE ${BOOM_SRC_DIR}/cpputil)

target_link_libraries(poisson ${BOOM_SRC_DIR}/bazel-bin/libboom.a)

install(TARGETS poisson DESTINATION ${CMAKE_BINARY_DIR}/bin)