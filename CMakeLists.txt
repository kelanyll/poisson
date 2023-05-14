cmake_minimum_required(VERSION 3.18.4)

set(CMAKE_CXX_STANDARD 17)

include(ExternalProject)

project(poisson)

add_executable(poisson src/main.cpp src/PoissonRegressionTrainer.hpp src/PoissonRegressionTrainer.cpp)

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
target_include_directories(poisson PRIVATE ${BOOM_SRC_DIR}/Models/Glm)
target_include_directories(poisson PRIVATE ${BOOM_SRC_DIR}/Models)
target_include_directories(poisson PRIVATE ${BOOM_SRC_DIR}/Models/Policies)

target_link_libraries(poisson PRIVATE ${BOOM_SRC_DIR}/bazel-bin/libboom.a)

find_package(DataFrame CONFIG REQUIRED)
target_include_directories(poisson PRIVATE ${DataFrame_INCLUDE_DIRS})
target_link_libraries(poisson PRIVATE DataFrame::DataFrame)

enable_testing()
    
find_package(GTest CONFIG REQUIRED)
target_link_libraries(poisson PRIVATE GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main)

# add_test(AllTestsInMain main)

install(TARGETS poisson DESTINATION ${CMAKE_BINARY_DIR}/bin)