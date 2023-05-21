cmake_minimum_required(VERSION 3.18.4)

set(CMAKE_CXX_STANDARD 17)

project(poisson)

add_executable(poisson src/main.cpp src/PoissonRegressionTrainer.hpp src/PoissonRegressionTrainer.cpp src/data-types.hpp src/goals-predict.hpp src/goals-predict.cpp src/util.hpp)

set(BOOM_SRC_DIR "${CMAKE_BINARY_DIR}/boom")

include(ExternalProject)
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

target_compile_options(poisson PRIVATE -g)

enable_testing()
    
find_package(GTest CONFIG REQUIRED)

add_executable(tests test/goals-predict_test.cpp src/goals-predict.cpp)

target_include_directories(tests PRIVATE ${CMAKE_SOURCE_DIR}/src)

target_include_directories(tests PRIVATE ${BOOM_SRC_DIR})
target_include_directories(tests PRIVATE ${BOOM_SRC_DIR}/cpputil)
target_include_directories(tests PRIVATE ${BOOM_SRC_DIR}/Models/Glm)
target_include_directories(tests PRIVATE ${BOOM_SRC_DIR}/Models)
target_include_directories(tests PRIVATE ${BOOM_SRC_DIR}/Models/Policies)
target_link_libraries(tests PRIVATE ${BOOM_SRC_DIR}/bazel-bin/libboom.a)

target_include_directories(tests PRIVATE ${DataFrame_INCLUDE_DIRS})
target_link_libraries(tests PRIVATE DataFrame::DataFrame)

target_link_libraries(tests PRIVATE GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(tests)

target_compile_options(tests PRIVATE -g)

install(TARGETS poisson DESTINATION ${CMAKE_BINARY_DIR}/bin)