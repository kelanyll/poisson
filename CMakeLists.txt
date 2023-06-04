cmake_minimum_required(VERSION 3.18.4)

set(CMAKE_CXX_STANDARD 17)

project(poisson)

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

add_library(core lib/src/PoissonRegressionTrainer.cpp lib/src/goals-predict.cpp lib/src/util.cpp)
target_include_directories(core PUBLIC lib/include)
target_include_directories(core PUBLIC ${BOOM_SRC_DIR})
target_include_directories(core PUBLIC ${BOOM_SRC_DIR}/cpputil)
target_include_directories(core PUBLIC ${BOOM_SRC_DIR}/Models/Glm)
target_include_directories(core PUBLIC ${BOOM_SRC_DIR}/Models)
target_include_directories(core PUBLIC ${BOOM_SRC_DIR}/Models/Policies)
target_link_libraries(core PUBLIC ${BOOM_SRC_DIR}/bazel-bin/libboom.a)
add_dependencies(core boom)
target_compile_options(core PUBLIC -g)

find_package(DataFrame CONFIG REQUIRED)
target_include_directories(core PUBLIC ${DataFrame_INCLUDE_DIRS})
target_link_libraries(core PUBLIC DataFrame::DataFrame)

add_executable(poisson poisson/src/main.cpp)
target_link_libraries(poisson PUBLIC core)

add_executable(tests test/goals-predict_test.cpp test/test-util.hpp)
target_link_libraries(tests PUBLIC core)

enable_testing()
find_package(GTest CONFIG REQUIRED)
target_link_libraries(tests PRIVATE GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(tests)

install(TARGETS poisson DESTINATION ${CMAKE_BINARY_DIR}/bin)