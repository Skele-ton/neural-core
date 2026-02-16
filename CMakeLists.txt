# Suggested configure commands (use separate build dirs to avoid cached-option surprises):
#   Release (default options: Matplot OFF, coverage OFF):
#     cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
#   Debug:
#     cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug
#   Coverage (GCC/Clang):
#     cmake -S . -B build-cov -DCMAKE_BUILD_TYPE=Debug -DENABLE_COVERAGE=ON
#   Enable plotting explicitly when needed:
#     -DENABLE_MATPLOT=ON
# Build:
#   cmake --build <build-dir> -j

cmake_minimum_required(VERSION 3.15)
project(NeuralCore LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# options
option(ENABLE_MATPLOT "Enable Matplot++ plotting" OFF)
option(ENABLE_COVERAGE "Enable coverage instrumentation + gcovr target" OFF)

# neural_core library target (consumed by app and tests)
add_library(neural_core STATIC
  src/model.cpp
  src/activations.cpp
  src/layers.cpp
  src/losses.cpp
  src/optimizers.cpp
  src/accuracy.cpp
  src/data.cpp
  src/scatter_plot.cpp
  src/core_utils.cpp
  src/rng.cpp
  src/matrix.cpp
)
target_compile_definitions(neural_core PRIVATE NEURAL_CORE_NO_MAIN)
target_include_directories(neural_core PUBLIC ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/include)

# run main app
# cmake --build <build-dir> --target run
# note: runs the full training entrypoint, which can take time and requires dataset files.
add_executable(neural_core_app
  train_fashion_mnist.cpp
)
target_link_libraries(neural_core_app PRIVATE neural_core)

add_custom_target(run
  COMMAND $<TARGET_FILE:neural_core_app>
  DEPENDS neural_core_app
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# run tests
# cmake --build <build-dir> --target check
enable_testing()

add_executable(tests
  tests/test_main.cpp
  tests/test_core.cpp
  tests/test_matrix.cpp
  tests/test_data_plot.cpp
  tests/test_activations.cpp
  tests/test_layers.cpp
  tests/test_losses.cpp
  tests/test_optimizers.cpp
  tests/test_accuracy.cpp
  tests/test_model.cpp
)
target_link_libraries(tests PRIVATE neural_core)

add_test(NAME tests COMMAND $<TARGET_FILE:tests>)
set_tests_properties(tests PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_custom_target(check
  COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -V
  DEPENDS tests
)

# warnings + optimization policy (target-based)
if (MSVC)
  set(NEURAL_CORE_WARNINGS /W4)
else()
  set(NEURAL_CORE_WARNINGS -Wall -Wextra)
endif()

target_compile_options(neural_core_app PRIVATE ${NEURAL_CORE_WARNINGS})
target_compile_options(tests PRIVATE ${NEURAL_CORE_WARNINGS})

if (NOT MSVC)
  target_compile_options(neural_core_app PRIVATE $<$<CONFIG:Release>:-O2>)
  target_compile_options(tests PRIVATE $<$<CONFIG:Release>:-O2>)

  target_compile_options(neural_core_app PRIVATE $<$<CONFIG:RelWithDebInfo>:-O2>)
  target_compile_options(tests PRIVATE $<$<CONFIG:RelWithDebInfo>:-O2>)
endif()

target_compile_definitions(neural_core_app PRIVATE $<$<CONFIG:Release>:NDEBUG>)
target_compile_definitions(tests PRIVATE $<$<CONFIG:Release>:NDEBUG>)

# run coverage (requires configure with -DENABLE_COVERAGE=ON)
# cmake --build <build-dir> --target coverage
# reports coverage for src/* and tests/* filters.
if (ENABLE_COVERAGE)
  if (NOT (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang"))
    message(FATAL_ERROR "Coverage is set up for GCC/Clang style '--coverage'.")
  endif()

  find_program(GCOVR_EXECUTABLE gcovr)
  if (NOT GCOVR_EXECUTABLE)
    message(FATAL_ERROR "gcovr not found. Install it (e.g. pip install gcovr) or disable ENABLE_COVERAGE.")
  endif()

  target_compile_options(neural_core PRIVATE -O0 -g --coverage)
  target_compile_options(neural_core_app PRIVATE -O0 -g --coverage)
  target_compile_options(tests PRIVATE -O0 -g --coverage)
  target_link_options(neural_core_app PRIVATE --coverage)
  target_link_options(tests PRIVATE --coverage)

  add_custom_target(coverage
    COMMAND ${CMAKE_CTEST_COMMAND} --quiet --output-on-failure
    COMMAND ${GCOVR_EXECUTABLE}
            -r ${CMAKE_SOURCE_DIR}
            --filter "${CMAKE_SOURCE_DIR}/src/"
            --filter "${CMAKE_SOURCE_DIR}/tests/"
            --exclude-noncode-lines
    DEPENDS tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )
endif()

# all-in-one: run + tests (+coverage if enabled; tests run again during coverage)
# cmake --build <build-dir> --target all-in-one
add_custom_target(all-in-one
  DEPENDS run check
)

if (ENABLE_COVERAGE)
  add_dependencies(all-in-one coverage)
endif()

# matplot++
if (ENABLE_MATPLOT)
  include(FetchContent)

  FetchContent_Declare(matplotplusplus
    GIT_REPOSITORY https://github.com/alandefreitas/matplotplusplus.git
    GIT_TAG v1.2.2
  )

  set(MATPLOTPP_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(MATPLOTPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)

  FetchContent_MakeAvailable(matplotplusplus)

  if (TARGET matplot)
    target_link_libraries(neural_core PUBLIC matplot)
  elseif (TARGET Matplot++::matplot)
    target_link_libraries(neural_core PUBLIC Matplot++::matplot)
  endif()

  target_compile_definitions(neural_core PUBLIC ENABLE_MATPLOT)
endif()
