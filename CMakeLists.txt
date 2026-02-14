# build without coverage:
# cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_MATPLOT=ON
#
# build with coverage:
# cmake -S . -B build-cov -DCMAKE_BUILD_TYPE=Debug -DENABLE_MATPLOT=ON -DENABLE_COVERAGE=ON
#
# build with debug (slow):
# cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug -DENABLE_MATPLOT=ON
#
# finish build:
# cmake --build <build-dir> -j

cmake_minimum_required(VERSION 3.15)
project(NeuralCore LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# options
option(ENABLE_MATPLOT "Enable Matplot++ plotting" OFF)
option(ENABLE_COVERAGE "Enable coverage instrumentation + gcovr target" OFF)

# library scaffold for incremental refactor
# keep app/tests unchanged for now; they still build from current sources
add_library(neural_core STATIC
  neural_core.cpp
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
add_executable(neural_core_app
  neural_core.cpp
  src/data.cpp
  src/scatter_plot.cpp
  src/core_utils.cpp
  src/rng.cpp
  src/matrix.cpp
)
target_include_directories(neural_core_app PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(neural_core_app PRIVATE ${CMAKE_SOURCE_DIR})

add_custom_target(run
  COMMAND $<TARGET_FILE:neural_core_app>
  DEPENDS neural_core_app
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# run tests
# cmake --build <build-dir> --target check
enable_testing()

add_executable(tests
  tests.cpp
  src/data.cpp
  src/scatter_plot.cpp
  src/core_utils.cpp
  src/rng.cpp
  src/matrix.cpp
)
target_include_directories(tests PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/include)

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
if (ENABLE_COVERAGE)
  if (NOT (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang"))
    message(FATAL_ERROR "Coverage is set up for GCC/Clang style '--coverage'.")
  endif()

  find_program(GCOVR_EXECUTABLE gcovr)
  if (NOT GCOVR_EXECUTABLE)
    message(FATAL_ERROR "gcovr not found. Install it (e.g. pip install gcovr) or disable ENABLE_COVERAGE.")
  endif()

  target_compile_options(tests PRIVATE -O0 -g --coverage)
  target_link_options(tests PRIVATE --coverage)

  add_custom_target(coverage
    COMMAND ${CMAKE_CTEST_COMMAND} --quiet --output-on-failure
    COMMAND ${GCOVR_EXECUTABLE}
            -r ${CMAKE_SOURCE_DIR}
            --filter "${CMAKE_SOURCE_DIR}/neural_core.cpp"
            --filter "${CMAKE_SOURCE_DIR}/tests.cpp"
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
    target_link_libraries(neural_core_app PRIVATE matplot)
  elseif (TARGET Matplot++::matplot)
    target_link_libraries(neural_core_app PRIVATE Matplot++::matplot)
  endif()

  target_compile_definitions(neural_core_app PRIVATE ENABLE_MATPLOT)
endif()
