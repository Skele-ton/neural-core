# Neural Core

A C++17 neural network project with:
- a reusable static library (`neural_core`)
- a training/demo executable (`neural_core_app`)
- a Doctest-based test suite (`tests`)

The app trains a small dense network on Fashion-MNIST and supports optional scatter plotting for generated data.

## Features

- `Matrix` core (`include/neural_core/matrix.hpp`)
  - dense 2D tensor container with indexing, transpose, argmax, row/column slicing, dot product, row shuffling, scalar mean, shape guards, and numeric conversion helpers.
- Model API (`include/neural_core/model.hpp`)
  - sequential model construction, `configure`, `train`, `evaluate`, `predict`, parameter get/set, parameter-file IO, and full model save/load.
- Layers (`include/neural_core/layers.hpp`)
  - `LayerInput`, `LayerDense` (with L1/L2 regularization support), and `LayerDropout`.
- Activations (`include/neural_core/activations.hpp`)
  - ReLU, Softmax, Sigmoid, and Linear with forward/backward + prediction helpers.
- Losses (`include/neural_core/losses.hpp`)
  - Categorical Cross-Entropy, Binary Cross-Entropy, Mean Squared Error, Mean Absolute Error, plus combined Softmax + Categorical Cross-Entropy backward path.
- Accuracy metrics (`include/neural_core/accuracy.hpp`)
  - categorical and regression accuracy with accumulated-pass tracking.
- Optimizers (`include/neural_core/optimizers.hpp`)
  - SGD, Adagrad, RMSprop, and Adam, with learning-rate decay and internal momentum/cache state handling.
- Data utilities (`include/neural_core/data.hpp`)
  - Fashion-MNIST load/normalize pipeline and synthetic dataset generators: spiral, vertical, and sine.
- RNG utilities (`include/neural_core/rng.hpp`)
  - deterministic seeding (`set_global_seed`, per-thread stream IDs) and random sampling helpers.
- Core numeric utilities (`include/neural_core/core_utils.hpp`)
  - overflow checking and whole-number validation helpers used across components.
- Plotting (`include/neural_core/scatter_plot.hpp`)
  - optional scatter plot rendering/writing through Matplot++ when `ENABLE_MATPLOT=ON`.
- End-to-end executable (`train_fashion_mnist.cpp`)
  - example workflow: optional plotting, Fashion-MNIST training, evaluation, prediction display, and serialization checks.
- Test suite (`tests/`)
  - Doctest-based unit/integration tests for matrix math, layers/activations/losses/optimizers/accuracy, model IO, data loaders, and plotting behavior.

## Requirements

- Linux environment
- C++ compiler with C++17 support (GCC or Clang)
- CMake 3.15+
- `doctest.h` for tests
  - Repo: <https://github.com/doctest/doctest>
  - This project includes `doctest.h` at repo root and expects that header to be available there.
- Fashion-MNIST dataset files for training and related tests
  - Repo: <https://github.com/zalandoresearch/fashion-mnist>
  - Required uncompressed files under `fashion_mnist/`:
    - `train-images-idx3-ubyte`
    - `train-labels-idx1-ubyte`
    - `t10k-images-idx3-ubyte`
    - `t10k-labels-idx1-ubyte`
- `gnuplot` for data plotting at runtime when `ENABLE_MATPLOT=ON`
- `gcovr` for coverage reporting when `ENABLE_COVERAGE=ON`

## Build

The build/run flow targets Linux only.

Recommended: use separate build directories for different option sets.

Release (defaults: Matplot OFF, coverage OFF):

```bash
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release -j
```

Debug:

```bash
cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build-debug -j
```

Release with plotting enabled:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_COVERAGE=OFF -DENABLE_MATPLOT=ON
cmake --build build -j
```

## Run

Run the training/demo executable:

```bash
./build/neural_core_app
```

Or via CMake target:

```bash
cmake --build build --target run
```

## Test

Run tests with CTest:

```bash
ctest --test-dir build --output-on-failure
```

Or via CMake target:

```bash
cmake --build build --target check
```

## Coverage

Enable coverage instrumentation and run coverage report:

```bash
cmake -S . -B build-cov -DCMAKE_BUILD_TYPE=Debug -DENABLE_COVERAGE=ON
cmake --build build-cov -j
cmake --build build-cov --target coverage
```

Notes:
- Coverage mode is set up for GCC/Clang-style `--coverage`.
- `gcovr` must be installed and available in `PATH`.

## Plotting

When `ENABLE_MATPLOT=ON`:
- CMake fetches Matplot++ during configure.
- Runtime plotting uses `gnuplot`.
- The app writes a generated scatter plot to `plot.png`.

If plotting is disabled (`ENABLE_MATPLOT=OFF`), plotting calls are skipped or throw clear runtime errors in API usage.

## Dataset Setup

Fashion-MNIST is a Zalando clothing image dataset with 10 labeled classes, similar in format to MNIST.  
This project uses it as a realistic supervised-learning example so the full training pipeline can be exercised on non-trivial data.

Create a `fashion_mnist/` directory in the project root and place the 4 required uncompressed IDX files listed in Requirements.

The training app and `fashion_mnist_create(...)` default to this directory:

```cpp
fashion_mnist_create(X_train, y_train, X_test, y_test); // default dir: "fashion_mnist"
```

You can pass a custom directory path:

```cpp
fashion_mnist_create(X_train, y_train, X_test, y_test, "path/to/fashion_mnist");
```

## Project Layout

```text
include/neural_core/   Public headers
src/                   Library implementation
tests/                 Doctest test files
train_fashion_mnist.cpp  App entry point (neural_core_app)
CMakeLists.txt
```

## Common Issues

- `fashion_mnist_create: dataset files not found under: ...`
  - Verify dataset directory and exact IDX filenames.
- Windows build/runtime behavior
  - Windows is not supported at the moment; use Linux for build and execution.
- Plotting errors or skipped plotting with `ENABLE_MATPLOT=ON`
  - Verify `gnuplot` is installed and available in `PATH`.
- Coverage configure/build failure
  - Verify compiler is GCC/Clang compatible for `--coverage`.
  - Verify `gcovr` is installed.
