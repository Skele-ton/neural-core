#include "neural_core/data.hpp"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "fashion_mnist/mnist_reader.hpp"
#include "neural_core/core_utils.hpp"
#include "neural_core/rng.hpp"

using std::acos;
using std::cos;
using std::runtime_error;
using std::sin;
using std::size_t;
using std::string;
using std::vector;

void fashion_mnist_create(
    Matrix& X_train_out, Matrix& y_train_out,
    Matrix& X_test_out, Matrix& y_test_out,
    const string& dir)
{
    namespace fs = std::filesystem;

    const string train_images = dir + "/train-images-idx3-ubyte";
    const string train_labels = dir + "/train-labels-idx1-ubyte";
    const string test_images = dir + "/t10k-images-idx3-ubyte";
    const string test_labels = dir + "/t10k-labels-idx1-ubyte";

    if (!fs::exists(train_images) || !fs::exists(train_labels) ||
        !fs::exists(test_images) || !fs::exists(test_labels)) {
        throw runtime_error("fashion_mnist_create: dataset files not found under: " + dir);
    }

    auto dataset = mnist::read_dataset<vector, vector, uint8_t, uint8_t>(dir);

    const size_t train_samples = dataset.training_images.size();
    const size_t test_samples = dataset.test_images.size();
    const size_t features = 784;

    X_train_out.assign(train_samples, features);
    y_train_out.assign(train_samples, 1);
    X_test_out.assign(test_samples, features);
    y_test_out.assign(test_samples, 1);

    for (size_t i = 0; i < train_samples; ++i) {
        for (size_t j = 0; j < features; ++j) {
            const double pixel = static_cast<double>(dataset.training_images[i][j]);
            X_train_out(i, j) = (pixel - 127.5) / 127.5;
        }

        y_train_out(i, 0) = static_cast<double>(dataset.training_labels[i]);
    }

    for (size_t i = 0; i < test_samples; ++i) {
        for (size_t j = 0; j < features; ++j) {
            const double pixel = static_cast<double>(dataset.test_images[i][j]);
            X_test_out(i, j) = (pixel - 127.5) / 127.5;
        }
        y_test_out(i, 0) = static_cast<double>(dataset.test_labels[i]);
    }

    X_train_out.shuffle_rows_with(y_train_out);
    X_test_out.shuffle_rows_with(y_test_out);
}

void generate_spiral_data(size_t samples_per_class, size_t classes, Matrix& X_out, Matrix& y_out)
{
    if (samples_per_class <= 1 || classes == 0) {
        throw runtime_error("generate_spiral_data: invalid arguments");
    }

    multiplication_overflow_check(classes, samples_per_class, "generate_spiral_data: total_samples overflow");

    const size_t total_samples = samples_per_class * classes;
    X_out.assign(total_samples, 2);
    y_out.assign(total_samples, 1);

    for (size_t class_idx = 0; class_idx < classes; ++class_idx) {
        const size_t class_offset = class_idx * samples_per_class;
        for (size_t i = 0; i < samples_per_class; ++i) {
            const double r = static_cast<double>(i) / static_cast<double>(samples_per_class - 1);
            double theta = static_cast<double>(class_idx) * 4.0 + r * 4.0;
            theta += random_gaussian() * 0.2;

            const double x = r * sin(theta);
            const double y = r * cos(theta);

            const size_t idx = class_offset + i;
            X_out(idx, 0) = x;
            X_out(idx, 1) = y;
            y_out(idx, 0) = static_cast<double>(class_idx);
        }
    }
}

void generate_vertical_data(size_t samples_per_class, size_t classes, Matrix& X_out, Matrix& y_out)
{
    if (samples_per_class == 0 || classes == 0) {
        throw runtime_error("generate_vertical_data: invalid arguments");
    }

    multiplication_overflow_check(classes, samples_per_class, "generate_vertical_data: total_samples overflow");

    const size_t total_samples = samples_per_class * classes;
    X_out.assign(total_samples, 2);
    y_out.assign(total_samples, 1);

    for (size_t class_idx = 0; class_idx < classes; ++class_idx) {
        const size_t class_offset = class_idx * samples_per_class;
        const double center_x = static_cast<double>(class_idx) / static_cast<double>(classes);

        for (size_t i = 0; i < samples_per_class; ++i) {
            const size_t idx = class_offset + i;

            const double x = center_x + random_gaussian() * 0.1;
            const double y = random_uniform();

            X_out(idx, 0) = x;
            X_out(idx, 1) = y;

            y_out(idx, 0) = static_cast<double>(class_idx);
        }
    }
}

void generate_sine_data(size_t samples, Matrix& X_out, Matrix& y_out)
{
    if (samples <= 1) {
        throw runtime_error("generate_sine_data: invalid arguments");
    }

    X_out.assign(samples, 1);
    y_out.assign(samples, 1);

    const double pi = acos(-1.0);

    for (size_t i = 0; i < samples; ++i) {
        const double x = static_cast<double>(i) / static_cast<double>(samples - 1);
        const double y = sin(2 * pi * x);

        X_out(i, 0) = x;
        y_out(i, 0) = y;
    }
}
