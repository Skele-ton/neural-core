#pragma once

#include <string>

#include "neural_core/matrix.hpp"

void fashion_mnist_create(
    Matrix& X_train_out, Matrix& y_train_out,
    Matrix& X_test_out, Matrix& y_test_out,
    const std::string& dir = "fashion_mnist");

void generate_spiral_data(std::size_t samples_per_class, std::size_t classes, Matrix& X_out, Matrix& y_out);
void generate_vertical_data(std::size_t samples_per_class, std::size_t classes, Matrix& X_out, Matrix& y_out);
void generate_sine_data(std::size_t samples, Matrix& X_out, Matrix& y_out);
