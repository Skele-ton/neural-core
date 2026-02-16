#include "neural_core/accuracy.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "neural_core/core_utils.hpp"

using std::abs;
using std::max;
using std::runtime_error;
using std::size_t;
using std::sqrt;

void Accuracy::init(const Matrix&)
{
}

void Accuracy::reset()
{
}

double Accuracy::calculate(const Matrix& y_pred, const Matrix& y_true)
{
    y_pred.require_non_empty("Accuracy::calculate: y_pred must be non-empty");
    y_true.require_non_empty("Accuracy::calculate: y_true must be non-empty");

    multiplication_overflow_check(y_pred.get_rows(), y_pred.get_cols(), "Accuracy::calculate: total overflow");
    const size_t correct = compare(y_pred, y_true);
    const size_t total = y_pred.get_cols() * y_pred.get_rows();

    accumulated_sum += correct;
    accumulated_count += total;

    return static_cast<double>(correct) / static_cast<double>(total);
}

double Accuracy::calculate_accumulated() const
{
    if (accumulated_count == 0) {
        throw runtime_error("Accuracy::calculate_accumulated: accumulated_count must be > 0");
    }
    return accumulated_sum / static_cast<double>(accumulated_count);
}

void Accuracy::new_pass()
{
    accumulated_sum = 0.0;
    accumulated_count = 0;
}

AccuracyCategorical::AccuracyCategorical(bool binary)
    : binary(binary)
{
}

bool AccuracyCategorical::get_binray() const
{
    return binary;
}

size_t AccuracyCategorical::compare(const Matrix& y_pred, const Matrix& y_true)
{
    if (!binary) y_pred.require_cols(1, "AccuracyCategorical::compare: categorical y_pred must have shape (N,1)");

    const size_t pred_rows = y_pred.get_rows();
    const size_t pred_cols = y_pred.get_cols();

    Matrix ground_truth;

    if (binary) {
        y_true.require_shape(pred_rows, pred_cols,
            "AccuracyCategorical::compare: for binary accuracy y_true must match y_pred shape");

        ground_truth = y_true;
    } else {
        if (y_true.is_col_vector() && y_true.get_rows() == pred_rows) {
            ground_truth = y_true;
        } else if (y_true.is_row_vector() && y_true.get_cols() == pred_rows) {
            ground_truth = y_true.transpose();
        } else if (y_true.get_rows() == pred_rows && y_true.get_cols() >= 2) {
            ground_truth = y_true.argmax();
        } else {
            throw runtime_error(
                "AccuracyCategorical::compare: for non-binary accuracy y_true must be sparse (N,1)/(1,N) or one-hot (N,C)");
        }

        ground_truth.require_shape(pred_rows, pred_cols,
            "AccuracyCategorical::compare: formatted y_true must match y_pred shape");
    }

    size_t correct = 0;

    for (size_t i = 0; i < pred_rows; ++i) {
        for (size_t j = 0; j < pred_cols; ++j) {
            if (y_pred.as_size_t(i, j) == ground_truth.as_size_t(i, j)) ++correct;
        }
    }

    return correct;
}

AccuracyRegression::AccuracyRegression(double precision_divisor)
    : precision_divisor(precision_divisor), precision(0.0), initialized(false)
{
    if (precision_divisor <= 0.0) {
        throw runtime_error("AccuracyRegression: precision_divisor must be positive");
    }
}

void AccuracyRegression::init(const Matrix& y_true)
{
    y_true.require_non_empty("AccuracyRegression::init: y_true must be non-empty");

    const size_t samples = y_true.get_rows();
    const size_t outputs = y_true.get_cols();

    multiplication_overflow_check(samples, outputs, "AccuracyRegression::init: n overflow");

    const size_t n = samples * outputs;

    double mean = 0.0;
    for (size_t i = 0; i < samples; ++i) {
        for (size_t j = 0; j < outputs; ++j) {
            mean += y_true(i, j);
        }
    }
    mean /= static_cast<double>(n);

    double var = 0.0;
    for (size_t i = 0; i < samples; ++i) {
        for (size_t j = 0; j < outputs; ++j) {
            const double d = y_true(i, j) - mean;
            var += d * d;
        }
    }
    var /= static_cast<double>(n);

    if (var < 0.0) var = 0.0;

    const double standard_deviation = sqrt(var);

    precision = max(standard_deviation / precision_divisor, 1e-7);

    initialized = true;
}

void AccuracyRegression::reset()
{
    initialized = false;
}

double AccuracyRegression::get_precision_divisor() const
{
    return precision_divisor;
}

size_t AccuracyRegression::compare(const Matrix& y_pred, const Matrix& y_true)
{
    y_pred.require_shape(y_true.get_rows(), y_true.get_cols(),
        "AccuracyRegression::compare: y_pred and y_true must have the same shape");

    if (!initialized) init(y_true);

    const size_t samples = y_true.get_rows();
    const size_t outputs = y_true.get_cols();

    size_t correct = 0;
    for (size_t i = 0; i < samples; ++i) {
        for (size_t j = 0; j < outputs; ++j) {
            if (abs(y_pred(i, j) - y_true(i, j)) < precision) {
                ++correct;
            }
        }
    }

    return correct;
}
