#include "neural_core/matrix.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "neural_core/core_utils.hpp"
#include "neural_core/rng.hpp"

using std::abs;
using std::cout;
using std::numeric_limits;
using std::round;
using std::runtime_error;
using std::size_t;
using std::swap;
using std::vector;

Matrix::Matrix()
    : rows(0), cols(0), data()
{
}

Matrix::Matrix(size_t r, size_t c, double value)
    : rows(0), cols(0), data()
{
    assign(r, c, value);
}

void Matrix::assign(size_t r, size_t c, double value)
{
    multiplication_overflow_check(r, c, "Matrix::assign: size overflow");

    rows = r;
    cols = c;
    data.assign(r * c, value);
}

double& Matrix::operator()(size_t r, size_t c)
{
    if (r >= rows || c >= cols) {
        throw runtime_error("Matrix::operator(): index out of bounds");
    }

    return data[r * cols + c];
}

double Matrix::operator()(size_t r, size_t c) const
{
    if (r >= rows || c >= cols) {
        throw runtime_error("Matrix::operator() const: index out of bounds");
    }

    return data[r * cols + c];
}

bool Matrix::is_empty() const
{
    return rows == 0 || cols == 0;
}

bool Matrix::is_row_vector() const
{
    return rows == 1 && cols > 0;
}

bool Matrix::is_col_vector() const
{
    return cols == 1 && rows > 0;
}

bool Matrix::is_vector() const
{
    return is_row_vector() || is_col_vector();
}

void Matrix::require_non_empty(const char* error_msg) const
{
    if (is_empty()) throw runtime_error(error_msg);
}

void Matrix::require_rows(size_t r, const char* error_msg) const
{
    if (rows != r) throw runtime_error(error_msg);
}

void Matrix::require_cols(size_t c, const char* error_msg) const
{
    if (cols != c) throw runtime_error(error_msg);
}

void Matrix::require_shape(size_t r, size_t c, const char* error_msg) const
{
    if (rows != r || cols != c) throw runtime_error(error_msg);
}

void Matrix::print() const
{
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            cout << (*this)(i, j);
            if (j + 1 != cols) {
                cout << ' ';
            }
        }
        cout << '\n';
    }
}

void Matrix::scale_by_scalar(size_t value)
{
    if (value == 0) {
        throw runtime_error("Matrix::scale_by_scalar: value cannot be zero 0");
    }

    const double inv = 1.0 / static_cast<double>(value);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            (*this)(i, j) *= inv;
        }
    }
}

Matrix Matrix::transpose() const
{
    if (is_empty()) return Matrix();

    Matrix result(cols, rows);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result(j, i) = (*this)(i, j);
        }
    }

    return result;
}

Matrix Matrix::argmax() const
{
    if (is_empty()) return Matrix();

    Matrix result(rows, 1);

    for (size_t i = 0; i < rows; ++i) {
        size_t biggest_j = 0;
        double biggest_v = (*this)(i, 0);

        for (size_t j = 1; j < cols; ++j) {
            const double v = (*this)(i, j);
            if (v > biggest_v) {
                biggest_v = v;
                biggest_j = j;
            }
        }

        result(i, 0) = static_cast<double>(biggest_j);
    }

    return result;
}

Matrix Matrix::slice_rows(size_t start, size_t end) const
{
    if (end < start || end > rows) {
        throw runtime_error("Matrix::slice_rows: invalid slice bounds");
    }

    Matrix result(end - start, cols);
    for (size_t i = start; i < end; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result(i - start, j) = (*this)(i, j);
        }
    }
    return result;
}

Matrix Matrix::slice_cols(size_t start, size_t end) const
{
    if (end < start || end > cols) {
        throw runtime_error("Matrix::slice_cols: invalid slice bounds");
    }

    Matrix result(rows, end - start);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = start; j < end; ++j) {
            result(i, j - start) = (*this)(i, j);
        }
    }
    return result;
}

size_t Matrix::as_size_t(size_t r, size_t c, double epsilon) const
{
    const double v = (*this)(r, c);
    if (!is_whole_number(v, epsilon)) {
        throw runtime_error("Matrix::as_size_t: value is not integer-like");
    }
    const double vr = round(v);

    if (vr < static_cast<double>(numeric_limits<size_t>::min()) ||
        vr > static_cast<double>(numeric_limits<size_t>::max())) {
        throw runtime_error("Matrix::as_size_t: integer out of range");
    }

    return static_cast<size_t>(vr);
}

double Matrix::scalar_mean() const
{
    require_non_empty("Matrix::scalar_mean: cannot find mean of empty matrix");

    double sum = 0.0;
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            sum += (*this)(i, j);
        }
    }

    return sum / (rows * cols);
}

void Matrix::shuffle_rows_with(Matrix& y)
{
    require_non_empty("shuffle_rows_with: base matrix must be non-empty");

    const bool y_row = (y.get_rows() == 1 && y.get_cols() == rows);
    const bool y_col = (y.get_cols() == 1 && y.get_rows() == rows);
    if (!y_row && !y_col) {
        throw runtime_error("shuffle_rows_with: y must be shape (1,N) or (N,1), where N = base matrix rows");
    }

    if (rows < 2) return;

    for (size_t i = rows - 1; i > 0; --i) {
        const size_t j = random_uniform_int(0, i);
        if (i == j) continue;

        for (size_t c = 0; c < cols; ++c) {
            swap((*this)(i, c), (*this)(j, c));
        }

        if (y_row) {
            swap(y(0, i), y(0, j));
        } else {
            swap(y(i, 0), y(j, 0));
        }
    }
}

Matrix Matrix::dot(const Matrix& a, const Matrix& b)
{
    if (a.is_empty() || b.is_empty()) {
        throw runtime_error("Matrix::dot: matrices must not be empty");
    }

    if (a.get_cols() != b.get_rows()) {
        throw runtime_error("Matrix::dot: matrices have incompatible shapes");
    }

    Matrix result(a.get_rows(), b.get_cols(), 0.0);

    for (size_t i = 0; i < a.get_rows(); ++i) {
        for (size_t k = 0; k < a.get_cols(); ++k) {
            const double aik = a(i, k);
            for (size_t j = 0; j < b.get_cols(); ++j) {
                result(i, j) += aik * b(k, j);
            }
        }
    }

    return result;
}

double Matrix::max_absolute_difference(const Matrix& a, const Matrix& b)
{
    a.require_shape(b.get_rows(), b.get_cols(), "Matrix::max_absolute_difference: shape mismatch");

    double m = 0.0;
    for (size_t i = 0; i < a.get_rows(); ++i) {
        for (size_t j = 0; j < a.get_cols(); ++j) {
            const double d = abs(a(i, j) - b(i, j));
            if (d > m) m = d;
        }
    }

    return m;
}

size_t Matrix::get_rows() const
{
    return rows;
}

size_t Matrix::get_cols() const
{
    return cols;
}

const vector<double>& Matrix::get_data() const
{
    return data;
}
