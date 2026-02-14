#pragma once

#include <cstddef>
#include <vector>

class Matrix
{
public:
    Matrix();
    Matrix(std::size_t r, std::size_t c, double value = 0.0);

    void assign(std::size_t r, std::size_t c, double value = 0.0);

    double& operator()(std::size_t r, std::size_t c);
    double operator()(std::size_t r, std::size_t c) const;

    bool is_empty() const;
    bool is_row_vector() const;
    bool is_col_vector() const;
    bool is_vector() const;

    void require_non_empty(const char* error_msg) const;
    void require_rows(std::size_t r, const char* error_msg) const;
    void require_cols(std::size_t c, const char* error_msg) const;
    void require_shape(std::size_t r, std::size_t c, const char* error_msg) const;

    void print() const;
    void scale_by_scalar(std::size_t value);

    Matrix transpose() const;
    Matrix argmax() const;

    Matrix slice_rows(std::size_t start, std::size_t end) const;
    Matrix slice_cols(std::size_t start, std::size_t end) const;

    std::size_t as_size_t(std::size_t r, std::size_t c, double epsilon = 1e-7) const;
    double scalar_mean() const;

    void shuffle_rows_with(Matrix& y);

    static Matrix dot(const Matrix& a, const Matrix& b);
    static double max_absolute_difference(const Matrix& a, const Matrix& b);

    std::size_t get_rows() const;
    std::size_t get_cols() const;
    const std::vector<double>& get_data() const;

private:
    std::size_t rows;
    std::size_t cols;
    std::vector<double> data;
};
