#include <iostream>
#include <iomanip>
#include <vector>
#include <stdexcept>
#include <random>
#include <cmath>
#include <limits>
#include <string>
#include <fstream>
#include <algorithm>

#include "fashion_mnist/mnist_reader.hpp"

using std::cout;
using std::ofstream;
using std::runtime_error;
using std::size_t;
using std::string;
using std::numeric_limits;
using std::mt19937;
using std::normal_distribution;
using std::uniform_real_distribution;
using std::fixed;
using std::setprecision;
using std::swap;
using std::min;
using std::max;
using std::round;
using std::sqrt;
using std::abs;
using std::sin;
using std::cos;
using std::acos;
using std::exp;
using std::log;
using std::isfinite;

inline bool is_whole_number(double v, double epsilon = 1e-7)
{
    return abs(v - round(v)) <= epsilon;
}

inline void multiplication_overflow_check(const size_t a, const size_t b, const char* error_msg) {
    if (a != 0 && b > numeric_limits<size_t>::max() / a) {
        throw runtime_error(error_msg);
    }
}

// TODO: make varaibles for classes private and add getters/setters where needed
// TODO: standardize the labels of the data plotting somehow (make the middle equal 0 or something similar)
// TODO: Make rng implementation thread-safe
// TODO: add predict method to the model class
// TODO: add get_params, set_params, save_params, load_params, save, load methods to the model class
//       maybe find a cpp library for reading/writing objects to files
// TODO: seperate project into multiple files


// global RNG
mt19937 g_rng(0);

// each generated number is centered around 0 with a standard deviation of 1
// unbounded - technically can be any value but it is more likely to be around 0
double random_gaussian()
{
    static normal_distribution<double> dist(0.0, 1.0);
    return dist(g_rng);
}

// generated numbers are in range [0, 1)
// they are equally likely to be any value in that range
double random_uniform()
{
    static uniform_real_distribution<double> uniform(0.0, 1.0);
    return uniform(g_rng);
}

class Matrix
{
public:
    size_t rows;
    size_t cols;
    std::vector<double> data;

    Matrix() : rows(0), cols(0), data() {}

    Matrix(size_t r, size_t c, double value = 0.0)
        : rows(0), cols(0), data()
    {
        assign(r, c, value);
    }

    void assign(size_t r, size_t c, double value = 0.0)
    {
        multiplication_overflow_check(r, c, "Matrix::assign: size overflow");

        rows = r;
        cols = c;
        data.assign(r * c, value);
    }

    double& operator()(size_t r, size_t c)
    {
        if (r >= rows || c >= cols) {
            throw runtime_error("Matrix::operator(): index out of bounds");
        }

        return data[r * cols + c];
    }

    double operator()(size_t r, size_t c) const
    {
        if (r >= rows || c >= cols) {
            throw runtime_error("Matrix::operator() const: index out of bounds");
        }

        return data[r * cols + c];
    }

    // === helper methods for shape validation
    bool is_empty() const { return rows == 0 || cols == 0; }

    bool is_row_vector() const { return rows == 1 && cols > 0; }

    bool is_col_vector() const { return cols == 1 && rows > 0; }

    bool is_vector() const { return is_row_vector() || is_col_vector(); }

    void require_non_empty(const char* error_msg) const
    {
        if (is_empty()) throw runtime_error(error_msg);
    }

    void require_rows(size_t r, const char* error_msg) const
    {
        if (rows != r) throw runtime_error(error_msg);
    }

    void require_cols(size_t c, const char* error_msg) const
    {
        if (cols != c) throw runtime_error(error_msg);
    }

    void require_shape(size_t r, size_t c, const char* error_msg) const
    {
        if (rows != r || cols != c) throw runtime_error(error_msg);
    }
    // =======================================

    void print() const
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

    void scale_by_scalar(size_t value)
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

    Matrix transpose() const
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

    Matrix argmax() const
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

    Matrix slice_rows(size_t start, size_t end) const
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

    Matrix slice_cols(size_t start, size_t end) const
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

    size_t as_size_t(size_t r, size_t c, double epsilon = 1e-7) const
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

    double scalar_mean() const
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

    static Matrix dot(const Matrix& a, const Matrix& b)
    {
        if (a.is_empty() || b.is_empty()) {
            throw runtime_error("Matrix::dot: matrices must not be empty");
        }

        if (a.cols != b.rows) {
            throw runtime_error("Matrix::dot: matrices have incompatible shapes");
        }

        Matrix result(a.rows, b.cols, 0.0);

        for (size_t i = 0; i < a.rows; ++i) {
            for (size_t k = 0; k < a.cols; ++k) {
                double aik = a(i, k);
                for (size_t j = 0; j < b.cols; ++j) {
                    result(i, j) += aik * b(k, j);
                }
            }
        }

        return result;
    }

    // method for shuffling input data, where the base matrix is 2D and y is it's 1D labels
    void shuffle_rows_with(Matrix& y)
    {
        require_non_empty("shuffle_rows_with: base matrix must be non-empty");

        const bool y_row = (y.rows == 1 && y.cols == rows);
        const bool y_col = (y.cols == 1 && y.rows == rows);
        if (!y_row && !y_col) {
            throw runtime_error("shuffle_rows_with: y must be shape (1,N) or (N,1), where N = base matrix rows");
        }

        if (rows < 2) return;

        for (size_t i = rows - 1; i > 0; --i) {
            std::uniform_int_distribution<size_t> dist(0, i);
            const size_t j = dist(g_rng);
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
};

// training data
void fashion_mnist_create(
    Matrix& X_train_out, Matrix& y_train_out,
    Matrix& X_test_out,  Matrix& y_test_out,
    const string& dir = "fashion_mnist")
{
    auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(dir);

    const size_t train_samples = dataset.training_images.size();
    const size_t test_samples  = dataset.test_images.size();
    const size_t features = 784; // 28x28 pixels per image

    X_train_out.assign(train_samples, features);
    y_train_out.assign(train_samples, 1);
    X_test_out.assign(test_samples, features);
    y_test_out.assign(test_samples, 1);

    // converting the loaded data from the dataset into matrices and normalizing values to [-1, 1]
    for (size_t i = 0; i < train_samples; ++i) {
        for (size_t j = 0; j < features; ++j) {
            double pixel = static_cast<double>(dataset.training_images[i][j]);
            X_train_out(i, j) = (pixel - 127.5) / 127.5;
        }

        y_train_out(i, 0) = static_cast<double>(dataset.training_labels[i]);
    }

    // same for the test data
    for (size_t i = 0; i < test_samples; ++i) {
        for (size_t j = 0; j < features; ++j) {
            double pixel = static_cast<double>(dataset.test_images[i][j]);
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

    size_t total_samples = samples_per_class * classes;
    X_out.assign(total_samples, 2);
    y_out.assign(total_samples, 1);

    for (size_t class_idx = 0; class_idx < classes; ++class_idx) {
        size_t class_offset = class_idx * samples_per_class;
        for (size_t i = 0; i < samples_per_class; ++i) {
            double r = static_cast<double>(i) / static_cast<double>(samples_per_class - 1);
            double theta = static_cast<double>(class_idx) * 4.0 + r * 4.0;
            theta += random_gaussian() * 0.2;

            double x = r * sin(theta);
            double y = r * cos(theta);

            size_t idx = class_offset + i;
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

    size_t total_samples = samples_per_class * classes;
    X_out.assign(total_samples, 2);
    y_out.assign(total_samples, 1);

    for (size_t class_idx = 0; class_idx < classes; ++class_idx) {
        size_t class_offset = class_idx * samples_per_class;
        double center_x = static_cast<double>(class_idx) / static_cast<double>(classes);

        for (size_t i = 0; i < samples_per_class; ++i) {
            size_t idx = class_offset + i;

            double x = center_x + random_gaussian() * 0.1;
            double y = random_uniform();

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

        const size_t idx = i;
        X_out(idx, 0) = x;
        y_out(idx, 0) = y;
    }
}

// plots generated data
void plot_scatter_svg(const string& path, const Matrix& points, const Matrix& labels = Matrix())
{
    points.require_non_empty("plot_scatter_svg: points must be non-empty");
    if (points.cols < 2) {
        throw runtime_error("plot_scatter_svg: invalid input data");
    }

    const size_t num_points = points.rows;

    const bool has_labels = !labels.is_empty();
    if (has_labels) {
        const bool column_vector_like = (labels.rows == num_points && labels.cols == 1);
        const bool row_vector_like = (labels.rows == 1 && labels.cols == num_points);
        if (!column_vector_like && !row_vector_like) {
            throw runtime_error("plot_scatter_svg: labels must be shape (N,1) or (1,N) where N = points.rows");
    }
}

    double xmin = points(0, 0), xmax = points(0, 0);
    double ymin = points(0, 1), ymax = points(0, 1);

    for (size_t point_index = 1; point_index < num_points; ++point_index) {
        const double point_x = points(point_index, 0);
        const double point_y = points(point_index, 1);
        xmin = min(xmin, point_x);
        ymin = min(ymin, point_y);
        xmax = max(xmax, point_x);
        ymax = max(ymax, point_y);
    }

    const double dist_x = abs(xmax - xmin);
    const double dist_y = abs(ymax - ymin);

    const double eps = 1e-12;
    if (dist_x < eps || dist_y < eps) {
        throw runtime_error("plot_scatter_svg: x and y must have non-zero distance between values");
    }

    const double pad_x = 0.08 * dist_x;
    const double pad_y = 0.08 * dist_y;

    xmin -= pad_x;
    ymin -= pad_y;
    xmax += pad_x;
    ymax += pad_y;

    const size_t svg_width = 900, svg_height = 700;
    const size_t margin_left = 70, margin_right = 30, margin_top = 30, margin_bottom = 70;
    const size_t plot_width = svg_width - margin_left - margin_right;
    const size_t plot_height = svg_height - margin_top - margin_bottom;

    auto map_x = [&](double raw_x) -> double {
        const double normalized_x = (raw_x - xmin) / (xmax - xmin);
        const double clamped_x = normalized_x < 0.0 ? 0.0 : (normalized_x > 1.0 ? 1.0 : normalized_x);
        return margin_left + clamped_x * plot_width;
    };
    auto map_y = [&](double raw_y) -> double {
        const double normalized_y = (raw_y - ymin) / (ymax - ymin);
        const double clamped_y = normalized_y < 0.0 ? 0.0 : (normalized_y > 1.0 ? 1.0 : normalized_y);
        return margin_top + (1.0 - clamped_y) * plot_height;
    };

    ofstream out(path);
    if (!out) throw runtime_error("plot_scatter_svg: given path is invalid");

    out << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << svg_width
        << "\" height=\"" << svg_height << "\" viewBox=\"0 0 " << svg_width << " " << svg_height << "\">\n";

    out << "<rect x=\"0\" y=\"0\" width=\"" << svg_width << "\" height=\"" << svg_height
        << "\" fill=\"white\"/>\n";

    out << "<rect x=\"" << margin_left << "\" y=\"" << margin_top << "\" width=\"" << plot_width << "\" height=\"" << plot_height
        << "\" fill=\"none\" stroke=\"#222\" stroke-width=\"1\"/>\n";

    const size_t ticks = 10;

    out << "<g stroke=\"#ddd\" stroke-width=\"1\">\n";
    for (size_t i = 1; i < ticks; ++i) {
        const double grid_x = margin_left + (plot_width * static_cast<double>(i) / ticks);
        const double grid_y = margin_top + (plot_height * static_cast<double>(i) / ticks);
        out << "<line x1=\"" << grid_x << "\" y1=\"" << margin_top << "\" x2=\"" << grid_x << "\" y2=\"" << (margin_top + plot_height) << "\"/>\n";
        out << "<line x1=\"" << margin_left << "\" y1=\"" << grid_y << "\" x2=\"" << (margin_left + plot_width) << "\" y2=\"" << grid_y << "\"/>\n";
    }
    out << "</g>\n";

    out << "<g fill=\"#222\" font-family=\"Arial\" font-size=\"12\">\n";
    out << fixed << setprecision(3);
    for (size_t i = 0; i <= ticks; ++i) {
        const double tick_value_x = xmin + (xmax - xmin) * static_cast<double>(i) / ticks;
        const double tick_value_y = ymin + (ymax - ymin) * static_cast<double>(i) / ticks;

        const double tick_pos_x = margin_left + (plot_width * static_cast<double>(i) / ticks);
        const double tick_pos_y = margin_top + plot_height - (plot_height * static_cast<double>(i) / ticks);
 
        out << "<text x=\"" << tick_pos_x << "\" y=\"" << (margin_top + plot_height + 22)
            << "\" text-anchor=\"middle\">" << tick_value_x << "</text>\n";
        out << "<text x=\"" << (margin_left - 10) << "\" y=\"" << (tick_pos_y + 4)
            << "\" text-anchor=\"end\">" << tick_value_y << "</text>\n";
    }
    out << "</g>\n";

    out << "<g>\n";
    const double r = 2.4;

    static const char* colors[] = {
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    };
    const size_t num_of_colors = static_cast<size_t>(sizeof(colors) / sizeof(colors[0]));

    for (size_t point_index = 0; point_index < num_points; ++point_index) {
        const double point_x = points(point_index, 0);
        const double point_y = points(point_index, 1);

        size_t class_id = 0;
        if (has_labels) {
            if (labels.rows == num_points) class_id = labels.as_size_t(point_index, 0);
            else class_id = labels.as_size_t(0, point_index);
        }
        const size_t color_id = class_id  % num_of_colors;
        const char* class_color = colors[color_id];

        out << "<circle cx=\"" << map_x(point_x) << "\" cy=\"" << map_y(point_y)
            << "\" r=\"" << r << "\" fill=\"" << class_color
            << "\" fill-opacity=\"0.85\"/>\n";
    }
    out << "</g>\n";

    out << "</svg>\n";
}

// activations
class Activation
{
public:
    Matrix inputs;
    Matrix output;
    Matrix dinputs;

    virtual ~Activation() = default;

    virtual void forward(const Matrix& inputs_batch) = 0;
    virtual void backward(const Matrix& dvalues) = 0;
    virtual Matrix predictions(const Matrix& outputs) const = 0;
};

class ActivationReLU : public Activation
{
public:

    void forward(const Matrix& inputs_batch) override
    {
        inputs_batch.require_non_empty("ActivationReLU::forward: inputs must be non-empty");

        inputs = inputs_batch;
        output.assign(inputs.rows, inputs.cols);
        for (size_t i = 0; i < inputs.rows; ++i) {
            for (size_t j = 0; j < inputs.cols; ++j) {
                output(i, j) = max(0.0, inputs(i, j));
            }
        }
    }

    void backward(const Matrix& dvalues) override
    {
        dvalues.require_non_empty("ActivationReLU::backward: dvalues must be non-empty");
        dvalues.require_shape(inputs.rows, inputs.cols,
            "ActivationReLU::backward: dvalues shape mismatch");

        dinputs = dvalues;
        for (size_t i = 0; i < inputs.rows; ++i) {
            for (size_t j = 0; j < inputs.cols; ++j) {
                if (inputs(i, j) <= 0.0) {
                    dinputs(i, j) = 0.0;
                }
            }
        }
    }

    Matrix predictions(const Matrix& outputs) const override { return outputs; }
};

// Softmax activation
class ActivationSoftmax : public Activation
{
public:
    void forward(const Matrix& inputs_batch) override
    {
        inputs_batch.require_non_empty("ActivationSoftmax::forward: inputs must be non-empty");

        inputs = inputs_batch;
        output.assign(inputs.rows, inputs.cols);

        for (size_t i = 0; i < inputs.rows; ++i) {
            double max_val = inputs(i, 0);
            for (size_t j = 1; j < inputs.cols; ++j) {
                double v = inputs(i, j);
                if (v > max_val) max_val = v;
            }

            double sum = 0.0;
            for (size_t j = 0; j < inputs.cols; ++j) {
                double e = exp(inputs(i, j) - max_val);
                output(i, j) = e;
                sum += e;
            }

            if (!isfinite(sum) || sum <= 0.0) {
                throw runtime_error("ActivationSoftmax: invalid sum of exponentials");
            }

            for (size_t j = 0; j < inputs.cols; ++j) {
                output(i, j) /= sum;
            }
        }
    }

    void backward(const Matrix& dvalues) override
    {
        dvalues.require_non_empty("ActivationSoftmax::backward: dvalues must be non-empty");
        dvalues.require_shape(output.rows, output.cols,
            "ActivationSoftmax::backward: dvalues shape mismatch");

        dinputs.assign(dvalues.rows, dvalues.cols);

        const size_t samples = dvalues.rows;
        const size_t classes = dvalues.cols;

        for (size_t i = 0; i < samples; ++i) { // O(C) per sample
            double alpha = 0.0;
            for (size_t k = 0; k < classes; ++k) {
                alpha += output(i, k) * dvalues(i, k);
            }
            for (size_t j = 0; j < classes; ++j) {
                dinputs(i, j) = output(i, j) * (dvalues(i, j) - alpha);
            }
        }
    }

    Matrix predictions(const Matrix& outputs) const override
    {
        outputs.require_non_empty("ActivationSoftmax::predictions: outputs must be non-empty");
        if(outputs.cols < 2) {
            throw runtime_error("ActivationSoftmax::predictions: computation of softmax predictions requires outputs.cols >= 2");
        }

        return outputs.argmax();
    }
};

// Sigmoid activation
class ActivationSigmoid : public Activation
{
public:
    void forward(const Matrix& inputs_batch) override
    {
        inputs_batch.require_non_empty("ActivationSigmoid::forward: inputs must be non-empty");

        inputs = inputs_batch;
        output.assign(inputs.rows, inputs.cols);
        for (size_t i = 0; i < inputs.rows; ++i) {
            for (size_t j = 0; j < inputs.cols; ++j) {
                // stable sigmoid - prevents overflow for large negative values
                const double inp = inputs(i, j);

                if (inp >= 0) {
                    output(i, j) = 1.0 / (1.0 + exp(-inp));
                } else {
                    const double inp_exp = exp(inp);
                    output(i, j) = inp_exp / (1.0 + inp_exp);
                }
            }
        }
    }

    void backward(const Matrix& dvalues) override
    {
        dvalues.require_non_empty("ActivationSigmoid::backward: dvalues must be non-empty");
        dvalues.require_shape(output.rows, output.cols,
            "ActivationSigmoid::backward: dvalues shape mismatch");

        dinputs.assign(dvalues.rows, dvalues.cols);
        for (size_t i = 0; i < dvalues.rows; ++i) {
            for (size_t j = 0; j < dvalues.cols; ++j) {
                const double s = output(i, j);
                dinputs(i, j) = dvalues(i, j) * (1.0 - s) * s;
            }
        }
    }

    Matrix predictions(const Matrix& outputs) const override
    {
        outputs.require_non_empty("ActivationSigmoid::predictions: outputs must be non-empty");

        Matrix preds(outputs.rows, outputs.cols);
        for (size_t i = 0; i < outputs.rows; ++i) {
            for (size_t j = 0; j < outputs.cols; ++j) {
                preds(i, j) = outputs(i, j) > 0.5 ? 1.0 : 0.0;
            }
        }
        return preds;
    }
};

// Linear activation (identity)
class ActivationLinear : public Activation
{
public:
    void forward(const Matrix& inputs_batch) override
    {
        inputs_batch.require_non_empty("ActivationLinear::forward: inputs must be non-empty");

        inputs = inputs_batch;
        output = inputs_batch;
    }

    void backward(const Matrix& dvalues) override
    {
        dvalues.require_non_empty("ActivationLinear::backward: dvalues must be non-empty");
        dvalues.require_shape(inputs.rows, inputs.cols,
            "ActivationLinear::backward: dvalues shape mismatch");
        
        dinputs = dvalues;
    }

    Matrix predictions(const Matrix& outputs) const override { return outputs; }
};

// layers
class Layer
{
public:
    Matrix inputs;
    Matrix output;
    Matrix dinputs;
    Activation* activation = nullptr;

    virtual ~Layer() = default;

    virtual void forward(const Matrix& inputs_batch, bool training) = 0;
    virtual void backward(const Matrix& dvalues, bool include_activation = true) = 0;
};

// dense layer with weights/biases and cached inputs/output
class LayerDense : public Layer
{
public:
    Matrix weights;
    Matrix biases;

    double weight_regularizer_l1;
    double weight_regularizer_l2;
    double bias_regularizer_l1;
    double bias_regularizer_l2;

    Matrix weight_momentums;
    Matrix bias_momentums;
    Matrix weight_cache;
    Matrix bias_cache;

    Matrix dweights;
    Matrix dbiases;

    LayerDense(size_t n_inputs, size_t n_neurons, Activation& activation_fn,
               double weight_regularizer_l1 = 0.0,
               double weight_regularizer_l2 = 0.0,
               double bias_regularizer_l1 = 0.0,
               double bias_regularizer_l2 = 0.0)
        : weights(n_inputs, n_neurons),
          biases(1, n_neurons, 0.0),
          weight_regularizer_l1(weight_regularizer_l1),
          weight_regularizer_l2(weight_regularizer_l2),
          bias_regularizer_l1(bias_regularizer_l1),
          bias_regularizer_l2(bias_regularizer_l2)
    {
        if (weight_regularizer_l1 < 0.0 || weight_regularizer_l2 < 0.0 ||
            bias_regularizer_l1 < 0.0 || bias_regularizer_l2 < 0.0) {
            throw runtime_error("LayerDense: regularizers must be non-negative");
        }

        if(n_inputs == 0 || n_neurons == 0) {
            throw runtime_error("LayerDense:  n_inputs and n_neurons must be > 0");
        }

        activation = &activation_fn;

        for (size_t input = 0; input < n_inputs; ++input) {
            for (size_t neuron = 0; neuron < n_neurons; ++neuron) {
                weights(input, neuron) = 0.1 * random_gaussian();
            }
        }
    }

    void forward(const Matrix& inputs_batch)
    {
        inputs = inputs_batch;

        weights.require_non_empty("LayerDense::forward: weights must be initialized");
        inputs.require_non_empty("LayerDense::forward: inputs must be non-empty");

        inputs.require_cols(weights.rows, "LayerDense::forward: inputs.cols must match weights.rows");
        biases.require_shape(1, weights.cols,
            "LayerDense::forward: biases must be shape (1, n_neurons)");

        output = Matrix::dot(inputs, weights);
        for (size_t i = 0; i < output.rows; ++i) {
            for (size_t j = 0; j < output.cols; ++j) {
                output(i, j) += biases(0, j);
            }
        }

        if (!activation) {
            throw runtime_error("LayerDense::forward: activation must be set");
        }
        activation->forward(output);
        output = activation->output;
    }
    
    // forward overload for dropout layer
    void forward(const Matrix& inputs_batch, bool) override { forward(inputs_batch); }

    void backward(const Matrix& dvalues, bool include_activation) override
    {
        dvalues.require_non_empty("LayerDense::backward: dvalues must be non-empty");
        dvalues.require_shape(inputs.rows, weights.cols,
            "LayerDense::backward: dvalues shape mismatch");

        Matrix dactivation;
        if (include_activation) {
            if (!activation) {
                throw runtime_error("LayerDense::backward: activation must be set");
            }
            activation->backward(dvalues);
            dactivation = activation->dinputs;
        } else {
            dactivation = dvalues;
        }
        
        const Matrix inputs_T = inputs.transpose();
        dweights = Matrix::dot(inputs_T, dactivation);

        dbiases.assign(1, biases.cols, 0.0);
        for (size_t i = 0; i < dactivation.rows; ++i) {
            for (size_t j = 0; j < dactivation.cols; ++j) {
                dbiases(0, j) += dactivation(i, j);
            }
        }

        // l1 and l2 regularization
        const bool has_w_l1 = weight_regularizer_l1 != 0.0;
        const bool has_w_l2 = weight_regularizer_l2 != 0.0;

        if (has_w_l1 || has_w_l2) {
            const double weight_l2_times_two = weight_regularizer_l2 * 2.0;

            for (size_t i = 0; i < weights.rows; ++i) {
                for (size_t j = 0; j < weights.cols; ++j) {
                    const double w = weights(i, j);

                    if (has_w_l1) dweights(i, j) += weight_regularizer_l1 * ((w >= 0.0) ? 1.0 : -1.0);
                    if (has_w_l2) dweights(i, j) += weight_l2_times_two * w;
                }
            }
        }

        const bool has_b_l1 = bias_regularizer_l1 != 0.0;
        const bool has_b_l2 = bias_regularizer_l2 != 0.0;

        if (has_b_l1 || has_b_l2) {
            const double bias_l2_times_two = bias_regularizer_l2 * 2.0;

            for (size_t j = 0; j < biases.cols; ++j) {
                const double b = biases(0, j);

                if (has_b_l1) dbiases(0, j) += bias_regularizer_l1 * ((b >= 0.0) ? 1.0 : -1.0);
                if (has_b_l2) dbiases(0, j) += bias_l2_times_two * b;
            }
        }

        const Matrix weights_T = weights.transpose();
        dinputs = Matrix::dot(dactivation, weights_T);
    }
};

// dropout layer
class LayerDropout : public Layer
{
public:
    double keep_rate;
    Matrix scaled_binary_mask;
    ActivationLinear activation_linear;

    explicit LayerDropout(double rate)
        : keep_rate(1.0 - rate)
    {
        if (keep_rate <= 0.0 || keep_rate > 1.0) {
            throw runtime_error("LayerDropout: rate must be in (0,1]");
        }

        activation = &activation_linear;
    }

    void forward(const Matrix& inputs_batch, bool training = true) override
    {
        inputs_batch.require_non_empty("LayerDropout::forward: inputs must be non-empty");

        inputs = inputs_batch;

        scaled_binary_mask.assign(inputs.rows, inputs.cols);
        output.assign(inputs.rows, inputs.cols);

        for (size_t i = 0; i < inputs.rows; ++i) {
            for (size_t j = 0; j < inputs.cols; ++j) {
                const double mask = training ? ((random_uniform() < keep_rate) ? (1.0 / keep_rate) : 0.0) : 1.0;
                scaled_binary_mask(i, j) = mask;
                output(i, j) = inputs(i, j) * mask;
            }
        }

        if (!activation) {
            throw runtime_error("LayerDropout::forward: activation must be set");
        }
        activation->forward(output);
        output = activation->output;
    }

    void backward(const Matrix& dvalues, bool include_activation) override
    {
        dvalues.require_non_empty("LayerDropout::backward: dvalues must be non-empty");
        dvalues.require_shape(scaled_binary_mask.rows, scaled_binary_mask.cols,
            "LayerDropout::backward: dvalues shape mismatch");

        Matrix dactivation;
        if (include_activation) {
            if (!activation) {
                throw runtime_error("LayerDropout::backward: activation must be set");
            }
            activation->backward(dvalues);
            dactivation = activation->dinputs;
        } else {
            dactivation = dvalues;
        }

        dinputs.assign(dactivation.rows, dactivation.cols);
        for (size_t i = 0; i < dactivation.rows; ++i) {
            for (size_t j = 0; j < dactivation.cols; ++j) {
                dinputs(i, j) = dactivation(i, j) * scaled_binary_mask(i, j);
            }
        }
    }
};

// input "layer" for storing inputs into the model. Doesn't inherit from the base layer class
class LayerInput
{
public:
    Matrix output;

    void forward(const Matrix& inputs_batch)
    {
        inputs_batch.require_non_empty("LayerInput::forward: inputs must be non-empty");
        output = inputs_batch;
    }
};

// loss functions
class Loss
{
public:
    Matrix dinputs;
    std::vector<LayerDense*> trainable_layers;
    double accumulated_sum = 0.0;
    size_t accumulated_count = 0;

    virtual ~Loss() = default;

    void remember_trainable_layers(const std::vector<LayerDense*>& layers)
    {
        trainable_layers = layers;
    }

    double calculate(const Matrix& output, const Matrix& y_true)
    {
        output.require_non_empty("Loss::calculate: output must be non-empty");
        y_true.require_non_empty("Loss::calculate: y_true must be non-empty");

        Matrix sample_losses = forward(output, y_true);

        sample_losses.require_shape(1, output.rows,
            "Loss::calculate: per-sample losses must be of shape (1,output.rows) after forward");

        double sum = 0.0;
        for (double v : sample_losses.data) sum += v;
        const double count = sample_losses.data.size();

        accumulated_sum += sum;
        accumulated_count += count;

        return sum / static_cast<double>(count);
    }

    double calculate(const Matrix& output, const Matrix& y_true, double& out_regularization_loss)
    {
        const double data_loss = calculate(output, y_true);
        out_regularization_loss = regularization_loss_self();
        return data_loss;
    }

    double calculate_accumulated() const
    {
        if (accumulated_count == 0) {
            throw runtime_error("Loss::calculate_accumulated: accumulated_count must be > 0");
        }
        return accumulated_sum / static_cast<double>(accumulated_count);
    }

    double calculate_accumulated(double& out_regularization_loss) const
    {
        const double data_loss = calculate_accumulated();
        out_regularization_loss = regularization_loss_self();
        return data_loss;
    }

    void new_pass()
    {
        accumulated_sum = 0.0;
        accumulated_count = 0;
    }

    double regularization_loss_self() const
    {
        double regularization = 0.0;
        for (const LayerDense* layer : trainable_layers) {
            if (!layer) continue;
            regularization += regularization_loss_layer(*layer);
        }
        return regularization;
    }

    static double regularization_loss_layer(const LayerDense& layer)
    {
        if (layer.weight_regularizer_l1 < 0.0 || layer.weight_regularizer_l2 < 0.0 ||
            layer.bias_regularizer_l1 < 0.0 || layer.bias_regularizer_l2 < 0.0) {
            throw runtime_error("Loss::regularization_loss_layer: regularizer coefficients must be non-negative");
        }

        double regularization = 0.0;

        const bool has_w_l1 = layer.weight_regularizer_l1 != 0.0;
        const bool has_w_l2 = layer.weight_regularizer_l2 != 0.0;

        if(has_w_l1 || has_w_l2) {
            layer.weights.require_non_empty("Loss::regularization_loss_layer: weights must be non-empty");

            double sum_abs = 0.0;
            double sum_sq  = 0.0;

            for (double weight : layer.weights.data) {
                if (has_w_l1) sum_abs += abs(weight);
                if (has_w_l2)  sum_sq  += weight * weight;
            }

            regularization += layer.weight_regularizer_l1 * sum_abs + layer.weight_regularizer_l2 * sum_sq;
        }

        const bool has_b_l1 = layer.bias_regularizer_l1 != 0.0;
        const bool has_b_l2 = layer.bias_regularizer_l2 != 0.0;

        if(has_b_l1 || has_b_l2) {
            layer.biases.require_non_empty("Loss::regularization_loss_layer: biases must be non-empty");
            layer.weights.require_non_empty("Loss::regularization_loss_layer: weights must be non-empty");

            layer.biases.require_shape(1, layer.weights.cols,
                "Loss::regularization_loss_layer: biases must have shape (1, n_neurons)");

            double sum_abs = 0.0;
            double sum_sq  = 0.0;

            for (double bias : layer.biases.data) {
                if (has_b_l1) sum_abs += abs(bias);
                if (has_b_l2) sum_sq  += bias * bias;
            }

            regularization += layer.bias_regularizer_l1 * sum_abs + layer.bias_regularizer_l2 * sum_sq;
        }

        return regularization;
    }

    virtual void backward(const Matrix& y_pred, const Matrix& y_true) = 0;

protected:
    virtual Matrix forward(const Matrix& output, const Matrix& y_true) const = 0;

    static double clamp(double p)
    {
        constexpr double eps = 1e-7;
        if (p < eps) return eps;
        if (p > 1.0 - eps) return 1.0 - eps;
        return p;
    }
};

class LossCategoricalCrossEntropy : public Loss
{
public:
    Matrix forward(const Matrix& y_pred, const Matrix& y_true) const override
    {
        if (y_pred.cols < 2) {
            throw runtime_error("LossCategoricalCrossEntropy::forward: y_pred.cols must be >= 2");
        }

        const size_t samples = y_pred.rows;
        const size_t classes = y_pred.cols;

        Matrix sample_losses(1, samples, 0.0);

        Matrix y_true_sparse;

        if (y_true.is_col_vector() && y_true.rows == samples) {
            y_true_sparse = y_true;
        } else if (y_true.is_row_vector() && y_true.cols == samples) {
            y_true_sparse = y_true.transpose();
        } else if (y_true.rows == samples && y_true.cols == classes) {
            y_true_sparse = y_true.argmax();
        } else {
            throw runtime_error("LossCategoricalCrossEntropy::forward: y_true must be sparse (N,1) or one-hot (N,C)");
        }

        y_true_sparse.require_shape(samples, 1,
            "LossCategoricalCrossEntropy::forward: y_true_sparse must have shape (N,1)");

        for (size_t i = 0; i < samples; ++i) {
            const size_t class_idx = y_true_sparse.as_size_t(i, 0);

            if (class_idx >= classes) {
                throw runtime_error("LossCategoricalCrossEntropy::forward: y_true class index out of range");
            }

            double confidence = clamp(y_pred(i, class_idx));
            sample_losses(0, i) = -log(confidence);
        }

        return sample_losses;
    }

    void backward(const Matrix& y_pred, const Matrix& y_true) override
    {
        y_pred.require_non_empty("LossCategoricalCrossEntropy::backward: y_pred must be non-empty");
        y_true.require_non_empty("LossCategoricalCrossEntropy::backward: y_true must be non-empty");

        if (y_pred.cols < 2) {
            throw runtime_error("LossCategoricalCrossEntropy::backward: y_pred.cols must be >= 2");
        }

        const size_t samples = y_pred.rows;
        const size_t classes  = y_pred.cols;

        dinputs.assign(samples, classes, 0.0);

        Matrix y_true_sparse;

        if (y_true.is_col_vector() && y_true.rows == samples) {
            y_true_sparse = y_true;
        } else if (y_true.is_row_vector() && y_true.cols == samples) {
            y_true_sparse = y_true.transpose();
        } else if (y_true.rows == samples && y_true.cols == classes) {
            y_true_sparse = y_true.argmax();
        } else {
            throw runtime_error("LossCategoricalCrossEntropy::backward: y_true must be sparse (N,1) or one-hot (N,C)");
        }

        y_true_sparse.require_shape(samples, 1,
            "LossCategoricalCrossEntropy::backward: y_true_sparse must have shape (N,1)");

        for (size_t i = 0; i < samples; ++i) {
            const size_t class_idx = y_true_sparse.as_size_t(i, 0);
            if (class_idx >= classes) {
                throw runtime_error("LossCategoricalCrossEntropy::backward: class index out of range");
            }

            dinputs(i, class_idx) = -1.0 / clamp(y_pred(i, class_idx));
        }

        dinputs.scale_by_scalar(samples);
    }
};

class LossBinaryCrossentropy : public Loss
{
public:
    Matrix forward(const Matrix& y_pred, const Matrix& y_true) const override
    {
        y_true.require_shape(y_pred.rows, y_pred.cols,
            "LossBinaryCrossentropy::forward: y_pred and y_true must have the same shape");
        
        const size_t samples = y_pred.rows;
        const size_t outputs = y_pred.cols;

        Matrix sample_losses(1, samples, 0.0);

        for (size_t i = 0; i < samples; ++i) {
            double loss_sum = 0.0;
            for (size_t j = 0; j < outputs; ++j) {
                double pred = clamp(y_pred(i, j));
                double truth = y_true(i, j);

                loss_sum += -(truth * log(pred) + (1.0 - truth) * log(1.0 - pred));
            }

            sample_losses(0, i) = loss_sum / static_cast<double>(outputs);
        }

        return sample_losses;
    }

    void backward(const Matrix& y_pred, const Matrix& y_true) override
    {
        y_pred.require_non_empty("LossBinaryCrossentropy::backward: y_pred must be non-empty");
        y_true.require_non_empty("LossBinaryCrossentropy::backward: y_true must be non-empty");

        y_true.require_shape(y_pred.rows, y_pred.cols,
            "LossBinaryCrossentropy::backward: y_pred and y_true must have the same shape");

        const size_t samples = y_pred.rows;
        const size_t outputs = y_pred.cols;

        dinputs.assign(samples, outputs, 0.0);

        for (size_t i = 0; i < samples; ++i) {
            for (size_t j = 0; j < outputs; ++j) {
                double pred = clamp(y_pred(i, j));
                double truth = y_true(i, j);

                dinputs(i, j) = -(truth / pred - (1.0 - truth) / (1.0 - pred)) / static_cast<double>(outputs);
            }
        }

        dinputs.scale_by_scalar(samples);
    }
};

class LossMeanSquaredError : public Loss
{
public:
    Matrix forward(const Matrix& y_pred, const Matrix& y_true) const override
    {
        y_true.require_shape(y_pred.rows, y_pred.cols,
            "LossMeanSquaredError::forward: y_pred and y_true must have the same shape");

        const size_t samples = y_pred.rows;
        const size_t outputs = y_pred.cols;

        Matrix sample_losses(1, samples, 0.0);

        for (size_t i = 0; i < samples; ++i) {
            double sum = 0.0;
            for (size_t j = 0; j < outputs; ++j) {
                const double diff = y_true(i, j) - y_pred(i, j);
                sum += diff * diff;
            }
            sample_losses(0, i) = sum / static_cast<double>(outputs);
        }

        return sample_losses;
    }

    void backward(const Matrix& y_pred, const Matrix& y_true) override
    {
        y_pred.require_non_empty("LossMeanSquaredError::backward: y_pred must be non-empty");
        y_true.require_non_empty("LossMeanSquaredError::backward: y_true must be non-empty");

        y_true.require_shape(y_pred.rows, y_pred.cols,
            "LossMeanSquaredError::backward: y_pred and y_true must have the same shape");

        const size_t samples = y_pred.rows;
        const size_t outputs = y_pred.cols;

        dinputs.assign(samples, outputs, 0.0);

        for (size_t i = 0; i < samples; ++i) {
            for (size_t j = 0; j < outputs; ++j) {
                dinputs(i, j) = -2.0 * (y_true(i, j) - y_pred(i, j)) / static_cast<double>(outputs);
            }
        }

        dinputs.scale_by_scalar(samples);
    }
};

class LossMeanAbsoluteError : public Loss
{
public:
    Matrix forward(const Matrix& y_pred, const Matrix& y_true) const override
    {
        y_true.require_shape(y_pred.rows, y_pred.cols,
            "LossMeanAbsoluteError::forward: y_pred and y_true must have the same shape");

        const size_t samples = y_pred.rows;
        const size_t outputs = y_pred.cols;

        Matrix sample_losses(1, samples, 0.0);

        for (size_t i = 0; i < samples; ++i) {
            double sum = 0.0;
            for (size_t j = 0; j < outputs; ++j) {
                sum += abs(y_true(i, j) - y_pred(i, j));
            }
            sample_losses(0, i) = sum / static_cast<double>(outputs);
        }

        return sample_losses;
    }

    void backward(const Matrix& y_pred, const Matrix& y_true) override
    {
        y_pred.require_non_empty("LossMeanAbsoluteError::backward: y_pred must be non-empty");
        y_true.require_non_empty("LossMeanAbsoluteError::backward: y_true must be non-empty");

        y_true.require_shape(y_pred.rows, y_pred.cols,
            "LossMeanAbsoluteError::backward: y_pred and y_true must have the same shape");

        const size_t samples = y_pred.rows;
        const size_t outputs = y_pred.cols;

        dinputs.assign(samples, outputs, 0.0);

        for (size_t i = 0; i < samples; ++i) {
            for (size_t j = 0; j < outputs; ++j) {
                const double diff = y_true(i, j) - y_pred(i, j);
                double grad = 0.0;
                // diff > 0 => |diff| = y_true - y_pred => d/dy_pred = -1
                if (diff > 0.0) grad = -1.0;
                // diff < 0 => |diff| = -(y_true - y_pred) = y_pred - y_true => d/dy_pred = +1
                else if (diff < 0.0) grad =  1.0;
                dinputs(i, j) = grad / static_cast<double>(outputs);
            }
        }

        dinputs.scale_by_scalar(samples);
    }
};

// Softmax classifier - combined Softmax activation and cross-entropy loss (backward only)
class ActivationSoftmaxLossCategoricalCrossEntropy
{
public:
    Matrix dinputs;

    void backward(const Matrix& y_pred, const Matrix& y_true)
    {
        y_pred.require_non_empty("ActivationSoftmaxLossCategoricalCrossEntropy::backward: y_pred must be non-empty");
        y_true.require_non_empty("ActivationSoftmaxLossCategoricalCrossEntropy::backward: y_true must be non-empty");

        if (y_pred.cols < 2) {
            throw runtime_error("ActivationSoftmaxLossCategoricalCrossEntropy::backward: y_pred.cols must be >= 2");
        }

        const size_t samples = y_pred.rows;
        const size_t classes = y_pred.cols;

        Matrix y_true_sparse;

        if (y_true.is_col_vector() && y_true.rows == samples) {
            y_true_sparse = y_true;
        } else if (y_true.is_row_vector() && y_true.cols == samples) {
            y_true_sparse = y_true.transpose();
        } else if (y_true.rows == samples && y_true.cols == classes) {
            y_true_sparse = y_true.argmax();
        } else {
            throw runtime_error("ActivationSoftmaxLossCategoricalCrossEntropy::backward: y_true must be sparse (N,1) or one-hot (N,C)");
        }

        y_true_sparse.require_shape(samples, 1,
            "ActivationSoftmaxLossCategoricalCrossEntropy::backward: y_true_sparse must have shape (N,1)");

        dinputs = y_pred;
        for (size_t i = 0; i < samples; ++i) {
            size_t class_idx = y_true_sparse.as_size_t(i, 0);
            if (class_idx >= classes) {
                throw runtime_error("ActivationSoftmaxLossCategoricalCrossEntropy::backward: class index out of range");
            }
            dinputs(i, class_idx) -= 1.0;
        }

        dinputs.scale_by_scalar(samples);
    }
};

// optimizers
class Optimizer
{
public:
    double learning_rate;
    double current_learning_rate;
    double decay;
    size_t iterations;

    Optimizer(double learning_rate, double decay)
        : learning_rate(learning_rate),
          current_learning_rate(learning_rate),
          decay(decay),
          iterations(0)
    {
        if (learning_rate <= 0.0) {
            throw runtime_error("Optimizer: learning_rate must be positive");
        }
        if (decay < 0.0) {
            throw runtime_error("Optimizer: decay must be non-negative");
        }
    }

    virtual ~Optimizer() = default;

    virtual void pre_update_params()
    {
        current_learning_rate = learning_rate;
        if (decay != 0.0) {
            current_learning_rate = learning_rate / (1.0 + decay * static_cast<double>(iterations));
        }
    }

    void post_update_params()
    {
        ++iterations;
    }

    virtual void update_params(LayerDense& layer) = 0;
};

// SGD optimizer
class OptimizerSGD : public Optimizer
{
public:
    double momentum;

    OptimizerSGD(double learning_rate = 1.0, double decay = 0.0, double momentum = 0.0)
        : Optimizer(learning_rate, decay),
          momentum(momentum)
    {
        if (momentum < 0.0) {
            throw runtime_error("OptimizerSGD: momentum must be non-negative");
        }
    }

    void update_params(LayerDense& layer) override
    {
        layer.weights.require_non_empty("OptimizerSGD::update_params: layer.weights must be non-empty");
        layer.biases.require_non_empty("OptimizerSGD::update_params: layer.biases must be non-empty");

        layer.biases.require_shape(1, layer.weights.cols, "OptimizerSGD::update_params: biases must have shape (1, n_neurons)");

        layer.dweights.require_shape(layer.weights.rows, layer.weights.cols,
            "OptimizerSGD::update_params: dweights must match weights shape");
        layer.dbiases.require_shape(1, layer.biases.cols,
            "OptimizerSGD::update_params: dbiases must match biases shape");

        const size_t w_rows = layer.weights.rows;
        const size_t w_cols = layer.weights.cols;
        const size_t b_cols = layer.biases.cols;

        const double minus_learning_rate = -current_learning_rate;

        if (momentum == 0.0) {
            for (size_t i = 0; i < w_rows; ++i) {
                for (size_t j = 0; j < w_cols; ++j) {
                    layer.weights(i, j) += minus_learning_rate * layer.dweights(i, j);
                }
            }
            for (size_t j = 0; j < b_cols; ++j) {
                layer.biases(0, j) += minus_learning_rate * layer.dbiases(0, j);
            }
        } else {
            if (layer.weight_momentums.rows != w_rows || layer.weight_momentums.cols != w_cols) {
                layer.weight_momentums.assign(w_rows, w_cols, 0.0);
            }
            if (layer.bias_momentums.rows != 1 || layer.bias_momentums.cols != b_cols) {
                layer.bias_momentums.assign(1, b_cols, 0.0);
            }

            for (size_t i = 0; i < w_rows; ++i) {
                for (size_t j = 0; j < w_cols; ++j) {
                    const double temp = momentum * layer.weight_momentums(i, j) + minus_learning_rate * layer.dweights(i, j);
                    layer.weight_momentums(i, j) = temp;
                    layer.weights(i, j) += temp;
                }
            }

            for (size_t j = 0; j < b_cols; ++j) {
                const double temp = momentum * layer.bias_momentums(0, j) + minus_learning_rate * layer.dbiases(0, j);
                layer.bias_momentums(0, j) = temp;
                layer.biases(0, j) += temp;
            }
        }
    }
};

// Adagrad optimizer
class OptimizerAdagrad : public Optimizer
{
public:
    double epsilon;

    OptimizerAdagrad(double learning_rate = 1.0, double decay = 0.0, double epsilon = 1e-7)
        : Optimizer(learning_rate, decay),
          epsilon(epsilon)
    {
        if (epsilon <= 0.0) {
            throw runtime_error("OptimizerAdagrad: epsilon must be positive");
        }
    }

    void update_params(LayerDense& layer) override
    {
        layer.weights.require_non_empty("OptimizerAdagrad::update_params: layer.weights must be non-empty");
        layer.biases.require_non_empty("OptimizerAdagrad::update_params: layer.biases must be non-empty");

        layer.biases.require_shape(1, layer.weights.cols,
            "OptimizerAdagrad::update_params: biases must have shape (1, n_neurons)");

        layer.dweights.require_shape(layer.weights.rows, layer.weights.cols,
            "OptimizerAdagrad::update_params: dweights must match weights shape");
        layer.dbiases.require_shape(1, layer.biases.cols,
            "OptimizerAdagrad::update_params: dbiases must match biases shape");

        const size_t w_rows = layer.weights.rows;
        const size_t w_cols = layer.weights.cols;
        const size_t b_cols = layer.biases.cols;

        if (layer.weight_cache.rows != w_rows || layer.weight_cache.cols != w_cols) {
            layer.weight_cache.assign(w_rows, w_cols, 0.0);
        }
        if (layer.bias_cache.rows != 1 || layer.bias_cache.cols != b_cols) {
            layer.bias_cache.assign(1, b_cols, 0.0);
        }

        const double minus_learning_rate = -current_learning_rate;

        for (size_t i = 0; i < w_rows; ++i) {
            for (size_t j = 0; j < w_cols; ++j) {
                const double g = layer.dweights(i, j);
                layer.weight_cache(i, j) += g * g;
                layer.weights(i, j) += minus_learning_rate * g / (sqrt(layer.weight_cache(i, j)) + epsilon);
            }
        }

        for (size_t j = 0; j < b_cols; ++j) {
            const double g = layer.dbiases(0, j);
            layer.bias_cache(0, j) += g * g;
            layer.biases(0, j) += minus_learning_rate * g / (sqrt(layer.bias_cache(0, j)) + epsilon);
        }
    }
};

// RMSprop optimizer
class OptimizerRMSprop : public Optimizer
{
public:
    double epsilon;
    double rho;

    OptimizerRMSprop(double learning_rate = 0.001, double decay = 0.0, double epsilon = 1e-7, double rho = 0.9)
        : Optimizer(learning_rate, decay),
          epsilon(epsilon),
          rho(rho)
    {
        if (epsilon <= 0.0) {
            throw runtime_error("OptimizerRMSprop: epsilon must be positive");
        }
        if (rho <= 0.0 || rho >= 1.0) {
            throw runtime_error("OptimizerRMSprop: rho must be in (0, 1)");
        }
    }

    void update_params(LayerDense& layer) override
    {
        layer.weights.require_non_empty("OptimizerRMSprop::update_params: layer.weights must be non-empty");
        layer.biases.require_non_empty("OptimizerRMSprop::update_params: layer.biases must be non-empty");

        layer.biases.require_shape(1, layer.weights.cols,
            "OptimizerRMSprop::update_params: biases must have shape (1, n_neurons)");

        layer.dweights.require_shape(layer.weights.rows, layer.weights.cols,
            "OptimizerRMSprop::update_params: dweights must match weights shape");
        layer.dbiases.require_shape(1, layer.biases.cols,
            "OptimizerRMSprop::update_params: dbiases must match biases shape");

        const size_t w_rows = layer.weights.rows;
        const size_t w_cols = layer.weights.cols;
        const size_t b_cols = layer.biases.cols;

        if (layer.weight_cache.rows != w_rows || layer.weight_cache.cols != w_cols) {
            layer.weight_cache.assign(w_rows, w_cols, 0.0);
        }
        if (layer.bias_cache.rows != 1 || layer.bias_cache.cols != b_cols) {
            layer.bias_cache.assign(1, b_cols, 0.0);
        }

        const double minus_learning_rate = -current_learning_rate;
        const double one_minus_rho = 1.0 - rho;

        for (size_t i = 0; i < w_rows; ++i) {
            for (size_t j = 0; j < w_cols; ++j) {
                const double g = layer.dweights(i, j);
                layer.weight_cache(i, j) = rho * layer.weight_cache(i, j) + one_minus_rho * g * g;
                layer.weights(i, j) += minus_learning_rate * g / (sqrt(layer.weight_cache(i, j)) + epsilon);
            }
        }

        for (size_t j = 0; j < b_cols; ++j) {
            const double g = layer.dbiases(0, j);
            layer.bias_cache(0, j) = rho * layer.bias_cache(0, j) + one_minus_rho * g * g;
            layer.biases(0, j) += minus_learning_rate * g / (sqrt(layer.bias_cache(0, j)) + epsilon);
        }
    }
};

// Adam optimizer
class OptimizerAdam : public Optimizer
{
public:
    double epsilon;
    double beta1;
    double beta2;

    OptimizerAdam(double learning_rate = 0.001, double decay = 0.0, double epsilon = 1e-7,
                  double beta1 = 0.9, double beta2 = 0.999)
        : Optimizer(learning_rate, decay),
          epsilon(epsilon),
          beta1(beta1),
          beta2(beta2),
          beta1_power(1.0),
          beta2_power(1.0)
    {
        if (epsilon <= 0.0) {
            throw runtime_error("OptimizerAdam: epsilon must be positive");
        }
        if (beta1 <= 0.0 || beta1 >= 1.0) {
            throw runtime_error("OptimizerAdam: beta1 must be in (0, 1)");
        }
        if (beta2 <= 0.0 || beta2 >= 1.0) {
            throw runtime_error("OptimizerAdam: beta2 must be in (0, 1)");
        }
    }

    void pre_update_params() override
    {
        Optimizer::pre_update_params();

        beta1_power *= beta1;
        beta2_power *= beta2;
    }

    void update_params(LayerDense& layer) override
    {
        layer.weights.require_non_empty("OptimizerAdam::update_params: layer.weights must be non-empty");
        layer.biases.require_non_empty("OptimizerAdam::update_params: layer.biases must be non-empty");

        layer.biases.require_shape(1, layer.weights.cols,
            "OptimizerAdam::update_params: biases must have shape (1, n_neurons)");

        layer.dweights.require_shape(layer.weights.rows, layer.weights.cols,
            "OptimizerAdam::update_params: dweights must match weights shape");
        layer.dbiases.require_shape(1, layer.biases.cols,
            "OptimizerAdam::update_params: dbiases must match biases shape");

        const size_t w_rows = layer.weights.rows;
        const size_t w_cols = layer.weights.cols;
        const size_t b_cols = layer.biases.cols;

        if (layer.weight_momentums.rows != w_rows || layer.weight_momentums.cols != w_cols) {
            layer.weight_momentums.assign(w_rows, w_cols, 0.0);
        }
        if (layer.weight_cache.rows != w_rows || layer.weight_cache.cols != w_cols) {
            layer.weight_cache.assign(w_rows, w_cols, 0.0);
        }
        if (layer.bias_momentums.rows != 1 || layer.bias_momentums.cols != b_cols) {
            layer.bias_momentums.assign(1, b_cols, 0.0);
        }
        if (layer.bias_cache.rows != 1 || layer.bias_cache.cols != b_cols) {
            layer.bias_cache.assign(1, b_cols, 0.0);
        }

        const double minus_learning_rate = -current_learning_rate;

        const double one_minus_beta1 = 1.0 - beta1;
        const double one_minus_beta2 = 1.0 - beta2;

        const double correction_applied_to_momentum = 1.0 - beta1_power;
        const double correction_applied_to_cache = 1.0 - beta2_power;

        if (correction_applied_to_momentum <= 0.0 || correction_applied_to_cache <= 0.0) {
            throw runtime_error(
                "OptimizerAdam::update_params: numerical issue in bias correction (pre_update_params not called?)");
        }

        for (size_t i = 0; i < w_rows; ++i) {
            for (size_t j = 0; j < w_cols; ++j) {
                const double g = layer.dweights(i, j);

                layer.weight_momentums(i, j) = beta1 * layer.weight_momentums(i, j) + one_minus_beta1 * g;
                layer.weight_cache(i, j) = beta2 * layer.weight_cache(i, j) + one_minus_beta2 * g * g;

                const double weight_momentum_corrected = layer.weight_momentums(i, j) / correction_applied_to_momentum;
                const double weight_cache_corrected = layer.weight_cache(i, j) / correction_applied_to_cache;

                layer.weights(i, j) += minus_learning_rate * weight_momentum_corrected / (sqrt(weight_cache_corrected) + epsilon);
            }
        }

        for (size_t j = 0; j < b_cols; ++j) {
            const double g = layer.dbiases(0, j);

            layer.bias_momentums(0, j) = beta1 * layer.bias_momentums(0, j) + one_minus_beta1 * g;
            layer.bias_cache(0, j) = beta2 * layer.bias_cache(0, j) + one_minus_beta2 * g * g;

            const double bias_momentum_corrected = layer.bias_momentums(0, j) / correction_applied_to_momentum;
            const double bias_cache_corrected = layer.bias_cache(0, j) / correction_applied_to_cache;

            layer.biases(0, j) += minus_learning_rate * bias_momentum_corrected / (sqrt(bias_cache_corrected) + epsilon);
        }
    }

private:
    double beta1_power;
    double beta2_power;
};

// accuracy
class Accuracy
{
public:
    size_t accumulated_sum = 0;
    size_t accumulated_count = 0;

    virtual ~Accuracy() = default;
    virtual void init(const Matrix&) {}
    virtual void reset() {}

    double calculate(const Matrix& y_pred, const Matrix& y_true) {
        y_pred.require_non_empty("Accuracy::calculate: y_pred must be non-empty");
        y_true.require_non_empty("Accuracy::calculate: y_true must be non-empty");

        multiplication_overflow_check(y_pred.rows, y_pred.cols, "Accuracy::calculate: total overflow");
        const size_t correct = compare(y_pred, y_true);
        const size_t total = y_pred.cols * y_pred.rows;
        
        accumulated_sum += correct;
        accumulated_count += total;

        return static_cast<double>(correct) / static_cast<double>(total);
    }

    double calculate_accumulated() const
    {
        if (accumulated_count == 0) {
            throw runtime_error("Accuracy::calculate_accumulated: accumulated_count must be > 0");
        }
        return accumulated_sum / static_cast<double>(accumulated_count);
    }

    void new_pass()
    {
        accumulated_sum = 0.0;
        accumulated_count = 0;
    }

protected:
    virtual size_t compare(const Matrix&, const Matrix&) = 0;
};

class AccuracyCategorical : public Accuracy
{
public:
    explicit AccuracyCategorical(bool binary = false)
        : binary(binary)
    {
    }

    size_t compare(const Matrix& y_pred, const Matrix& y_true) override
    {
        if(!binary) y_pred.require_cols(1, "AccuracyCategorical::compare: categorical y_pred must have shape (N,1)");

        const size_t pred_rows = y_pred.rows;
        const size_t pred_cols = y_pred.cols;

        Matrix ground_truth;

        if (binary) {
            y_true.require_shape(pred_rows, pred_cols,
                "AccuracyCategorical::compare: for binary accuracy y_true must match y_pred shape");

            ground_truth = y_true;
        } else {
            if (y_true.is_col_vector() && y_true.rows == pred_rows) {
                ground_truth = y_true;
            } else if (y_true.is_row_vector() && y_true.cols == pred_rows) {
                ground_truth = y_true.transpose();
            } else if (y_true.rows == pred_rows && y_true.cols >= 2) {
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

private:
    bool binary;
};

class AccuracyRegression : public Accuracy
{
public:
    explicit AccuracyRegression(double precision_divisor = 250.0)
        : precision_divisor(precision_divisor), precision(0.0), initialized(false)
    {
        if (precision_divisor <= 0.0) {
            throw runtime_error("AccuracyRegression: precision_divisor must be positive");
        }
    }

    // calculate resuable precision with standard deviation
    void init(const Matrix& y_true) override
    {
        y_true.require_non_empty("AccuracyRegression::init: y_true must be non-empty");

        const size_t samples = y_true.rows;
        const size_t outputs = y_true.cols;

        multiplication_overflow_check(samples, outputs, "AccuracyRegression::init: n overflow");

        const size_t n = samples * outputs;

        // calculate mean
        double mean = 0.0;
        for (size_t i = 0; i < samples; ++i) {
            for (size_t j = 0; j < outputs; ++j) {
                mean += y_true(i, j);
            }
        }
        mean /= static_cast<double>(n);

        // calculate variance using the previously calculated mean
        double var = 0.0;
        for (size_t i = 0; i < samples; ++i) {
            for (size_t j = 0; j < outputs; ++j) {
                const double d = y_true(i, j) - mean;
                var += d * d;
            }
        }
        var /= static_cast<double>(n);

        // clamp to prevent floating error
        if (var < 0.0) var = 0.0;

        // calculate standard deviation (the square root of the variance)
        const double standard_deviation = sqrt(var);

        // calculate precision
        precision = max(standard_deviation / precision_divisor, 1e-7);

        initialized = true;
    }

    void reset() override { initialized = false; }

    size_t compare(const Matrix& y_pred, const Matrix& y_true) override
    {
        y_pred.require_shape(y_true.rows, y_true.cols,
            "AccuracyRegression::compare: y_pred and y_true must have the same shape");

        if (!initialized) init(y_true);
        
        const size_t samples = y_true.rows;
        const size_t outputs = y_true.cols;

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

private:
    double precision_divisor;
    double precision;
    bool initialized;
};

// Model wrapper
class Model
{
public:
    LayerInput input_layer;
    std::vector<Layer*> layers;
    std::vector<LayerDense*> trainable_layers;

    Loss* loss = nullptr;
    bool loss_is_cce = false;
    Accuracy* accuracy = nullptr;
    Optimizer* optimizer = nullptr;
    bool finalized = false;

    Matrix output;
    Matrix last_predictions;

    void add(Layer& layer)
    {
        layers.push_back(&layer);
        if (auto* dense = dynamic_cast<LayerDense*>(&layer)) {
            // currently trainable layers check only for dense layers
            // if any other layer types (outside of dropout) are added later this method would need to be updated
            trainable_layers.push_back(dense);
        }
    }

    void set(Loss& loss_obj, Accuracy& accuracy_obj, Optimizer* optimizer_obj = nullptr)
    {
        loss = &loss_obj;
        accuracy = &accuracy_obj;
        optimizer = optimizer_obj;        

        loss_is_cce = (dynamic_cast<LossCategoricalCrossEntropy*>(loss) != nullptr);
    }

    void set(Loss& loss_obj, Accuracy& accuracy_obj, Optimizer& optimizer_obj)
    {
        set(loss_obj, accuracy_obj, &optimizer_obj);
    }

    void finalize()
    {
        if (!loss || !accuracy) {
            throw runtime_error("Model::finalize: loss and accuracy must be set");
        }
        if (layers.empty()) {
            throw runtime_error("Model::finalize: no layers added");
        }

        loss->remember_trainable_layers(trainable_layers);
        finalized = true;
    }

    void train(const Matrix& X, const Matrix& y, size_t epochs = 1, size_t batch_size = 0,
               size_t print_every = 100, const Matrix* X_val = nullptr, const Matrix* y_val = nullptr)
    {
        if (!loss || !accuracy) {
            throw runtime_error("Model::train: loss and accuracy must be set");
        }
        if (!optimizer) {
            throw runtime_error("Model::train: optimizer must be set");
        }
        if (layers.empty()) {
            throw runtime_error("Model::train: no layers added");
        }

        X.require_non_empty("Model::train: X must be non-empty");
        y.require_non_empty("Model::train: y must be non-empty");

        if (!finalized) finalize();

        accuracy->init(y);

        size_t steps = 1;
        if (batch_size != 0) {
            steps = X.rows / batch_size;
            if (steps * batch_size < X.rows) ++steps;
        }

        for (size_t epoch = 1; epoch <= epochs; ++epoch) {
            cout << "epoch: " << epoch << '\n';

            loss->new_pass();
            accuracy->new_pass();

            for (size_t step = 0; step < steps; ++step) {
                Matrix batch_X;
                Matrix batch_y;

                if (batch_size == 0) {
                    batch_X = X;
                    batch_y = y;
                } else {
                    const size_t start = step * batch_size;
                    const size_t end = min(start + batch_size, X.rows);
                    batch_X = X.slice_rows(start, end);
                    batch_y = y.slice_rows(start, end);
                }

                forward_pass(batch_X, true);

                // calculating loss and accuracy
                double reg_loss = 0.0;
                const double data_loss = loss->calculate(output, batch_y, reg_loss);
                const double total_loss = data_loss + reg_loss;

                last_predictions = predictions_from_output(output);
                const double acc = accuracy->calculate(last_predictions, batch_y);

                // backward pass
                const bool use_combined = output_is_softmax() && loss_is_cce;
                const Matrix* dvalues;
                auto iter = layers.rbegin();

                if (use_combined) {
                    combined_softmax_ce.backward(output, batch_y);
                    dvalues = &combined_softmax_ce.dinputs;

                    Layer* last_layer = *iter;
                    auto* last_dense = dynamic_cast<LayerDense*>(last_layer);
                    if (!last_dense) {
                        throw runtime_error("Model::train: combined softmax loss requires final layer to be LayerDense");
                    }

                    last_dense->backward(*dvalues, false);
                    dvalues = &last_dense->dinputs;
                    ++iter;
                } else {
                    loss->backward(output, batch_y);
                    dvalues = &loss->dinputs;
                }

                for (; iter != layers.rend(); ++iter) {
                    (*iter)->backward(*dvalues);
                    dvalues = &(*iter)->dinputs;
                }

                // using optimizers
                optimizer->pre_update_params();
                for (LayerDense* dense : trainable_layers) {
                    optimizer->update_params(*dense);
                }
                optimizer->post_update_params();

                // printing
                if (print_every != 0 && ((step % print_every) == 0 || step == steps - 1)) {
                    cout << "  step: " << step
                         << ", accuracy: " << acc
                         << ", loss: " << total_loss
                         << " (data loss: " << data_loss
                         << ", regularization loss: " << reg_loss
                         << ")"
                         << ", learning rate: " << optimizer->current_learning_rate
                         << '\n';
                }
            }

            double epoch_reg_loss = 0.0;
            const double epoch_data_loss = loss->calculate_accumulated(epoch_reg_loss);
            const double epoch_loss = epoch_data_loss + epoch_reg_loss;
            const double epoch_accuracy = accuracy->calculate_accumulated();
            cout << "training - accuracy: " << epoch_accuracy
                 << ", loss: " << epoch_loss
                 << " (data loss: " << epoch_data_loss
                 << ", regularization loss: " << epoch_reg_loss
                 << ")"
                 << ", learning rate: " << optimizer->current_learning_rate
                 << '\n';

            if (X_val && y_val) evaluate(*X_val, *y_val, batch_size, false);
        }
    }

    void  evaluate(const Matrix& X, const Matrix& y, size_t batch_size = 0, bool reinit = true)
    {
        if (!loss || !accuracy) {
            throw runtime_error("Model::evaluate: loss, and accuracy must be set");
        }
        if (layers.empty()) {
            throw runtime_error("Model::evaluate: no layers added");
        }

        X.require_non_empty("Model::evaluate: X must be non-empty");
        y.require_non_empty("Model::evaluate: y must be non-empty");

        if (!finalized) finalize();

        if (reinit) accuracy->init(y);
        loss->new_pass();
        accuracy->new_pass();

        size_t steps = 1;
        if (batch_size != 0) {
            steps = X.rows / batch_size;
            if (steps * batch_size < X.rows) ++steps;
        }

        for (size_t step = 0; step < steps; ++step) {
            Matrix batch_X;
            Matrix batch_y;

            if (batch_size == 0) {
                batch_X = X;
                batch_y = y;
            } else {
                const size_t start = step * batch_size;
                const size_t end = min(start + batch_size, X.rows);
                batch_X = X.slice_rows(start, end);
                batch_y = y.slice_rows(start, end);
            }

            forward_pass(batch_X, false);
            loss->calculate(output, batch_y);
            last_predictions = predictions_from_output(output);
            accuracy->calculate(last_predictions, batch_y);
        }

        const double val_loss = loss->calculate_accumulated();
        const double val_acc = accuracy->calculate_accumulated();

        cout << "validation - accuracy: " << val_acc
             << ", loss: " << val_loss
             << '\n';
    }

private:
    ActivationSoftmaxLossCategoricalCrossEntropy combined_softmax_ce;

    void forward_pass(const Matrix& X, const bool training)
    {
        input_layer.forward(X);

        const Matrix* current = &input_layer.output;
        for (Layer* layer : layers) {
            layer->forward(*current, training);
            current = &layer->output;
        }

        output = *current;
    }

    bool output_is_softmax() const
    {
        if (layers.empty()) return false;
        const Activation* activation = layers.back()->activation;
        return activation && (dynamic_cast<const ActivationSoftmax*>(activation) != nullptr);
    }

    Matrix predictions_from_output(const Matrix& outputs) const
    {
        if (layers.empty()) return outputs;
        const Activation* activation = layers.back()->activation;
        if (!activation) return outputs;
        return activation->predictions(outputs);
    }
};

#ifndef NNFS_NO_MAIN
int main()
{
    Matrix X;
    Matrix y;
    Matrix X_test;
    Matrix y_test;

    fashion_mnist_create(X, y, X_test, y_test);

    ActivationReLU activation1;
    LayerDense dense1(X.cols, 128, activation1, 0.0, 5e-4, 0.0, 5e-4);

    ActivationReLU activation2;
    LayerDense dense2(128, 128, activation2, 0.0, 5e-4, 0.0, 5e-4);

    ActivationSoftmax activation3;
    LayerDense dense3(128, 10, activation3);

    LossCategoricalCrossEntropy loss;
    AccuracyCategorical accuracy;
    OptimizerAdam optimizer(1e-3);

    Model model;
    model.add(dense1);
    model.add(dense2);
    model.add(dense3);
    model.set(loss, accuracy, optimizer);
    model.finalize();

    model.train(X, y, 2, 128, 100, &X_test, &y_test);

    cout << "extra validation after shuffling to check if the model assumes sorted labels:\n";
    X_test.shuffle_rows_with(y_test);
    model.evaluate(X_test, y_test, 128);

    return 0;
}
#endif
