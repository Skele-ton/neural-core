#include <iostream>
#include <iomanip>
#include <vector>
#include <stdexcept>
#include <random>
#include <cmath>
#include <limits>
#include <cstdio>
#include <string>
#include <fstream>
#include <type_traits>
#include <functional>
#include <algorithm>
#include <list>

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
using std::min;
using std::max;
using std::round;
using std::sqrt;
using std::pow;
using std::abs;
using std::sin;
using std::cos;
using std::acos;
using std::exp;
using std::log;
using std::isfinite;
using std::is_same_v;

static inline bool is_whole_number(double v, double epsilon = 1e-7)
{
    return abs(v - round(v)) <= epsilon;
}

static inline void multiplication_overflow_check(const size_t a, const size_t b, const char* error_msg) {
    if (a != 0 && b > numeric_limits<size_t>::max() / a) {
        throw runtime_error(error_msg);
    }
}

// TODO: Make rng implementation thread-safe

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

    // === helper methods for error checks
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
    // ===================================

    void scale_by_scalar(size_t samples)
    {
        if (samples == 0) {
            throw runtime_error("Matrix::scale_by_scalar: samples must be bigger than 0");
        }

        const double inv = 1.0 / static_cast<double>(samples);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                (*this)(i, j) *= inv;
            }
        }
    }

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

    bool is_empty() const { return rows == 0 || cols == 0; }

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

        Matrix result(1, rows);

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

            result(0, i) = static_cast<double>(biggest_j);
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

    Matrix from_one_dimensional_as_column() const
    {
        if (is_empty()) return Matrix();

        if(rows != 1) {
            throw runtime_error("Matrix::from_one_dimensional_as_column: the matrix has more than one row");
        }

        Matrix result(cols, 1);
        for (size_t i = 0; i < cols; ++i) {
            result(i, 0) = static_cast<double>(data[i]);
        }

        return result;
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
};

// training data
void generate_spiral_data(size_t samples_per_class, size_t classes, Matrix& X_out, Matrix& y_out)
{
    if (samples_per_class <= 1 || classes == 0) {
        throw runtime_error("generate_spiral_data: invalid arguments");
    }

    multiplication_overflow_check(classes, samples_per_class, "generate_spiral_data: total_samples overflow");

    size_t total_samples = samples_per_class * classes;
    X_out.assign(total_samples, 2);
    y_out.assign(1, total_samples);

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
            y_out(0, idx) = static_cast<double>(class_idx);
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
    y_out.assign(1, total_samples);

    for (size_t class_idx = 0; class_idx < classes; ++class_idx) {
        size_t class_offset = class_idx * samples_per_class;
        double center_x = static_cast<double>(class_idx) / static_cast<double>(classes);

        for (size_t i = 0; i < samples_per_class; ++i) {
            size_t idx = class_offset + i;

            double x = center_x + random_gaussian() * 0.1;
            double y = random_uniform();

            X_out(idx, 0) = x;
            X_out(idx, 1) = y;

            y_out(0, idx) = static_cast<double>(class_idx);
        }
    }
}

void generate_sine_data(size_t samples, Matrix& X_out, Matrix& y_out)
{
    if (samples <= 1) {
        throw runtime_error("generate_sine_data: invalid arguments");
    }

    X_out.assign(samples, 1);
    y_out.assign(1, samples);

    const double pi = acos(-1.0);

    for (size_t i = 0; i < samples; ++i) {
        const double x = static_cast<double>(i) / static_cast<double>(samples - 1);
        const double y = sin(2 * pi * x);

        const size_t idx = i;
        X_out(idx, 0) = x;
        y_out(0, idx) = y;
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
        labels.require_shape(1, num_points,
            "plot_scatter_svg: labels must be shape (1,points.rows)");
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
    if(dist_x < eps || dist_y < eps) {
        throw runtime_error("plot_scatter_svg: x and y must have non-zero range");
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

        const size_t class_id = has_labels ? labels.as_size_t(0, point_index) : 0;
        const size_t color_id = class_id  % num_of_colors;
        const char* class_color = colors[color_id];

        out << "<circle cx=\"" << map_x(point_x) << "\" cy=\"" << map_y(point_y)
            << "\" r=\"" << r << "\" fill=\"" << class_color
            << "\" fill-opacity=\"0.85\"/>\n";
    }
    out << "</g>\n";

    out << "</svg>\n";
}

// Dense layer with weights/biases and cached inputs/output
class LayerDense
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

    Matrix output;
    Matrix inputs;

    Matrix dweights;
    Matrix dbiases;
    Matrix dinputs;

    LayerDense(size_t n_inputs, size_t n_neurons,
               double weight_regularizer_l1 = 0.0,
               double weight_regularizer_l2 = 0.0,
               double bias_regularizer_l1 = 0.0,
               double bias_regularizer_l2 = 0.0)
        : weights(n_inputs, n_neurons),
          biases(1, n_neurons, 0.0),
          weight_regularizer_l1(weight_regularizer_l1),
          weight_regularizer_l2(weight_regularizer_l2),
          bias_regularizer_l1(bias_regularizer_l1),
          bias_regularizer_l2(bias_regularizer_l2),
          output(),
          inputs()
    {
        if (weight_regularizer_l1 < 0.0 || weight_regularizer_l2 < 0.0 ||
            bias_regularizer_l1 < 0.0 || bias_regularizer_l2 < 0.0) {
            throw runtime_error("LayerDense: regularizers must be non-negative");
        }

        if(n_inputs == 0 || n_neurons == 0) {
            throw runtime_error("LayerDense:  n_inputs and n_neurons must be > 0");
        }

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
    }
    
    // overload for dropout layer
    void forward(const Matrix& inputs_batch, bool) { forward(inputs_batch); }

    void backward(const Matrix& dvalues)
    {
        dvalues.require_non_empty("LayerDense::backward: dvalues must be non-empty");
        dvalues.require_shape(inputs.rows, weights.cols,
            "LayerDense::backward: dvalues shape mismatch");

        const Matrix inputs_T = inputs.transpose();
        dweights = Matrix::dot(inputs_T, dvalues);

        dbiases.assign(1, biases.cols, 0.0);
        for (size_t i = 0; i < dvalues.rows; ++i) {
            for (size_t j = 0; j < dvalues.cols; ++j) {
                dbiases(0, j) += dvalues(i, j);
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
        dinputs = Matrix::dot(dvalues, weights_T);
    }
};

// Dropout layer
class LayerDropout
{
public:
    double keep_rate;
    Matrix scaled_binary_mask;
    Matrix output;
    Matrix inputs;
    Matrix dinputs;

    explicit LayerDropout(double rate)
        : keep_rate(1.0 - rate)
    {
        if (keep_rate <= 0.0 || keep_rate > 1.0) {
            throw runtime_error("LayerDropout: rate must be in (0,1]");
        }
    }

    void forward(const Matrix& inputs_batch, bool training = true)
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
    }

    void backward(const Matrix& dvalues)
    {
        dvalues.require_non_empty("LayerDropout::backward: dvalues must be non-empty");
        dvalues.require_shape(scaled_binary_mask.rows, scaled_binary_mask.cols,
            "LayerDropout::backward: dvalues shape mismatch");

        dinputs.assign(dvalues.rows, dvalues.cols);
        for (size_t i = 0; i < dvalues.rows; ++i) {
            for (size_t j = 0; j < dvalues.cols; ++j) {
                dinputs(i, j) = dvalues(i, j) * scaled_binary_mask(i, j);
            }
        }
    }
};

// Input "layer" to provide consistent interface for model
class LayerInput
{
public:
    Matrix input;
    Matrix output;

    void forward(const Matrix& inputs_batch)
    {
        inputs_batch.require_non_empty("LayerInput::forward: inputs must be non-empty");
        input = inputs_batch;
        output = inputs_batch;
    }

    // overload for dropout layer
    void forward(const Matrix& inputs_batch, bool) { forward(inputs_batch); }
};

// activations
class ActivationReLU
{
public:
    Matrix inputs;
    Matrix output;
    Matrix dinputs;

    void forward(const Matrix& inputs_batch)
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

    // overload for dropout layer
    void forward(const Matrix& inputs_batch, bool) { forward(inputs_batch); }

    void backward(const Matrix& dvalues)
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

    Matrix predictions(const Matrix& outputs) const { return outputs; }
};

// Softmax activation
class ActivationSoftmax
{
public:
    Matrix inputs;
    Matrix output;
    Matrix dinputs;

    void forward(const Matrix& inputs_batch)
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

    // overload for dropout layer
    void forward(const Matrix& inputs_batch, bool) { forward(inputs_batch); }

    void backward(const Matrix& dvalues)
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

    Matrix predictions(const Matrix& outputs) const
    {
        outputs.require_non_empty("ActivationSoftmax::predictions: outputs must be non-empty");
        return outputs.argmax();
    }
};

// Sigmoid activation
class ActivationSigmoid
{
public:
    Matrix inputs;
    Matrix output;
    Matrix dinputs;

    void forward(const Matrix& inputs_batch)
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

    // overload for dropout layer
    void forward(const Matrix& inputs_batch, bool) { forward(inputs_batch); }

    void backward(const Matrix& dvalues)
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

    Matrix predictions(const Matrix& outputs) const
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
class ActivationLinear
{
public:
    Matrix inputs;
    Matrix output;
    Matrix dinputs;

    void forward(const Matrix& inputs_batch)
    {
        inputs_batch.require_non_empty("ActivationLinear::forward: inputs must be non-empty");

        inputs = inputs_batch;
        output = inputs_batch;
    }

    // overload for dropout layer
    void forward(const Matrix& inputs_batch, bool) { forward(inputs_batch); }

    void backward(const Matrix& dvalues)
    {
        dvalues.require_non_empty("ActivationLinear::backward: dvalues must be non-empty");
        dvalues.require_shape(inputs.rows, inputs.cols,
            "ActivationLinear::backward: dvalues shape mismatch");
        
        dinputs = dvalues;
    }

    Matrix predictions(const Matrix& outputs) const { return outputs; }
};

// loss functions
class Loss
{
public:
    Matrix dinputs;

    virtual ~Loss() = default;

    double calculate(const Matrix& output, const Matrix& y_true) const
    {
        output.require_non_empty("Loss::calculate: output must be non-empty");
        y_true.require_non_empty("Loss::calculate: y_true must be non-empty");

        Matrix sample_losses = forward(output, y_true);

        sample_losses.require_shape(1, output.rows,
            "Loss::calculate: per-sample losses must be of shape (1,output.rows) after forward");

        return sample_losses.scalar_mean();
    }

    static double regularization_loss(const LayerDense& layer)
    {
        if (layer.weight_regularizer_l1 < 0.0 || layer.weight_regularizer_l2 < 0.0 ||
            layer.bias_regularizer_l1 < 0.0 || layer.bias_regularizer_l2 < 0.0) {
            throw runtime_error("Loss::regularization_loss: regularizer coefficients must be non-negative");
        }

        double regularization = 0.0;

        const bool has_w_l1 = layer.weight_regularizer_l1 != 0.0;
        const bool has_w_l2 = layer.weight_regularizer_l2 != 0.0;

        if(has_w_l1 || has_w_l2) {
            layer.weights.require_non_empty("Loss::regularization_loss: weights must be non-empty");

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
            layer.biases.require_non_empty("Loss::regularization_loss: biases must be non-empty");
            layer.weights.require_non_empty("Loss::regularization_loss: weights must be non-empty");

            layer.biases.require_shape(1, layer.weights.cols,
                "Loss::regularization_loss: biases must have shape (1, n_neurons)");

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
        const size_t samples = y_pred.rows;
        const size_t classes = y_pred.cols;

        Matrix sample_losses(1, samples, 0.0);

        Matrix y_true_sparse;

        if(y_true.rows == 1) {
            y_true.require_cols(samples, "LossCategoricalCrossEntropy::forward: sparse y_true must have shape (1, N)");

            y_true_sparse = y_true;
        } else {
            y_true.require_shape(samples, classes,
                "LossCategoricalCrossEntropy::forward: one-hot y_true must match y_pred shape");

            y_true_sparse = y_true.argmax();

            y_true_sparse.require_shape(1, samples,
                "LossCategoricalCrossEntropy::forward: argmax(y_true) must return shape (1, N)");
        }

        for (size_t i = 0; i < samples; ++i) {
            const size_t class_idx = y_true_sparse.as_size_t(0, i);

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

        const size_t samples = y_pred.rows;
        const size_t classes  = y_pred.cols;

        dinputs.assign(samples, classes, 0.0);

        Matrix y_true_sparse;

        if(y_true.rows == 1) {
            y_true.require_cols(samples, "LossCategoricalCrossEntropy::backward: sparse y_true must have shape (1, N)");

            y_true_sparse = y_true;
        } else {
            y_true.require_shape(samples, classes,
                "LossCategoricalCrossEntropy::backward: one-hot y_true must match y_pred shape");

            y_true_sparse = y_true.argmax();

            y_true_sparse.require_shape(1, samples,
                "LossCategoricalCrossEntropy::backward: argmax(y_true) must return shape (1, N)");
        }

        for (size_t i = 0; i < samples; ++i) {
            const size_t class_idx = y_true_sparse.as_size_t(0, i);
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
            "LossBinaryCrossentropy::backward: shapes of y_pred and y_true must match");

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
            "LossMeanSquaredError::backward: shapes of y_pred and y_true must match");

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
            "LossMeanAbsoluteError::backward: shapes of y_pred and y_true must match");

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

        const size_t samples = y_pred.rows;
        const size_t classes = y_pred.cols;

        Matrix y_true_sparse;

        if(y_true.rows == 1) {
            y_true.require_cols(samples, "ActivationSoftmaxLossCategoricalCrossEntropy::backward: sparse y_true must have shape (1, N)");

            y_true_sparse = y_true;
        } else {
            y_true.require_shape(samples, classes,
                "ActivationSoftmaxLossCategoricalCrossEntropy::backward: one-hot y_true must match y_pred shape");

            y_true_sparse = y_true.argmax();

            y_true_sparse.require_shape(1, samples,
                "ActivationSoftmaxLossCategoricalCrossEntropy::backward: argmax(y_true) must return shape (1, N)");
        }

        dinputs = y_pred;
        for (size_t i = 0; i < samples; ++i) {
            size_t class_idx = y_true_sparse.as_size_t(0, i);
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

// data generators produce wrong shape of data (N, 1) instead of (1, N) - fix!
// accuracy
class Accuracy
{
public:
    virtual ~Accuracy() = default;
    virtual void init(const Matrix&) {}
    virtual void reset() {}

    double calculate(const Matrix& y_pred, const Matrix& y_true) {
        y_pred.require_non_empty("Accuracy::calculate: y_pred must be non-empty");
        y_true.require_non_empty("Accuracy::calculate: y_true must be non-empty");

        multiplication_overflow_check(y_pred.rows, y_pred.cols, "Accuracy::calculate: total overflow");
        const size_t total = y_pred.cols * y_pred.rows;
        const size_t correct = compare(y_pred, y_true);

        return static_cast<double>(correct) / static_cast<double>(total);
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
        if(!binary) y_pred.require_rows(1, "AccuracyCategorical::compare: categorical y_pred must have shape (1, N)");

        Matrix ground_truth;

        const size_t rows = y_pred.rows;
        const size_t cols = y_pred.cols;

        if(!binary && y_true.rows != 1) { // one-hot non-binary
            y_true.require_rows(cols, "AccuracyCategorical::compare: one-hot y_true must have shape (y_pred.cols, C)");
            if (y_true.cols < 2) {
                throw runtime_error("AccuracyCategorical::compare: one-hot y_true must have at least 2 columns");
            }

            ground_truth = y_true.argmax();

            ground_truth.require_shape(1, cols,
                "AccuracyCategorical::compare: argmax(y_true) must return shape (1, N)");
        } else { // sparse non-binary and binary
            y_true.require_shape(rows, cols,
                "AccuracyCategorical::compare: y_true must match y_pred shape");

            ground_truth = y_true;
        }

        size_t correct = 0;

        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
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
    struct LayerHandle
    {
        std::function<void(const Matrix&, bool)> forward;
        std::function<void(const Matrix&)> backward;
        std::function<const Matrix&()> output;
        std::function<const Matrix&()> dinputs;
        std::function<Matrix(const Matrix&)> predictions;
        bool output_is_softmax = false;
    };

    LayerInput input_layer;
    std::vector<LayerHandle> layers;
    std::vector<LayerDense*> trainable_layers;

    Loss* loss = nullptr;
    bool loss_is_cce = false;
    Optimizer* optimizer = nullptr;
    Accuracy* accuracy = nullptr;

    Matrix output;
    Matrix last_predictions;

    template <typename LayerType>
    void add(LayerType& layer)
    {
        LayerHandle handle;

        handle.forward = [&layer](const Matrix& inputs, bool training) { layer.forward(inputs, training); };
        handle.backward = [&layer](const Matrix& dvalues) { layer.backward(dvalues); };
        handle.output = [&layer]() -> const Matrix& { return layer.output; };
        handle.dinputs = [&layer]() -> const Matrix& { return layer.dinputs; };

        if constexpr (is_same_v<LayerType, ActivationSoftmax> ||
                      is_same_v<LayerType, ActivationSigmoid> ||
                      is_same_v<LayerType, ActivationLinear> ||
                      is_same_v<LayerType, ActivationReLU>) {
            handle.predictions = [&layer](const Matrix& outputs) -> Matrix { return layer.predictions(outputs); };
        } else {
            handle.predictions = [](const Matrix& outputs) -> Matrix { return outputs; };
        }

        handle.output_is_softmax = is_same_v<LayerType, ActivationSoftmax>;

        layers.push_back(handle);

        if constexpr (is_same_v<LayerType, LayerDense>) {
            trainable_layers.push_back(&layer);
        }
    }

    void set(Loss& loss_obj, Optimizer& optimizer_obj, Accuracy& accuracy_obj)
    {
        loss = &loss_obj;
        optimizer = &optimizer_obj;
        accuracy = &accuracy_obj;

        loss_is_cce = (dynamic_cast<LossCategoricalCrossEntropy*>(loss) != nullptr);
    }

    void train(const Matrix& X, const Matrix& y, size_t epochs, size_t print_every = 100)
    {
        if (!loss || !optimizer || !accuracy) {
            throw runtime_error("Model::train: loss, optimizer, and accuracy must be set");
        }
        if (layers.empty()) {
            throw runtime_error("Model::train: no layers added");
        }

        X.require_non_empty("Model::train: X must be non-empty");
        y.require_non_empty("Model::train: y must be non-empty");

        accuracy->init(y);

        for (size_t epoch = 0; epoch <= epochs; ++epoch) {
            optimizer->pre_update_params();

            forward_pass(X, true);

            // calculating and printing loss and accuracy
            Metrics metrics = calculate_metrics(output, y);

            if (print_every != 0 && (epoch % print_every) == 0) {
                cout << "epoch: " << epoch
                        << ", accuracy: " << metrics.accuracy
                        << ", loss: " << metrics.total_loss
                        << " (data_loss: " << metrics.data_loss
                        << ", reg_loss: " << metrics.reg_loss
                        << ")"
                        << ", lr: " << optimizer->current_learning_rate
                        << '\n';
            }

            // backward pass
            const bool use_combined = layers.back().output_is_softmax && loss_is_cce;
            const Matrix* dvalues;
            auto it = layers.rbegin();

            if (use_combined) {
                combined_softmax_ce.backward(output, y);
                dvalues = &combined_softmax_ce.dinputs;

                it++;
            } else {
                loss->backward(output, y);
                dvalues = &loss->dinputs;
            }

            for (; it != layers.rend(); ++it) {
                it->backward(*dvalues);
                dvalues = &it->dinputs();
            }

            // using optimizer
            for (LayerDense* dense : trainable_layers) {
                optimizer->update_params(*dense);
            }

            optimizer->post_update_params();
        }
    }

    double evaluate(const Matrix& X, const Matrix& y, double& out_loss)
    {
        if (!loss || !accuracy) {
            throw runtime_error("Model::evaluate: loss, and accuracy must be set");
        }
        if (layers.empty()) {
            throw runtime_error("Model::evaluate: no layers added");
        }

        X.require_non_empty("Model::evaluate: X must be non-empty");
        y.require_non_empty("Model::evaluate: y must be non-empty");

        accuracy->init(y);

        forward_pass(X, false);

        Metrics metrics = calculate_metrics(output, y);

        out_loss = metrics.data_loss;

        cout << "validation, accuracy: " << metrics.accuracy
         << ", loss: " << out_loss
         << '\n';

        return metrics.accuracy;
    }

private:
    ActivationSoftmaxLossCategoricalCrossEntropy combined_softmax_ce;

    struct Metrics
    {
        double data_loss = 0.0;
        double reg_loss = 0.0;
        double total_loss = 0.0;
        double accuracy = 0.0;
    };

    void forward_pass(const Matrix& X, const bool training)
    {
        input_layer.forward(X, training);

            const Matrix* current = &input_layer.output;
            for (auto& layer : layers) {
                layer.forward(*current, training);
                current = &layer.output();
            }

        output = *current;
    }

    Metrics calculate_metrics(const Matrix& output, const Matrix& y) {
        Metrics m;

        m.data_loss = loss->calculate(output, y);

        m.reg_loss = 0.0;
        for (LayerDense* dense : trainable_layers) {
            m.reg_loss += Loss::regularization_loss(*dense);
        }

        m.total_loss = m.data_loss + m.reg_loss;

        last_predictions = layers.back().predictions(output);
        m.accuracy = accuracy->calculate(last_predictions, y);

        return m;
    }
};

#ifndef NNFS_NO_MAIN
int main()
{
    Matrix X;
    Matrix y;
    generate_sine_data(1000, X, y);

    Matrix plot_points(X.rows, 2);
    for (size_t i = 0; i < X.rows; ++i) {
        plot_points(i, 0) = X(i, 0);
        plot_points(i, 1) = y(0, i);
    }
    plot_scatter_svg("plot.svg", plot_points);

    LayerDense dense1(1, 64, 0.0, 5e-4, 0.0, 5e-4);
    ActivationReLU activation1;

    LayerDense dense2(64, 64, 0.0, 5e-4, 0.0, 5e-4);
    ActivationReLU activation2;

    LayerDense dense3(64, 1);
    ActivationLinear activation3;

    LossMeanSquaredError loss_function;
    OptimizerAdam optimizer(0.005, 1e-3);
    AccuracyRegression accuracy(75.0);

    Model model;
    model.add(dense1);
    model.add(activation1);
    model.add(dense2);
    model.add(activation2);
    model.add(dense3);
    model.add(activation3);
    model.set(loss_function, optimizer, accuracy);

    cout << fixed << setprecision(5);
    model.train(X, y.from_one_dimensional_as_column(), 10000, 100);

    Matrix X_test;
    Matrix y_test;
    generate_sine_data(100, X_test, y_test);

    double test_loss = 0.0;
    model.evaluate(X_test, y_test.from_one_dimensional_as_column(), test_loss);
    return 0;
}
#endif
