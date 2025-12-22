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

using std::cout;
using std::runtime_error;
using std::size_t;
using std::string;
using std::numeric_limits;
using std::mt19937;
using std::normal_distribution;
using std::uniform_real_distribution;
using std::to_string;
using std::fixed;
using std::setprecision;
using std::min;
using std::max;
using std::sin;
using std::cos;
using std::exp;
using std::log;
using std::isfinite;

using VecD = std::vector<double>;
using VecI = std::vector<int>;

class Matrix
{
public:
    size_t rows;
    size_t cols;
    VecD data;

    Matrix() : rows(0), cols(0), data() {}

    Matrix(size_t r, size_t c, double value = 0.0)
        : rows(r), cols(c), data(r * c, value) {}

    void assign(size_t r, size_t c, double value = 0.0)
    {
        rows = r;
        cols = c;
        data.assign(r * c, value);
    }

    bool is_empty() const
    {
        return rows == 0 || cols == 0;
    }

    double& operator()(size_t r, size_t c)
    {
        return data[r * cols + c];
    }

    double operator()(size_t r, size_t c) const
    {
        return data[r * cols + c];
    }
};

using MatD = Matrix;

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

// math
MatD matmul(const MatD& a, const MatD& b)
{
    if (a.is_empty() || b.is_empty()) {
        throw runtime_error("matmul: matrices must not be empty");
    }

    if (a.cols != b.rows) {
        throw runtime_error("matmul: incompatible shapes");
    }

    MatD result(a.rows, b.cols, 0.0);

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

MatD transpose(const MatD& m)
{
    if (m.is_empty()) {
        return MatD();
    }

    MatD result(m.cols, m.rows);

    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            result(j, i) = m(i, j);
        }
    }

    return result;
}

MatD clip_matrix(const MatD& m, double min_value, double max_value)
{
    if (min_value > max_value) {
        throw runtime_error("clip_matrix: min_value must not exceed max_value");
    }

    MatD result(m.rows, m.cols);
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            double v = m(i, j);
            if (v < min_value) {
                v = min_value;
            } else if (v > max_value) {
                v = max_value;
            }
            result(i, j) = v;
        }
    }

    return result;
}

double mean(const VecD& values)
{
    if (values.empty()) {
        throw runtime_error("mean: cannot compute mean of empty vector");
    }

    double sum = 0.0;
    for (double v : values) {
        sum += v;
    }

    return sum / static_cast<double>(values.size());
}

// training data
void generate_spiral_data(int samples_per_class, int classes, MatD& X_out, VecI& y_out)
{
    if (samples_per_class <= 1 || classes <= 0) {
        throw runtime_error("generate_spiral_data: invalid arguments");
    }

    int total_samples = samples_per_class * classes;
    X_out.assign(static_cast<size_t>(total_samples), 2);
    y_out.assign(static_cast<size_t>(total_samples), 0);

    for (int class_idx = 0; class_idx < classes; ++class_idx) {
        int class_offset = class_idx * samples_per_class;
        for (int i = 0; i < samples_per_class; ++i) {
            double r = static_cast<double>(i) / (samples_per_class - 1);
            double theta = static_cast<double>(class_idx) * 4.0 + r * 4.0;
            theta += random_gaussian() * 0.2;

            double x = r * sin(theta);
            double y = r * cos(theta);

            int idx = class_offset + i;
            size_t s_idx = static_cast<size_t>(idx);
            X_out(s_idx, 0) = x;
            X_out(s_idx, 1) = y;
            y_out[s_idx] = class_idx;
        }
    }
}

void generate_vertical_data(int samples_per_class, int classes, MatD& X_out, VecI& y_out)
{
    if (samples_per_class <= 0 || classes <= 0) {
        throw runtime_error("generate_vertical_data: invalid arguments");
    }

    int total_samples = samples_per_class * classes;
    X_out.assign(static_cast<size_t>(total_samples), 2);
    y_out.assign(static_cast<size_t>(total_samples), 0);

    for (int class_idx = 0; class_idx < classes; ++class_idx) {
        int class_offset = class_idx * samples_per_class;
        double center_x = static_cast<double>(class_idx) / static_cast<double>(classes);

        for (int i = 0; i < samples_per_class; ++i) {
            size_t s_idx = static_cast<size_t>(class_offset + i);

            double x = center_x + random_gaussian() * 0.1;
            double y = random_uniform();

            X_out(s_idx, 0) = x;
            X_out(s_idx, 1) = y;
            y_out[s_idx] = class_idx;
        }
    }
}

// plots generated data
static double clamp01(double v) { return v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v); }

static string class_color(int c)
{
    static const char* k[] = {
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    };
    const int n = static_cast<int>(sizeof(k) / sizeof(k[0]));
    return k[((c % n) + n) % n];
}

void plot_scatter_svg(const string& path, const MatD& points, const VecI& labels = {})
{
    if (points.is_empty() || points.rows == 0 || points.cols < 2) {
        throw runtime_error("plot_scatter_svg: invalid input data");
    }

    const size_t num_points = points.rows;

    double xmin = points(0, 0), xmax = points(0, 0);
    double ymin = points(0, 1), ymax = points(0, 1);

    for (size_t point_index = 1; point_index < num_points; ++point_index) {
        const double point_x = points(point_index, 0);
        const double point_y = points(point_index, 1);
        xmin = min(xmin, point_x); xmax = max(xmax, point_x);
        ymin = min(ymin, point_y); ymax = max(ymax, point_y);
    }

    const double dist_x = xmax - xmin;
    const double dist_y = ymax - ymin;
    const double pad_x = dist_x > 0.0 ? 0.08 * dist_x : 1.0;
    const double pad_y = dist_y > 0.0 ? 0.08 * dist_y : 1.0;

    xmin -= pad_x; xmax += pad_x;
    ymin -= pad_y; ymax += pad_y;

    const int svg_width = 900, svg_height = 700;
    const int margin_left = 70, margin_right = 30, margin_top = 30, margin_bottom = 70;
    const int plot_width = svg_width - margin_left - margin_right;
    const int plot_height = svg_height - margin_top - margin_bottom;

    auto map_x = [&](double raw_x) -> double {
        const double normalized_x = (raw_x - xmin) / (xmax - xmin);
        return margin_left + clamp01(normalized_x) * plot_width;
    };
    auto map_y = [&](double raw_y) -> double {
        const double normalized_y = (raw_y - ymin) / (ymax - ymin);
        return margin_top + (1.0 - clamp01(normalized_y)) * plot_height;
    };

    std::ofstream out(path);
    if (!out) {
        throw runtime_error("plot_scatter_svg: given path is invalid");
    };

    out << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << svg_width
        << "\" height=\"" << svg_height << "\" viewBox=\"0 0 " << svg_width << " " << svg_height << "\">\n";

    out << "<rect x=\"0\" y=\"0\" width=\"" << svg_width << "\" height=\"" << svg_height
        << "\" fill=\"white\"/>\n";

    out << "<rect x=\"" << margin_left << "\" y=\"" << margin_top << "\" width=\"" << plot_width << "\" height=\"" << plot_height
        << "\" fill=\"none\" stroke=\"#222\" stroke-width=\"1\"/>\n";

    const int ticks = 10;

    out << "<g stroke=\"#ddd\" stroke-width=\"1\">\n";
    for (int i = 1; i < ticks; ++i) {
        const double grid_x = margin_left + (plot_width * (double)i / ticks);
        const double grid_y = margin_top + (plot_height * (double)i / ticks);
        out << "<line x1=\"" << grid_x << "\" y1=\"" << margin_top << "\" x2=\"" << grid_x << "\" y2=\"" << (margin_top + plot_height) << "\"/>\n";
        out << "<line x1=\"" << margin_left << "\" y1=\"" << grid_y << "\" x2=\"" << (margin_left + plot_width) << "\" y2=\"" << grid_y << "\"/>\n";
    }
    out << "</g>\n";

    out << "<g fill=\"#222\" font-family=\"Arial\" font-size=\"12\">\n";
    for (int i = 0; i <= ticks; ++i) {
        const double tick_value_x = xmin + (xmax - xmin) * (double)i / ticks;
        const double tick_value_y = ymin + (ymax - ymin) * (double)i / ticks;

        const double tick_pos_x = margin_left + (plot_width * (double)i / ticks);
        const double tick_pos_y = margin_top + plot_height - (plot_height * (double)i / ticks);

        out << std::fixed << std::setprecision(3);
        out << "<text x=\"" << tick_pos_x << "\" y=\"" << (margin_top + plot_height + 22)
            << "\" text-anchor=\"middle\">" << tick_value_x << "</text>\n";
        out << "<text x=\"" << (margin_left - 10) << "\" y=\"" << (tick_pos_y + 4)
            << "\" text-anchor=\"end\">" << tick_value_y << "</text>\n";
    }
    out << "</g>\n";

    out << "<g>\n";
    const double r = 2.4;
    const bool has_labels = (labels.size() == num_points);

    for (std::size_t point_index = 0; point_index < num_points; ++point_index) {
        const double point_x = points(point_index, 0);
        const double point_y = points(point_index, 1);
        const int class_id = has_labels ? labels[point_index] : 0;

        out << "<circle cx=\"" << map_x(point_x) << "\" cy=\"" << map_y(point_y)
            << "\" r=\"" << r << "\" fill=\"" << class_color(class_id)
            << "\" fill-opacity=\"0.85\"/>\n";
    }
    out << "</g>\n";

    out << "</svg>\n";
}

// neurons
MatD layer_forward_batch(const MatD& inputs, const MatD& weights, const VecD& biases)
{
    if (weights.rows != biases.size()) {
        throw runtime_error("layer_forward_batch: weights.rows must match biases.size()");
    }

    MatD weights_T = transpose(weights);
    MatD outputs = matmul(inputs, weights_T);

    for (size_t i = 0; i < outputs.rows; ++i) {
        for (size_t j = 0; j < outputs.cols; ++j) {
            outputs(i, j) += biases[j];
        }
    }

    return outputs;
}

// Dense layer with weights/biases and cached inputs/output
class LayerDense
{
public:
    MatD weights;
    VecD biases;
    MatD output;
    MatD inputs;

    LayerDense(size_t n_inputs, size_t n_neurons)
        : weights(n_neurons, n_inputs),
          biases(n_neurons, 0.0),
          output(),
          inputs()
    {
        for (size_t neuron = 0; neuron < n_neurons; ++neuron) {
            for (size_t input = 0; input < n_inputs; ++input) {
                weights(neuron, input) = 0.01 * random_gaussian();
            }
        }
    }

    void forward(const MatD& inputs_batch)
    {
        inputs = inputs_batch;
        output = layer_forward_batch(inputs, weights, biases);
    }
};

// activations
// ReLU activation (forward only)
class ActivationReLU
{
public:
    MatD inputs;
    MatD output;

    void forward(const MatD& inputs_batch)
    {
        inputs = inputs_batch;
        output.assign(inputs.rows, inputs.cols);
        for (size_t i = 0; i < inputs.rows; ++i) {
            for (size_t j = 0; j < inputs.cols; ++j) {
                double v = inputs(i, j);
                if (v > 0.0) {
                    output(i, j) = v;
                } else {
                    output(i, j) = 0.0;
                }
            }
        }
    }
};

// Softmax activation (forward only)
class ActivationSoftmax
{
public:
    MatD inputs;
    MatD output;

    void forward(const MatD& inputs_batch)
    {
        inputs = inputs_batch;
        output.assign(inputs.rows, inputs.cols);

        for (size_t i = 0; i < inputs.rows; ++i) {
            double max_val = inputs(i, 0);
            for (size_t j = 1; j < inputs.cols; ++j) {
                double v = inputs(i, j);
                if (v > max_val) {
                    max_val = v;
                }
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
};

// loss
class Loss
{
public:
    virtual ~Loss() = default;

    double calculate(const MatD& output, const VecI& y_true) const
    {
        VecD sample_losses = forward(output, y_true);
        return mean(sample_losses);
    }

    double calculate(const MatD& output, const MatD& y_true) const
    {
        VecD sample_losses = forward(output, y_true);
        return mean(sample_losses);
    }

protected: 
    virtual VecD forward(const MatD& output, const VecI& y_true) const = 0;
    virtual VecD forward(const MatD& output, const MatD& y_true) const = 0;
};

class LossCategoricalCrossEntropy : public Loss
{
public:
    // sparse labels
    VecD forward(const MatD& y_pred, const VecI& y_true) const override
    {
        if (y_pred.rows != y_true.size()) {
            throw runtime_error("LossCategoricalCrossEntropy: y_pred.rows must match y_true.size()");
        }

        MatD clipped = clip_matrix(y_pred, 1e-7, 1.0 - 1e-7);
        VecD losses(y_pred.rows, 0.0);

        for (size_t i = 0; i < y_pred.rows; ++i) {
            int class_idx = y_true[i];
            if (class_idx < 0 || static_cast<size_t>(class_idx) >= y_pred.cols) {
                throw runtime_error("LossCategoricalCrossEntropy: class index out of range");
            }

            double confidence = clipped(i, static_cast<size_t>(class_idx));
            losses[i] = -log(confidence);
        }

        return losses;
    }

    // one-hot labels path
    VecD forward(const MatD& y_pred, const MatD& y_true) const override
    {
        if (y_pred.rows != y_true.rows || y_pred.cols != y_true.cols) {
            throw runtime_error("LossCategoricalCrossEntropy: y_pred and y_true must have the same shape");
        }

        MatD clipped = clip_matrix(y_pred, 1e-7, 1.0 - 1e-7);
        VecD losses(y_pred.rows, 0.0);

        for (size_t i = 0; i < y_pred.rows; ++i) {
            double confidence = 0.0;
            for (size_t j = 0; j < y_pred.cols; ++j) {
                confidence += clipped(i, j) * y_true(i, j);
            }

            losses[i] = -log(confidence);
        }

        return losses;
    }
};

// accuracy
// for sparse integer labels
double classification_accuracy(const MatD& y_pred, const VecI& y_true)
{
    if (y_pred.rows != y_true.size()) {
        throw runtime_error("classification_accuracy: y_pred.rows must match y_true.size()");
    }
    if (y_pred.rows == 0 || y_pred.cols == 0) {
        throw runtime_error("classification_accuracy: y_pred must be non-empty");
    }

    size_t correct = 0;
    for (size_t i = 0; i < y_pred.rows; ++i) {
        size_t pred_class = 0;
        double max_pred = y_pred(i, 0);
        for (size_t j = 1; j < y_pred.cols; ++j) {
            double v = y_pred(i, j);
            if (v > max_pred) {
                max_pred = v;
                pred_class = j;
            }
        }

        if (pred_class == static_cast<size_t>(y_true[i])) {
            ++correct;
        }
    }

    return static_cast<double>(correct) / static_cast<double>(y_pred.rows);
}

// for one-hot labels (argmax on both)
double classification_accuracy(const MatD& y_pred, const MatD& y_true)
{
    if (y_pred.rows != y_true.rows || y_pred.cols != y_true.cols) {
        throw runtime_error("classification_accuracy: y_pred and y_true must have the same shape");
    }
    if (y_pred.rows == 0 || y_pred.cols == 0) {
        throw runtime_error("classification_accuracy: y_pred must be non-empty");
    }

    size_t correct = 0;
    for (size_t i = 0; i < y_pred.rows; ++i) {
        size_t pred_class = 0;
        double max_pred = y_pred(i, 0);
        for (size_t j = 1; j < y_pred.cols; ++j) {
            double v = y_pred(i, j);
            if (v > max_pred) {
                max_pred = v;
                pred_class = j;
            }
        }

        size_t true_class = 0;
        double max_true = y_true(i, 0);
        for (size_t j = 1; j < y_true.cols; ++j) {
            double v = y_true(i, j);
            if (v > max_true) {
                max_true = v;
                true_class = j;
                if (max_true == 1.0) {
                    break;
                }
            }
        }

        if (pred_class == true_class) {
            ++correct;
        }
    }

    return static_cast<double>(correct) / static_cast<double>(y_pred.rows);
}

#ifndef NNFS_NO_MAIN
int main()
{
    MatD X;
    VecI y;
    generate_vertical_data(100, 3, X, y);
    plot_scatter_svg("plot.svg", X, y);

    LayerDense dense1(2, 3);
    ActivationReLU activation1;
    LayerDense dense2(3, 3);
    ActivationSoftmax activation2;
    LossCategoricalCrossEntropy loss_function;

    double lowest_loss = numeric_limits<double>::infinity();
    MatD best_dense1_weights = dense1.weights;
    VecD best_dense1_biases = dense1.biases;
    MatD best_dense2_weights = dense2.weights;
    VecD best_dense2_biases = dense2.biases;

    const double step_size = 0.05;
    const int iterations = 10000;

    for (int iteration = 0; iteration < iterations; ++iteration) {
        for (size_t i = 0; i < dense1.weights.rows; ++i) {
            for (size_t j = 0; j < dense1.weights.cols; ++j) {
                dense1.weights(i, j) += step_size * random_gaussian();
            }
        }

        for (double& bias : dense1.biases) {
            bias += step_size * random_gaussian();
        }

        for (size_t i = 0; i < dense2.weights.rows; ++i) {
            for (size_t j = 0; j < dense2.weights.cols; ++j) {
                dense2.weights(i, j) += step_size * random_gaussian();
            }
        }

        for (double& bias : dense2.biases) {
            bias += step_size * random_gaussian();
        }

        dense1.forward(X);
        activation1.forward(dense1.output);
        dense2.forward(activation1.output);
        activation2.forward(dense2.output);

        double loss = loss_function.calculate(activation2.output, y);
        double acc = classification_accuracy(activation2.output, y);

        if (loss < lowest_loss) {
            cout << "New set of weights in iteration " << iteration
                 << ", loss: " << loss
                 << ", acc: " << acc << '\n';

            lowest_loss = loss;
            best_dense1_weights = dense1.weights;
            best_dense1_biases = dense1.biases;
            best_dense2_weights = dense2.weights;
            best_dense2_biases = dense2.biases;
        } else {
            dense1.weights = best_dense1_weights;
            dense1.biases = best_dense1_biases;
            dense2.weights = best_dense2_weights;
            dense2.biases = best_dense2_biases;
        }
    }

    return 0;
}
#endif
