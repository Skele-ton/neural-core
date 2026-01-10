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

using std::cout;
using std::ofstream;
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
using std::sqrt;
using std::pow;
using std::sin;
using std::cos;
using std::exp;
using std::log;
using std::isfinite;
using std::is_same_v;

using VecD = std::vector<double>;
using VecI = std::vector<int>;

void print_vector(const VecD& v)
{
    for (size_t i = 0; i < v.size(); ++i) {
        cout << v[i];
        if (i + 1 != v.size()) {
            cout << ' ';
        }
    }
    cout << '\n';
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

    Matrix transpose() const
    {
        if (is_empty()) {
            return Matrix();
        }

        Matrix result(cols, rows);

        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result(j, i) = (*this)(i, j);
            }
        }

        return result;
    }

    Matrix clip(double min_value, double max_value) const
    {
        if (min_value > max_value) {
            throw runtime_error("clip: min_value must not exceed max_value");
        }

        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                double v = (*this)(i, j);
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

    void scale_by_scalar(size_t samples)
    {
        if (samples == 0) {
            throw runtime_error("scale_by_scalar: samples must be bigger than 0");
        }

        const double inv = 1.0 / static_cast<double>(samples);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                (*this)(i, j) *= inv;
            }
        }
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

MatD matrix_dot(const MatD& a, const MatD& b)
{
    if (a.is_empty() || b.is_empty()) {
        throw runtime_error("matrix_dot: matrices must not be empty");
    }

    if (a.cols != b.rows) {
        throw runtime_error("matrix_dot: incompatible shapes");
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

// accuracy for sparse and one-hot labels
template <typename YTrue>
double classification_accuracy(const MatD& y_pred, const YTrue& y_true)
{
    static_assert(is_same_v<YTrue, VecI> || is_same_v<YTrue, MatD>,
                  "classification_accuracy: y_true must be VecI (sparse) or MatD (one-hot)");

    if (y_pred.rows == 0 || y_pred.cols == 0) {
        throw runtime_error("classification_accuracy: y_pred must be non-empty");
    }

    if constexpr (is_same_v<YTrue, VecI>) {
        if (y_pred.rows != y_true.size()) {
            throw runtime_error("classification_accuracy: y_pred.rows must match y_true.size()");
        }
    } else {
        if (y_pred.rows != y_true.rows || y_pred.cols != y_true.cols) {
            throw runtime_error("classification_accuracy: y_pred and y_true must have the same shape");
        }
    }

    size_t correct = 0;
    const size_t samples = y_pred.rows;
    const size_t classes = y_pred.cols;

    for (size_t i = 0; i < samples; ++i) {
        size_t pred_class = 0;
        double max_pred = y_pred(i, 0);
        for (size_t j = 1; j < classes; ++j) {
            const double v = y_pred(i, j);
            if (v > max_pred) {
                max_pred = v;
                pred_class = j;
            }
        }

        // true_class depends on label format
        size_t true_class = 0;
        if constexpr (is_same_v<YTrue, VecI>) {
            const int y_true_size_t_check = y_true[i];
            if (y_true_size_t_check < 0 || static_cast<size_t>(y_true_size_t_check) >= classes) {
                throw runtime_error("classification_accuracy: class index out of range");
            }
            true_class = static_cast<size_t>(y_true_size_t_check);
        } else {
            double max_true = y_true(i, 0);
            for (size_t j = 1; j < classes; ++j) {
                const double v = y_true(i, j);
                if (v > max_true) {
                    max_true = v;
                    true_class = j;
                    if (max_true == 1.0) break;
                }
            }
        }

        if (pred_class == true_class) ++correct;
    }

    return static_cast<double>(correct) / static_cast<double>(samples);
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

    ofstream out(path);
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

        out << fixed << setprecision(3);
        out << "<text x=\"" << tick_pos_x << "\" y=\"" << (margin_top + plot_height + 22)
            << "\" text-anchor=\"middle\">" << tick_value_x << "</text>\n";
        out << "<text x=\"" << (margin_left - 10) << "\" y=\"" << (tick_pos_y + 4)
            << "\" text-anchor=\"end\">" << tick_value_y << "</text>\n";
    }
    out << "</g>\n";

    out << "<g>\n";
    const double r = 2.4;
    const bool has_labels = (labels.size() == num_points);

    for (size_t point_index = 0; point_index < num_points; ++point_index) {
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

// Dense layer with weights/biases and cached inputs/output
class LayerDense
{
public:
    MatD weights;
    VecD biases;
    MatD weight_momentums;
    VecD bias_momentums;
    MatD weight_cache;
    VecD bias_cache;
    MatD output;
    MatD inputs;
    MatD dweights;
    VecD dbiases;
    MatD dinputs;

    LayerDense(size_t n_inputs, size_t n_neurons)
        : weights(n_inputs, n_neurons),
          biases(n_neurons, 0.0),
          output(),
          inputs()
    {
        for (size_t input = 0; input < n_inputs; ++input) {
            for (size_t neuron = 0; neuron < n_neurons; ++neuron) {
                weights(input, neuron) = 0.01 * random_gaussian();
            }
        }
    }

    void forward(const MatD& inputs_batch)
    {
        inputs = inputs_batch;
        if (weights.is_empty()) {
            throw runtime_error("LayerDense::forward: weights must be initialized");
        }
        if (inputs.cols != weights.rows) {
            throw runtime_error("LayerDense::forward: inputs.cols must match weights.rows");
        }
        if (biases.size() != weights.cols) {
            throw runtime_error("LayerDense::forward: biases.size() must match weights.cols");
        }

        output = matrix_dot(inputs, weights);
        for (size_t i = 0; i < output.rows; ++i) {
            for (size_t j = 0; j < output.cols; ++j) {
                output(i, j) += biases[j];
            }
        }
    }

    void backward(const MatD& dvalues)
    {
        if (dvalues.rows != inputs.rows || dvalues.cols != weights.cols) {
            throw runtime_error("LayerDense::backward: dvalues shape mismatch");
        }

        MatD inputs_T = inputs.transpose();
        dweights = matrix_dot(inputs_T, dvalues);

        dbiases.assign(biases.size(), 0.0);
        for (size_t i = 0; i < dvalues.rows; ++i) {
            for (size_t j = 0; j < dvalues.cols; ++j) {
                dbiases[j] += dvalues(i, j);
            }
        }

        MatD weights_T = weights.transpose();
        dinputs = matrix_dot(dvalues, weights_T);
    }
};

// activations
// ReLU activation (forward only)
class ActivationReLU
{
public:
    MatD inputs;
    MatD output;
    MatD dinputs;

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

    void backward(const MatD& dvalues)
    {
        if (dvalues.rows != inputs.rows || dvalues.cols != inputs.cols) {
            throw runtime_error("ActivationReLU::backward: dvalues shape mismatch");
        }

        dinputs = dvalues;
        for (size_t i = 0; i < inputs.rows; ++i) {
            for (size_t j = 0; j < inputs.cols; ++j) {
                if (inputs(i, j) <= 0.0) {
                    dinputs(i, j) = 0.0;
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
    MatD dinputs;

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

    void backward(const MatD& dvalues)
    {
        if (dvalues.rows != output.rows || dvalues.cols != output.cols) {
            throw runtime_error("ActivationSoftmax::backward: dvalues shape mismatch");
        }

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
    }

    virtual ~Optimizer() = default;

    void pre_update_params()
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
    }

    void update_params(LayerDense& layer) override
    {
        if (momentum == 0.0) {
            for (size_t i = 0; i < layer.weights.rows; ++i) {
                for (size_t j = 0; j < layer.weights.cols; ++j) {
                    layer.weights(i, j) += -current_learning_rate * layer.dweights(i, j);
                }
            }
            for (size_t j = 0; j < layer.biases.size(); ++j) {
                layer.biases[j] += -current_learning_rate * layer.dbiases[j];
            }
        } else {
            if (layer.weight_momentums.rows != layer.weights.rows || layer.weight_momentums.cols != layer.weights.cols) {
                layer.weight_momentums.assign(layer.weights.rows, layer.weights.cols, 0.0);
            }
            if (layer.bias_momentums.size() != layer.biases.size()) {
                layer.bias_momentums.assign(layer.biases.size(), 0.0);
            }

            for (size_t i = 0; i < layer.weights.rows; ++i) {
                for (size_t j = 0; j < layer.weights.cols; ++j) {
                    const double temp = momentum * layer.weight_momentums(i, j) -
                                        current_learning_rate * layer.dweights(i, j);
                    layer.weight_momentums(i, j) = temp;
                    layer.weights(i, j) += temp;
                }
            }

            for (size_t j = 0; j < layer.biases.size(); ++j) {
                const double temp = momentum * layer.bias_momentums[j] - current_learning_rate * layer.dbiases[j];
                layer.bias_momentums[j] = temp;
                layer.biases[j] += temp;
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
    }

    void update_params(LayerDense& layer) override
    {
        if (layer.weight_cache.rows != layer.weights.rows || layer.weight_cache.cols != layer.weights.cols) {
            layer.weight_cache.assign(layer.weights.rows, layer.weights.cols, 0.0);
        }
        if (layer.bias_cache.size() != layer.biases.size()) {
            layer.bias_cache.assign(layer.biases.size(), 0.0);
        }

        for (size_t i = 0; i < layer.weights.rows; ++i) {
            for (size_t j = 0; j < layer.weights.cols; ++j) {
                layer.weight_cache(i, j) += layer.dweights(i, j) * layer.dweights(i, j);
                layer.weights(i, j) += -current_learning_rate * layer.dweights(i, j) /
                                       (sqrt(layer.weight_cache(i, j)) + epsilon);
            }
        }

        for (size_t j = 0; j < layer.biases.size(); ++j) {
            layer.bias_cache[j] += layer.dbiases[j] * layer.dbiases[j];
            layer.biases[j] += -current_learning_rate * layer.dbiases[j] / (sqrt(layer.bias_cache[j]) + epsilon);
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
    }

    void update_params(LayerDense& layer) override
    {
        if (layer.weight_cache.rows != layer.weights.rows || layer.weight_cache.cols != layer.weights.cols) {
            layer.weight_cache.assign(layer.weights.rows, layer.weights.cols, 0.0);
        }
        if (layer.bias_cache.size() != layer.biases.size()) {
            layer.bias_cache.assign(layer.biases.size(), 0.0);
        }

        for (size_t i = 0; i < layer.weights.rows; ++i) {
            for (size_t j = 0; j < layer.weights.cols; ++j) {
                layer.weight_cache(i, j) = rho * layer.weight_cache(i, j) +
                                           (1.0 - rho) * layer.dweights(i, j) * layer.dweights(i, j);
                layer.weights(i, j) += -current_learning_rate * layer.dweights(i, j) /
                                       (sqrt(layer.weight_cache(i, j)) + epsilon);
            }
        }

        for (size_t j = 0; j < layer.biases.size(); ++j) {
            layer.bias_cache[j] = rho * layer.bias_cache[j] + (1.0 - rho) * layer.dbiases[j] * layer.dbiases[j];
            layer.biases[j] += -current_learning_rate * layer.dbiases[j] / (sqrt(layer.bias_cache[j]) + epsilon);
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
          beta2(beta2)
    {
    }

    void update_params(LayerDense& layer) override
    {
        if (layer.weight_momentums.rows != layer.weights.rows || layer.weight_momentums.cols != layer.weights.cols) {
            layer.weight_momentums.assign(layer.weights.rows, layer.weights.cols, 0.0);
        }
        if (layer.weight_cache.rows != layer.weights.rows || layer.weight_cache.cols != layer.weights.cols) {
            layer.weight_cache.assign(layer.weights.rows, layer.weights.cols, 0.0);
        }
        if (layer.bias_momentums.size() != layer.biases.size()) {
            layer.bias_momentums.assign(layer.biases.size(), 0.0);
        }
        if (layer.bias_cache.size() != layer.biases.size()) {
            layer.bias_cache.assign(layer.biases.size(), 0.0);
        }

        const double correction_applied_to_momentum = 1.0 - pow(beta1, static_cast<double>(iterations + 1));
        const double correction_applied_to_cache = 1.0 - pow(beta2, static_cast<double>(iterations + 1));

        for (size_t i = 0; i < layer.weights.rows; ++i) {
            for (size_t j = 0; j < layer.weights.cols; ++j) {
                layer.weight_momentums(i, j) = beta1 * layer.weight_momentums(i, j) +
                                               (1.0 - beta1) * layer.dweights(i, j);
                layer.weight_cache(i, j) = beta2 * layer.weight_cache(i, j) +
                                           (1.0 - beta2) * layer.dweights(i, j) * layer.dweights(i, j);

                const double weight_momentum_corrected = layer.weight_momentums(i, j) / correction_applied_to_momentum;
                const double weight_cache_corrected = layer.weight_cache(i, j) / correction_applied_to_cache;

                layer.weights(i, j) += -current_learning_rate * weight_momentum_corrected /
                                       (sqrt(weight_cache_corrected) + epsilon);
            }
        }

        for (size_t j = 0; j < layer.biases.size(); ++j) {
            layer.bias_momentums[j] = beta1 * layer.bias_momentums[j] +
                                      (1.0 - beta1) * layer.dbiases[j];
            layer.bias_cache[j] = beta2 * layer.bias_cache[j] +
                                  (1.0 - beta2) * layer.dbiases[j] * layer.dbiases[j];

            const double bias_momentum_corrected = layer.bias_momentums[j] / correction_applied_to_momentum;
            const double bias_cache_corrected = layer.bias_cache[j] / correction_applied_to_cache;

            layer.biases[j] += -current_learning_rate * bias_momentum_corrected /
                               (sqrt(bias_cache_corrected) + epsilon);
        }
    }
};

// loss functions
class Loss
{
public:
    virtual ~Loss() = default;

    template <typename YTrue>
    double calculate(const MatD& output, const YTrue& y_true) const
    {
        static_assert(is_same_v<YTrue, VecI> || is_same_v<YTrue, MatD>,
            "Loss::calculate: y_true must be VecI (sparse) or MatD (one-hot)");

        VecD sample_losses = forward(output, y_true);
        return mean_sample_losses(sample_losses);
    }

protected:
    static double mean_sample_losses(const VecD& sample_losses)
    {
        if (sample_losses.empty()) {
            throw runtime_error("Loss::mean_sample_losses: sample_losses must contain at least one element");
        }

        double sum = 0.0;
        for (double sample_loss : sample_losses) {
            sum += sample_loss;
        }

        return sum / static_cast<double>(sample_losses.size());
    }

    virtual VecD forward(const MatD& output, const VecI& y_true) const = 0;
    virtual VecD forward(const MatD& output, const MatD& y_true) const = 0;
};

class LossCategoricalCrossEntropy : public Loss
{
public:
    MatD dinputs;

    // sparse labels
    VecD forward(const MatD& y_pred, const VecI& y_true) const override
    {
        if (y_pred.rows != y_true.size()) {
            throw runtime_error("LossCategoricalCrossEntropy: y_pred.rows must match y_true.size()");
        }

        MatD clipped = y_pred.clip(1e-7, 1.0 - 1e-7);
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

    // one-hot labels
    VecD forward(const MatD& y_pred, const MatD& y_true) const override
    {
        if (y_pred.rows != y_true.rows || y_pred.cols != y_true.cols) {
            throw runtime_error("LossCategoricalCrossEntropy: y_pred and y_true must have the same shape");
        }

        MatD clipped = y_pred.clip(1e-7, 1.0 - 1e-7);
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

    // sparse labels
    void backward(const MatD& dvalues, const VecI& y_true)
    {
        if (dvalues.rows != y_true.size()) {
            throw runtime_error("LossCategoricalCrossEntropy::backward: dvalues.rows must match y_true.size()");
        }

        const size_t samples = dvalues.rows;
        const size_t labels  = dvalues.cols;

        if (samples == 0) {
            throw runtime_error("LossCategoricalCrossEntropy::backward: dvalues must contain at least one sample");
        }

        dinputs.assign(samples, labels, 0.0);

        for (size_t i = 0; i < samples; ++i) {
            const int class_idx = y_true[i];
            if (class_idx < 0 || static_cast<size_t>(class_idx) >= labels) {
                throw runtime_error("LossCategoricalCrossEntropy::backward: class index out of range");
            }

            const size_t c = static_cast<size_t>(class_idx);
            double p = dvalues(i, c);

            p = clamp_prob(p);

            dinputs(i, c) = -1.0 / p;
        }

        dinputs.scale_by_scalar(samples);
    }

    // one-hot labels
    void backward(const MatD& dvalues, const MatD& y_true)
    {
        if (dvalues.rows != y_true.rows || dvalues.cols != y_true.cols) {
            throw runtime_error("LossCategoricalCrossEntropy::backward: shapes of dvalues and y_true must match");
        }

        const size_t samples = dvalues.rows;
        const size_t labels  = dvalues.cols;

        if (samples == 0) {
            throw runtime_error("LossCategoricalCrossEntropy::backward: dvalues must contain at least one sample");
        }

        dinputs.assign(samples, labels, 0.0);

        for (size_t i = 0; i < samples; ++i) {
            for (size_t j = 0; j < labels; ++j) {
                double p = dvalues(i, j);

                p = clamp_prob(p);

                dinputs(i, j) = -y_true(i, j) / p;
            }
        }

        dinputs.scale_by_scalar(samples);
    }

private:
    static double clamp_prob(double p)
    {
        constexpr double eps = 1e-7;
        if (p < eps) return eps;
        if (p > 1.0 - eps) return 1.0 - eps;
        return p;
    }
};

// Softmax classifier - combined Softmax activation and cross-entropy loss
class ActivationSoftmaxLossCategoricalCrossEntropy
{
public:
    ActivationSoftmax activation;
    LossCategoricalCrossEntropy loss;
    MatD output;
    MatD dinputs;

    template <typename YTrue>
    double forward(const MatD& inputs, const YTrue& y_true)
    {
        static_assert(is_same_v<YTrue, VecI> || is_same_v<YTrue, MatD>,
            "y_true must be VecI (sparse) or MatD (one-hot)");

        activation.forward(inputs);
        output = activation.output;
        return loss.calculate(output, y_true);
    }

    // sparse labels
    void backward(const MatD& dvalues, const VecI& y_true)
    {
        if (dvalues.rows != y_true.size()) {
            throw runtime_error("ActivationSoftmaxLossCategoricalCrossEntropy::backward: dvalues.rows must match y_true.size()");
        }

        const size_t samples = dvalues.rows;
        const size_t labels = dvalues.cols;

        dinputs = dvalues;
        for (size_t i = 0; i < samples; ++i) {
            int class_idx = y_true[i];
            if (class_idx < 0 || static_cast<size_t>(class_idx) >= labels) {
                throw runtime_error("ActivationSoftmaxLossCategoricalCrossEntropy::backward: class index out of range");
            }
            dinputs(i, static_cast<size_t>(class_idx)) -= 1.0;
        }

        dinputs.scale_by_scalar(samples);
    }

    // one-hot labels (turns them into sparse and then calls the other backward method)
    void backward(const MatD& dvalues, const MatD& y_true)
    {
        if (dvalues.rows != y_true.rows || dvalues.cols != y_true.cols) {
            throw runtime_error("ActivationSoftmaxLossCategoricalCrossEntropy::backward: shapes of dvalues and y_true must match");
        }

        VecI y_true_sparse(y_true.rows, 0);
        for (size_t i = 0; i < y_true.rows; ++i) {
            size_t class_idx = 0;
            double max_val = y_true(i, 0);
            for (size_t j = 1; j < y_true.cols; ++j) {
                double v = y_true(i, j);
                if (v > max_val) {
                    max_val = v;
                    class_idx = j;
                    if (max_val == 1.0) break;
                }
            }
            y_true_sparse[i] = static_cast<int>(class_idx);
        }

        backward(dvalues, y_true_sparse);
    }
};

#ifndef NNFS_NO_MAIN
int main()
{
    MatD X;
    VecI y;
    generate_spiral_data(100, 3, X, y);
    plot_scatter_svg("plot.svg", X, y);
    cout << "Saved scatter plot to plot.svg\n";

    LayerDense dense1(2, 64);
    ActivationReLU activation1;
    LayerDense dense2(64, 3);
    ActivationSoftmaxLossCategoricalCrossEntropy loss_activation;
    OptimizerAdam optimizer(0.05, 5e-7);

    cout << fixed << setprecision(3);

    for (size_t epoch = 0; epoch <= 10000; ++epoch) {
        dense1.forward(X);
        activation1.forward(dense1.output);
        dense2.forward(activation1.output);

        double loss = loss_activation.forward(dense2.output, y);
        double accuracy = classification_accuracy(loss_activation.output, y);

        if (epoch % 100 == 0) {
            cout << "epoch: " << epoch
                 << ", acc: " << accuracy
                 << ", loss: " << loss
                 << ", learning_rate: " << setprecision(10) << optimizer.current_learning_rate
                 << setprecision(3)
                 << '\n';
        }

        loss_activation.backward(loss_activation.output, y);
        dense2.backward(loss_activation.dinputs);
        activation1.backward(dense2.dinputs);
        dense1.backward(activation1.dinputs);

        optimizer.pre_update_params();
        optimizer.update_params(dense1);
        optimizer.update_params(dense2);
        optimizer.post_update_params();
    }

    return 0;
}
#endif
