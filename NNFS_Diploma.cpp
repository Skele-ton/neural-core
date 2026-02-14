#include <iostream>
#include <vector>
#include <stdexcept>
#include <random>
#include <cmath>
#include <limits>
#include <string>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <algorithm>
#include <atomic>

#include "fashion_mnist/mnist_reader.hpp"

#ifdef ENABLE_MATPLOT
#include <matplot/matplot.h>
#endif

using std::cout;
using std::is_trivially_copyable_v;
using std::istream;
using std::ifstream;
using std::ostream;
using std::ofstream;
using std::streamsize;
using std::ios;
using std::vector;
using std::unique_ptr;
using std::runtime_error;
using std::size_t;
using std::string;
using std::atomic;
using std::numeric_limits;
using std::mt19937;
using std::normal_distribution;
using std::uniform_real_distribution;
using std::uniform_int_distribution;
using std::seed_seq;
using std::random_device;
using std::exception;
using std::memory_order_relaxed;
using std::memory_order_release;
using std::memory_order_acquire;
using std::swap;
using std::move;
using std::memcmp;
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
using std::transform;
using std::tolower;
using std::make_unique;

inline bool is_whole_number(double v, double epsilon = 1e-7)
{
    return abs(v - round(v)) <= epsilon;
}

inline void multiplication_overflow_check(const size_t a, const size_t b, const char* error_msg) {
    if (a != 0 && b > numeric_limits<size_t>::max() / a) {
        throw runtime_error(error_msg);
    }
}

// thread-safe rng (thread-local mt19937)
// deterministic per-thread streams with global seed + per-thread stream id
static atomic<uint32_t> g_seed_value{0};
static atomic<bool> g_use_deterministic_seed{false};
static atomic<uint32_t> g_seed_epoch{0};

static thread_local uint32_t t_stream_id = 0;
static thread_local bool t_stream_id_set = false;

// call once per thread before any random draws in that thread
void set_thread_stream_id(uint32_t stream_id)
{
    t_stream_id = stream_id;
    t_stream_id_set = true;
}

// make rng deterministic
// call once before any random draws (at the start of the program) to make outcomes deterministic
void set_global_seed(uint32_t seed)
{
    g_seed_value.store(seed, memory_order_relaxed);
    g_use_deterministic_seed.store(true, memory_order_relaxed);
    g_seed_epoch.fetch_add(1, memory_order_release);
}

// make rng non-deterministic
void set_nondeterministic_seed()
{
    g_use_deterministic_seed.store(false, memory_order_relaxed);
    g_seed_epoch.fetch_add(1, memory_order_release);
}

static mt19937& thread_rng()
{
    thread_local mt19937 rng;
    thread_local uint32_t local_epoch = numeric_limits<uint32_t>::max();

    const uint32_t current_epoch = g_seed_epoch.load(memory_order_acquire);
    if (local_epoch != current_epoch) {
        local_epoch = current_epoch;

        if (g_use_deterministic_seed.load(memory_order_relaxed)) {
            if (!t_stream_id_set) {
                throw runtime_error(
                    "thread_rng: deterministic mode requires set_thread_stream_id() to be called once per thread before any random draws");
            }

            const uint32_t seed = g_seed_value.load(memory_order_relaxed);
            seed_seq seq{seed, current_epoch, t_stream_id, 0x9e3779b9u, 0x85ebca6bu};
            rng.seed(seq);
        } else {
            const uint32_t stream_id = t_stream_id_set ? t_stream_id : 0u;

            random_device rd;
            seed_seq seq{rd(), rd(), stream_id, 0x9e3779b9u, 0x85ebca6bu};
            rng.seed(seq);
        }
    }

    return rng;
}

// each generated number is centered around 0 with a standard deviation of 1
// unbounded - technically can be any value but it is more likely to be around 0
double random_gaussian()
{
    thread_local normal_distribution<double> dist(0.0, 1.0);

    thread_local uint32_t dist_epoch = numeric_limits<uint32_t>::max();
    const uint32_t current_epoch = g_seed_epoch.load(memory_order_acquire);
    if (dist_epoch != current_epoch) {
        dist_epoch = current_epoch;
        dist.reset();
    }

    return dist(thread_rng());
}

// generated numbers are in range [0, 1)
// they are equally likely to be any value in that range
double random_uniform()
{
    thread_local uniform_real_distribution<double> uniform(0.0, 1.0);
    return uniform(thread_rng());
}

// generated numbers are in range [min_value, max_value]
// they are equally likely to be any value in that range
size_t random_uniform_int(size_t min_value, size_t max_value)
{
    if (min_value > max_value) {
        throw runtime_error("random_uniform_int: min_value cannot exceed max_value");
    }
    uniform_int_distribution<size_t> dist(min_value, max_value);
    return dist(thread_rng());
}

// Matrix class based on the standard C++ vector
class Matrix
{
public:
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

    // helper methods for shape validation
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
    // ===================================

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

    // method for shuffling input data, where the base matrix is 2D and y is it's 1D labels
    void shuffle_rows_with(Matrix& y)
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

    static Matrix dot(const Matrix& a, const Matrix& b)
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
                double aik = a(i, k);
                for (size_t j = 0; j < b.get_cols(); ++j) {
                    result(i, j) += aik * b(k, j);
                }
            }
        }

        return result;
    }

    static double max_absolute_difference(const Matrix& a, const Matrix& b)
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

    size_t get_rows() const { return rows; }
    size_t get_cols() const { return cols; }
    const vector<double>& get_data() const { return data; }

private:
    size_t rows;
    size_t cols;
    vector<double> data;
};

// training data
void fashion_mnist_create(
    Matrix& X_train_out, Matrix& y_train_out,
    Matrix& X_test_out,  Matrix& y_test_out,
    const string& dir = "fashion_mnist")
{
    namespace fs = std::filesystem;

    const string train_images = dir + "/train-images-idx3-ubyte";
    const string train_labels = dir + "/train-labels-idx1-ubyte";
    const string test_images = dir + "/t10k-images-idx3-ubyte";
    const string test_labels = dir + "/t10k-labels-idx1-ubyte";

    if (!fs::exists(train_images) || !fs::exists(train_labels) ||
        !fs::exists(test_images)  || !fs::exists(test_labels)) {
        throw runtime_error("fashion_mnist_create: dataset files not found under: " + dir);
    }

    auto dataset = mnist::read_dataset<vector, vector, uint8_t, uint8_t>(dir);

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

// plot generated data
void scatter_plot(const string& path, const Matrix& points, const Matrix& labels = Matrix())
{
#ifndef ENABLE_MATPLOT
    (void)path;
    (void)points;
    (void)labels;
    throw runtime_error("scatter_plot: built without Matplot++ (ENABLE_MATPLOT=OFF)");
#else
    if (path.empty()) {
        throw runtime_error("scatter_plot: given path is invalid");
    }

    points.require_non_empty("scatter_plot: points must be non-empty");
    if (points.get_cols() < 2) {
        throw runtime_error("scatter_plot: points must have at least 2 columns");
    }

    const size_t num_points = points.get_rows();

    const bool has_labels = !labels.is_empty();
    Matrix labels_sparse;
    if (has_labels) {
        if (labels.is_col_vector() && labels.get_rows() == num_points) {
            labels_sparse = labels;
        } else if (labels.is_row_vector() && labels.get_cols() == num_points) {
            labels_sparse = labels.transpose();
        } else {
            throw runtime_error("scatter_plot: labels must be a 1D vector with shape (N,1) or (1,N), where N = points.get_rows()");
        }

        labels_sparse.require_shape(num_points, 1,
            "scatter_plot: normalized labels must have shape (N,1)");
    }

    double xmin = points(0, 0);
    double xmax = points(0, 0);
    double ymin = points(0, 1);
    double ymax = points(0, 1);
    for (size_t point_index = 1; point_index < num_points; ++point_index) {
        const double x = points(point_index, 0);
        const double y = points(point_index, 1);
        xmin = min(xmin, x);
        xmax = max(xmax, x);
        ymin = min(ymin, y);
        ymax = max(ymax, y);
    }

    const double x_span = xmax - xmin;
    const double y_span = ymax - ymin;
    const double pad_ratio = 0.05;
    const double min_pad = 1e-6;
    const double x_pad = max(x_span * pad_ratio, min_pad);
    const double y_pad = max(y_span * pad_ratio, min_pad);

    vector<vector<double>> xs(1);
    vector<vector<double>> ys(1);
    for (size_t point_index = 0; point_index < num_points; ++point_index) {
        size_t class_id = 0;
        if (has_labels) {
            class_id = labels_sparse.as_size_t(point_index, 0);
        }

        if (class_id >= xs.size()) {
            xs.resize(class_id + 1);
            ys.resize(class_id + 1);
        }

        xs[class_id].push_back(points(point_index, 0));
        ys[class_id].push_back(points(point_index, 1));
    }

    matplot::figure(true);
    matplot::hold(matplot::on);

    for (size_t class_idx = 0; class_idx < xs.size(); ++class_idx) {
        if (xs[class_idx].empty()) continue;
        auto p = matplot::plot(xs[class_idx], ys[class_idx], ".");
        p->marker_size(3.0);
    }

    matplot::title("Scatter Plot");
    matplot::xlabel("x");
    matplot::ylabel("y");
    matplot::axis(matplot::equal);
    matplot::xlim({xmin - x_pad, xmax + x_pad});
    matplot::ylim({ymin - y_pad, ymax + y_pad});
    matplot::grid(matplot::on);

    matplot::save(path);
#endif
}

// activations
class Activation
{
public:
    virtual ~Activation() = default;

    virtual void forward(const Matrix& inputs_batch) = 0;
    virtual void backward(const Matrix& dvalues) = 0;
    virtual Matrix predictions(const Matrix& outputs) const = 0;

    const Matrix& get_inputs() const { return inputs; }
    const Matrix& get_output() const { return output; }
    const Matrix& get_dinputs() const { return dinputs; }

protected:
    Matrix inputs;
    Matrix output;
    Matrix dinputs;
};

class ActivationReLU : public Activation
{
public:
    void forward(const Matrix& inputs_batch) override
    {
        inputs_batch.require_non_empty("ActivationReLU::forward: inputs must be non-empty");

        inputs = inputs_batch;
        output.assign(inputs.get_rows(), inputs.get_cols());
        for (size_t i = 0; i < inputs.get_rows(); ++i) {
            for (size_t j = 0; j < inputs.get_cols(); ++j) {
                output(i, j) = max(0.0, inputs(i, j));
            }
        }
    }

    void backward(const Matrix& dvalues) override
    {
        dvalues.require_non_empty("ActivationReLU::backward: dvalues must be non-empty");
        dvalues.require_shape(inputs.get_rows(), inputs.get_cols(),
            "ActivationReLU::backward: dvalues shape mismatch");

        dinputs = dvalues;
        for (size_t i = 0; i < inputs.get_rows(); ++i) {
            for (size_t j = 0; j < inputs.get_cols(); ++j) {
                if (inputs(i, j) <= 0.0) {
                    dinputs(i, j) = 0.0;
                }
            }
        }
    }

    Matrix predictions(const Matrix& outputs) const override { return outputs; }
};

class ActivationSoftmax : public Activation
{
public:
    void forward(const Matrix& inputs_batch) override
    {
        inputs_batch.require_non_empty("ActivationSoftmax::forward: inputs must be non-empty");

        inputs = inputs_batch;
        output.assign(inputs.get_rows(), inputs.get_cols());

        for (size_t i = 0; i < inputs.get_rows(); ++i) {
            double max_val = inputs(i, 0);
            for (size_t j = 1; j < inputs.get_cols(); ++j) {
                double v = inputs(i, j);
                if (v > max_val) max_val = v;
            }

            double sum = 0.0;
            for (size_t j = 0; j < inputs.get_cols(); ++j) {
                double e = exp(inputs(i, j) - max_val);
                output(i, j) = e;
                sum += e;
            }

            if (!isfinite(sum) || sum <= 0.0) {
                throw runtime_error("ActivationSoftmax: invalid sum of exponentials");
            }

            for (size_t j = 0; j < inputs.get_cols(); ++j) {
                output(i, j) /= sum;
            }
        }
    }

    void backward(const Matrix& dvalues) override
    {
        dvalues.require_non_empty("ActivationSoftmax::backward: dvalues must be non-empty");
        dvalues.require_shape(output.get_rows(), output.get_cols(),
            "ActivationSoftmax::backward: dvalues shape mismatch");

        dinputs.assign(dvalues.get_rows(), dvalues.get_cols());

        const size_t samples = dvalues.get_rows();
        const size_t classes = dvalues.get_cols();

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
        if(outputs.get_cols() < 2) {
            throw runtime_error("ActivationSoftmax::predictions: computation of softmax predictions requires outputs.get_cols() >= 2");
        }

        return outputs.argmax();
    }
};

class ActivationSigmoid : public Activation
{
public:
    void forward(const Matrix& inputs_batch) override
    {
        inputs_batch.require_non_empty("ActivationSigmoid::forward: inputs must be non-empty");

        inputs = inputs_batch;
        output.assign(inputs.get_rows(), inputs.get_cols());
        for (size_t i = 0; i < inputs.get_rows(); ++i) {
            for (size_t j = 0; j < inputs.get_cols(); ++j) {
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
        dvalues.require_shape(output.get_rows(), output.get_cols(),
            "ActivationSigmoid::backward: dvalues shape mismatch");

        dinputs.assign(dvalues.get_rows(), dvalues.get_cols());
        for (size_t i = 0; i < dvalues.get_rows(); ++i) {
            for (size_t j = 0; j < dvalues.get_cols(); ++j) {
                const double s = output(i, j);
                dinputs(i, j) = dvalues(i, j) * (1.0 - s) * s;
            }
        }
    }

    Matrix predictions(const Matrix& outputs) const override
    {
        outputs.require_non_empty("ActivationSigmoid::predictions: outputs must be non-empty");

        Matrix preds(outputs.get_rows(), outputs.get_cols());
        for (size_t i = 0; i < outputs.get_rows(); ++i) {
            for (size_t j = 0; j < outputs.get_cols(); ++j) {
                preds(i, j) = outputs(i, j) > 0.5 ? 1.0 : 0.0;
            }
        }
        return preds;
    }
};

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
        dvalues.require_shape(inputs.get_rows(), inputs.get_cols(),
            "ActivationLinear::backward: dvalues shape mismatch");
        
        dinputs = dvalues;
    }

    Matrix predictions(const Matrix& outputs) const override { return outputs; }
};

// layers
class Layer
{
public:
    virtual ~Layer() = default;

    virtual void forward(const Matrix& inputs_batch, bool training) = 0;
    virtual void backward(const Matrix& dvalues, bool include_activation) = 0;

    const Matrix& get_inputs() const { return inputs; }
    const Matrix& get_output() const { return output; }
    const Matrix& get_dinputs() const { return dinputs; }
    const Activation* get_activation() const { return activation; }

protected:
    Matrix inputs;
    Matrix output;
    Matrix dinputs;
    Activation* activation = nullptr;

    void forward(const Matrix& inputs_batch) { forward(inputs_batch, true); }
    void backward(const Matrix& dvalues) { backward(dvalues, true); }
};

// dense layer with weights, biases and an activation function
class LayerDense : public Layer
{
public:
    using Layer::forward;
    using Layer::backward;

    Matrix weights;
    Matrix biases;

    Matrix weight_momentums;
    Matrix bias_momentums;
    Matrix weight_cache;
    Matrix bias_cache;

    LayerDense(size_t n_neurons, const string& activation_name,
               double weight_regularizer_l1 = 0.0,
               double weight_regularizer_l2 = 0.0,
               double bias_regularizer_l1 = 0.0,
               double bias_regularizer_l2 = 0.0)
        : weights(),
        biases(1, n_neurons, 0.0),
        weight_regularizer_l1(weight_regularizer_l1),
        weight_regularizer_l2(weight_regularizer_l2),
        bias_regularizer_l1(bias_regularizer_l1),
        bias_regularizer_l2(bias_regularizer_l2),
        starting_n_neurons(n_neurons)
    {
        if (weight_regularizer_l1 < 0.0 || weight_regularizer_l2 < 0.0 ||
            bias_regularizer_l1 < 0.0 || bias_regularizer_l2 < 0.0) {
            throw runtime_error("LayerDense: regularizers must be non-negative");
        }
        if (n_neurons == 0) {
            throw runtime_error("LayerDense: n_neurons must be > 0");
        }

        string name = activation_name;
        transform(name.begin(), name.end(), name.begin(),
                       [](unsigned char c) { return static_cast<char>(tolower(c)); });

        if (name == "relu") {
            activation = &activation_relu;
        } else if (name == "softmax") {
            activation = &activation_softmax;
        } else if (name == "sigmoid") {
            activation = &activation_sigmoid;
        } else if (name == "linear") {
            activation = &activation_linear;
        } else {
            throw runtime_error("LayerDense: unknown activation. use relu, softmax, sigmoid or linear");
        }
    }

    void forward(const Matrix& inputs_batch, bool) override
    {
        inputs = inputs_batch;

        inputs.require_non_empty("LayerDense::forward: inputs must be non-empty");

        // filling the weights in the shape of starting_n_neurons x inputs.get_cols(), in case they are empty
        // will be ran when it's the first time data is passed through the layer
        // but also if the weights are emptied afterwards and data is passed through again
        if (weights.is_empty()) {
            size_t n_inputs = inputs.get_cols();

            weights.assign(n_inputs, starting_n_neurons);
            for (size_t input = 0; input < n_inputs; ++input) {
                for (size_t neuron = 0; neuron < starting_n_neurons; ++neuron) {
                    weights(input, neuron) = 0.1 * random_gaussian();
                }
            }
        }

        inputs.require_cols(weights.get_rows(), "LayerDense::forward: inputs.get_cols() must match weights.get_rows()");
        biases.require_shape(1, weights.get_cols(),
            "LayerDense::forward: biases must be shape (1, n_neurons)");

        output = Matrix::dot(inputs, weights);
        for (size_t i = 0; i < output.get_rows(); ++i) {
            for (size_t j = 0; j < output.get_cols(); ++j) {
                output(i, j) += biases(0, j);
            }
        }

        activation->forward(output);
        output = activation->get_output();
    }

    void backward(const Matrix& dvalues, bool include_activation) override
    {
        inputs.require_non_empty("LayerDense::backward: inputs must be non-empty (forward not called?)");
        weights.require_non_empty("LayerDense::backward: weights must be initialized (forward not called?)");

        dvalues.require_non_empty("LayerDense::backward: dvalues must be non-empty");
        dvalues.require_shape(inputs.get_rows(), weights.get_cols(),
            "LayerDense::backward: dvalues shape mismatch");

        Matrix dactivation;
        if (include_activation) {
            activation->backward(dvalues);
            dactivation = activation->get_dinputs();
        } else {
            dactivation = dvalues;
        }
        
        const Matrix inputs_T = inputs.transpose();
        dweights = Matrix::dot(inputs_T, dactivation);

        dbiases.assign(1, biases.get_cols(), 0.0);
        for (size_t i = 0; i < dactivation.get_rows(); ++i) {
            for (size_t j = 0; j < dactivation.get_cols(); ++j) {
                dbiases(0, j) += dactivation(i, j);
            }
        }

        // l1 and l2 regularization
        const bool has_w_l1 = weight_regularizer_l1 != 0.0;
        const bool has_w_l2 = weight_regularizer_l2 != 0.0;

        if (has_w_l1 || has_w_l2) {
            const double weight_l2_times_two = weight_regularizer_l2 * 2.0;

            for (size_t i = 0; i < weights.get_rows(); ++i) {
                for (size_t j = 0; j < weights.get_cols(); ++j) {
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

            for (size_t j = 0; j < biases.get_cols(); ++j) {
                const double b = biases(0, j);

                if (has_b_l1) dbiases(0, j) += bias_regularizer_l1 * ((b >= 0.0) ? 1.0 : -1.0);
                if (has_b_l2) dbiases(0, j) += bias_l2_times_two * b;
            }
        }

        const Matrix weights_T = weights.transpose();
        dinputs = Matrix::dot(dactivation, weights_T);
    }

    const Matrix& get_dweights() const { return dweights; }
    const Matrix& get_dbiases() const { return dbiases; }

    double get_weight_regularizer_l1() const { return weight_regularizer_l1; }
    double get_weight_regularizer_l2() const { return weight_regularizer_l2; }
    double get_bias_regularizer_l1() const { return bias_regularizer_l1; }
    double get_bias_regularizer_l2() const { return bias_regularizer_l2; }

    double get_starting_n_neurons() const { return starting_n_neurons; }

private:
    Matrix dweights;
    Matrix dbiases;

    double weight_regularizer_l1;
    double weight_regularizer_l2;
    double bias_regularizer_l1;
    double bias_regularizer_l2;

    size_t starting_n_neurons;

    ActivationReLU activation_relu;
    ActivationSoftmax activation_softmax;
    ActivationSigmoid activation_sigmoid;
    ActivationLinear activation_linear;
};

// dropout layer for "disabling" part of the neurons during training
class LayerDropout : public Layer
{
public:
    using Layer::forward;
    using Layer::backward;

    explicit LayerDropout(double dropout_rate)
        : keep_rate(1.0 - dropout_rate)
    {
        if (keep_rate <= 0.0 || keep_rate > 1.0) {
            throw runtime_error("LayerDropout: dropout_rate must be in [0,1)");
        }

        activation = &activation_linear;
    }

    void forward(const Matrix& inputs_batch, bool training) override
    {
        inputs_batch.require_non_empty("LayerDropout::forward: inputs must be non-empty");

        inputs = inputs_batch;

        scaled_binary_mask.assign(inputs.get_rows(), inputs.get_cols());
        output.assign(inputs.get_rows(), inputs.get_cols());

        for (size_t i = 0; i < inputs.get_rows(); ++i) {
            for (size_t j = 0; j < inputs.get_cols(); ++j) {
                const double mask = training ? ((random_uniform() < keep_rate) ? (1.0 / keep_rate) : 0.0) : 1.0;
                scaled_binary_mask(i, j) = mask;
                output(i, j) = inputs(i, j) * mask;
            }
        }

        activation->forward(output);
        output = activation->get_output();
    }

    void backward(const Matrix& dvalues, bool) override
    {
        dvalues.require_non_empty("LayerDropout::backward: dvalues must be non-empty");
        dvalues.require_shape(scaled_binary_mask.get_rows(), scaled_binary_mask.get_cols(),
            "LayerDropout::backward: dvalues shape mismatch");

        activation->backward(dvalues);
        Matrix dactivation = activation->get_dinputs();

        dinputs.assign(dactivation.get_rows(), dactivation.get_cols());
        for (size_t i = 0; i < dactivation.get_rows(); ++i) {
            for (size_t j = 0; j < dactivation.get_cols(); ++j) {
                dinputs(i, j) = dactivation(i, j) * scaled_binary_mask(i, j);
            }
        }
    }

    double get_keep_rate() const { return keep_rate; }

private:
    double keep_rate;
    Matrix scaled_binary_mask;
    ActivationLinear activation_linear;
};

// input "layer" for storing inputs into the model
// doesn't inherit from the base layer class
class LayerInput
{
public:
    void forward(const Matrix& inputs_batch)
    {
        inputs_batch.require_non_empty("LayerInput::forward: inputs must be non-empty");
        output = inputs_batch;
    }

    const Matrix& get_output() const { return output; }

private:
    Matrix output;
};

// loss functions
class Loss
{
public:
    virtual ~Loss() = default;

    double calculate(const Matrix& output, const Matrix& y_true)
    {
        output.require_non_empty("Loss::calculate: output must be non-empty");
        y_true.require_non_empty("Loss::calculate: y_true must be non-empty");

        Matrix sample_losses = forward(output, y_true);

        sample_losses.require_shape(1, output.get_rows(),
            "Loss::calculate: per-sample losses must be of shape (1,output.get_rows()) after forward");

        double sum = 0.0;
        for (double v : sample_losses.get_data()) sum += v;
        const double count = sample_losses.get_data().size();

        accumulated_sum += sum;
        accumulated_count += count;

        return sum / static_cast<double>(count);
    }

    double calculate(const Matrix& output, const Matrix& y_true, double& out_regularization_loss,
                     const vector<LayerDense*>& layers)
    {
        const double data_loss = calculate(output, y_true);
        out_regularization_loss = regularization_loss(layers);
        return data_loss;
    }

    double calculate_accumulated() const
    {
        if (accumulated_count == 0) {
            throw runtime_error("Loss::calculate_accumulated: accumulated_count must be > 0");
        }
        return accumulated_sum / static_cast<double>(accumulated_count);
    }

    double calculate_accumulated(double& out_regularization_loss, const vector<LayerDense*>& layers) const
    {
        const double data_loss = calculate_accumulated();
        out_regularization_loss = regularization_loss(layers);
        return data_loss;
    }

    void new_pass()
    {
        accumulated_sum = 0.0;
        accumulated_count = 0;
    }

    virtual void backward(const Matrix& y_pred, const Matrix& y_true) = 0;

    const Matrix& get_dinputs() const { return dinputs; }

protected:
    Matrix dinputs;

    virtual Matrix forward(const Matrix& output, const Matrix& y_true) const = 0;

    static double clamp(double p)
    {
        constexpr double eps = 1e-7;
        if (p < eps) return eps;
        if (p > 1.0 - eps) return 1.0 - eps;
        return p;
    }

private:
    double accumulated_sum = 0.0;
    size_t accumulated_count = 0;

    static double regularization_loss(const LayerDense& layer)
    {
        double regularization = 0.0;

        const bool has_w_l1 = layer.get_weight_regularizer_l1() != 0.0;
        const bool has_w_l2 = layer.get_weight_regularizer_l2() != 0.0;

        if(has_w_l1 || has_w_l2) {
            layer.weights.require_non_empty("Loss::regularization_loss: weights must be non-empty");

            double sum_abs = 0.0;
            double sum_sq  = 0.0;

            for (double weight : layer.weights.get_data()) {
                if (has_w_l1) sum_abs += abs(weight);
                if (has_w_l2)  sum_sq  += weight * weight;
            }

            regularization += layer.get_weight_regularizer_l1() * sum_abs + layer.get_weight_regularizer_l2() * sum_sq;
        }

        const bool has_b_l1 = layer.get_bias_regularizer_l1() != 0.0;
        const bool has_b_l2 = layer.get_bias_regularizer_l2() != 0.0;

        if(has_b_l1 || has_b_l2) {
            layer.biases.require_non_empty("Loss::regularization_loss: biases must be non-empty");
            layer.weights.require_non_empty("Loss::regularization_loss: weights must be non-empty");

            layer.biases.require_shape(1, layer.weights.get_cols(),
                "Loss::regularization_loss: biases must have shape (1, n_neurons)");

            double sum_abs = 0.0;
            double sum_sq  = 0.0;

            for (double bias : layer.biases.get_data()) {
                if (has_b_l1) sum_abs += abs(bias);
                if (has_b_l2) sum_sq  += bias * bias;
            }

            regularization += layer.get_bias_regularizer_l1() * sum_abs + layer.get_bias_regularizer_l2() * sum_sq;
        }

        return regularization;
    }

    static double regularization_loss(const vector<LayerDense*>& layers)
    {
        double regularization = 0.0;
        for (const LayerDense* layer : layers) {
            if (!layer) continue;
            regularization += regularization_loss(*layer);
        }
        return regularization;
    }
};

class LossCategoricalCrossEntropy : public Loss
{
public:
    void backward(const Matrix& y_pred, const Matrix& y_true) override
    {
        y_pred.require_non_empty("LossCategoricalCrossEntropy::backward: y_pred must be non-empty");
        y_true.require_non_empty("LossCategoricalCrossEntropy::backward: y_true must be non-empty");

        if (y_pred.get_cols() < 2) {
            throw runtime_error("LossCategoricalCrossEntropy::backward: y_pred.get_cols() must be >= 2");
        }

        const size_t samples = y_pred.get_rows();
        const size_t classes  = y_pred.get_cols();

        dinputs.assign(samples, classes, 0.0);

        Matrix y_true_sparse;

        if (y_true.is_col_vector() && y_true.get_rows() == samples) {
            y_true_sparse = y_true;
        } else if (y_true.is_row_vector() && y_true.get_cols() == samples) {
            y_true_sparse = y_true.transpose();
        } else if (y_true.get_rows() == samples && y_true.get_cols() == classes) {
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

protected:
    Matrix forward(const Matrix& y_pred, const Matrix& y_true) const override
    {
        if (y_pred.get_cols() < 2) {
            throw runtime_error("LossCategoricalCrossEntropy::forward: y_pred.get_cols() must be >= 2");
        }

        const size_t samples = y_pred.get_rows();
        const size_t classes = y_pred.get_cols();

        Matrix sample_losses(1, samples, 0.0);

        Matrix y_true_sparse;

        if (y_true.is_col_vector() && y_true.get_rows() == samples) {
            y_true_sparse = y_true;
        } else if (y_true.is_row_vector() && y_true.get_cols() == samples) {
            y_true_sparse = y_true.transpose();
        } else if (y_true.get_rows() == samples && y_true.get_cols() == classes) {
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
};

class LossBinaryCrossentropy : public Loss
{
public:
    void backward(const Matrix& y_pred, const Matrix& y_true) override
    {
        y_pred.require_non_empty("LossBinaryCrossentropy::backward: y_pred must be non-empty");
        y_true.require_non_empty("LossBinaryCrossentropy::backward: y_true must be non-empty");

        y_true.require_shape(y_pred.get_rows(), y_pred.get_cols(),
            "LossBinaryCrossentropy::backward: y_pred and y_true must have the same shape");

        const size_t samples = y_pred.get_rows();
        const size_t outputs = y_pred.get_cols();

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

protected:
    Matrix forward(const Matrix& y_pred, const Matrix& y_true) const override
    {
        y_true.require_shape(y_pred.get_rows(), y_pred.get_cols(),
            "LossBinaryCrossentropy::forward: y_pred and y_true must have the same shape");
        
        const size_t samples = y_pred.get_rows();
        const size_t outputs = y_pred.get_cols();

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
};

class LossMeanSquaredError : public Loss
{
public:
    void backward(const Matrix& y_pred, const Matrix& y_true) override
    {
        y_pred.require_non_empty("LossMeanSquaredError::backward: y_pred must be non-empty");
        y_true.require_non_empty("LossMeanSquaredError::backward: y_true must be non-empty");

        y_true.require_shape(y_pred.get_rows(), y_pred.get_cols(),
            "LossMeanSquaredError::backward: y_pred and y_true must have the same shape");

        const size_t samples = y_pred.get_rows();
        const size_t outputs = y_pred.get_cols();

        dinputs.assign(samples, outputs, 0.0);

        for (size_t i = 0; i < samples; ++i) {
            for (size_t j = 0; j < outputs; ++j) {
                dinputs(i, j) = -2.0 * (y_true(i, j) - y_pred(i, j)) / static_cast<double>(outputs);
            }
        }

        dinputs.scale_by_scalar(samples);
    }

private:
    Matrix forward(const Matrix& y_pred, const Matrix& y_true) const override
    {
        y_true.require_shape(y_pred.get_rows(), y_pred.get_cols(),
            "LossMeanSquaredError::forward: y_pred and y_true must have the same shape");

        const size_t samples = y_pred.get_rows();
        const size_t outputs = y_pred.get_cols();

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
};

class LossMeanAbsoluteError : public Loss
{
public:
    void backward(const Matrix& y_pred, const Matrix& y_true) override
    {
        y_pred.require_non_empty("LossMeanAbsoluteError::backward: y_pred must be non-empty");
        y_true.require_non_empty("LossMeanAbsoluteError::backward: y_true must be non-empty");

        y_true.require_shape(y_pred.get_rows(), y_pred.get_cols(),
            "LossMeanAbsoluteError::backward: y_pred and y_true must have the same shape");

        const size_t samples = y_pred.get_rows();
        const size_t outputs = y_pred.get_cols();

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

protected:
    Matrix forward(const Matrix& y_pred, const Matrix& y_true) const override
    {
        y_true.require_shape(y_pred.get_rows(), y_pred.get_cols(),
            "LossMeanAbsoluteError::forward: y_pred and y_true must have the same shape");

        const size_t samples = y_pred.get_rows();
        const size_t outputs = y_pred.get_cols();

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
};

// Softmax classifier (backward only) - combined Softmax activation and cross-entropy loss
class ActivationSoftmaxLossCategoricalCrossEntropy
{
public:
    void backward(const Matrix& y_pred, const Matrix& y_true)
    {
        y_pred.require_non_empty("ActivationSoftmaxLossCategoricalCrossEntropy::backward: y_pred must be non-empty");
        y_true.require_non_empty("ActivationSoftmaxLossCategoricalCrossEntropy::backward: y_true must be non-empty");

        if (y_pred.get_cols() < 2) {
            throw runtime_error("ActivationSoftmaxLossCategoricalCrossEntropy::backward: y_pred.get_cols() must be >= 2");
        }

        const size_t samples = y_pred.get_rows();
        const size_t classes = y_pred.get_cols();

        Matrix y_true_sparse;

        if (y_true.is_col_vector() && y_true.get_rows() == samples) {
            y_true_sparse = y_true;
        } else if (y_true.is_row_vector() && y_true.get_cols() == samples) {
            y_true_sparse = y_true.transpose();
        } else if (y_true.get_rows() == samples && y_true.get_cols() == classes) {
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

    const Matrix& get_dinputs() const { return dinputs; }

private:
    Matrix dinputs;
};

// optimizers
class Optimizer
{
public:
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

    double get_learning_rate() const { return learning_rate; }
    double get_current_learning_rate() const { return current_learning_rate; }
    double get_decay() const { return decay; }
    double get_iterations() const { return iterations; }

protected:
    double learning_rate;
    double current_learning_rate;
    double decay;
    size_t iterations;
};

class OptimizerSGD : public Optimizer
{
public:
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

        layer.biases.require_shape(1, layer.weights.get_cols(), "OptimizerSGD::update_params: biases must have shape (1, n_neurons)");

        layer.get_dweights().require_shape(layer.weights.get_rows(), layer.weights.get_cols(),
            "OptimizerSGD::update_params: dweights must match weights shape");

        const size_t w_rows = layer.weights.get_rows();
        const size_t w_cols = layer.weights.get_cols();
        const size_t b_cols = layer.biases.get_cols();

        const double minus_learning_rate = -current_learning_rate;

        if (momentum == 0.0) {
            for (size_t i = 0; i < w_rows; ++i) {
                for (size_t j = 0; j < w_cols; ++j) {
                    layer.weights(i, j) += minus_learning_rate * layer.get_dweights()(i, j);
                }
            }
            for (size_t j = 0; j < b_cols; ++j) {
                layer.biases(0, j) += minus_learning_rate * layer.get_dbiases()(0, j);
            }
        } else {
            if (layer.weight_momentums.get_rows() != w_rows || layer.weight_momentums.get_cols() != w_cols) {
                layer.weight_momentums.assign(w_rows, w_cols, 0.0);
            }
            if (layer.bias_momentums.get_rows() != 1 || layer.bias_momentums.get_cols() != b_cols) {
                layer.bias_momentums.assign(1, b_cols, 0.0);
            }

            for (size_t i = 0; i < w_rows; ++i) {
                for (size_t j = 0; j < w_cols; ++j) {
                    const double temp = momentum * layer.weight_momentums(i, j) + minus_learning_rate * layer.get_dweights()(i, j);
                    layer.weight_momentums(i, j) = temp;
                    layer.weights(i, j) += temp;
                }
            }

            for (size_t j = 0; j < b_cols; ++j) {
                const double temp = momentum * layer.bias_momentums(0, j) + minus_learning_rate * layer.get_dbiases()(0, j);
                layer.bias_momentums(0, j) = temp;
                layer.biases(0, j) += temp;
            }
        }
    }

private:
    double momentum;
};

class OptimizerAdagrad : public Optimizer
{
public:
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

        layer.biases.require_shape(1, layer.weights.get_cols(),
            "OptimizerAdagrad::update_params: biases must have shape (1, n_neurons)");

        layer.get_dweights().require_shape(layer.weights.get_rows(), layer.weights.get_cols(),
            "OptimizerAdagrad::update_params: dweights must match weights shape");

        const size_t w_rows = layer.weights.get_rows();
        const size_t w_cols = layer.weights.get_cols();
        const size_t b_cols = layer.biases.get_cols();

        if (layer.weight_cache.get_rows() != w_rows || layer.weight_cache.get_cols() != w_cols) {
            layer.weight_cache.assign(w_rows, w_cols, 0.0);
        }
        if (layer.bias_cache.get_rows() != 1 || layer.bias_cache.get_cols() != b_cols) {
            layer.bias_cache.assign(1, b_cols, 0.0);
        }

        const double minus_learning_rate = -current_learning_rate;

        for (size_t i = 0; i < w_rows; ++i) {
            for (size_t j = 0; j < w_cols; ++j) {
                const double g = layer.get_dweights()(i, j);
                layer.weight_cache(i, j) += g * g;
                layer.weights(i, j) += minus_learning_rate * g / (sqrt(layer.weight_cache(i, j)) + epsilon);
            }
        }

        for (size_t j = 0; j < b_cols; ++j) {
            const double g = layer.get_dbiases()(0, j);
            layer.bias_cache(0, j) += g * g;
            layer.biases(0, j) += minus_learning_rate * g / (sqrt(layer.bias_cache(0, j)) + epsilon);
        }
    }

private:
    double epsilon;
};

class OptimizerRMSprop : public Optimizer
{
public:
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

        layer.biases.require_shape(1, layer.weights.get_cols(),
            "OptimizerRMSprop::update_params: biases must have shape (1, n_neurons)");

        layer.get_dweights().require_shape(layer.weights.get_rows(), layer.weights.get_cols(),
            "OptimizerRMSprop::update_params: dweights must match weights shape");

        const size_t w_rows = layer.weights.get_rows();
        const size_t w_cols = layer.weights.get_cols();
        const size_t b_cols = layer.biases.get_cols();

        if (layer.weight_cache.get_rows() != w_rows || layer.weight_cache.get_cols() != w_cols) {
            layer.weight_cache.assign(w_rows, w_cols, 0.0);
        }
        if (layer.bias_cache.get_rows() != 1 || layer.bias_cache.get_cols() != b_cols) {
            layer.bias_cache.assign(1, b_cols, 0.0);
        }

        const double minus_learning_rate = -current_learning_rate;
        const double one_minus_rho = 1.0 - rho;

        for (size_t i = 0; i < w_rows; ++i) {
            for (size_t j = 0; j < w_cols; ++j) {
                const double g = layer.get_dweights()(i, j);
                layer.weight_cache(i, j) = rho * layer.weight_cache(i, j) + one_minus_rho * g * g;
                layer.weights(i, j) += minus_learning_rate * g / (sqrt(layer.weight_cache(i, j)) + epsilon);
            }
        }

        for (size_t j = 0; j < b_cols; ++j) {
            const double g = layer.get_dbiases()(0, j);
            layer.bias_cache(0, j) = rho * layer.bias_cache(0, j) + one_minus_rho * g * g;
            layer.biases(0, j) += minus_learning_rate * g / (sqrt(layer.bias_cache(0, j)) + epsilon);
        }
    }

private:
    double epsilon;
    double rho;
};

class OptimizerAdam : public Optimizer
{
public:
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

        layer.biases.require_shape(1, layer.weights.get_cols(),
            "OptimizerAdam::update_params: biases must have shape (1, n_neurons)");

        layer.get_dweights().require_shape(layer.weights.get_rows(), layer.weights.get_cols(),
            "OptimizerAdam::update_params: dweights must match weights shape");

        const size_t w_rows = layer.weights.get_rows();
        const size_t w_cols = layer.weights.get_cols();
        const size_t b_cols = layer.biases.get_cols();

        if (layer.weight_momentums.get_rows() != w_rows || layer.weight_momentums.get_cols() != w_cols) {
            layer.weight_momentums.assign(w_rows, w_cols, 0.0);
        }
        if (layer.weight_cache.get_rows() != w_rows || layer.weight_cache.get_cols() != w_cols) {
            layer.weight_cache.assign(w_rows, w_cols, 0.0);
        }
        if (layer.bias_momentums.get_rows() != 1 || layer.bias_momentums.get_cols() != b_cols) {
            layer.bias_momentums.assign(1, b_cols, 0.0);
        }
        if (layer.bias_cache.get_rows() != 1 || layer.bias_cache.get_cols() != b_cols) {
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
                const double g = layer.get_dweights()(i, j);

                layer.weight_momentums(i, j) = beta1 * layer.weight_momentums(i, j) + one_minus_beta1 * g;
                layer.weight_cache(i, j) = beta2 * layer.weight_cache(i, j) + one_minus_beta2 * g * g;

                const double weight_momentum_corrected = layer.weight_momentums(i, j) / correction_applied_to_momentum;
                const double weight_cache_corrected = layer.weight_cache(i, j) / correction_applied_to_cache;

                layer.weights(i, j) += minus_learning_rate * weight_momentum_corrected / (sqrt(weight_cache_corrected) + epsilon);
            }
        }

        for (size_t j = 0; j < b_cols; ++j) {
            const double g = layer.get_dbiases()(0, j);

            layer.bias_momentums(0, j) = beta1 * layer.bias_momentums(0, j) + one_minus_beta1 * g;
            layer.bias_cache(0, j) = beta2 * layer.bias_cache(0, j) + one_minus_beta2 * g * g;

            const double bias_momentum_corrected = layer.bias_momentums(0, j) / correction_applied_to_momentum;
            const double bias_cache_corrected = layer.bias_cache(0, j) / correction_applied_to_cache;

            layer.biases(0, j) += minus_learning_rate * bias_momentum_corrected / (sqrt(bias_cache_corrected) + epsilon);
        }
    }

private:
    double epsilon;
    double beta1;
    double beta2;
    double beta1_power;
    double beta2_power;
};

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

        multiplication_overflow_check(y_pred.get_rows(), y_pred.get_cols(), "Accuracy::calculate: total overflow");
        const size_t correct = compare(y_pred, y_true);
        const size_t total = y_pred.get_cols() * y_pred.get_rows();
        
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

private:
    size_t accumulated_sum = 0;
    size_t accumulated_count = 0;
};

class AccuracyCategorical : public Accuracy
{
public:
    explicit AccuracyCategorical(bool binary = false)
        : binary(binary)
    {
    }

    bool get_binray() const { return binary; }

protected:
    size_t compare(const Matrix& y_pred, const Matrix& y_true) override
    {
        if(!binary) y_pred.require_cols(1, "AccuracyCategorical::compare: categorical y_pred must have shape (N,1)");

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

        const size_t samples = y_true.get_rows();
        const size_t outputs = y_true.get_cols();

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

    double get_precision_divisor() const { return precision_divisor; }

protected:
    size_t compare(const Matrix& y_pred, const Matrix& y_true) override
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

private:
    double precision_divisor;
    double precision;
    bool initialized;
};

// Model wrapper
class Model
{
public:
    Model() = default;

    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    Model(Model&&) noexcept = default;
    Model& operator=(Model&&) noexcept = default;

    void clear()
    {
        input_layer = LayerInput();
        layers.clear();
        architecture.clear();

        loss = nullptr;
        accuracy = nullptr;
        optimizer = nullptr;
        loss_is_cce = false;

        invalidate_compiled_state();
    }

    LayerDense& add_dense(size_t n_neurons, const string& activation_name,
                          double w_l1 = 0.0, double w_l2 = 0.0, double b_l1 = 0.0, double b_l2 = 0.0)
    {
        const ActivationKind k = string_to_activation_kind(activation_name);
        return add_dense_to_model(n_neurons, k, w_l1, w_l2, b_l1, b_l2);
    }

    LayerDropout& add_dropout(double dropout_rate)
    {
        LayerDesc d;
        d.kind = LayerKind::Dropout;
        d.dropout.dropout_rate = dropout_rate;

        architecture.push_back(d);

        auto layer = make_unique<LayerDropout>(dropout_rate);
        LayerDropout& ref = *layer;
        layers.push_back(move(layer));

        invalidate_compiled_state();

        return ref;
    }

    void configure(Loss& loss_obj, Accuracy& accuracy_obj, Optimizer* optimizer_obj = nullptr)
    {
        loss = &loss_obj;
        accuracy = &accuracy_obj;
        optimizer = optimizer_obj;
        loss_is_cce = (dynamic_cast<LossCategoricalCrossEntropy*>(loss) != nullptr);

        if (finalized) update_combined_flag();
    }

    void configure(Loss& loss_obj, Accuracy& accuracy_obj, Optimizer& optimizer_obj)
    {
        configure(loss_obj, accuracy_obj, &optimizer_obj);
    }

    void finalize()
    {
        if (layers.empty()) {
            throw runtime_error("Model::finalize: no layers added");
        }
        if (architecture.empty()) {
            throw runtime_error("Model::finalize: internal error (empty architecture)");
        }
        if (layers.size() != architecture.size()) {
            throw runtime_error("Model::finalize: internal error (layers/spec size mismatch)");
        }
        if (architecture.back().kind == LayerKind::Dropout) {
            throw runtime_error("Model::finalize: final layer cannot be dropout");
        }

        rebuild_caches();
        update_combined_flag();

        finalized = true;
    }

    void train(const Matrix& X, const Matrix& y, size_t epochs = 1, size_t batch_size = 0,
               size_t print_every = 100, const Matrix* X_val = nullptr, const Matrix* y_val = nullptr)
    {
        if (!finalized) finalize();

        if (!loss || !accuracy || !optimizer) {
            throw runtime_error("Model::train: loss, accuracy and optimizer must be set");
        }

        X.require_non_empty("Model::train: X must be non-empty");
        y.require_non_empty("Model::train: y must be non-empty");

        if (batch_size > X.get_rows()) {
            throw runtime_error("Model::train: batch_size cannot exceed number of samples aka X.get_rows()");
        }

        const size_t samples = X.get_rows();

        Matrix Y;
        if (y.get_rows() == 1 && y.get_cols() == samples) {
            Y = y.transpose();
        } else {
            y.require_rows(samples, "Model::train: non-row-vector y must have same number of rows as X");
            Y = y;
        }

        accuracy->init(Y);

        const size_t steps = calc_steps(samples, batch_size);

        for (size_t epoch = 1; epoch <= epochs; ++epoch) {
            cout << "epoch: " << epoch << '\n';

            loss->new_pass();
            accuracy->new_pass();

            for (size_t step = 0; step < steps; ++step) {
                Matrix batch_X, batch_y;
                slice_batch(X, &Y, step, batch_size, batch_X, &batch_y);

                forward_pass(batch_X, true);

                double reg_loss = 0.0;
                const double data_loss = loss->calculate(output, batch_y, reg_loss, trainable_layers);
                const double total_loss = data_loss + reg_loss;

                last_predictions = last_activation->predictions(output);
                const double acc = accuracy->calculate(last_predictions, batch_y);

                const Matrix* dvalues = nullptr;
                auto iter = layers.rbegin();

                if (use_combined_softmax_ce) {
                    combined_softmax_ce.backward(output, batch_y);
                    dvalues = &combined_softmax_ce.get_dinputs();

                    Layer* last_layer = iter->get();
                    last_layer->backward(*dvalues, false);
                    dvalues = &last_layer->get_dinputs();
                    ++iter;
                } else {
                    loss->backward(output, batch_y);
                    dvalues = &loss->get_dinputs();
                }

                for (; iter != layers.rend(); ++iter) {
                    (*iter)->backward(*dvalues, true);
                    dvalues = &(*iter)->get_dinputs();
                }

                optimizer->pre_update_params();
                for (LayerDense* dense : trainable_layers) {
                    optimizer->update_params(*dense);
                }
                optimizer->post_update_params();

                if (print_every != 0 && ((step % print_every) == 0 || step == steps - 1)) {
                    cout << "  step: " << step
                         << ", accuracy: " << acc
                         << ", loss: " << total_loss
                         << " (data loss: " << data_loss
                         << ", regularization loss: " << reg_loss
                         << ")"
                         << ", learning rate: " << optimizer->get_current_learning_rate()
                         << '\n';
                }
            }

            double epoch_reg_loss = 0.0;
            const double epoch_data_loss = loss->calculate_accumulated(epoch_reg_loss, trainable_layers);
            const double epoch_loss = epoch_data_loss + epoch_reg_loss;
            const double epoch_accuracy = accuracy->calculate_accumulated();

            cout << "training - accuracy: " << epoch_accuracy
                 << ", loss: " << epoch_loss
                 << " (data loss: " << epoch_data_loss
                 << ", regularization loss: " << epoch_reg_loss
                 << ")"
                 << ", learning rate: " << optimizer->get_current_learning_rate()
                 << '\n';

            if (X_val && y_val) evaluate(*X_val, *y_val, batch_size, false);
        }
    }

    void evaluate(const Matrix& X, const Matrix& y, size_t batch_size = 0, bool reinit = true)
    {
        if (!finalized) finalize();

        if (!loss || !accuracy) {
            throw runtime_error("Model::evaluate: loss and accuracy must be set");
        }

        X.require_non_empty("Model::evaluate: X must be non-empty");
        y.require_non_empty("Model::evaluate: y must be non-empty");

        if (batch_size > X.get_rows()) {
            throw runtime_error("Model::evaluate: batch_size cannot exceed number of samples aka X.get_rows()");
        }

        const size_t samples = X.get_rows();

        Matrix Y;
        if (y.get_rows() == 1 && y.get_cols() == samples) {
            Y = y.transpose();
        } else {
            y.require_rows(samples, "Model::evaluate: non-row-vector y must have same number of rows as X");
            Y = y;
        }

        if (reinit) accuracy->init(Y);

        loss->new_pass();
        accuracy->new_pass();

        const size_t steps = calc_steps(samples, batch_size);

        for (size_t step = 0; step < steps; ++step) {
            Matrix batch_X, batch_y;
            slice_batch(X, &Y, step, batch_size, batch_X, &batch_y);

            forward_pass(batch_X, false);
            loss->calculate(output, batch_y);
            last_predictions = last_activation->predictions(output);
            accuracy->calculate(last_predictions, batch_y);
        }

        const double val_loss = loss->calculate_accumulated();
        const double val_acc = accuracy->calculate_accumulated();

        cout << "validation - accuracy: " << val_acc
             << ", loss: " << val_loss
             << '\n';
    }

    Matrix predict(const Matrix& X, size_t batch_size = 0)
    {
        if (!finalized) finalize();

        X.require_non_empty("Model::predict: X must be non-empty");

        if (batch_size > X.get_rows()) {
            throw runtime_error("Model::predict: batch_size cannot exceed number of samples aka X.get_rows()");
        }

        const size_t samples = X.get_rows();
        const size_t steps = calc_steps(samples, batch_size);

        Matrix predictions;
        bool preds_init = false;

        for (size_t step = 0; step < steps; ++step) {
            Matrix batch_X;
            slice_batch(X, nullptr, step, batch_size, batch_X, nullptr);

            forward_pass(batch_X, false);
            last_predictions = last_activation->predictions(output);

            last_predictions.require_rows(batch_X.get_rows(), "Model::predict: predictions must have same rows as batch_X");

            if (!preds_init) {
                predictions.assign(samples, last_predictions.get_cols());
                preds_init = true;
            } else {
                last_predictions.require_cols(predictions.get_cols(), "Model::predict: predictions cols mismatch across batches");
            }

            const size_t start = step * batch_size;
            for (size_t i = 0; i < last_predictions.get_rows(); ++i) {
                for (size_t j = 0; j < last_predictions.get_cols(); ++j) {
                    predictions(start + i, j) = last_predictions(i, j);
                }
            }
        }

        return predictions;
    }

    void save_params(const string& path)
    {
        if (path.empty()) throw runtime_error("Model::save_params: invalid path");
        if (!finalized) finalize();

        require_dense_params_initialized();

        ofstream f(path, ios::binary);
        if (!f) throw runtime_error("Model::save_params: failed to open file");

        constexpr char magic[11] = {'M','O','D','E','L','P','A','R','A','M','S'};
        write_exact(f, magic, sizeof(magic));

        const uint64_t dense_count = static_cast<uint64_t>(dense_layers_count());
        write_trivial(f, dense_count);

        for (const LayerDense* layer : trainable_layers) {
            write_matrix(f, layer->weights);
            write_matrix(f, layer->biases);
        }
    }

    void load_params(const string& path)
    {
        if (path.empty()) throw runtime_error("Model::load_params: invalid path");
        if (!finalized) finalize();

        ifstream f(path, ios::binary);
        if (!f) throw runtime_error("Model::load_params: failed to open file");

        constexpr char magic_expected[11] = {'M','O','D','E','L','P','A','R','A','M','S'};
        char magic[11]{};
        read_exact(f, magic, sizeof(magic));

        if (memcmp(magic, magic_expected, sizeof(magic)) != 0) {
            throw runtime_error("Model::load_params: invalid file (magic mismatch)");
        }

        const uint64_t dense_count_file = read_trivial<uint64_t>(f);
        if (dense_count_file > static_cast<uint64_t>(numeric_limits<size_t>::max())) {
            throw runtime_error("Model::load_params: dense layer count exceeds size_t range");
        }
        const size_t dense_count_model = dense_layers_count();
        if (dense_count_file != static_cast<uint64_t>(dense_count_model)) {
            throw runtime_error("Model::load_params: dense layer count mismatch");
        }

        vector<Matrix> params;
        multiplication_overflow_check(dense_count_model, 2, "Model::load_params: params reserve size overflow");
        params.reserve(dense_count_model * 2);

        for (size_t i = 0; i < dense_count_model; ++i) {
            params.push_back(read_matrix(f)); // weights
            params.push_back(read_matrix(f)); // biases
        }

        set_params(params);
    }

    void save(const string& path)
    {
        if (path.empty()) throw runtime_error("Model::save: invalid path");
        if (!finalized) finalize();

        require_dense_params_initialized();

        ofstream f(path, ios::binary);
        if (!f) throw runtime_error("Model::save: failed to open file");

        constexpr char magic[11] = {'M','O','D','E','L','O','B','J','E','C','T'};
        write_exact(f, magic, sizeof(magic));

        const uint64_t layers_count = static_cast<uint64_t>(architecture.size());
        write_trivial(f, layers_count);

        for (const LayerDesc& d : architecture) {
            write_trivial<uint8_t>(f, static_cast<uint8_t>(d.kind));

            if (d.kind == LayerKind::Dense) {
                write_trivial<uint64_t>(f, static_cast<uint64_t>(d.dense.n_neurons));
                write_trivial<uint8_t>(f, static_cast<uint8_t>(d.dense.activation));
                write_trivial<double>(f, d.dense.w_l1);
                write_trivial<double>(f, d.dense.w_l2);
                write_trivial<double>(f, d.dense.b_l1);
                write_trivial<double>(f, d.dense.b_l2);
            } else if (d.kind == LayerKind::Dropout) {
                write_trivial<double>(f, d.dropout.dropout_rate);
            } else {
                throw runtime_error("Model::save: unsupported layer kind");
            }
        }

        const uint64_t dense_count = static_cast<uint64_t>(dense_layers_count());
        write_trivial(f, dense_count);

        for (const LayerDense* layer : trainable_layers) {
            write_matrix(f, layer->weights);
            write_matrix(f, layer->biases);
        }
    }

    static Model load(const string& path)
    {
        if (path.empty()) throw runtime_error("Model::load: invalid path");

        ifstream f(path, ios::binary);
        if (!f) throw runtime_error("Model::load: failed to open file");

        constexpr char magic_expected[11] = {'M','O','D','E','L','O','B','J','E','C','T'};
        char magic[11]{};
        read_exact(f, magic, sizeof(magic));

        if (memcmp(magic, magic_expected, sizeof(magic)) != 0) {
            throw runtime_error("Model::load: invalid file (magic mismatch)");
        }

        const uint64_t layers_count = read_trivial<uint64_t>(f);
        if (layers_count == 0) throw runtime_error("Model::load: empty architecture");
        if (layers_count > static_cast<uint64_t>(numeric_limits<size_t>::max())) {
            throw runtime_error("Model::load: layer count exceeds size_t range");
        }

        vector<LayerDesc> arch;
        arch.reserve(static_cast<size_t>(layers_count));

        for (uint64_t i = 0; i < layers_count; ++i) {
            LayerDesc d;

            const uint8_t kind_u8 = read_trivial<uint8_t>(f);
            if (kind_u8 > static_cast<uint8_t>(LayerKind::Dropout)) {
                throw runtime_error("Model serialization: invalid LayerKind");
            }
            d.kind = static_cast<LayerKind>(kind_u8);

            if (d.kind == LayerKind::Dense) {
                const uint64_t n_neurons_u64 = read_trivial<uint64_t>(f);
                if (n_neurons_u64 == 0) {
                    throw runtime_error("Model::load: dense layer n_neurons must be > 0");
                }
                if (n_neurons_u64 > static_cast<uint64_t>(numeric_limits<size_t>::max())) {
                    throw runtime_error("Model::load: dense layer n_neurons exceeds size_t range");
                }
                d.dense.n_neurons = static_cast<size_t>(n_neurons_u64);

                const uint8_t activation_u8 = read_trivial<uint8_t>(f);
                if (activation_u8 > static_cast<uint8_t>(ActivationKind::Linear)) {
                    throw runtime_error("Model serialization: invalid ActivationKind");
                }
                d.dense.activation = static_cast<ActivationKind>(activation_u8);

                d.dense.w_l1 = read_trivial<double>(f);
                d.dense.w_l2 = read_trivial<double>(f);
                d.dense.b_l1 = read_trivial<double>(f);
                d.dense.b_l2 = read_trivial<double>(f);
            } else if (d.kind == LayerKind::Dropout) {
                d.dropout.dropout_rate = read_trivial<double>(f);
            } else {
                throw runtime_error("Model::load: unsupported layer kind");
            }

            arch.push_back(d);
        }

        const uint64_t dense_count_file = read_trivial<uint64_t>(f);
        if (dense_count_file > static_cast<uint64_t>(numeric_limits<size_t>::max())) {
            throw runtime_error("Model::load: dense layer count exceeds size_t range");
        }

        size_t dense_count_arch = 0;
        for (const LayerDesc& d : arch) {
            if (d.kind == LayerKind::Dense) ++dense_count_arch;
        }
        if (dense_count_file != static_cast<uint64_t>(dense_count_arch)) {
            throw runtime_error("Model::load: dense layer count mismatch");
        }

        vector<Matrix> params;
        multiplication_overflow_check(dense_count_arch, 2, "Model::load: params reserve size overflow");
        params.reserve(dense_count_arch * 2);

        for (size_t i = 0; i < dense_count_arch; ++i) {
            params.push_back(read_matrix(f));
            params.push_back(read_matrix(f));
        }

        Model m;
        m.set_architecture(arch); // private, ok here
        m.set_params(params);     // finalizes
        return m;
    }

    vector<Matrix> get_params() const
    {
        vector<Matrix> params;

        for (const auto& p : layers) {
            const auto* dense = dynamic_cast<const LayerDense*>(p.get());
            if (!dense) continue;

            if (dense->weights.is_empty() || dense->biases.is_empty()) {
                throw runtime_error("Model::get_params: weights/biases not initialized (run predict/train or load params first)");
            }

            params.push_back(dense->weights);
            params.push_back(dense->biases);
        }

        return params;
    }

    void set_params(const vector<Matrix>& params)
    {
        if (!finalized) finalize();

        size_t dense_count = dense_layers_count();
        if (dense_count != trainable_layers.size()) {
            throw runtime_error("Model::set_params: internal error (dense_count != trainable_layers.size())");
        }

        if (params.size() != dense_count * 2) {
            throw runtime_error("Model::set_params: params size must be 2 * number_of_dense_layers");
        }

        size_t dense_idx = 0;
        for (const LayerDesc& d : architecture) {
            if (d.kind != LayerKind::Dense) continue;

            const Matrix& w = params[dense_idx * 2];
            const Matrix& b = params[dense_idx * 2 + 1];

            w.require_non_empty("Model::set_params: weights must be non-empty");
            b.require_non_empty("Model::set_params: biases must be non-empty");

            if (w.get_cols() != d.dense.n_neurons) {
                throw runtime_error("Model::set_params: weights cols must match architecture n_neurons");
            }
            b.require_shape(1, d.dense.n_neurons,
                            "Model::set_params: biases must have shape (1, n_neurons)");

            if (dense_idx > 0) {
                const Matrix& prev_w = params[(dense_idx - 1) * 2];
                prev_w.require_cols(w.get_rows(),
                                    "Model::set_params: consecutive dense weight matrices must be shape-compatible");
            }

            LayerDense* layer = trainable_layers[dense_idx];
            if (!layer) {
                throw runtime_error("Model::set_params: internal error (null trainable layer)");
            }

            layer->weights = w;
            layer->biases  = b;

            ++dense_idx;
        }
    }

    const Matrix& get_output() const { return output; }
    const Matrix& get_last_predictions() const { return last_predictions; }

private:
    enum class LayerKind { Dense, Dropout };
    enum class ActivationKind { ReLU, Softmax, Sigmoid, Linear };

    struct DenseDesc
    {
        size_t n_neurons = 0;
        ActivationKind activation = ActivationKind::ReLU;
        double w_l1 = 0.0, w_l2 = 0.0, b_l1 = 0.0, b_l2 = 0.0;
    };

    struct DropoutDesc
    {
        double dropout_rate = 0.0;
    };

    struct LayerDesc
    {
        LayerKind kind = LayerKind::Dense;
        DenseDesc dense;
        DropoutDesc dropout;
    };

    LayerInput input_layer;
    vector<unique_ptr<Layer>> layers;
    vector<LayerDense*> trainable_layers;

    vector<LayerDesc> architecture;

    Loss* loss = nullptr;
    bool loss_is_cce = false;
    Accuracy* accuracy = nullptr;
    Optimizer* optimizer = nullptr;

    bool finalized = false;

    const Activation* last_activation = nullptr;
    ActivationKind last_activation_kind = ActivationKind::Linear;
    bool use_combined_softmax_ce = false;

    Matrix output;
    Matrix last_predictions;

    ActivationSoftmaxLossCategoricalCrossEntropy combined_softmax_ce;

    static ActivationKind string_to_activation_kind(string name)
    {
        transform(name.begin(), name.end(), name.begin(),
                  [](unsigned char c) { return static_cast<char>(tolower(c)); });

        if (name == "relu") return ActivationKind::ReLU;
        if (name == "softmax") return ActivationKind::Softmax;
        if (name == "sigmoid") return ActivationKind::Sigmoid;
        if (name == "linear") return ActivationKind::Linear;

        throw runtime_error("Model unknown activation. use relu, softmax, sigmoid or linear");
    }

    static string activation_kind_to_string(ActivationKind k)
    {
        switch (k) {
        case ActivationKind::ReLU: return "relu";
        case ActivationKind::Softmax: return "softmax";
        case ActivationKind::Sigmoid: return "sigmoid";
        case ActivationKind::Linear: return "linear";
        default: throw runtime_error("Model: unknown ActivationKind");
        }
    }

    void require_dense_params_initialized() const
    {
        for (const LayerDense* layer : trainable_layers) {
            if (!layer) throw runtime_error("Model: internal error (null dense layer)");
            if (layer->weights.is_empty() || layer->biases.is_empty()) {
                throw runtime_error("Model: dense weights/biases are not initialized (run train/predict or load params first)");
            }
        }
    }

    size_t dense_layers_count() const
    {
        size_t n = 0;
        for (const LayerDesc& d : architecture) {
            if (d.kind == LayerKind::Dense) ++n;
        }
        return n;
    }

    void invalidate_compiled_state()
    {
        finalized = false;

        trainable_layers.clear();

        last_activation = nullptr;
        last_activation_kind = ActivationKind::Linear;
        use_combined_softmax_ce = false;

        output = Matrix();
        last_predictions = Matrix();
    }

    LayerDense& add_dense_to_model(size_t n_neurons, ActivationKind activation_kind,
                          double w_l1 = 0.0, double w_l2 = 0.0, double b_l1 = 0.0, double b_l2 = 0.0)
    {
        LayerDesc d;
        d.kind = LayerKind::Dense;
        d.dense.n_neurons = n_neurons;
        d.dense.activation = activation_kind;
        d.dense.w_l1 = w_l1;
        d.dense.w_l2 = w_l2;
        d.dense.b_l1 = b_l1;
        d.dense.b_l2 = b_l2;

        architecture.push_back(d);

        auto layer = make_unique<LayerDense>(
            n_neurons, activation_kind_to_string(activation_kind),
            w_l1, w_l2, b_l1, b_l2);

        LayerDense& ref = *layer;
        layers.push_back(move(layer));

        invalidate_compiled_state();

        return ref;
    }

    void set_architecture(const vector<LayerDesc>& arch)
    {
        if (arch.empty()) {
            throw runtime_error("Model::set_architecture: empty architecture");
        }
        if (arch.back().kind == LayerKind::Dropout) {
            throw runtime_error("Model::set_architecture: final layer cannot be dropout");
        }

        vector<unique_ptr<Layer>> new_layers;
        new_layers.reserve(arch.size());

        for (const LayerDesc& d : arch) {
            if (d.kind == LayerKind::Dense) {
                new_layers.push_back(make_unique<LayerDense>(
                    d.dense.n_neurons,
                    activation_kind_to_string(d.dense.activation),
                    d.dense.w_l1, d.dense.w_l2, d.dense.b_l1, d.dense.b_l2));
            } else if (d.kind == LayerKind::Dropout) {
                new_layers.push_back(make_unique<LayerDropout>(d.dropout.dropout_rate));
            } else {
                throw runtime_error("Model::set_architecture: unsupported layer kind");
            }
        }

        clear();
        layers.swap(new_layers);
        architecture = arch;
    }

    void rebuild_caches()
    {
        if (layers.size() != architecture.size()) {
            throw runtime_error("Model: internal error (layers/spec size mismatch)");
        }

        trainable_layers.clear();

        for (auto& p : layers) {
            if (auto* dense = dynamic_cast<LayerDense*>(p.get())) {
                trainable_layers.push_back(dense);
            }
        }

        last_activation = layers.back()->get_activation();
        if (!last_activation) {
            throw runtime_error("Model: last layer has null activation");
        }

        const LayerDesc& last = architecture.back();
        if (last.kind != LayerKind::Dense) {
            throw runtime_error("Model: internal error (last layer kind is not Dense)");
        }
        last_activation_kind = last.dense.activation;
    }

    void update_combined_flag()
    {
        use_combined_softmax_ce = false;
        if (!loss) return;
        use_combined_softmax_ce = (last_activation_kind == ActivationKind::Softmax) && loss_is_cce;
    }

    static size_t calc_steps(size_t samples, size_t batch_size)
    {
        if (batch_size == 0) return 1;

        size_t steps = samples / batch_size;
        if (steps * batch_size < samples) ++steps;
        return steps;
    }

    static void slice_batch(const Matrix& X, const Matrix* Y, size_t step,
                            size_t batch_size, Matrix& batch_X, Matrix* batch_Y)
    {
        if (batch_size == 0) {
            batch_X = X;
            if (Y && batch_Y) *batch_Y = *Y;
            return;
        }

        const size_t start = step * batch_size;
        const size_t end = min(start + batch_size, X.get_rows());
        batch_X = X.slice_rows(start, end);
        if (Y && batch_Y) *batch_Y = Y->slice_rows(start, end);
    }

    void forward_pass(const Matrix& X, const bool training)
    {
        input_layer.forward(X);

        const Matrix* current = &input_layer.get_output();
        for (auto& p : layers) {
            p->forward(*current, training);
            current = &p->get_output();
        }

        output = *current;
    }

    static void write_exact(ostream& os, const void* data, size_t bytes)
    {
        os.write(static_cast<const char*>(data), static_cast<streamsize>(bytes));
        if (!os) throw runtime_error("Model serialization: write failed");
    }

    template<class T>
    static void write_trivial(ostream& os, const T& v)
    {
        static_assert(is_trivially_copyable_v<T>);
        write_exact(os, &v, sizeof(T));
    }

    static void read_exact(istream& is, void* data, size_t bytes)
    {
        is.read(static_cast<char*>(data), static_cast<streamsize>(bytes));
        if (!is) throw runtime_error("Model serialization: read failed");
    }
    
    template<class T>
    static T read_trivial(istream& is)
    {
        static_assert(is_trivially_copyable_v<T>);
        T v{};
        read_exact(is, &v, sizeof(T));
        return v;
    }

    static void write_matrix(ostream& os, const Matrix& m)
    {
        const uint64_t r = static_cast<uint64_t>(m.get_rows());
        const uint64_t c = static_cast<uint64_t>(m.get_cols());
        write_trivial(os, r);
        write_trivial(os, c);

        for (size_t i = 0; i < m.get_rows(); ++i) {
            for (size_t j = 0; j < m.get_cols(); ++j) {
                const double x = m(i, j);
                write_trivial(os, x);
            }
        }
    }

    static Matrix read_matrix(istream& is)
    {
        const uint64_t r64 = read_trivial<uint64_t>(is);
        const uint64_t c64 = read_trivial<uint64_t>(is);

        if (r64 > static_cast<uint64_t>(numeric_limits<size_t>::max()) ||
            c64 > static_cast<uint64_t>(numeric_limits<size_t>::max())) {
            throw runtime_error("Model serialization: matrix shape exceeds size_t range");
        }

        const size_t r = static_cast<size_t>(r64);
        const size_t c = static_cast<size_t>(c64);

        Matrix m;
        m.assign(r, c);

        for (size_t i = 0; i < r; ++i) {
            for (size_t j = 0; j < c; ++j) {
                m(i, j) = read_trivial<double>(is);
            }
        }
        return m;
    }
};

#ifndef NNFS_NO_MAIN
int main()
{
    try {
        const string params_path = "model_params.bin";
        const string model_path  = "full_model.bin";

        // setting up rng
        set_global_seed(0);
        set_thread_stream_id(0);

        // testing scatter plotting with generated data
        Matrix X_generated;
        Matrix y_generated;
        generate_spiral_data(1000, 3, X_generated, y_generated);

        const string path = "plot.png";
        scatter_plot(path, X_generated, y_generated);
        cout << "data plotting complete. generated data plot is in file: "
             << path << "\n";

        // loading real dataset
        Matrix X, y, X_test, y_test;
        fashion_mnist_create(X, y, X_test, y_test);

        // building architecture inside model
        Model model;
        
        model.add_dense(128, "relu", 0.0, 5e-4, 0.0, 5e-4);
        model.add_dense(128, "relu", 0.0, 5e-4, 0.0, 5e-4);
        model.add_dropout(0.1);
        model.add_dense(10, "softmax");

        // attaching training objects
        LossCategoricalCrossEntropy loss;
        AccuracyCategorical accuracy;
        OptimizerAdam optimizer(1e-3);

        model.configure(loss, accuracy, optimizer);

        // training and evaluating model
        model.train(X, y, 1, 128, 100, &X_test, &y_test);

        // printing some predictions
        const Matrix preds_model = model.predict(X_test, 128);
        const size_t pred_rows = preds_model.get_rows();

        cout << "predict output shape: " << pred_rows
            << "x" << preds_model.get_cols() << '\n';

        const vector<string> label_names = {
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
        };

        cout << "sample predictions:\n";
        const size_t samples_to_check = min<size_t>(20, pred_rows);
        for (size_t i = 0; i < samples_to_check; ++i) {
            const size_t pred_id = (pred_rows == 1) ? preds_model.as_size_t(0, i) : preds_model.as_size_t(i, 0);
            const size_t true_id = (y_test.get_rows() == 1) ? y_test.as_size_t(0, i) : y_test.as_size_t(i, 0);

            cout << "  sample " << i
                << " - predicted: " << label_names[pred_id] << " (" << pred_id << ")"
                << ", actual: " << label_names[true_id] << " (" << true_id << ")"
                << '\n';
        }

        model.evaluate(X_test, y_test, 128);

        // setting and getting parameters
        vector<Matrix> params = model.get_params();
        model.set_params(params);

        Matrix preds_model_set_params = model.predict(X_test, 128);
        cout << "\ndifference in predictions after saving the model parameters to a variable and setting them back: "
             << Matrix::max_absolute_difference(preds_model, preds_model_set_params) << '\n'
             << "evaluation after setting model parameters:\n";

        model.evaluate(X_test, y_test, 128);

        // saving and loading parameters to/from file
        model.save_params(params_path);

        Model m2;
        m2.add_dense(128, "relu", 0.0, 5e-4, 0.0, 5e-4);
        m2.add_dense(128, "relu", 0.0, 5e-4, 0.0, 5e-4);
        m2.add_dropout(0.1);
        m2.add_dense(10, "softmax");

        m2.load_params(params_path);

        const Matrix preds_model_loaded_params = m2.predict(X_test, 128);
        cout << "\ndifference in predictions after saving the model parameters to a file and loading them: "
             << Matrix::max_absolute_difference(preds_model, preds_model_loaded_params) << '\n'
             << "evaluation after loading in model parameters:\n";

        m2.configure(loss, accuracy);
        m2.evaluate(X_test, y_test, 128);

        // saving and loading full model to/from file
        model.save(model_path);
        Model m3 = Model::load(model_path);

        const Matrix preds_model_loaded_full = m3.predict(X_test, 128);
        cout << "\ndifference in predictions after saving the full model to a file and loading it: "
             << Matrix::max_absolute_difference(preds_model, preds_model_loaded_full) << '\n'
             << "evaluation after loading in full model:\n";

        m3.configure(loss, accuracy);
        m3.evaluate(X_test, y_test, 128);

        return 0;
    }
    catch (const exception& e) {
        cout << "ERROR: " << e.what() << '\n';

        return 1;
    }
}
#endif
