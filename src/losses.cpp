#include "neural_core/losses.hpp"

#include <cmath>
#include <stdexcept>

using std::abs;
using std::log;
using std::runtime_error;
using std::size_t;
using std::vector;

double Loss::calculate(const Matrix& output, const Matrix& y_true)
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

double Loss::calculate(const Matrix& output, const Matrix& y_true, double& out_regularization_loss,
                       const vector<LayerDense*>& layers)
{
    const double data_loss = calculate(output, y_true);
    out_regularization_loss = regularization_loss(layers);
    return data_loss;
}

double Loss::calculate_accumulated() const
{
    if (accumulated_count == 0) {
        throw runtime_error("Loss::calculate_accumulated: accumulated_count must be > 0");
    }
    return accumulated_sum / static_cast<double>(accumulated_count);
}

double Loss::calculate_accumulated(double& out_regularization_loss, const vector<LayerDense*>& layers) const
{
    const double data_loss = calculate_accumulated();
    out_regularization_loss = regularization_loss(layers);
    return data_loss;
}

void Loss::new_pass()
{
    accumulated_sum = 0.0;
    accumulated_count = 0;
}

const Matrix& Loss::get_dinputs() const
{
    return dinputs;
}

double Loss::clamp(double p)
{
    constexpr double eps = 1e-7;
    if (p < eps) return eps;
    if (p > 1.0 - eps) return 1.0 - eps;
    return p;
}

double Loss::regularization_loss(const LayerDense& layer)
{
    double regularization = 0.0;

    const bool has_w_l1 = layer.get_weight_regularizer_l1() != 0.0;
    const bool has_w_l2 = layer.get_weight_regularizer_l2() != 0.0;

    if (has_w_l1 || has_w_l2) {
        layer.weights.require_non_empty("Loss::regularization_loss: weights must be non-empty");

        double sum_abs = 0.0;
        double sum_sq = 0.0;

        for (double weight : layer.weights.get_data()) {
            if (has_w_l1) sum_abs += abs(weight);
            if (has_w_l2) sum_sq += weight * weight;
        }

        regularization += layer.get_weight_regularizer_l1() * sum_abs + layer.get_weight_regularizer_l2() * sum_sq;
    }

    const bool has_b_l1 = layer.get_bias_regularizer_l1() != 0.0;
    const bool has_b_l2 = layer.get_bias_regularizer_l2() != 0.0;

    if (has_b_l1 || has_b_l2) {
        layer.biases.require_non_empty("Loss::regularization_loss: biases must be non-empty");
        layer.weights.require_non_empty("Loss::regularization_loss: weights must be non-empty");

        layer.biases.require_shape(1, layer.weights.get_cols(),
            "Loss::regularization_loss: biases must have shape (1, n_neurons)");

        double sum_abs = 0.0;
        double sum_sq = 0.0;

        for (double bias : layer.biases.get_data()) {
            if (has_b_l1) sum_abs += abs(bias);
            if (has_b_l2) sum_sq += bias * bias;
        }

        regularization += layer.get_bias_regularizer_l1() * sum_abs + layer.get_bias_regularizer_l2() * sum_sq;
    }

    return regularization;
}

double Loss::regularization_loss(const vector<LayerDense*>& layers)
{
    double regularization = 0.0;
    for (const LayerDense* layer : layers) {
        if (!layer) continue;
        regularization += regularization_loss(*layer);
    }
    return regularization;
}

void LossCategoricalCrossEntropy::backward(const Matrix& y_pred, const Matrix& y_true)
{
    y_pred.require_non_empty("LossCategoricalCrossEntropy::backward: y_pred must be non-empty");
    y_true.require_non_empty("LossCategoricalCrossEntropy::backward: y_true must be non-empty");

    if (y_pred.get_cols() < 2) {
        throw runtime_error("LossCategoricalCrossEntropy::backward: y_pred.get_cols() must be >= 2");
    }

    const size_t samples = y_pred.get_rows();
    const size_t classes = y_pred.get_cols();

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

Matrix LossCategoricalCrossEntropy::forward(const Matrix& y_pred, const Matrix& y_true) const
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

        const double confidence = clamp(y_pred(i, class_idx));
        sample_losses(0, i) = -log(confidence);
    }

    return sample_losses;
}

void LossBinaryCrossentropy::backward(const Matrix& y_pred, const Matrix& y_true)
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
            const double pred = clamp(y_pred(i, j));
            const double truth = y_true(i, j);

            dinputs(i, j) = -(truth / pred - (1.0 - truth) / (1.0 - pred)) / static_cast<double>(outputs);
        }
    }

    dinputs.scale_by_scalar(samples);
}

Matrix LossBinaryCrossentropy::forward(const Matrix& y_pred, const Matrix& y_true) const
{
    y_true.require_shape(y_pred.get_rows(), y_pred.get_cols(),
        "LossBinaryCrossentropy::forward: y_pred and y_true must have the same shape");

    const size_t samples = y_pred.get_rows();
    const size_t outputs = y_pred.get_cols();

    Matrix sample_losses(1, samples, 0.0);

    for (size_t i = 0; i < samples; ++i) {
        double loss_sum = 0.0;
        for (size_t j = 0; j < outputs; ++j) {
            const double pred = clamp(y_pred(i, j));
            const double truth = y_true(i, j);

            loss_sum += -(truth * log(pred) + (1.0 - truth) * log(1.0 - pred));
        }

        sample_losses(0, i) = loss_sum / static_cast<double>(outputs);
    }

    return sample_losses;
}

void LossMeanSquaredError::backward(const Matrix& y_pred, const Matrix& y_true)
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

Matrix LossMeanSquaredError::forward(const Matrix& y_pred, const Matrix& y_true) const
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

void LossMeanAbsoluteError::backward(const Matrix& y_pred, const Matrix& y_true)
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
            if (diff > 0.0) grad = -1.0;
            else if (diff < 0.0) grad = 1.0;
            dinputs(i, j) = grad / static_cast<double>(outputs);
        }
    }

    dinputs.scale_by_scalar(samples);
}

Matrix LossMeanAbsoluteError::forward(const Matrix& y_pred, const Matrix& y_true) const
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

void ActivationSoftmaxLossCategoricalCrossEntropy::backward(const Matrix& y_pred, const Matrix& y_true)
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
        const size_t class_idx = y_true_sparse.as_size_t(i, 0);
        if (class_idx >= classes) {
            throw runtime_error("ActivationSoftmaxLossCategoricalCrossEntropy::backward: class index out of range");
        }
        dinputs(i, class_idx) -= 1.0;
    }

    dinputs.scale_by_scalar(samples);
}

const Matrix& ActivationSoftmaxLossCategoricalCrossEntropy::get_dinputs() const
{
    return dinputs;
}
