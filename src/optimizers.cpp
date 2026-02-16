#include "neural_core/optimizers.hpp"

#include <cmath>
#include <stdexcept>

using std::runtime_error;
using std::size_t;
using std::sqrt;

Optimizer::Optimizer(double learning_rate, double decay)
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

void Optimizer::pre_update_params()
{
    current_learning_rate = learning_rate;
    if (decay != 0.0) {
        current_learning_rate = learning_rate / (1.0 + decay * static_cast<double>(iterations));
    }
}

void Optimizer::post_update_params()
{
    ++iterations;
}

double Optimizer::get_learning_rate() const
{
    return learning_rate;
}

double Optimizer::get_current_learning_rate() const
{
    return current_learning_rate;
}

double Optimizer::get_decay() const
{
    return decay;
}

double Optimizer::get_iterations() const
{
    return iterations;
}

OptimizerSGD::OptimizerSGD(double learning_rate, double decay, double momentum)
    : Optimizer(learning_rate, decay),
      momentum(momentum)
{
    if (momentum < 0.0) {
        throw runtime_error("OptimizerSGD: momentum must be non-negative");
    }
}

void OptimizerSGD::update_params(LayerDense& layer)
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

OptimizerAdagrad::OptimizerAdagrad(double learning_rate, double decay, double epsilon)
    : Optimizer(learning_rate, decay),
      epsilon(epsilon)
{
    if (epsilon <= 0.0) {
        throw runtime_error("OptimizerAdagrad: epsilon must be positive");
    }
}

void OptimizerAdagrad::update_params(LayerDense& layer)
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

OptimizerRMSprop::OptimizerRMSprop(double learning_rate, double decay, double epsilon, double rho)
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

void OptimizerRMSprop::update_params(LayerDense& layer)
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

OptimizerAdam::OptimizerAdam(double learning_rate, double decay, double epsilon,
                             double beta1, double beta2)
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

void OptimizerAdam::pre_update_params()
{
    Optimizer::pre_update_params();

    beta1_power *= beta1;
    beta2_power *= beta2;
}

void OptimizerAdam::update_params(LayerDense& layer)
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
