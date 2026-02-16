#include "neural_core/layers.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>

#include "neural_core/rng.hpp"

using std::runtime_error;
using std::size_t;
using std::string;
using std::tolower;
using std::transform;

const Matrix& Layer::get_inputs() const
{
    return inputs;
}

const Matrix& Layer::get_output() const
{
    return output;
}

const Matrix& Layer::get_dinputs() const
{
    return dinputs;
}

const Activation* Layer::get_activation() const
{
    return activation;
}

void Layer::forward(const Matrix& inputs_batch)
{
    forward(inputs_batch, true);
}

void Layer::backward(const Matrix& dvalues)
{
    backward(dvalues, true);
}

LayerDense::LayerDense(size_t n_neurons, const string& activation_name,
                       double weight_regularizer_l1,
                       double weight_regularizer_l2,
                       double bias_regularizer_l1,
                       double bias_regularizer_l2)
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

void LayerDense::forward(const Matrix& inputs_batch, bool)
{
    inputs = inputs_batch;

    inputs.require_non_empty("LayerDense::forward: inputs must be non-empty");

    if (weights.is_empty()) {
        const size_t n_inputs = inputs.get_cols();

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

void LayerDense::backward(const Matrix& dvalues, bool include_activation)
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

const Matrix& LayerDense::get_dweights() const
{
    return dweights;
}

const Matrix& LayerDense::get_dbiases() const
{
    return dbiases;
}

double LayerDense::get_weight_regularizer_l1() const
{
    return weight_regularizer_l1;
}

double LayerDense::get_weight_regularizer_l2() const
{
    return weight_regularizer_l2;
}

double LayerDense::get_bias_regularizer_l1() const
{
    return bias_regularizer_l1;
}

double LayerDense::get_bias_regularizer_l2() const
{
    return bias_regularizer_l2;
}

double LayerDense::get_starting_n_neurons() const
{
    return starting_n_neurons;
}

LayerDropout::LayerDropout(double dropout_rate)
    : keep_rate(1.0 - dropout_rate)
{
    if (keep_rate <= 0.0 || keep_rate > 1.0) {
        throw runtime_error("LayerDropout: dropout_rate must be in [0,1)");
    }

    activation = &activation_linear;
}

void LayerDropout::forward(const Matrix& inputs_batch, bool training)
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

void LayerDropout::backward(const Matrix& dvalues, bool)
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

double LayerDropout::get_keep_rate() const
{
    return keep_rate;
}

void LayerInput::forward(const Matrix& inputs_batch)
{
    inputs_batch.require_non_empty("LayerInput::forward: inputs must be non-empty");
    output = inputs_batch;
}

const Matrix& LayerInput::get_output() const
{
    return output;
}
