#include "neural_core/activations.hpp"

#include <cmath>
#include <stdexcept>

using std::exp;
using std::isfinite;
using std::max;
using std::runtime_error;
using std::size_t;

const Matrix& Activation::get_inputs() const
{
    return inputs;
}

const Matrix& Activation::get_output() const
{
    return output;
}

const Matrix& Activation::get_dinputs() const
{
    return dinputs;
}

void ActivationReLU::forward(const Matrix& inputs_batch)
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

void ActivationReLU::backward(const Matrix& dvalues)
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

Matrix ActivationReLU::predictions(const Matrix& outputs) const
{
    return outputs;
}

void ActivationSoftmax::forward(const Matrix& inputs_batch)
{
    inputs_batch.require_non_empty("ActivationSoftmax::forward: inputs must be non-empty");

    inputs = inputs_batch;
    output.assign(inputs.get_rows(), inputs.get_cols());

    for (size_t i = 0; i < inputs.get_rows(); ++i) {
        double max_val = inputs(i, 0);
        for (size_t j = 1; j < inputs.get_cols(); ++j) {
            const double v = inputs(i, j);
            if (v > max_val) max_val = v;
        }

        double sum = 0.0;
        for (size_t j = 0; j < inputs.get_cols(); ++j) {
            const double e = exp(inputs(i, j) - max_val);
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

void ActivationSoftmax::backward(const Matrix& dvalues)
{
    dvalues.require_non_empty("ActivationSoftmax::backward: dvalues must be non-empty");
    dvalues.require_shape(output.get_rows(), output.get_cols(),
        "ActivationSoftmax::backward: dvalues shape mismatch");

    dinputs.assign(dvalues.get_rows(), dvalues.get_cols());

    const size_t samples = dvalues.get_rows();
    const size_t classes = dvalues.get_cols();

    for (size_t i = 0; i < samples; ++i) {
        double alpha = 0.0;
        for (size_t k = 0; k < classes; ++k) {
            alpha += output(i, k) * dvalues(i, k);
        }
        for (size_t j = 0; j < classes; ++j) {
            dinputs(i, j) = output(i, j) * (dvalues(i, j) - alpha);
        }
    }
}

Matrix ActivationSoftmax::predictions(const Matrix& outputs) const
{
    outputs.require_non_empty("ActivationSoftmax::predictions: outputs must be non-empty");
    if (outputs.get_cols() < 2) {
        throw runtime_error("ActivationSoftmax::predictions: computation of softmax predictions requires outputs.get_cols() >= 2");
    }

    return outputs.argmax();
}

void ActivationSigmoid::forward(const Matrix& inputs_batch)
{
    inputs_batch.require_non_empty("ActivationSigmoid::forward: inputs must be non-empty");

    inputs = inputs_batch;
    output.assign(inputs.get_rows(), inputs.get_cols());
    for (size_t i = 0; i < inputs.get_rows(); ++i) {
        for (size_t j = 0; j < inputs.get_cols(); ++j) {
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

void ActivationSigmoid::backward(const Matrix& dvalues)
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

Matrix ActivationSigmoid::predictions(const Matrix& outputs) const
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

void ActivationLinear::forward(const Matrix& inputs_batch)
{
    inputs_batch.require_non_empty("ActivationLinear::forward: inputs must be non-empty");

    inputs = inputs_batch;
    output = inputs_batch;
}

void ActivationLinear::backward(const Matrix& dvalues)
{
    dvalues.require_non_empty("ActivationLinear::backward: dvalues must be non-empty");
    dvalues.require_shape(inputs.get_rows(), inputs.get_cols(),
        "ActivationLinear::backward: dvalues shape mismatch");

    dinputs = dvalues;
}

Matrix ActivationLinear::predictions(const Matrix& outputs) const
{
    return outputs;
}
