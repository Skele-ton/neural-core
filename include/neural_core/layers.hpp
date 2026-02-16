#pragma once

#include <cstddef>
#include <string>

#include "neural_core/activations.hpp"

class Layer
{
public:
    virtual ~Layer() = default;

    virtual void forward(const Matrix& inputs_batch, bool training) = 0;
    virtual void backward(const Matrix& dvalues, bool include_activation) = 0;

    const Matrix& get_inputs() const;
    const Matrix& get_output() const;
    const Matrix& get_dinputs() const;
    const Activation* get_activation() const;

protected:
    Matrix inputs;
    Matrix output;
    Matrix dinputs;
    Activation* activation = nullptr;

    void forward(const Matrix& inputs_batch);
    void backward(const Matrix& dvalues);
};

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

    LayerDense(std::size_t n_neurons, const std::string& activation_name,
               double weight_regularizer_l1 = 0.0,
               double weight_regularizer_l2 = 0.0,
               double bias_regularizer_l1 = 0.0,
               double bias_regularizer_l2 = 0.0);

    void forward(const Matrix& inputs_batch, bool training) override;
    void backward(const Matrix& dvalues, bool include_activation) override;

    const Matrix& get_dweights() const;
    const Matrix& get_dbiases() const;

    double get_weight_regularizer_l1() const;
    double get_weight_regularizer_l2() const;
    double get_bias_regularizer_l1() const;
    double get_bias_regularizer_l2() const;

    double get_starting_n_neurons() const;

private:
    Matrix dweights;
    Matrix dbiases;

    double weight_regularizer_l1;
    double weight_regularizer_l2;
    double bias_regularizer_l1;
    double bias_regularizer_l2;

    std::size_t starting_n_neurons;

    ActivationReLU activation_relu;
    ActivationSoftmax activation_softmax;
    ActivationSigmoid activation_sigmoid;
    ActivationLinear activation_linear;
};

class LayerDropout : public Layer
{
public:
    using Layer::forward;
    using Layer::backward;

    explicit LayerDropout(double dropout_rate);

    void forward(const Matrix& inputs_batch, bool training) override;
    void backward(const Matrix& dvalues, bool include_activation) override;

    double get_keep_rate() const;

private:
    double keep_rate;
    Matrix scaled_binary_mask;
    ActivationLinear activation_linear;
};

class LayerInput
{
public:
    void forward(const Matrix& inputs_batch);

    const Matrix& get_output() const;

private:
    Matrix output;
};
