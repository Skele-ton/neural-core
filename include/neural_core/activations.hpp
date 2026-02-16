#pragma once

#include "neural_core/matrix.hpp"

class Activation
{
public:
    virtual ~Activation() = default;

    virtual void forward(const Matrix& inputs_batch) = 0;
    virtual void backward(const Matrix& dvalues) = 0;
    virtual Matrix predictions(const Matrix& outputs) const = 0;

    const Matrix& get_inputs() const;
    const Matrix& get_output() const;
    const Matrix& get_dinputs() const;

protected:
    Matrix inputs;
    Matrix output;
    Matrix dinputs;
};

class ActivationReLU : public Activation
{
public:
    void forward(const Matrix& inputs_batch) override;
    void backward(const Matrix& dvalues) override;
    Matrix predictions(const Matrix& outputs) const override;
};

class ActivationSoftmax : public Activation
{
public:
    void forward(const Matrix& inputs_batch) override;
    void backward(const Matrix& dvalues) override;
    Matrix predictions(const Matrix& outputs) const override;
};

class ActivationSigmoid : public Activation
{
public:
    void forward(const Matrix& inputs_batch) override;
    void backward(const Matrix& dvalues) override;
    Matrix predictions(const Matrix& outputs) const override;
};

class ActivationLinear : public Activation
{
public:
    void forward(const Matrix& inputs_batch) override;
    void backward(const Matrix& dvalues) override;
    Matrix predictions(const Matrix& outputs) const override;
};
