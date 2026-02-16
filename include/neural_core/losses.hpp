#pragma once

#include <cstddef>
#include <vector>

#include "neural_core/layers.hpp"

class Loss
{
public:
    virtual ~Loss() = default;

    double calculate(const Matrix& output, const Matrix& y_true);
    double calculate(const Matrix& output, const Matrix& y_true, double& out_regularization_loss,
                     const std::vector<LayerDense*>& layers);

    double calculate_accumulated() const;
    double calculate_accumulated(double& out_regularization_loss, const std::vector<LayerDense*>& layers) const;

    void new_pass();

    virtual void backward(const Matrix& y_pred, const Matrix& y_true) = 0;

    const Matrix& get_dinputs() const;

protected:
    Matrix dinputs;

    virtual Matrix forward(const Matrix& output, const Matrix& y_true) const = 0;

    static double clamp(double p);

private:
    double accumulated_sum = 0.0;
    std::size_t accumulated_count = 0;

    static double regularization_loss(const LayerDense& layer);
    static double regularization_loss(const std::vector<LayerDense*>& layers);
};

class LossCategoricalCrossEntropy : public Loss
{
public:
    void backward(const Matrix& y_pred, const Matrix& y_true) override;

protected:
    Matrix forward(const Matrix& y_pred, const Matrix& y_true) const override;
};

class LossBinaryCrossentropy : public Loss
{
public:
    void backward(const Matrix& y_pred, const Matrix& y_true) override;

protected:
    Matrix forward(const Matrix& y_pred, const Matrix& y_true) const override;
};

class LossMeanSquaredError : public Loss
{
public:
    void backward(const Matrix& y_pred, const Matrix& y_true) override;

private:
    Matrix forward(const Matrix& y_pred, const Matrix& y_true) const override;
};

class LossMeanAbsoluteError : public Loss
{
public:
    void backward(const Matrix& y_pred, const Matrix& y_true) override;

protected:
    Matrix forward(const Matrix& y_pred, const Matrix& y_true) const override;
};

class ActivationSoftmaxLossCategoricalCrossEntropy
{
public:
    void backward(const Matrix& y_pred, const Matrix& y_true);

    const Matrix& get_dinputs() const;

private:
    Matrix dinputs;
};
