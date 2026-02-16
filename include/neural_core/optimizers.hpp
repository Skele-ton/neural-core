#pragma once

#include <cstddef>

#include "neural_core/layers.hpp"

class Optimizer
{
public:
    Optimizer(double learning_rate, double decay);
    virtual ~Optimizer() = default;

    virtual void pre_update_params();
    void post_update_params();

    virtual void update_params(LayerDense& layer) = 0;

    double get_learning_rate() const;
    double get_current_learning_rate() const;
    double get_decay() const;
    double get_iterations() const;

protected:
    double learning_rate;
    double current_learning_rate;
    double decay;
    std::size_t iterations;
};

class OptimizerSGD : public Optimizer
{
public:
    OptimizerSGD(double learning_rate = 1.0, double decay = 0.0, double momentum = 0.0);

    void update_params(LayerDense& layer) override;

private:
    double momentum;
};

class OptimizerAdagrad : public Optimizer
{
public:
    OptimizerAdagrad(double learning_rate = 1.0, double decay = 0.0, double epsilon = 1e-7);

    void update_params(LayerDense& layer) override;

private:
    double epsilon;
};

class OptimizerRMSprop : public Optimizer
{
public:
    OptimizerRMSprop(double learning_rate = 0.001, double decay = 0.0, double epsilon = 1e-7, double rho = 0.9);

    void update_params(LayerDense& layer) override;

private:
    double epsilon;
    double rho;
};

class OptimizerAdam : public Optimizer
{
public:
    OptimizerAdam(double learning_rate = 0.001, double decay = 0.0, double epsilon = 1e-7,
                  double beta1 = 0.9, double beta2 = 0.999);

    void pre_update_params() override;
    void update_params(LayerDense& layer) override;

private:
    double epsilon;
    double beta1;
    double beta2;
    double beta1_power;
    double beta2_power;
};
