#include "tests/test_common.hpp"

TEST_CASE("Optimizer validates parameters and updates learning rate")
{
    struct DummyOpt : Optimizer {
        using Optimizer::Optimizer;
        void update_params(LayerDense&) override {}
    };

    CHECK_THROWS_WITH_AS(DummyOpt(0.0, 0.0),
                         "Optimizer: learning_rate must be positive",
                         runtime_error);
    CHECK_THROWS_WITH_AS(DummyOpt(1.0, -0.1),
                         "Optimizer: decay must be non-negative",
                         runtime_error);

    DummyOpt opt(1.0, 0.5);
    CHECK(opt.get_learning_rate() == doctest::Approx(1.0));
    CHECK(opt.get_decay() == doctest::Approx(0.5));
    CHECK(opt.get_current_learning_rate() == doctest::Approx(1.0));
    CHECK(opt.get_iterations() == 0);

    opt.pre_update_params();
    CHECK(opt.get_learning_rate() == doctest::Approx(1.0));
    CHECK(opt.get_decay() == doctest::Approx(0.5));
    CHECK(opt.get_current_learning_rate() == doctest::Approx(1.0));
    opt.post_update_params();
    CHECK(opt.get_iterations() == 1);

    opt.pre_update_params();
    CHECK(opt.get_current_learning_rate() == doctest::Approx(1.0 / (1.0 + 0.5 * 1.0)));
    opt.post_update_params();
    CHECK(opt.get_iterations() == 2);

    LayerDense dummy(1, "linear");
    opt.update_params(dummy);
}

TEST_CASE("OptimizerSGD updates weights with and without momentum")
{
    LayerDense layer = make_dense_with_grads(1.0, 0.5, 2.0, 1.0);

    OptimizerSGD sgd_no_momentum(0.1, 0.0, 0.0);
    sgd_no_momentum.update_params(layer);
    CHECK(layer.weights(0, 0) == doctest::Approx(0.8));
    CHECK(layer.biases(0, 0) == doctest::Approx(0.4));

    LayerDense layer_m = make_dense_with_grads(1.0, 0.0, 2.0, 1.0);
    OptimizerSGD sgd_momentum(0.1, 0.0, 0.9);
    sgd_momentum.update_params(layer_m);

    CHECK(layer_m.weight_momentums.get_rows() == 1);
    CHECK(layer_m.weight_momentums.get_cols() == 1);
    CHECK(layer_m.weight_momentums(0, 0) == doctest::Approx(-0.2));
    CHECK(layer_m.bias_momentums(0, 0) == doctest::Approx(-0.1));
    CHECK(layer_m.weights(0, 0) == doctest::Approx(0.8));
    CHECK(layer_m.biases(0, 0) == doctest::Approx(-0.1));
}

TEST_CASE("OptimizerSGD validates parameters and shapes")
{
    CHECK_THROWS_WITH_AS(OptimizerSGD(1.0, 0.0, -0.1),
                         "OptimizerSGD: momentum must be non-negative",
                         runtime_error);

    OptimizerSGD sgd(1.0, 0.0, 0.0);
    LayerDense layer(1, "linear");
    layer.weights.assign(0, 0);
    CHECK_THROWS_WITH_AS(sgd.update_params(layer),
                         "OptimizerSGD::update_params: layer.weights must be non-empty",
                         runtime_error);

    layer.weights.assign(1, 1);
    layer.biases.assign(0, 0);
    CHECK_THROWS_WITH_AS(sgd.update_params(layer),
                         "OptimizerSGD::update_params: layer.biases must be non-empty",
                         runtime_error);

    layer.biases.assign(1, 2);
    CHECK_THROWS_WITH_AS(sgd.update_params(layer),
                         "OptimizerSGD::update_params: biases must have shape (1, n_neurons)",
                         runtime_error);

    layer.biases.assign(1, 1);
    layer.weights.assign(2, 1);
    layer.forward(Matrix(1, 2, 1.0));
    layer.backward(Matrix(1, 1, 1.0));
    layer.weights.assign(1, 1);
    CHECK_THROWS_WITH_AS(sgd.update_params(layer),
                         "OptimizerSGD::update_params: dweights must match weights shape",
                         runtime_error);
}

TEST_CASE("OptimizerAdagrad accumulates cache and scales updates")
{
    LayerDense layer = make_dense_with_grads(1.0, 0.1, 2.0, 3.0);
    OptimizerAdagrad adagrad(1.0, 0.0, 1e-7);
    adagrad.update_params(layer);

    CHECK(layer.weight_cache(0, 0) == doctest::Approx(4.0));
    CHECK(layer.bias_cache(0, 0) == doctest::Approx(9.0));
    CHECK(layer.weights(0, 0) == doctest::Approx(0.0).epsilon(1e-6));
    CHECK(layer.biases(0, 0) == doctest::Approx(-0.9).epsilon(1e-6));
}

TEST_CASE("OptimizerAdagrad validates parameters and shapes")
{
    CHECK_THROWS_WITH_AS(OptimizerAdagrad(1.0, 0.0, 0.0),
                         "OptimizerAdagrad: epsilon must be positive",
                         runtime_error);

    OptimizerAdagrad opt(1.0, 0.0, 1e-7);
    LayerDense layer(1, "linear");
    layer.weights.assign(0, 0);
    CHECK_THROWS_WITH_AS(opt.update_params(layer),
                         "OptimizerAdagrad::update_params: layer.weights must be non-empty",
                         runtime_error);

    layer.weights.assign(1, 1);
    layer.biases.assign(0, 0);
    CHECK_THROWS_WITH_AS(opt.update_params(layer),
                         "OptimizerAdagrad::update_params: layer.biases must be non-empty",
                         runtime_error);

    layer.biases.assign(1, 2);
    CHECK_THROWS_WITH_AS(opt.update_params(layer),
                         "OptimizerAdagrad::update_params: biases must have shape (1, n_neurons)",
                         runtime_error);

    layer.biases.assign(1, 1);
    layer.weights.assign(2, 1);
    layer.forward(Matrix(1, 2, 1.0));
    layer.backward(Matrix(1, 1, 1.0));
    layer.weights.assign(1, 1);
    CHECK_THROWS_WITH_AS(opt.update_params(layer),
                         "OptimizerAdagrad::update_params: dweights must match weights shape",
                         runtime_error);
}

TEST_CASE("OptimizerRMSprop applies exponential cache and updates")
{
    LayerDense layer = make_dense_with_grads(1.0, 0.5, 2.0, 1.0);
    OptimizerRMSprop rms(1.0, 0.0, 1e-7, 0.5);
    rms.update_params(layer);

    CHECK(layer.weight_cache(0, 0) == doctest::Approx(2.0));
    CHECK(layer.bias_cache(0, 0) == doctest::Approx(0.5));
    CHECK(layer.weights(0, 0) == doctest::Approx(-0.41421356).epsilon(1e-6));
    CHECK(layer.biases(0, 0) == doctest::Approx(-0.91421356).epsilon(1e-6));
}

TEST_CASE("OptimizerRMSprop validates parameters and shapes")
{
    CHECK_THROWS_WITH_AS(OptimizerRMSprop(1.0, 0.0, 0.0, 0.9),
                         "OptimizerRMSprop: epsilon must be positive",
                         runtime_error);
    CHECK_THROWS_WITH_AS(OptimizerRMSprop(1.0, 0.0, 1e-7, 1.0),
                         "OptimizerRMSprop: rho must be in (0, 1)",
                         runtime_error);

    OptimizerRMSprop opt(1.0, 0.0, 1e-7, 0.9);
    LayerDense layer(1, "linear");
    layer.weights.assign(0, 0);
    CHECK_THROWS_WITH_AS(opt.update_params(layer),
                         "OptimizerRMSprop::update_params: layer.weights must be non-empty",
                         runtime_error);

    layer.weights.assign(1, 1);
    layer.biases.assign(0, 0);
    CHECK_THROWS_WITH_AS(opt.update_params(layer),
                         "OptimizerRMSprop::update_params: layer.biases must be non-empty",
                         runtime_error);

    layer.biases.assign(1, 2);
    CHECK_THROWS_WITH_AS(opt.update_params(layer),
                         "OptimizerRMSprop::update_params: biases must have shape (1, n_neurons)",
                         runtime_error);

    layer.biases.assign(1, 1);
    layer.weights.assign(2, 1);
    layer.forward(Matrix(1, 2, 1.0));
    layer.backward(Matrix(1, 1, 1.0));
    layer.weights.assign(1, 1);
    CHECK_THROWS_WITH_AS(opt.update_params(layer),
                         "OptimizerRMSprop::update_params: dweights must match weights shape",
                         runtime_error);
}

TEST_CASE("OptimizerAdam validates parameters and bias correction")
{
    CHECK_THROWS_WITH_AS(OptimizerAdam(1.0, 0.0, 0.0),
                         "OptimizerAdam: epsilon must be positive",
                         runtime_error);
    CHECK_THROWS_WITH_AS(OptimizerAdam(1.0, 0.0, 1e-7, 0.0, 0.9),
                         "OptimizerAdam: beta1 must be in (0, 1)",
                         runtime_error);
    CHECK_THROWS_WITH_AS(OptimizerAdam(1.0, 0.0, 1e-7, 0.9, 1.0),
                         "OptimizerAdam: beta2 must be in (0, 1)",
                         runtime_error);

    LayerDense layer = make_dense_with_grads(1.0, 0.0, 2.0, 2.0);
    OptimizerAdam adam(0.1, 0.0, 1e-7, 0.5, 0.5);
    CHECK_THROWS_WITH_AS(adam.update_params(layer),
                         "OptimizerAdam::update_params: numerical issue in bias correction (pre_update_params not called?)",
                         runtime_error);
}

TEST_CASE("OptimizerAdam updates momentums and caches with bias correction")
{
    LayerDense layer = make_dense_with_grads(1.0, 0.0, 2.0, 2.0);
    OptimizerAdam adam(0.1, 0.0, 1e-7, 0.5, 0.5);
    adam.pre_update_params();
    adam.update_params(layer);

    CHECK(layer.weight_momentums(0, 0) == doctest::Approx(1.0));
    CHECK(layer.weight_cache(0, 0) == doctest::Approx(2.0));
    CHECK(layer.bias_momentums(0, 0) == doctest::Approx(1.0));
    CHECK(layer.bias_cache(0, 0) == doctest::Approx(2.0));
    CHECK(layer.weights(0, 0) == doctest::Approx(0.9).epsilon(1e-6));
    CHECK(layer.biases(0, 0) == doctest::Approx(-0.1).epsilon(1e-6));
}

TEST_CASE("OptimizerAdam validates shapes")
{
    OptimizerAdam opt(1.0, 0.0, 1e-7);
    LayerDense layer(1, "linear");
    layer.weights.assign(0, 0);
    CHECK_THROWS_WITH_AS(opt.update_params(layer),
                         "OptimizerAdam::update_params: layer.weights must be non-empty",
                         runtime_error);

    layer.weights.assign(1, 1);
    layer.biases.assign(0, 0);
    CHECK_THROWS_WITH_AS(opt.update_params(layer),
                         "OptimizerAdam::update_params: layer.biases must be non-empty",
                         runtime_error);

    layer.biases.assign(1, 2);
    CHECK_THROWS_WITH_AS(opt.update_params(layer),
                         "OptimizerAdam::update_params: biases must have shape (1, n_neurons)",
                         runtime_error);

    layer.biases.assign(1, 1);
    layer.weights.assign(2, 1);
    layer.forward(Matrix(1, 2, 1.0));
    layer.backward(Matrix(1, 1, 1.0));
    layer.weights.assign(1, 1);
    CHECK_THROWS_WITH_AS(opt.update_params(layer),
                         "OptimizerAdam::update_params: dweights must match weights shape",
                         runtime_error);
}

