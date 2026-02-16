#include "tests/test_common.hpp"

TEST_CASE("LayerDense constructor validates inputs")
{
    CHECK_THROWS_WITH_AS(LayerDense(0, "relu"),
                         "LayerDense: n_neurons must be > 0",
                         runtime_error);
    CHECK_THROWS_WITH_AS(LayerDense(1, "relu", -0.1),
                         "LayerDense: regularizers must be non-negative",
                         runtime_error);

    LayerDense ok(2, "relu", 0.1, 0.2, 0.3, 0.4);
    CHECK(ok.get_starting_n_neurons() == doctest::Approx(2.0));
    CHECK(ok.get_weight_regularizer_l1() == doctest::Approx(0.1));
    CHECK(ok.get_weight_regularizer_l2() == doctest::Approx(0.2));
    CHECK(ok.get_bias_regularizer_l1() == doctest::Approx(0.3));
    CHECK(ok.get_bias_regularizer_l2() == doctest::Approx(0.4));
    CHECK(dynamic_cast<const ActivationReLU*>(ok.get_activation()) != nullptr);
}

TEST_CASE("LayerDense supports multiple activation names")
{
    LayerDense softmax_layer(1, "softmax");
    CHECK(dynamic_cast<const ActivationSoftmax*>(softmax_layer.get_activation()) != nullptr);

    LayerDense sigmoid_layer(1, "sigmoid");
    CHECK(dynamic_cast<const ActivationSigmoid*>(sigmoid_layer.get_activation()) != nullptr);

    LayerDense linear_layer(1, "linear");
    CHECK(dynamic_cast<const ActivationLinear*>(linear_layer.get_activation()) != nullptr);

    CHECK_THROWS_WITH_AS(LayerDense(1, "unknown"),
                         "LayerDense: unknown activation. use relu, softmax, sigmoid or linear",
                         runtime_error);
}

TEST_CASE("LayerDense forward matches known example and stores inputs")
{
    Matrix inputs(3, 4);
    inputs(0, 0) = 1.0;  inputs(0, 1) = 2.0;  inputs(0, 2) = 3.0;  inputs(0, 3) = 2.5;
    inputs(1, 0) = 2.0;  inputs(1, 1) = 5.0;  inputs(1, 2) = -1.0; inputs(1, 3) = 2.0;
    inputs(2, 0) = -1.5; inputs(2, 1) = 2.7;  inputs(2, 2) = 3.3;  inputs(2, 3) = -0.8;

    LayerDense layer(3, "linear");
    layer.weights.assign(4, 3);
    layer.weights(0, 0) = 0.2;  layer.weights(0, 1) = 0.5;   layer.weights(0, 2) = -0.26;
    layer.weights(1, 0) = 0.8;  layer.weights(1, 1) = -0.91; layer.weights(1, 2) = -0.27;
    layer.weights(2, 0) = -0.5; layer.weights(2, 1) = 0.26;  layer.weights(2, 2) = 0.17;
    layer.weights(3, 0) = 1.0;  layer.weights(3, 1) = -0.5;  layer.weights(3, 2) = 0.87;
    layer.biases.assign(1, 3);
    layer.biases(0, 0) = 2.0; layer.biases(0, 1) = 3.0; layer.biases(0, 2) = 0.5;

    layer.forward(inputs);
    CHECK(layer.get_output().get_rows() == 3);
    CHECK(layer.get_output().get_cols() == 3);

    CHECK(layer.get_output()(0, 0) == doctest::Approx(4.8));
    CHECK(layer.get_output()(0, 1) == doctest::Approx(1.21));
    CHECK(layer.get_output()(0, 2) == doctest::Approx(2.385));
    CHECK(layer.get_output()(1, 0) == doctest::Approx(8.9));
    CHECK(layer.get_output()(1, 1) == doctest::Approx(-1.81));
    CHECK(layer.get_output()(1, 2) == doctest::Approx(0.2));
    CHECK(layer.get_output()(2, 0) == doctest::Approx(1.41));
    CHECK(layer.get_output()(2, 1) == doctest::Approx(1.051));
    CHECK(layer.get_output()(2, 2) == doctest::Approx(0.026));

    CHECK(layer.get_inputs().get_rows() == inputs.get_rows());
    CHECK(layer.get_inputs().get_cols() == inputs.get_cols());
}

TEST_CASE("LayerDense forward validates inputs and shapes")
{
    Matrix inputs(1, 2, 1.0);
    Matrix empty;
    LayerDense layer(1, "linear");
    CHECK_THROWS_WITH_AS(layer.forward(empty),
                         "LayerDense::forward: inputs must be non-empty",
                         runtime_error);

    Matrix bad_inputs(1, 3, 1.0);
    layer.weights.assign(2, 1);
    CHECK_THROWS_WITH_AS(layer.forward(bad_inputs),
                         "LayerDense::forward: inputs.get_cols() must match weights.get_rows()",
                         runtime_error);

    layer.biases.assign(1, 2);
    CHECK_THROWS_WITH_AS(layer.forward(inputs),
                         "LayerDense::forward: biases must be shape (1, n_neurons)",
                         runtime_error);
}

TEST_CASE("LayerDense backward computes gradients and validates shapes")
{
    Matrix inputs(2, 2);
    inputs(0, 0) = 1.0; inputs(0, 1) = 2.0;
    inputs(1, 0) = 3.0; inputs(1, 1) = 4.0;

    LayerDense layer(2, "linear");
    layer.weights.assign(2, 2);
    layer.weights(0, 0) = 1.0; layer.weights(0, 1) = 0.0;
    layer.weights(1, 0) = 0.0; layer.weights(1, 1) = 1.0;
    layer.biases.assign(1, 2);
    layer.biases(0, 0) = 0.0; layer.biases(0, 1) = 0.0;

    layer.forward(inputs);

    Matrix dvalues(2, 2);
    dvalues(0, 0) = 1.0; dvalues(0, 1) = 2.0;
    dvalues(1, 0) = 3.0; dvalues(1, 1) = 4.0;

    layer.backward(dvalues);

    CHECK(layer.get_dweights()(0, 0) == doctest::Approx(10.0));
    CHECK(layer.get_dweights()(0, 1) == doctest::Approx(14.0));
    CHECK(layer.get_dweights()(1, 0) == doctest::Approx(14.0));
    CHECK(layer.get_dweights()(1, 1) == doctest::Approx(20.0));

    CHECK(layer.get_dbiases()(0, 0) == doctest::Approx(4.0));
    CHECK(layer.get_dbiases()(0, 1) == doctest::Approx(6.0));

    CHECK(layer.get_dinputs()(0, 0) == doctest::Approx(1.0));
    CHECK(layer.get_dinputs()(0, 1) == doctest::Approx(2.0));
    CHECK(layer.get_dinputs()(1, 0) == doctest::Approx(3.0));
    CHECK(layer.get_dinputs()(1, 1) == doctest::Approx(4.0));

    Matrix empty;
    CHECK_THROWS_WITH_AS(layer.backward(empty),
                         "LayerDense::backward: dvalues must be non-empty",
                         runtime_error);

    Matrix bad_dvalues(1, 1, 0.0);
    CHECK_THROWS_WITH_AS(layer.backward(bad_dvalues),
                         "LayerDense::backward: dvalues shape mismatch",
                         runtime_error);
}

TEST_CASE("LayerDense backward applies L1 and L2 regularization")
{
    LayerDense layer(2, "linear", 0.3, 0.7, 0.5, 0.9);
    layer.weights.assign(2, 2);
    layer.weights(0, 0) = -1.0; layer.weights(0, 1) = 2.0;
    layer.weights(1, 0) = -3.0; layer.weights(1, 1) = 4.0;
    layer.biases.assign(1, 2);
    layer.biases(0, 0) = -5.0; layer.biases(0, 1) = 6.0;

    Matrix inputs(1, 2);
    inputs(0, 0) = 1.0; inputs(0, 1) = 1.0;
    layer.forward(inputs);

    Matrix dvalues(1, 2);
    dvalues(0, 0) = 0.1; dvalues(0, 1) = -0.2;
    layer.backward(dvalues);

    CHECK(layer.get_dweights()(0, 0) == doctest::Approx(-1.6));
    CHECK(layer.get_dweights()(0, 1) == doctest::Approx(2.9));
    CHECK(layer.get_dweights()(1, 0) == doctest::Approx(-4.4));
    CHECK(layer.get_dweights()(1, 1) == doctest::Approx(5.7));

    CHECK(layer.get_dbiases()(0, 0) == doctest::Approx(-9.4));
    CHECK(layer.get_dbiases()(0, 1) == doctest::Approx(11.1));
}

TEST_CASE("LayerDropout constructor validates rate")
{
    CHECK_THROWS_WITH_AS(LayerDropout(1.0),
                         "LayerDropout: dropout_rate must be in [0,1)",
                         runtime_error);
    CHECK_THROWS_WITH_AS(LayerDropout(-0.1),
                         "LayerDropout: dropout_rate must be in [0,1)",
                         runtime_error);

    LayerDropout ok(0.2);
    CHECK(ok.get_keep_rate() == doctest::Approx(0.8));
}

TEST_CASE("LayerDropout forward scales activations with training flag")
{
    Matrix inputs(1, 3);
    inputs(0, 0) = 2.0;
    inputs(0, 1) = -3.0;
    inputs(0, 2) = 4.0;

    LayerDropout harsh_dropout(0.999); // keep rate = 0.001

    harsh_dropout.forward(inputs, false);
    CHECK(harsh_dropout.get_output()(0, 0) == doctest::Approx(inputs(0, 0)));
    CHECK(harsh_dropout.get_output()(0, 1) == doctest::Approx(inputs(0, 1)));
    CHECK(harsh_dropout.get_output()(0, 2) == doctest::Approx(inputs(0, 2)));
    
    const double keep_rate = 0.8;
    LayerDropout regular_dropout(1.0 - keep_rate);
    regular_dropout.forward(inputs);
    for (size_t i = 0; i < 3; ++i) {
        const double mask = regular_dropout.get_output()(0, i) / inputs(0, i);
        CHECK((mask == doctest::Approx(0.0) || mask == doctest::Approx(1.0 / keep_rate)));
    }
}

TEST_CASE("LayerDropout forward/backward validates shapes")
{
    LayerDropout dropout(0.2);
    Matrix empty;
    CHECK_THROWS_WITH_AS(dropout.forward(empty),
                         "LayerDropout::forward: inputs must be non-empty",
                         runtime_error);

    Matrix inputs(1, 2);
    inputs(0, 0) = 5.0;
    inputs(0, 1) = -2.0;
    dropout.forward(inputs);

    Matrix dvalues(1, 2);
    dvalues(0, 0) = 3.0; dvalues(0, 1) = 4.0;
    dropout.backward(dvalues);

    CHECK(dropout.get_dinputs().get_rows() == 1);
    CHECK(dropout.get_dinputs().get_cols() == 2);

    const double mask0 = dropout.get_output()(0, 0) / inputs(0, 0);
    const double mask1 = dropout.get_output()(0, 1) / inputs(0, 1);

    CHECK(dropout.get_dinputs()(0, 0) == doctest::Approx(dvalues(0, 0) * mask0));
    CHECK(dropout.get_dinputs()(0, 1) == doctest::Approx(dvalues(0, 1) * mask1));

    Matrix bad_dvalues(2, 2, 0.0);
    CHECK_THROWS_WITH_AS(dropout.backward(bad_dvalues),
                         "LayerDropout::backward: dvalues shape mismatch",
                         runtime_error);

    Matrix empty_dvalues;
    CHECK_THROWS_WITH_AS(dropout.backward(empty_dvalues),
                         "LayerDropout::backward: dvalues must be non-empty",
                         runtime_error);
}

TEST_CASE("LayerInput forwards inputs and validates emptiness")
{
    Matrix inputs(2, 2);
    inputs(0, 0) = 1.0; inputs(0, 1) = 2.0;
    inputs(1, 0) = 3.0; inputs(1, 1) = 4.0;

    LayerInput input_layer;
    input_layer.forward(inputs);
    CHECK(input_layer.get_output().get_rows() == 2);
    CHECK(input_layer.get_output().get_cols() == 2);
    CHECK(input_layer.get_output()(0, 1) == doctest::Approx(2.0));
    CHECK(input_layer.get_output()(1, 1) == doctest::Approx(4.0));

    Matrix empty;
    CHECK_THROWS_WITH_AS(input_layer.forward(empty),
                         "LayerInput::forward: inputs must be non-empty",
                         runtime_error);
}

