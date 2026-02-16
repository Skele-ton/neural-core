#include "tests/test_common.hpp"

TEST_CASE("ActivationReLU forward and backward")
{
    Matrix inputs(2, 3);
    inputs(0, 0) = -1.0; inputs(0, 1) = 0.0; inputs(0, 2) = 2.5;
    inputs(1, 0) = 3.0;  inputs(1, 1) = -0.1; inputs(1, 2) = 0.0;

    ActivationReLU activation;
    activation.forward(inputs);
    CHECK(activation.get_inputs()(1, 0) == doctest::Approx(3.0));
    CHECK(activation.get_output()(0, 0) == doctest::Approx(0.0));
    CHECK(activation.get_output()(0, 2) == doctest::Approx(2.5));
    CHECK(activation.get_output()(1, 0) == doctest::Approx(3.0));

    Matrix dvalues(2, 2);
    dvalues(0, 0) = 5.0; dvalues(0, 1) = 6.0;
    dvalues(1, 0) = 7.0; dvalues(1, 1) = 8.0;

    Matrix inputs2(2, 2);
    inputs2(0, 0) = -1.0; inputs2(0, 1) = 1.0;
    inputs2(1, 0) = 0.0;  inputs2(1, 1) = 2.0;
    activation.forward(inputs2);
    activation.backward(dvalues);
    CHECK(activation.get_dinputs()(0, 0) == doctest::Approx(0.0));
    CHECK(activation.get_dinputs()(0, 1) == doctest::Approx(6.0));
    CHECK(activation.get_dinputs()(1, 0) == doctest::Approx(0.0));
    CHECK(activation.get_dinputs()(1, 1) == doctest::Approx(8.0));

    Matrix bad;
    CHECK_THROWS_WITH_AS(activation.forward(bad),
                         "ActivationReLU::forward: inputs must be non-empty",
                         runtime_error);

    Matrix bad_dvalues(1, 1, 0.0);
    activation.forward(Matrix(1, 2, 1.0));
    CHECK_THROWS_WITH_AS(activation.backward(bad_dvalues),
                         "ActivationReLU::backward: dvalues shape mismatch",
                         runtime_error);
}

TEST_CASE("ActivationReLU predictions passthrough")
{
    Matrix inputs(1, 2);
    inputs(0, 0) = -1.0;
    inputs(0, 1) = 2.0;

    ActivationReLU activation;
    activation.forward(inputs);
    CHECK(activation.get_output()(0, 0) == doctest::Approx(0.0));
    CHECK(activation.get_output()(0, 1) == doctest::Approx(2.0));

    Matrix preds = activation.predictions(activation.get_output());
    CHECK(preds.get_rows() == activation.get_output().get_rows());
    CHECK(preds.get_cols() == activation.get_output().get_cols());
    CHECK(preds(0, 1) == doctest::Approx(2.0));
}

TEST_CASE("ActivationSoftmax computes probabilities and predictions")
{
    Matrix inputs(2, 3);
    inputs(0, 0) = 0.0; inputs(0, 1) = 1.0; inputs(0, 2) = 2.0;
    inputs(1, 0) = 0.0; inputs(1, 1) = 0.0; inputs(1, 2) = 0.0;

    ActivationSoftmax activation;
    activation.forward(inputs);

    CHECK(activation.get_output()(0, 0) == doctest::Approx(0.0900306));
    CHECK(activation.get_output()(0, 1) == doctest::Approx(0.2447285));
    CHECK(activation.get_output()(0, 2) == doctest::Approx(0.6652409));
    CHECK(activation.get_output()(1, 0) == doctest::Approx(1.0 / 3.0));

    Matrix preds = activation.predictions(activation.get_output());
    CHECK(preds.get_rows() == 2);
    CHECK(preds.get_cols() == 1);
    CHECK(preds(0, 0) == doctest::Approx(2.0));
    CHECK(preds(1, 0) == doctest::Approx(0.0));

    Matrix bad;
    CHECK_THROWS_WITH_AS(activation.forward(bad),
                         "ActivationSoftmax::forward: inputs must be non-empty",
                         runtime_error);
}

TEST_CASE("ActivationSoftmax predictions require at least 2 columns")
{
    ActivationSoftmax activation;
    Matrix outputs(2, 1, 0.5);
    CHECK_THROWS_WITH_AS(activation.predictions(outputs),
                         "ActivationSoftmax::predictions: computation of softmax predictions requires outputs.get_cols() >= 2",
                         runtime_error);
}

TEST_CASE("ActivationSoftmax backward validates shapes and sums")
{
    Matrix inputs(1, 2);
    inputs(0, 0) = 0.0; inputs(0, 1) = 0.0;

    ActivationSoftmax activation;
    activation.forward(inputs);

    Matrix dvalues(1, 2);
    dvalues(0, 0) = 1.0; dvalues(0, 1) = -1.0;
    activation.backward(dvalues);
    CHECK(activation.get_dinputs()(0, 0) == doctest::Approx(0.5));
    CHECK(activation.get_dinputs()(0, 1) == doctest::Approx(-0.5));

    Matrix bad_dvalues(2, 2, 0.0);
    CHECK_THROWS_WITH_AS(activation.backward(bad_dvalues),
                         "ActivationSoftmax::backward: dvalues shape mismatch",
                         runtime_error);

    Matrix empty;
    CHECK_THROWS_WITH_AS(activation.backward(empty),
                         "ActivationSoftmax::backward: dvalues must be non-empty",
                         runtime_error);
}

TEST_CASE("ActivationSoftmax rejects invalid exponentials and predictions")
{
    Matrix inputs(1, 1);
    inputs(0, 0) = -numeric_limits<double>::infinity();
    ActivationSoftmax activation;
    CHECK_THROWS_WITH_AS(activation.forward(inputs),
                         "ActivationSoftmax: invalid sum of exponentials",
                         runtime_error);

    Matrix empty;
    CHECK_THROWS_WITH_AS(activation.predictions(empty),
                         "ActivationSoftmax::predictions: outputs must be non-empty",
                         runtime_error);
}

TEST_CASE("ActivationSigmoid forward/backward and predictions")
{
    Matrix inputs(2, 2);
    inputs(0, 0) = 0.0; inputs(0, 1) = 1.0;
    inputs(1, 0) = -1.0; inputs(1, 1) = 2.0;

    ActivationSigmoid activation;
    activation.forward(inputs);
    CHECK(activation.get_output()(0, 0) == doctest::Approx(0.5));
    CHECK(activation.get_output()(0, 1) == doctest::Approx(1.0 / (1.0 + exp(-1.0))));

    Matrix upstream(1, 2, 1.0);
    Matrix inputs2(1, 2);
    inputs2(0, 0) = 0.0; inputs2(0, 1) = -2.0;
    activation.forward(inputs2);
    activation.backward(upstream);
    const double s0 = 1.0 / (1.0 + exp(-inputs2(0, 0)));
    const double s1 = 1.0 / (1.0 + exp(-inputs2(0, 1)));
    CHECK(activation.get_dinputs()(0, 0) == doctest::Approx((1.0 - s0) * s0));
    CHECK(activation.get_dinputs()(0, 1) == doctest::Approx((1.0 - s1) * s1));

    Matrix preds_in(1, 2);
    preds_in(0, 0) = 0.51;
    preds_in(0, 1) = 0.5;
    Matrix preds = activation.predictions(preds_in);
    CHECK(preds(0, 0) == doctest::Approx(1.0));
    CHECK(preds(0, 1) == doctest::Approx(0.0));

    Matrix bad;
    CHECK_THROWS_WITH_AS(activation.forward(bad),
                         "ActivationSigmoid::forward: inputs must be non-empty",
                         runtime_error);
    CHECK_THROWS_WITH_AS(activation.predictions(bad),
                         "ActivationSigmoid::predictions: outputs must be non-empty",
                         runtime_error);
}

TEST_CASE("ActivationSigmoid backward validates shapes")
{
    Matrix inputs(1, 1, 0.0);
    ActivationSigmoid activation;
    activation.forward(inputs);

    Matrix bad_dvalues(2, 1, 1.0);
    CHECK_THROWS_WITH_AS(activation.backward(bad_dvalues),
                         "ActivationSigmoid::backward: dvalues shape mismatch",
                         runtime_error);

    Matrix empty;
    CHECK_THROWS_WITH_AS(activation.backward(empty),
                         "ActivationSigmoid::backward: dvalues must be non-empty",
                         runtime_error);
}

TEST_CASE("ActivationLinear forward/backward")
{
    Matrix inputs(1, 3);
    inputs(0, 0) = -1.0; inputs(0, 1) = 0.5; inputs(0, 2) = 2.0;

    ActivationLinear activation;
    activation.forward(inputs);
    CHECK(activation.get_output()(0, 0) == doctest::Approx(-1.0));
    CHECK(activation.get_output()(0, 2) == doctest::Approx(2.0));

    Matrix upstream(1, 2);
    upstream(0, 0) = 3.0; upstream(0, 1) = -4.0;
    Matrix inputs2(1, 2);
    inputs2(0, 0) = 0.1; inputs2(0, 1) = -0.2;
    activation.forward(inputs2);
    activation.backward(upstream);
    CHECK(activation.get_dinputs()(0, 0) == doctest::Approx(3.0));
    CHECK(activation.get_dinputs()(0, 1) == doctest::Approx(-4.0));

    Matrix bad;
    CHECK_THROWS_WITH_AS(activation.forward(bad),
                         "ActivationLinear::forward: inputs must be non-empty",
                         runtime_error);

    Matrix bad_dvalues(2, 2, 0.0);
    activation.forward(inputs2);
    CHECK_THROWS_WITH_AS(activation.backward(bad_dvalues),
                         "ActivationLinear::backward: dvalues shape mismatch",
                         runtime_error);
    CHECK_THROWS_WITH_AS(activation.backward(bad),
                         "ActivationLinear::backward: dvalues must be non-empty",
                         runtime_error);
}

