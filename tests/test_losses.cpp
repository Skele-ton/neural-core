#include "tests/test_common.hpp"

TEST_CASE("Loss::calculate validates non-empty inputs")
{
    struct DummyLoss : Loss {
        Matrix forward(const Matrix& output, const Matrix&) const override
        {
            Matrix losses(1, output.get_rows(), 1.0);
            return losses;
        }
        void backward(const Matrix&, const Matrix&) override {}
    } loss;

    Matrix empty;
    CHECK_THROWS_WITH_AS(loss.calculate(empty, empty),
                         "Loss::calculate: output must be non-empty",
                         runtime_error);

    Matrix output(1, 1, 0.0);
    CHECK_THROWS_WITH_AS(loss.calculate(output, empty),
                         "Loss::calculate: y_true must be non-empty",
                         runtime_error);

    Matrix y(1, 1, 0.0);
    CHECK(loss.calculate(output, y) == doctest::Approx(1.0));
    loss.backward(output, y);
}

TEST_CASE("Loss accumulated calculations and new_pass")
{
    struct DummyLoss : Loss {
        Matrix forward(const Matrix& output, const Matrix&) const override
        {
            Matrix losses(1, output.get_rows(), 2.0);
            return losses;
        }
        void backward(const Matrix&, const Matrix&) override {}
    } loss;

    Matrix out(2, 1, 0.0);
    Matrix y(2, 1, 0.0);

    CHECK_THROWS_WITH_AS(loss.calculate_accumulated(),
                         "Loss::calculate_accumulated: accumulated_count must be > 0",
                         runtime_error);

    double reg = 0.0;
    CHECK(loss.calculate(out, y) == doctest::Approx(2.0));
    loss.backward(out, y);
    vector<LayerDense*> empty_layers;
    CHECK(loss.calculate(out, y, reg, empty_layers) == doctest::Approx(2.0));
    CHECK(reg == doctest::Approx(0.0));

    double reg_accum = 0.0;
    CHECK(loss.calculate_accumulated(reg_accum, empty_layers) == doctest::Approx(2.0));
    CHECK(reg_accum == doctest::Approx(0.0));

    loss.new_pass();
    CHECK_THROWS_WITH_AS(loss.calculate_accumulated(),
                         "Loss::calculate_accumulated: accumulated_count must be > 0",
                         runtime_error);
}

TEST_CASE("Loss regularization is computed through calculate() and validates shapes")
{
    struct DummyLoss : Loss {
        Matrix forward(const Matrix& output, const Matrix&) const override
        {
            Matrix losses(1, output.get_rows(), 0.0);
            return losses;
        }
        void backward(const Matrix&, const Matrix&) override {}
    } loss;

    LayerDense layer(2, "linear", 0.3, 0.7, 0.5, 0.9);
    layer.weights.assign(2, 2);
    layer.weights(0, 0) = 1.0;  layer.weights(0, 1) = -2.0;
    layer.weights(1, 0) = -3.0; layer.weights(1, 1) = 4.0;
    layer.biases.assign(1, 2);
    layer.biases(0, 0) = 3.0; layer.biases(0, 1) = -4.0;

    vector<LayerDense*> layers = { &layer };

    Matrix output(2, 1, 0.0);
    Matrix y(2, 1, 0.0);
    double reg = 0.0;
    loss.calculate(output, y, reg, layers);
    CHECK(reg == doctest::Approx(50.0));

    LayerDense no_weights(1, "linear", 0.1, 0.0, 0.0, 0.0);
    no_weights.weights.assign(0, 0);
    vector<LayerDense*> layers_no_weights = { &no_weights };
    CHECK_THROWS_WITH_AS(loss.calculate(output, y, reg, layers_no_weights),
                         "Loss::regularization_loss: weights must be non-empty",
                         runtime_error);

    LayerDense bad_bias_shape(2, "linear", 0.0, 0.0, 0.1, 0.0);
    bad_bias_shape.weights.assign(2, 2);
    bad_bias_shape.biases.assign(1, 1);
    vector<LayerDense*> layers_bad_bias = { &bad_bias_shape };
    CHECK_THROWS_WITH_AS(loss.calculate(output, y, reg, layers_bad_bias),
                         "Loss::regularization_loss: biases must have shape (1, n_neurons)",
                         runtime_error);

    loss.backward(output, y);
}

TEST_CASE("LossCategoricalCrossEntropy forward matches sparse and one-hot labels")
{
    Matrix preds(3, 3);
    preds(0, 0) = 0.7; preds(0, 1) = 0.1; preds(0, 2) = 0.2;
    preds(1, 0) = 0.1; preds(1, 1) = 0.5; preds(1, 2) = 0.4;
    preds(2, 0) = 0.02; preds(2, 1) = 0.9; preds(2, 2) = 0.08;

    Matrix sparse(3, 1);
    sparse(0, 0) = 0.0;
    sparse(1, 0) = 1.0;
    sparse(2, 0) = 1.0;

    LossCategoricalCrossEntropy loss;
    double mean_loss = loss.calculate(preds, sparse);
    double expected = (-log(0.7) - log(0.5) - log(0.9)) / 3.0;
    CHECK(mean_loss == doctest::Approx(expected));

    Matrix one_hot(3, 3, 0.0);
    one_hot(0, 0) = 1.0;
    one_hot(1, 1) = 1.0;
    one_hot(2, 1) = 1.0;
    double mean_loss_oh = loss.calculate(preds, one_hot);
    CHECK(mean_loss_oh == doctest::Approx(expected));

    Matrix sparse_row(1, 3);
    sparse_row(0, 0) = 0.0;
    sparse_row(0, 1) = 1.0;
    sparse_row(0, 2) = 1.0;
    CHECK(loss.calculate(preds, sparse_row) == doctest::Approx(expected));
}

TEST_CASE("LossCategoricalCrossEntropy forward clamps and validates shapes")
{
    Matrix preds(1, 3);
    preds(0, 0) = 1.0;
    preds(0, 1) = 0.0;
    preds(0, 2) = 0.0;

    Matrix sparse(1, 1);
    sparse(0, 0) = 0.0;

    LossCategoricalCrossEntropy loss;
    double mean_loss = loss.calculate(preds, sparse);
    CHECK(mean_loss == doctest::Approx(1.0e-7).epsilon(1e-3));

    preds(0, 0) = 0.0;
    preds(0, 1) = 0.5;
    preds(0, 2) = 0.5;
    mean_loss = loss.calculate(preds, sparse);
    CHECK(mean_loss == doctest::Approx(16.11809565095832).epsilon(1e-6));

    Matrix bad(2, 2, 0.0);
    CHECK_THROWS_WITH_AS(loss.calculate(preds, bad),
                         "LossCategoricalCrossEntropy::forward: y_true must be sparse (N,1) or one-hot (N,C)",
                         runtime_error);

    Matrix oob(1, 1);
    oob(0, 0) = 3.0;
    CHECK_THROWS_WITH_AS(loss.calculate(preds, oob),
                         "LossCategoricalCrossEntropy::forward: y_true class index out of range",
                         runtime_error);

    Matrix bad_preds(1, 1, 0.5);
    CHECK_THROWS_WITH_AS(loss.calculate(bad_preds, sparse),
                         "LossCategoricalCrossEntropy::forward: y_pred.get_cols() must be >= 2",
                         runtime_error);
}

TEST_CASE("LossCategoricalCrossEntropy backward computes gradients and validates inputs")
{
    Matrix preds(2, 2);
    preds(0, 0) = 1.0; preds(0, 1) = 0.0;
    preds(1, 0) = 0.2; preds(1, 1) = 0.8;

    Matrix sparse(2, 1);
    sparse(0, 0) = 0.0;
    sparse(1, 0) = 1.0;

    LossCategoricalCrossEntropy loss;
    loss.backward(preds, sparse);
    CHECK(loss.get_dinputs()(0, 0) == doctest::Approx(-0.50000005).epsilon(1e-6));
    CHECK(loss.get_dinputs()(1, 1) == doctest::Approx(-0.625));

    Matrix one_hot(2, 2, 0.0);
    one_hot(0, 0) = 1.0;
    one_hot(1, 1) = 1.0;
    preds(0, 0) = 0.0; preds(0, 1) = 1.0;
    preds(1, 0) = 0.6; preds(1, 1) = 0.4;
    loss.backward(preds, one_hot);
    CHECK(loss.get_dinputs()(0, 0) == doctest::Approx(-5000000.0).epsilon(1e-6));
    CHECK(loss.get_dinputs()(1, 1) == doctest::Approx(-1.25).epsilon(1e-9));

    Matrix empty;
    CHECK_THROWS_WITH_AS(loss.backward(empty, one_hot),
                         "LossCategoricalCrossEntropy::backward: y_pred must be non-empty",
                         runtime_error);
    CHECK_THROWS_WITH_AS(loss.backward(preds, empty),
                         "LossCategoricalCrossEntropy::backward: y_true must be non-empty",
                         runtime_error);

    Matrix bad(3, 1, 0.0);
    CHECK_THROWS_WITH_AS(loss.backward(preds, bad),
                         "LossCategoricalCrossEntropy::backward: y_true must be sparse (N,1) or one-hot (N,C)",
                         runtime_error);

    Matrix oob(1, 2);
    oob(0, 0) = 2.0; oob(0, 1) = 0.0;
    CHECK_THROWS_WITH_AS(loss.backward(preds, oob),
                         "LossCategoricalCrossEntropy::backward: class index out of range",
                         runtime_error);

    Matrix bad_preds(1, 1, 0.5);
    CHECK_THROWS_WITH_AS(loss.backward(bad_preds, sparse),
                         "LossCategoricalCrossEntropy::backward: y_pred.get_cols() must be >= 2",
                         runtime_error);
}

TEST_CASE("LossBinaryCrossentropy computes mean loss and gradients")
{
    Matrix preds(2, 2);
    preds(0, 0) = 0.9; preds(0, 1) = 0.2;
    preds(1, 0) = 0.3; preds(1, 1) = 0.6;

    Matrix targets(2, 2);
    targets(0, 0) = 1.0; targets(0, 1) = 0.0;
    targets(1, 0) = 0.0; targets(1, 1) = 1.0;

    LossBinaryCrossentropy loss;
    double mean_loss = loss.calculate(preds, targets);
    CHECK(mean_loss == doctest::Approx(0.2990011586691898));

    loss.backward(preds, targets);
    CHECK(loss.get_dinputs()(0, 0) == doctest::Approx(-0.27777778));
    CHECK(loss.get_dinputs()(0, 1) == doctest::Approx(0.3125));
    CHECK(loss.get_dinputs()(1, 0) == doctest::Approx(0.35714286));
    CHECK(loss.get_dinputs()(1, 1) == doctest::Approx(-0.41666667));
}

TEST_CASE("LossBinaryCrossentropy validates shapes and non-empty inputs")
{
    Matrix preds(1, 2, 0.5);
    Matrix targets(2, 2, 0.5);
    LossBinaryCrossentropy loss;
    CHECK_THROWS_WITH_AS(loss.calculate(preds, targets),
                         "LossBinaryCrossentropy::forward: y_pred and y_true must have the same shape",
                         runtime_error);

    Matrix empty;
    CHECK_THROWS_WITH_AS(loss.backward(empty, targets),
                         "LossBinaryCrossentropy::backward: y_pred must be non-empty",
                         runtime_error);
    CHECK_THROWS_WITH_AS(loss.backward(preds, empty),
                         "LossBinaryCrossentropy::backward: y_true must be non-empty",
                         runtime_error);

    Matrix targets2(1, 1, 0.5);
    CHECK_THROWS_WITH_AS(loss.backward(preds, targets2),
                         "LossBinaryCrossentropy::backward: y_pred and y_true must have the same shape",
                         runtime_error);
}

TEST_CASE("LossMeanSquaredError computes average squared error and gradients")
{
    Matrix preds(2, 1);
    preds(0, 0) = 0.0; preds(1, 0) = 1.0;

    Matrix targets(2, 1);
    targets(0, 0) = 1.0; targets(1, 0) = 0.0;

    LossMeanSquaredError loss;
    double mean_loss = loss.calculate(preds, targets);
    CHECK(mean_loss == doctest::Approx(1.0));

    loss.backward(preds, targets);
    CHECK(loss.get_dinputs()(0, 0) == doctest::Approx(-1.0));
    CHECK(loss.get_dinputs()(1, 0) == doctest::Approx(1.0));
}

TEST_CASE("LossMeanSquaredError validates shapes and non-empty inputs")
{
    LossMeanSquaredError loss;
    Matrix preds(1, 1, 0.0);
    Matrix targets(2, 1, 0.0);

    CHECK_THROWS_WITH_AS(loss.calculate(preds, targets),
                         "LossMeanSquaredError::forward: y_pred and y_true must have the same shape",
                         runtime_error);

    Matrix empty;
    CHECK_THROWS_WITH_AS(loss.backward(empty, targets),
                         "LossMeanSquaredError::backward: y_pred must be non-empty",
                         runtime_error);
    CHECK_THROWS_WITH_AS(loss.backward(preds, empty),
                         "LossMeanSquaredError::backward: y_true must be non-empty",
                         runtime_error);
    CHECK_THROWS_WITH_AS(loss.backward(preds, targets),
                         "LossMeanSquaredError::backward: y_pred and y_true must have the same shape",
                         runtime_error);
}

TEST_CASE("LossMeanAbsoluteError computes average absolute error and gradients")
{
    Matrix preds(2, 1);
    preds(0, 0) = 0.0; preds(1, 0) = 1.0;

    Matrix targets(2, 1);
    targets(0, 0) = 1.0; targets(1, 0) = 0.0;

    LossMeanAbsoluteError loss;
    double mean_loss = loss.calculate(preds, targets);
    CHECK(mean_loss == doctest::Approx(1.0));

    loss.backward(preds, targets);
    CHECK(loss.get_dinputs()(0, 0) == doctest::Approx(-0.5));
    CHECK(loss.get_dinputs()(1, 0) == doctest::Approx(0.5));
}

TEST_CASE("LossMeanAbsoluteError validates shapes and non-empty inputs")
{
    LossMeanAbsoluteError loss;
    Matrix preds(1, 1, 0.0);
    Matrix targets(2, 1, 0.0);

    CHECK_THROWS_WITH_AS(loss.calculate(preds, targets),
                         "LossMeanAbsoluteError::forward: y_pred and y_true must have the same shape",
                         runtime_error);

    Matrix empty;
    CHECK_THROWS_WITH_AS(loss.backward(empty, targets),
                         "LossMeanAbsoluteError::backward: y_pred must be non-empty",
                         runtime_error);
    CHECK_THROWS_WITH_AS(loss.backward(preds, empty),
                         "LossMeanAbsoluteError::backward: y_true must be non-empty",
                         runtime_error);
    CHECK_THROWS_WITH_AS(loss.backward(preds, targets),
                         "LossMeanAbsoluteError::backward: y_pred and y_true must have the same shape",
                         runtime_error);
}

TEST_CASE("ActivationSoftmaxLossCategoricalCrossEntropy backward supports sparse and one-hot labels")
{
    Matrix preds(2, 3);
    preds(0, 0) = 0.7; preds(0, 1) = 0.2; preds(0, 2) = 0.1;
    preds(1, 0) = 0.1; preds(1, 1) = 0.8; preds(1, 2) = 0.1;

    Matrix sparse(2, 1);
    sparse(0, 0) = 0.0;
    sparse(1, 0) = 1.0;

    ActivationSoftmaxLossCategoricalCrossEntropy combo;
    combo.backward(preds, sparse);

    Matrix expected = preds;
    expected(0, 0) -= 1.0;
    expected(1, 1) -= 1.0;
    for (size_t i = 0; i < expected.get_rows(); ++i) {
        for (size_t j = 0; j < expected.get_cols(); ++j) {
            expected(i, j) /= static_cast<double>(expected.get_rows());
            CHECK(combo.get_dinputs()(i, j) == doctest::Approx(expected(i, j)));
        }
    }

    Matrix one_hot(2, 3, 0.0);
    one_hot(0, 2) = 1.0;
    one_hot(1, 0) = 1.0;
    combo.backward(preds, one_hot);
    expected = preds;
    expected(0, 2) -= 1.0;
    expected(1, 0) -= 1.0;
    for (size_t i = 0; i < expected.get_rows(); ++i) {
        for (size_t j = 0; j < expected.get_cols(); ++j) {
            expected(i, j) /= static_cast<double>(expected.get_rows());
            CHECK(combo.get_dinputs()(i, j) == doctest::Approx(expected(i, j)));
        }
    }

    Matrix sparse_row(1, 2);
    sparse_row(0, 0) = 0.0;
    sparse_row(0, 1) = 1.0;
    combo.backward(preds, sparse_row);
    expected = preds;
    expected(0, 0) -= 1.0;
    expected(1, 1) -= 1.0;
    for (size_t i = 0; i < expected.get_rows(); ++i) {
        for (size_t j = 0; j < expected.get_cols(); ++j) {
            expected(i, j) /= static_cast<double>(expected.get_rows());
            CHECK(combo.get_dinputs()(i, j) == doctest::Approx(expected(i, j)));
        }
    }
}

TEST_CASE("ActivationSoftmaxLossCategoricalCrossEntropy backward validates inputs")
{
    ActivationSoftmaxLossCategoricalCrossEntropy combo;
    Matrix empty;
    Matrix preds(1, 2, 0.5);
    Matrix sparse(1, 1, 0.0);

    CHECK_THROWS_WITH_AS(combo.backward(empty, sparse),
                         "ActivationSoftmaxLossCategoricalCrossEntropy::backward: y_pred must be non-empty",
                         runtime_error);
    CHECK_THROWS_WITH_AS(combo.backward(preds, empty),
                         "ActivationSoftmaxLossCategoricalCrossEntropy::backward: y_true must be non-empty",
                         runtime_error);

    Matrix bad(2, 2, 0.0);
    CHECK_THROWS_WITH_AS(combo.backward(preds, bad),
                         "ActivationSoftmaxLossCategoricalCrossEntropy::backward: y_true must be sparse (N,1) or one-hot (N,C)",
                         runtime_error);

    Matrix oob(1, 1);
    oob(0, 0) = 2.0;
    CHECK_THROWS_WITH_AS(combo.backward(preds, oob),
                         "ActivationSoftmaxLossCategoricalCrossEntropy::backward: class index out of range",
                         runtime_error);

    Matrix bad_preds(1, 1, 0.5);
    CHECK_THROWS_WITH_AS(combo.backward(bad_preds, sparse),
                         "ActivationSoftmaxLossCategoricalCrossEntropy::backward: y_pred.get_cols() must be >= 2",
                         runtime_error);
}

