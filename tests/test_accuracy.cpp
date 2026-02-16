#include "tests/test_common.hpp"

TEST_CASE("Accuracy base init/reset default implementations run")
{
    AccuracyCategorical acc;
    Matrix y(1, 1, 0.0);
    acc.init(y);
    acc.reset();
}

TEST_CASE("Accuracy accumulated calculations and new_pass")
{
    AccuracyCategorical acc;
    Matrix preds(2, 1);
    preds(0, 0) = 0.0;
    preds(1, 0) = 1.0;
    Matrix y(2, 1);
    y(0, 0) = 0.0;
    y(1, 0) = 0.0;

    CHECK_THROWS_WITH_AS(acc.calculate_accumulated(),
                         "Accuracy::calculate_accumulated: accumulated_count must be > 0",
                         runtime_error);

    CHECK(acc.calculate(preds, y) == doctest::Approx(0.5));
    CHECK(acc.calculate_accumulated() == doctest::Approx(0.5));

    acc.new_pass();
    CHECK_THROWS_WITH_AS(acc.calculate_accumulated(),
                         "Accuracy::calculate_accumulated: accumulated_count must be > 0",
                         runtime_error);
}

TEST_CASE("AccuracyCategorical computes accuracy for sparse and one-hot labels")
{
    AccuracyCategorical acc;

    Matrix preds(3, 1);
    preds(0, 0) = 0.0;
    preds(1, 0) = 1.0;
    preds(2, 0) = 2.0;

    Matrix sparse(3, 1);
    sparse(0, 0) = 0.0;
    sparse(1, 0) = 1.0;
    sparse(2, 0) = 2.0;
    CHECK(acc.calculate(preds, sparse) == doctest::Approx(1.0));

    Matrix pred_single(1, 1);
    pred_single(0, 0) = 2.0;
    Matrix one_hot(1, 3, 0.0);
    one_hot(0, 2) = 1.0;
    CHECK(acc.calculate(pred_single, one_hot) == doctest::Approx(1.0));

    Matrix sparse_row(1, 3);
    sparse_row(0, 0) = 0.0;
    sparse_row(0, 1) = 1.0;
    sparse_row(0, 2) = 2.0;
    CHECK(acc.calculate(preds, sparse_row) == doctest::Approx(1.0));
}

TEST_CASE("AccuracyCategorical supports binary and validates shapes")
{
    AccuracyCategorical binary_acc(true);
    CHECK(binary_acc.get_binray() == true);
    Matrix preds(2, 2);
    preds(0, 0) = 1.0; preds(0, 1) = 0.0;
    preds(1, 0) = 1.0; preds(1, 1) = 0.0;

    Matrix targets(2, 2);
    targets(0, 0) = 1.0; targets(0, 1) = 0.0;
    targets(1, 0) = 1.0; targets(1, 1) = 0.0;
    CHECK(binary_acc.calculate(preds, targets) == doctest::Approx(1.0));

    AccuracyCategorical acc;
    CHECK(acc.get_binray() == false);
    Matrix bad_preds(2, 2, 0.0);
    CHECK_THROWS_WITH_AS(acc.calculate(bad_preds, targets),
                         "AccuracyCategorical::compare: categorical y_pred must have shape (N,1)",
                         runtime_error);

    Matrix pred_labels(3, 1);
    pred_labels(0, 0) = 0.0;
    pred_labels(1, 0) = 1.0;
    pred_labels(2, 0) = 2.0;

    Matrix categorical_bad(2, 2, 0.0);
    CHECK_THROWS_WITH_AS(acc.calculate(pred_labels, categorical_bad),
                         "AccuracyCategorical::compare: for non-binary accuracy y_true must be sparse (N,1)/(1,N) or one-hot (N,C)",
                         runtime_error);

    Matrix binary_bad(2, 1, 0.0);
    CHECK_THROWS_WITH_AS(binary_acc.calculate(preds, binary_bad),
                         "AccuracyCategorical::compare: for binary accuracy y_true must match y_pred shape",
                         runtime_error);
}

TEST_CASE("AccuracyCategorical validates non-empty inputs")
{
    AccuracyCategorical acc;
    Matrix empty;
    CHECK_THROWS_WITH_AS(acc.calculate(empty, empty),
                         "Accuracy::calculate: y_pred must be non-empty",
                         runtime_error);

    Matrix preds(1, 1, 0.0);
    CHECK_THROWS_WITH_AS(acc.calculate(preds, empty),
                         "Accuracy::calculate: y_true must be non-empty",
                         runtime_error);
}

TEST_CASE("AccuracyRegression computes precision and accuracy")
{
    AccuracyRegression acc(10.0);
    Matrix y_true(2, 1);
    y_true(0, 0) = 1.0;
    y_true(1, 0) = 3.0;

    Matrix preds(2, 1);
    preds(0, 0) = 1.05;
    preds(1, 0) = 2.95;

    CHECK(acc.calculate(preds, y_true) == doctest::Approx(1.0));

    acc.reset();
    CHECK(acc.calculate(preds, y_true) == doctest::Approx(1.0));
}

TEST_CASE("AccuracyRegression validates inputs and shapes")
{
    CHECK_THROWS_WITH_AS(AccuracyRegression(0.0),
                         "AccuracyRegression: precision_divisor must be positive",
                         runtime_error);

    AccuracyRegression acc(10.0);
    CHECK(acc.get_precision_divisor() == doctest::Approx(10.0));
    Matrix empty;
    CHECK_THROWS_WITH_AS(acc.init(empty),
                         "AccuracyRegression::init: y_true must be non-empty",
                         runtime_error);

    Matrix preds(1, 1, 0.0);
    Matrix targets(2, 1, 0.0);
    CHECK_THROWS_WITH_AS(acc.calculate(preds, targets),
                         "AccuracyRegression::compare: y_pred and y_true must have the same shape",
                         runtime_error);
}


TEST_CASE("Base class virtual destructors run")
{
    Loss* loss = new LossCategoricalCrossEntropy();
    delete loss;

    Optimizer* opt = new OptimizerSGD(1.0, 0.0, 0.0);
    delete opt;

    Accuracy* acc = new AccuracyCategorical();
    delete acc;
}

