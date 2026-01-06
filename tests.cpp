#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include <sstream>
#include <iterator>

#define NNFS_NO_MAIN
#include "NNFS_Diploma.cpp"

//matrix tests
TEST_CASE("Matrix basic operations")
{
    MatD m(2, 3, 1.5);
    CHECK(m.rows == 2);
    CHECK(m.cols == 3);
    CHECK(m(0, 0) == doctest::Approx(1.5));
    CHECK(m(1, 2) == doctest::Approx(1.5));

    m(0, 1) = 4.2;
    CHECK(m(0, 1) == doctest::Approx(4.2));

    m.assign(3, 2, 0.0);
    CHECK(m.rows == 3);
    CHECK(m.cols == 2);
    CHECK(m(2, 1) == doctest::Approx(0.0));
}

TEST_CASE("Matrix empty and transpose on empty")
{
    MatD m;
    CHECK(m.is_empty() == true);

    MatD t = m.transpose();
    CHECK(t.is_empty() == true);
}

TEST_CASE("Matrix print writes expected output")
{
    std::ostringstream oss;
    auto* old_buf = cout.rdbuf(oss.rdbuf());

    MatD m(2, 2);
    m(0, 0) = 1.0; m(0, 1) = 2.0;
    m(1, 0) = 3.0; m(1, 1) = 4.5;
    m.print();

    cout.rdbuf(old_buf);

    CHECK(oss.str() == "1 2\n3 4.5\n");
}

TEST_CASE("Matrix transpose swaps rows and columns and values")
{
    MatD m(2, 3);
    m(0, 0) = 1.0; m(0, 1) = 2.0; m(0, 2) = 3.0;
    m(1, 0) = 4.0; m(1, 1) = 5.0; m(1, 2) = 6.0;

    MatD t = m.transpose();
    CHECK(t.rows == 3);
    CHECK(t.cols == 2);

    CHECK(t(0, 0) == doctest::Approx(1.0));
    CHECK(t(0, 1) == doctest::Approx(4.0));
    CHECK(t(1, 0) == doctest::Approx(2.0));
    CHECK(t(1, 1) == doctest::Approx(5.0));
    CHECK(t(2, 0) == doctest::Approx(3.0));
    CHECK(t(2, 1) == doctest::Approx(6.0));
}

TEST_CASE("Matrix clip clamps values to bounds")
{
    MatD m(2, 3);
    m(0, 0) = -1.0; m(0, 1) = 0.2; m(0, 2) = 1.5;
    m(1, 0) = 0.8; m(1, 1) = 0.0; m(1, 2) = 2.0;

    MatD clipped = m.clip(0.1, 1.0);

    CHECK(clipped(0, 0) == doctest::Approx(0.1));
    CHECK(clipped(0, 1) == doctest::Approx(0.2));
    CHECK(clipped(0, 2) == doctest::Approx(1.0));
    CHECK(clipped(1, 0) == doctest::Approx(0.8));
    CHECK(clipped(1, 1) == doctest::Approx(0.1));
    CHECK(clipped(1, 2) == doctest::Approx(1.0));
}

TEST_CASE("Matrix clip throws when min exceeds max")
{
    MatD m(1, 1, 0.0);
    CHECK_THROWS_WITH_AS(m.clip(2.0, 1.0),
                         "clip: min_value must not exceed max_value",
                         runtime_error);
}

TEST_CASE("Matrix scale_by_scalar throws on zero samples")
{
    MatD m(1, 1, 2.0);
    CHECK_THROWS_WITH_AS(m.scale_by_scalar(0),
                         "scale_by_scalar: samples must be bigger than 0",
                         runtime_error);
}

// matrix_dot tests
TEST_CASE("matrix_dot multiplies small matrices correctly")
{
    MatD a(2, 3);
    a(0, 0) = 1.0; a(0, 1) = 2.0; a(0, 2) = 3.0;
    a(1, 0) = 4.0; a(1, 1) = 5.0; a(1, 2) = 6.0;

    MatD b(3, 2);
    b(0, 0) = 7.0; b(0, 1) = 8.0;
    b(1, 0) = 9.0; b(1, 1) = 10.0;
    b(2, 0) = 11.0; b(2, 1) = 12.0;

    MatD c = matrix_dot(a, b);
    CHECK(c.rows == 2);
    CHECK(c.cols == 2);

    CHECK(c(0, 0) == doctest::Approx(58.0));
    CHECK(c(0, 1) == doctest::Approx(64.0));
    CHECK(c(1, 0) == doctest::Approx(139.0));
    CHECK(c(1, 1) == doctest::Approx(154.0));
}

TEST_CASE("matrix_dot throws on empty matrices")
{
    MatD a;
    MatD b(1, 1, 1.0);
    CHECK_THROWS_WITH_AS(matrix_dot(a, b),
                         "matrix_dot: matrices must not be empty",
                         runtime_error);

    MatD c(1, 1, 1.0);
    MatD d;
    CHECK_THROWS_WITH_AS(matrix_dot(c, d),
                         "matrix_dot: matrices must not be empty",
                         runtime_error);
}

TEST_CASE("matrix_dot throws on incompatible shapes")
{
    MatD a(2, 3);
    MatD b(4, 1);
    CHECK_THROWS_WITH_AS(matrix_dot(a, b),
                         "matrix_dot: incompatible shapes",
                         runtime_error);
}

// print vector test
TEST_CASE("print_vector writes expected output")
{
    std::ostringstream oss;
    auto* old_buf = cout.rdbuf(oss.rdbuf());

    VecD v = {1.0, 2.5, -3.0};
    print_vector(v);

    cout.rdbuf(old_buf);

    CHECK(oss.str() == "1 2.5 -3\n");
}

// classification_accuracy tests
TEST_CASE("classification_accuracy computes correct value for sparse labels")
{
    MatD preds(3, 3);
    preds(0, 0) = 0.7; preds(0, 1) = 0.1; preds(0, 2) = 0.2;
    preds(1, 0) = 0.1; preds(1, 1) = 0.5; preds(1, 2) = 0.4;
    preds(2, 0) = 0.2; preds(2, 1) = 0.3; preds(2, 2) = 0.5;

    VecI targets = {0, 1, 2};

    double acc = classification_accuracy(preds, targets);
    CHECK(acc == doctest::Approx(1.0));
}

TEST_CASE("classification_accuracy throws on sparse size mismatch")
{
    MatD preds(2, 3);
    preds(0, 0) = 0.7; preds(0, 1) = 0.2; preds(0, 2) = 0.1;
    preds(1, 0) = 0.2; preds(1, 1) = 0.3; preds(1, 2) = 0.5;

    VecI targets = {0}; // mismatch

    CHECK_THROWS_WITH_AS(classification_accuracy(preds, targets),
                         "classification_accuracy: y_pred.rows must match y_true.size()",
                         runtime_error);
}

TEST_CASE("classification_accuracy throws on sparse empty predictions")
{
    MatD preds(1, 0);
    VecI targets = {0};
    CHECK_THROWS_WITH_AS(classification_accuracy(preds, targets),
                         "classification_accuracy: y_pred must be non-empty",
                         runtime_error);
}

TEST_CASE("classification_accuracy throws on sparse class index out of range")
{
    MatD preds(1, 2);
    preds(0, 0) = 0.5; preds(0, 1) = 0.5;
    VecI targets = {2}; // invalid class

    CHECK_THROWS_WITH_AS(classification_accuracy(preds, targets),
                         "classification_accuracy: class index out of range",
                         runtime_error);
}

TEST_CASE("classification_accuracy computes correct value for one-hot labels")
{
    MatD preds(2, 3);
    preds(0, 0) = 0.6; preds(0, 1) = 0.3; preds(0, 2) = 0.1;
    preds(1, 0) = 0.2; preds(1, 1) = 0.5; preds(1, 2) = 0.3;

    MatD targets(2, 3, 0.0);
    targets(0, 0) = 1.0;
    targets(1, 1) = 1.0;

    double acc = classification_accuracy(preds, targets);
    CHECK(acc == doctest::Approx(1.0));
}

TEST_CASE("classification_accuracy throws on one-hot shape mismatch")
{
    MatD preds(1, 2);
    preds(0, 0) = 0.5; preds(0, 1) = 0.5;

    MatD targets(2, 2, 0.0); // mismatched rows
    targets(0, 0) = 1.0;
    targets(1, 1) = 1.0;

    CHECK_THROWS_WITH_AS(classification_accuracy(preds, targets),
                         "classification_accuracy: y_pred and y_true must have the same shape",
                         runtime_error);
}

TEST_CASE("classification_accuracy throws on one-hot empty predictions")
{
    MatD preds(0, 0);
    MatD targets(0, 0);
    CHECK_THROWS_WITH_AS(classification_accuracy(preds, targets),
                         "classification_accuracy: y_pred must be non-empty",
                         runtime_error);
}

// generate_spiral_data tests
TEST_CASE("generate_spiral_data shapes and labels are correct")
{
    MatD X;
    VecI y;

    generate_spiral_data(10, 3, X, y);

    CHECK(X.rows == 30);
    CHECK(X.cols == 2);
    CHECK(y.size() == 30);

    // labels are 0,1,2
    for (int label : y) {
        CHECK(label >= 0);
        CHECK(label < 3);
    }

    // values are finite and not all identical
    double min_v = X(0, 0);
    double max_v = X(0, 0);
    for (size_t i = 0; i < X.rows; ++i) {
        for (size_t j = 0; j < X.cols; ++j) {
            double v = X(i, j);
            CHECK(std::isfinite(v));
            if (v < min_v) min_v = v;
            if (v > max_v) max_v = v;
        }
    }
    CHECK(max_v != min_v);
}

TEST_CASE("generate_spiral_data throws on invalid arguments")
{
    MatD X;
    VecI y;

    CHECK_THROWS_WITH_AS(generate_spiral_data(1, 3, X, y),
                         "generate_spiral_data: invalid arguments",
                         runtime_error);
    CHECK_THROWS_WITH_AS(generate_spiral_data(10, 0, X, y),
                         "generate_spiral_data: invalid arguments",
                         runtime_error);
}

TEST_CASE("random_uniform produces values in unit interval")
{
    g_rng.seed(0);
    double first = random_uniform();
    double second = random_uniform();

    CHECK(first >= 0.0);
    CHECK(first < 1.0);
    CHECK(second >= 0.0);
    CHECK(second < 1.0);
}

// generate_vertical_data tests
TEST_CASE("generate_vertical_data throws on invalid arguments")
{
    MatD X;
    VecI y;

    CHECK_THROWS_WITH_AS(generate_vertical_data(0, 3, X, y),
                         "generate_vertical_data: invalid arguments",
                         runtime_error);
    CHECK_THROWS_WITH_AS(generate_vertical_data(5, 0, X, y),
                         "generate_vertical_data: invalid arguments",
                         runtime_error);
}

TEST_CASE("generate_vertical_data fills samples with labels")
{
    MatD X;
    VecI y;

    g_rng.seed(0);
    generate_vertical_data(4, 3, X, y);

    CHECK(X.rows == 12);
    CHECK(X.cols == 2);
    CHECK(y.size() == 12);

    for (size_t i = 0; i < y.size(); ++i) {
        CHECK(y[i] >= 0);
        CHECK(y[i] < 3);
        CHECK(std::isfinite(X(i, 0)));
        CHECK(std::isfinite(X(i, 1)));
    }

    double min_x = X(0, 0), max_x = X(0, 0);
    double min_y = X(0, 1), max_y = X(0, 1);
    for (size_t i = 1; i < X.rows; ++i) {
        min_x = std::min(min_x, X(i, 0));
        max_x = std::max(max_x, X(i, 0));
        min_y = std::min(min_y, X(i, 1));
        max_y = std::max(max_y, X(i, 1));
    }

    CHECK(max_x != min_x);
    CHECK(max_y != min_y);
}

// plot_scatter_svg tests
TEST_CASE("plot_scatter_svg validates inputs and paths")
{
    MatD empty;
    CHECK_THROWS_WITH_AS(plot_scatter_svg("unused.svg", empty),
                         "plot_scatter_svg: invalid input data",
                         runtime_error);

    MatD wrong_cols(1, 1, 0.0);
    CHECK_THROWS_WITH_AS(plot_scatter_svg("unused.svg", wrong_cols),
                         "plot_scatter_svg: invalid input data",
                         runtime_error);

    MatD points(1, 2, 0.5);
    VecI labels = {0};
    CHECK_THROWS_WITH_AS(plot_scatter_svg("/nonexistent_dir/plot.svg", points, labels),
                         "plot_scatter_svg: given path is invalid",
                         runtime_error);
}

TEST_CASE("plot_scatter_svg writes circles with optional labels")
{
    MatD points(3, 2);
    points(0, 0) = 0.0; points(0, 1) = 0.0;
    points(1, 0) = 1.0; points(1, 1) = 0.5;
    points(2, 0) = 0.5; points(2, 1) = 1.0;

    VecI labels = {0, 1, 2};
    const std::string path = "test_plot.svg";

    plot_scatter_svg(path, points, labels);

    std::ifstream file(path);
    REQUIRE(file.good());

    std::string line;
    bool saw_svg = false;
    int circles = 0;
    while (std::getline(file, line)) {
        if (!saw_svg && line.find("<svg") != std::string::npos) {
            saw_svg = true;
        }
        if (line.find("<circle") != std::string::npos) {
            ++circles;
        }
    }

    CHECK(saw_svg == true);
    CHECK(circles == 3);
    std::remove(path.c_str());
}

TEST_CASE("plot_scatter_svg handles unlabeled points")
{
    MatD points(2, 2);
    points(0, 0) = -1.0; points(0, 1) = -1.0;
    points(1, 0) = 2.0;  points(1, 1) = 3.0;

    const std::string path = "test_plot_unlabeled.svg";
    plot_scatter_svg(path, points);

    std::ifstream file(path);
    REQUIRE(file.good());

    std::string contents((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    CHECK(contents.find("<circle") != std::string::npos);
    CHECK(contents.find("#1f77b4") != std::string::npos); // default color for class 0
    std::remove(path.c_str());
}

// LayerDense forward tests (using embedded batch math)
TEST_CASE("LayerDense forward matches known example")
{
    // inputs: 3 samples x 4 features
    MatD inputs(3, 4);
    inputs(0, 0) = 1.0; inputs(0, 1) = 2.0; inputs(0, 2) = 3.0; inputs(0, 3) = 2.5;
    inputs(1, 0) = 2.0; inputs(1, 1) = 5.0; inputs(1, 2) = -1.0; inputs(1, 3) = 2.0;
    inputs(2, 0) = -1.5; inputs(2, 1) = 2.7; inputs(2, 2) = 3.3; inputs(2, 3) = -0.8;

    LayerDense layer(4, 3);
    layer.weights.assign(4, 3);
    layer.weights(0, 0) = 0.2;  layer.weights(0, 1) = 0.5;   layer.weights(0, 2) = -0.26;
    layer.weights(1, 0) = 0.8;  layer.weights(1, 1) = -0.91; layer.weights(1, 2) = -0.27;
    layer.weights(2, 0) = -0.5; layer.weights(2, 1) = 0.26;  layer.weights(2, 2) = 0.17;
    layer.weights(3, 0) = 1.0;  layer.weights(3, 1) = -0.5;  layer.weights(3, 2) = 0.87;
    layer.biases = {2.0, 3.0, 0.5};

    layer.forward(inputs);
    CHECK(layer.output.rows == 3);
    CHECK(layer.output.cols == 3);

    CHECK(layer.output(0, 0) == doctest::Approx(4.8));
    CHECK(layer.output(0, 1) == doctest::Approx(1.21));
    CHECK(layer.output(0, 2) == doctest::Approx(2.385));

    CHECK(layer.output(1, 0) == doctest::Approx(8.9));
    CHECK(layer.output(1, 1) == doctest::Approx(-1.81));
    CHECK(layer.output(1, 2) == doctest::Approx(0.2));

    CHECK(layer.output(2, 0) == doctest::Approx(1.41));
    CHECK(layer.output(2, 1) == doctest::Approx(1.051));
    CHECK(layer.output(2, 2) == doctest::Approx(0.026));

    // inputs stored correctly
    CHECK(layer.inputs.rows == inputs.rows);
    CHECK(layer.inputs.cols == inputs.cols);
}

// LayerDense tests
TEST_CASE("LayerDense forward with custom weights and biases")
{
    // prepare inputs as above example
    MatD inputs(3, 4);
    inputs(0, 0) = 1.0; inputs(0, 1) = 2.0; inputs(0, 2) = 3.0; inputs(0, 3) = 2.5;
    inputs(1, 0) = 2.0; inputs(1, 1) = 5.0; inputs(1, 2) = -1.0; inputs(1, 3) = 2.0;
    inputs(2, 0) = -1.5; inputs(2, 1) = 2.7; inputs(2, 2) = 3.3; inputs(2, 3) = -0.8;

    LayerDense layer(4, 3);        // will override weights/biases
    layer.weights.assign(4, 3);    // 4 inputs x 3 neurons

    layer.weights(0, 0) = 0.2;  layer.weights(0, 1) = 0.5;   layer.weights(0, 2) = -0.26;
    layer.weights(1, 0) = 0.8;  layer.weights(1, 1) = -0.91; layer.weights(1, 2) = -0.27;
    layer.weights(2, 0) = -0.5; layer.weights(2, 1) = 0.26;  layer.weights(2, 2) = 0.17;
    layer.weights(3, 0) = 1.0;  layer.weights(3, 1) = -0.5;  layer.weights(3, 2) = 0.87;

    layer.biases = {2.0, 3.0, 0.5};

    layer.forward(inputs);

    CHECK(layer.output.rows == 3);
    CHECK(layer.output.cols == 3);

    CHECK(layer.output(0, 0) == doctest::Approx(4.8));
    CHECK(layer.output(0, 1) == doctest::Approx(1.21));
    CHECK(layer.output(0, 2) == doctest::Approx(2.385));

    CHECK(layer.output(1, 0) == doctest::Approx(8.9));
    CHECK(layer.output(1, 1) == doctest::Approx(-1.81));
    CHECK(layer.output(1, 2) == doctest::Approx(0.2));

    CHECK(layer.output(2, 0) == doctest::Approx(1.41));
    CHECK(layer.output(2, 1) == doctest::Approx(1.051));
    CHECK(layer.output(2, 2) == doctest::Approx(0.026));

    // inputs stored correctly
    CHECK(layer.inputs.rows == inputs.rows);
    CHECK(layer.inputs.cols == inputs.cols);
}

TEST_CASE("LayerDense output shape matches inputs and neuron count")
{
    MatD X;
    VecI y;
    generate_spiral_data(5, 2, X, y); // 10 samples x 2 features

    LayerDense dense(2, 4); // 4 neurons
    dense.forward(X);

    CHECK(dense.output.rows == X.rows);
    CHECK(dense.output.cols == 4);
}

TEST_CASE("LayerDense forward throws when weights are empty")
{
    MatD inputs(1, 1, 1.0);
    LayerDense layer(1, 1);
    layer.weights.assign(0, 0); // make weights empty

    CHECK_THROWS_WITH_AS(layer.forward(inputs),
                         "LayerDense::forward: weights must be initialized",
                         runtime_error);
}

TEST_CASE("LayerDense forward throws on input/weight shape mismatch")
{
    MatD inputs(1, 3);       // 1 x 3
    LayerDense layer(2, 2);  // expects 2 inputs
    CHECK_THROWS_WITH_AS(layer.forward(inputs),
                         "LayerDense::forward: inputs.cols must match weights.rows",
                         runtime_error);
}

TEST_CASE("LayerDense forward throws on bias size mismatch")
{
    MatD inputs(1, 2);
    LayerDense layer(2, 3);
    layer.biases = {1.0, 2.0}; // size 2, but weights.cols = 3
    CHECK_THROWS_WITH_AS(layer.forward(inputs),
                         "LayerDense::forward: biases.size() must match weights.cols",
                         runtime_error);
}

TEST_CASE("LayerDense backward computes gradients")
{
    MatD inputs(2, 2);
    inputs(0, 0) = 1.0; inputs(0, 1) = 2.0;
    inputs(1, 0) = 3.0; inputs(1, 1) = 4.0;

    LayerDense layer(2, 2);
    layer.weights.assign(2, 2);
    layer.weights(0, 0) = 1.0; layer.weights(0, 1) = 0.0;
    layer.weights(1, 0) = 0.0; layer.weights(1, 1) = 1.0;
    layer.biases = {0.0, 0.0};

    layer.forward(inputs);

    MatD dvalues(2, 2);
    dvalues(0, 0) = 1.0; dvalues(0, 1) = 2.0;
    dvalues(1, 0) = 3.0; dvalues(1, 1) = 4.0;

    layer.backward(dvalues);

    CHECK(layer.dweights(0, 0) == doctest::Approx(10.0));
    CHECK(layer.dweights(0, 1) == doctest::Approx(14.0));
    CHECK(layer.dweights(1, 0) == doctest::Approx(14.0));
    CHECK(layer.dweights(1, 1) == doctest::Approx(20.0));

    CHECK(layer.dbiases[0] == doctest::Approx(4.0));
    CHECK(layer.dbiases[1] == doctest::Approx(6.0));

    CHECK(layer.dinputs(0, 0) == doctest::Approx(1.0));
    CHECK(layer.dinputs(0, 1) == doctest::Approx(2.0));
    CHECK(layer.dinputs(1, 0) == doctest::Approx(3.0));
    CHECK(layer.dinputs(1, 1) == doctest::Approx(4.0));
}

TEST_CASE("LayerDense backward throws on shape mismatch")
{
    MatD inputs(1, 2);
    inputs(0, 0) = 1.0; inputs(0, 1) = 2.0;

    LayerDense layer(2, 2);
    layer.forward(inputs);

    MatD bad_dvalues(1, 1, 0.0); // wrong number of columns
    CHECK_THROWS_WITH_AS(layer.backward(bad_dvalues),
                         "LayerDense::backward: dvalues shape mismatch",
                         runtime_error);
}

// ActivationReLU tests
TEST_CASE("ActivationReLU sets negatives to zero and keeps positives")
{
    MatD inputs(2, 3);
    inputs(0, 0) = -1.0; inputs(0, 1) = 0.0; inputs(0, 2) = 2.5;
    inputs(1, 0) = 3.0; inputs(1, 1) = -0.1; inputs(1, 2) = 0.0;

    ActivationReLU activation;
    activation.forward(inputs);

    // shape preserved
    CHECK(activation.output.rows == inputs.rows);
    CHECK(activation.output.cols == inputs.cols);

    // inputs stored correctly
    CHECK(activation.inputs.rows == inputs.rows);
    CHECK(activation.inputs.cols == inputs.cols);
    for (size_t i = 0; i < inputs.rows; ++i) {
        for (size_t j = 0; j < inputs.cols; ++j) {
            CHECK(activation.inputs(i, j) == doctest::Approx(inputs(i, j)));
        }
    }

    // negatives -> 0, non-negatives unchanged
    CHECK(activation.output(0, 0) == doctest::Approx(0.0));   // -1.0 -> 0
    CHECK(activation.output(0, 1) == doctest::Approx(0.0));   // 0.0 stays 0
    CHECK(activation.output(0, 2) == doctest::Approx(2.5));   // 2.5 stays 2.5

    CHECK(activation.output(1, 0) == doctest::Approx(3.0));   // 3.0 stays 3.0
    CHECK(activation.output(1, 1) == doctest::Approx(0.0));   // -0.1 -> 0
    CHECK(activation.output(1, 2) == doctest::Approx(0.0));   // 0.0 stays 0
}

TEST_CASE("ActivationReLU backward zeroes gradients where inputs were non-positive")
{
    MatD inputs(2, 2);
    inputs(0, 0) = -1.0; inputs(0, 1) = 1.0;
    inputs(1, 0) = 0.0;  inputs(1, 1) = 2.0;

    ActivationReLU activation;
    activation.forward(inputs);

    MatD dvalues(2, 2);
    dvalues(0, 0) = 5.0; dvalues(0, 1) = 6.0;
    dvalues(1, 0) = 7.0; dvalues(1, 1) = 8.0;

    activation.backward(dvalues);

    CHECK(activation.dinputs(0, 0) == doctest::Approx(0.0));
    CHECK(activation.dinputs(0, 1) == doctest::Approx(6.0));
    CHECK(activation.dinputs(1, 0) == doctest::Approx(0.0));
    CHECK(activation.dinputs(1, 1) == doctest::Approx(8.0));
}

TEST_CASE("ActivationReLU backward throws on shape mismatch")
{
    MatD inputs(1, 1, 1.0);
    ActivationReLU activation;
    activation.forward(inputs);

    MatD bad_dvalues(1, 2, 0.0);
    CHECK_THROWS_WITH_AS(activation.backward(bad_dvalues),
                         "ActivationReLU::backward: dvalues shape mismatch",
                         runtime_error);
}

// ActivationSoftmax tests
TEST_CASE("ActivationSoftmax computes correct probabilities per row")
{
    MatD inputs(2, 3);

    // Row 0: [0, 1, 2] -> softmax ≈ [0.09003, 0.24473, 0.66524]
    inputs(0, 0) = 0.0; inputs(0, 1) = 1.0; inputs(0, 2) = 2.0;

    // Row 1: [0, 0, 0] -> softmax = [1/3, 1/3, 1/3]
    inputs(1, 0) = 0.0; inputs(1, 1) = 0.0; inputs(1, 2) = 0.0;

    ActivationSoftmax activation;
    activation.forward(inputs);

    // shape preserved
    CHECK(activation.output.rows == inputs.rows);
    CHECK(activation.output.cols == inputs.cols);

    // inputs stored correctly
    CHECK(activation.inputs.rows == inputs.rows);
    CHECK(activation.inputs.cols == inputs.cols);
    for (size_t i = 0; i < inputs.rows; ++i) {
        for (size_t j = 0; j < inputs.cols; ++j) {
            CHECK(activation.inputs(i, j) == doctest::Approx(inputs(i, j)));
        }
    }

    // Row 0 probabilities (approximate known softmax)
    CHECK(activation.output(0, 0) == doctest::Approx(0.0900306));
    CHECK(activation.output(0, 1) == doctest::Approx(0.2447285));
    CHECK(activation.output(0, 2) == doctest::Approx(0.6652409));

    // Row 1 probabilities (uniform)
    CHECK(activation.output(1, 0) == doctest::Approx(1.0 / 3.0));
    CHECK(activation.output(1, 1) == doctest::Approx(1.0 / 3.0));
    CHECK(activation.output(1, 2) == doctest::Approx(1.0 / 3.0));

    // Each row sums to 1
    for (size_t i = 0; i < activation.output.rows; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < activation.output.cols; ++j) {
            sum += activation.output(i, j);
        }
        CHECK(sum == doctest::Approx(1.0));
    }
}

TEST_CASE("ActivationSoftmax throws when exponentials sum is non-finite")
{
    MatD inputs(1, 1);
    inputs(0, 0) = -std::numeric_limits<double>::infinity();
    ActivationSoftmax activation;
    CHECK_THROWS_WITH_AS(activation.forward(inputs),
                         "ActivationSoftmax: invalid sum of exponentials",
                         runtime_error);
}

TEST_CASE("ActivationSoftmax backward computes gradients")
{
    MatD inputs(1, 2);
    inputs(0, 0) = 0.0; inputs(0, 1) = 0.0;

    ActivationSoftmax activation;
    activation.forward(inputs);

    MatD dvalues(1, 2);
    dvalues(0, 0) = 1.0; dvalues(0, 1) = -1.0;

    activation.backward(dvalues);

    CHECK(activation.dinputs(0, 0) == doctest::Approx(0.5));
    CHECK(activation.dinputs(0, 1) == doctest::Approx(-0.5));
}

TEST_CASE("ActivationSoftmax backward throws on shape mismatch")
{
    MatD inputs(1, 2);
    inputs(0, 0) = 0.1; inputs(0, 1) = 0.2;

    ActivationSoftmax activation;
    activation.forward(inputs);

    MatD bad_dvalues(2, 2, 0.0); // wrong number of rows
    CHECK_THROWS_WITH_AS(activation.backward(bad_dvalues),
                         "ActivationSoftmax::backward: dvalues shape mismatch",
                         runtime_error);
}

// LossCategoricalCrossEntropy tests
TEST_CASE("LossCategoricalCrossEntropy matches known example with sparse labels")
{
    MatD preds(3, 3);
    preds(0, 0) = 0.7; preds(0, 1) = 0.1; preds(0, 2) = 0.2;
    preds(1, 0) = 0.1; preds(1, 1) = 0.5; preds(1, 2) = 0.4;
    preds(2, 0) = 0.02; preds(2, 1) = 0.9; preds(2, 2) = 0.08;

    // sparse
    VecI targets = {0, 1, 1};

    LossCategoricalCrossEntropy loss;
    double mean_loss = loss.calculate(preds, targets);

    CHECK(mean_loss == doctest::Approx(0.38506088005216804));
}

TEST_CASE("LossCategoricalCrossEntropy matches known example with one-hot labels")
{
    MatD preds(3, 3);
    preds(0, 0) = 0.7; preds(0, 1) = 0.1; preds(0, 2) = 0.2;
    preds(1, 0) = 0.1; preds(1, 1) = 0.5; preds(1, 2) = 0.4;
    preds(2, 0) = 0.02; preds(2, 1) = 0.9; preds(2, 2) = 0.08;

    // one-hot
    MatD targets(3, 3, 0.0);
    targets(0, 0) = 1.0;
    targets(1, 1) = 1.0;
    targets(2, 1) = 1.0;

    LossCategoricalCrossEntropy loss;
    double mean_loss = loss.calculate(preds, targets);

    CHECK(mean_loss == doctest::Approx(0.38506088005216804));
}

TEST_CASE("LossCategoricalCrossEntropy clips to avoid log(0)")
{
    MatD preds(1, 3);
    preds(0, 0) = 1.0;
    preds(0, 1) = 0.0;
    preds(0, 2) = 0.0;

    VecI targets = {0};

    LossCategoricalCrossEntropy loss;
    double mean_loss = loss.calculate(preds, targets);

    CHECK(mean_loss == doctest::Approx(1.0e-7).epsilon(1e-3));
}

TEST_CASE("LossCategoricalCrossEntropy clips zero confidence for true class")
{
    MatD preds(1, 3);
    preds(0, 0) = 0.0;
    preds(0, 1) = 0.5;
    preds(0, 2) = 0.5;

    VecI targets = {0};

    LossCategoricalCrossEntropy loss;
    double mean_loss = loss.calculate(preds, targets);

    // Expect clipped to -log(1e-7)
    CHECK(mean_loss == doctest::Approx(16.11809565095832).epsilon(1e-6));
}

TEST_CASE("LossCategoricalCrossEntropy throws on sparse label count mismatch")
{
    MatD preds(2, 3);
    preds(0, 0) = 0.7; preds(0, 1) = 0.1; preds(0, 2) = 0.2;
    preds(1, 0) = 0.1; preds(1, 1) = 0.5; preds(1, 2) = 0.4;

    VecI targets = {0}; // size mismatch vs preds.rows

    LossCategoricalCrossEntropy loss;
    CHECK_THROWS_WITH_AS(loss.calculate(preds, targets),
                         "LossCategoricalCrossEntropy: y_pred.rows must match y_true.size()",
                         runtime_error);
}

TEST_CASE("LossCategoricalCrossEntropy mean_sample_losses throws on zero samples")
{
    MatD preds;
    preds.assign(0, 2, 0.0);
    VecI targets;

    LossCategoricalCrossEntropy loss;
    CHECK_THROWS_WITH_AS(loss.calculate(preds, targets),
                         "Loss::mean_sample_losses: sample_losses must contain at least one element",
                         runtime_error);
}

TEST_CASE("LossCategoricalCrossEntropy throws on sparse label out of range")
{
    MatD preds(1, 2);
    preds(0, 0) = 0.5; preds(0, 1) = 0.5;
    VecI targets = {2}; // invalid index

    LossCategoricalCrossEntropy loss;
    CHECK_THROWS_WITH_AS(loss.calculate(preds, targets),
                         "LossCategoricalCrossEntropy: class index out of range",
                         runtime_error);
}

TEST_CASE("LossCategoricalCrossEntropy throws on one-hot shape mismatch")
{
    MatD preds(2, 2);
    preds(0, 0) = 0.7; preds(0, 1) = 0.3;
    preds(1, 0) = 0.1; preds(1, 1) = 0.9;

    MatD targets(1, 2, 0.0); // mismatched rows

    LossCategoricalCrossEntropy loss;
    CHECK_THROWS_WITH_AS(loss.calculate(preds, targets),
                         "LossCategoricalCrossEntropy: y_pred and y_true must have the same shape",
                         runtime_error);
}

TEST_CASE("LossCategoricalCrossEntropy backward (sparse) clamps probabilities")
{
    MatD preds(2, 2);
    preds(0, 0) = 1.0; preds(0, 1) = 0.0; // will clamp to 1 - eps and eps
    preds(1, 0) = 0.2; preds(1, 1) = 0.8;
    VecI targets = {0, 1};

    LossCategoricalCrossEntropy loss;
    loss.backward(preds, targets);

    CHECK(loss.dinputs(0, 0) == doctest::Approx(-0.50000005).epsilon(1e-6));
    CHECK(loss.dinputs(0, 1) == doctest::Approx(0.0));
    CHECK(loss.dinputs(1, 0) == doctest::Approx(0.0));
    CHECK(loss.dinputs(1, 1) == doctest::Approx(-0.625));
}

TEST_CASE("LossCategoricalCrossEntropy backward (one-hot) clamps low probabilities")
{
    MatD preds(2, 2);
    preds(0, 0) = 0.0; preds(0, 1) = 1.0; // clamp low branch
    preds(1, 0) = 0.6; preds(1, 1) = 0.4;

    MatD targets(2, 2, 0.0);
    targets(0, 0) = 1.0;
    targets(1, 1) = 1.0;

    LossCategoricalCrossEntropy loss;
    loss.backward(preds, targets);

    CHECK(loss.dinputs(0, 0) == doctest::Approx(-5000000.0).epsilon(1e-6)); // -1e7 / 2
    CHECK(loss.dinputs(0, 1) == doctest::Approx(0.0));
    CHECK(loss.dinputs(1, 0) == doctest::Approx(0.0));
    CHECK(loss.dinputs(1, 1) == doctest::Approx(-1.25).epsilon(1e-9));
}

TEST_CASE("LossCategoricalCrossEntropy backward throws on sparse shape mismatch")
{
    MatD preds(1, 2, 0.5);
    VecI targets = {0, 1}; // size mismatch

    LossCategoricalCrossEntropy loss;
    CHECK_THROWS_WITH_AS(loss.backward(preds, targets),
                         "LossCategoricalCrossEntropy::backward: dvalues.rows must match y_true.size()",
                         runtime_error);
}

TEST_CASE("LossCategoricalCrossEntropy backward throws on zero samples sparse path")
{
    MatD preds;
    preds.assign(0, 2, 0.0);
    VecI targets;

    LossCategoricalCrossEntropy loss;
    CHECK_THROWS_WITH_AS(loss.backward(preds, targets),
                         "LossCategoricalCrossEntropy::backward: dvalues must contain at least one sample",
                         runtime_error);
}

TEST_CASE("LossCategoricalCrossEntropy backward throws on sparse class out of range")
{
    MatD preds(1, 1, 0.5);
    VecI targets = {1}; // invalid index

    LossCategoricalCrossEntropy loss;
    CHECK_THROWS_WITH_AS(loss.backward(preds, targets),
                         "LossCategoricalCrossEntropy::backward: class index out of range",
                         runtime_error);
}

TEST_CASE("LossCategoricalCrossEntropy backward throws on one-hot shape mismatch")
{
    MatD preds(1, 2, 0.5);
    MatD targets(2, 2, 0.0); // mismatched rows

    LossCategoricalCrossEntropy loss;
    CHECK_THROWS_WITH_AS(loss.backward(preds, targets),
                         "LossCategoricalCrossEntropy::backward: shapes of dvalues and y_true must match",
                         runtime_error);
}

TEST_CASE("LossCategoricalCrossEntropy backward throws on zero samples one-hot path")
{
    MatD preds;
    preds.assign(0, 2, 0.0);
    MatD targets;
    targets.assign(0, 2, 0.0);

    LossCategoricalCrossEntropy loss;
    CHECK_THROWS_WITH_AS(loss.backward(preds, targets),
                         "LossCategoricalCrossEntropy::backward: dvalues must contain at least one sample",
                         runtime_error);
}

// Combined softmax + cross-entropy tests
TEST_CASE("ActivationSoftmaxLossCategoricalCrossEntropy end-to-end with sparse labels")
{
    MatD logits(2, 3);
    logits(0, 0) = 1.0; logits(0, 1) = 0.0; logits(0, 2) = 0.0;
    logits(1, 0) = 0.0; logits(1, 1) = 1.0; logits(1, 2) = 0.0;

    VecI y_true = {0, 2};

    ActivationSoftmaxLossCategoricalCrossEntropy combo;
    double loss = combo.forward(logits, y_true);
    CHECK(loss > 0.0);

    MatD expected = combo.output;
    const size_t samples = expected.rows;
    for (size_t i = 0; i < samples; ++i) {
        expected(i, static_cast<size_t>(y_true[i])) -= 1.0;
    }
    for (size_t i = 0; i < expected.rows; ++i) {
        for (size_t j = 0; j < expected.cols; ++j) {
            expected(i, j) /= static_cast<double>(samples);
        }
    }

    combo.backward(combo.output, y_true);

    for (size_t i = 0; i < expected.rows; ++i) {
        for (size_t j = 0; j < expected.cols; ++j) {
            CHECK(combo.dinputs(i, j) == doctest::Approx(expected(i, j)));
        }
    }
}

TEST_CASE("ActivationSoftmaxLossCategoricalCrossEntropy supports one-hot labels")
{
    MatD logits(1, 3);
    logits(0, 0) = -1.0; logits(0, 1) = 2.0; logits(0, 2) = 0.5;

    MatD y_true(1, 3, 0.0);
    y_true(0, 1) = 1.0;

    ActivationSoftmaxLossCategoricalCrossEntropy combo;
    double loss = combo.forward(logits, y_true);
    CHECK(loss > 0.0);

    MatD expected = combo.output;
    expected(0, 1) -= 1.0;

    combo.backward(combo.output, y_true);

    CHECK(combo.dinputs.rows == 1);
    CHECK(combo.dinputs.cols == 3);
    for (size_t j = 0; j < combo.dinputs.cols; ++j) {
        CHECK(combo.dinputs(0, j) == doctest::Approx(expected(0, j)));
    }
}

TEST_CASE("ActivationSoftmaxLossCategoricalCrossEntropy backward throws on sparse shape mismatch")
{
    MatD logits(2, 2);
    logits(0, 0) = 0.1; logits(0, 1) = 0.9;
    logits(1, 0) = 0.2; logits(1, 1) = 0.8;

    VecI y_true = {0, 1};

    ActivationSoftmaxLossCategoricalCrossEntropy combo;
    combo.forward(logits, y_true);

    MatD bad_dvalues(1, 2, 0.0); // wrong number of rows
    CHECK_THROWS_WITH_AS(combo.backward(bad_dvalues, y_true),
                         "ActivationSoftmaxLossCategoricalCrossEntropy::backward: dvalues.rows must match y_true.size()",
                         runtime_error);
}

TEST_CASE("ActivationSoftmaxLossCategoricalCrossEntropy backward throws on invalid sparse class index")
{
    MatD dvalues(1, 2, 0.5);
    VecI bad_labels = {3}; // out of range for 2 classes

    ActivationSoftmaxLossCategoricalCrossEntropy combo;
    CHECK_THROWS_WITH_AS(combo.backward(dvalues, bad_labels),
                         "ActivationSoftmaxLossCategoricalCrossEntropy::backward: class index out of range",
                         runtime_error);
}

TEST_CASE("ActivationSoftmaxLossCategoricalCrossEntropy backward throws on one-hot shape mismatch")
{
    MatD dvalues(1, 2, 0.5);
    MatD y_true(2, 2, 0.0); // mismatched rows

    ActivationSoftmaxLossCategoricalCrossEntropy combo;
    CHECK_THROWS_WITH_AS(combo.backward(dvalues, y_true),
                         "ActivationSoftmaxLossCategoricalCrossEntropy::backward: shapes of dvalues and y_true must match",
                         runtime_error);
}
