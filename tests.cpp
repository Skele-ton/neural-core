#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#define NNFS_NO_MAIN
#include "NNFS_Diploma.cpp"

// Matrix tests
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
    CHECK(m.empty() == true);

    MatD t = transpose(m);
    CHECK(t.empty() == true);
}

// matmul tests
TEST_CASE("matmul multiplies small matrices correctly")
{
    MatD a(2, 3);
    a(0, 0) = 1.0; a(0, 1) = 2.0; a(0, 2) = 3.0;
    a(1, 0) = 4.0; a(1, 1) = 5.0; a(1, 2) = 6.0;

    MatD b(3, 2);
    b(0, 0) = 7.0;  b(0, 1) = 8.0;
    b(1, 0) = 9.0;  b(1, 1) = 10.0;
    b(2, 0) = 11.0; b(2, 1) = 12.0;

    MatD c = matmul(a, b);
    CHECK(c.rows == 2);
    CHECK(c.cols == 2);

    CHECK(c(0, 0) == doctest::Approx(58.0));
    CHECK(c(0, 1) == doctest::Approx(64.0));
    CHECK(c(1, 0) == doctest::Approx(139.0));
    CHECK(c(1, 1) == doctest::Approx(154.0));
}

TEST_CASE("matmul throws on empty matrices")
{
    MatD a;
    MatD b(1, 1, 1.0);
    CHECK_THROWS_AS(matmul(a, b), runtime_error);

    MatD c(1, 1, 1.0);
    MatD d;
    CHECK_THROWS_AS(matmul(c, d), runtime_error);
}

TEST_CASE("matmul throws on incompatible shapes")
{
    MatD a(2, 3);
    MatD b(4, 1);
    CHECK_THROWS_AS(matmul(a, b), runtime_error);
}

// transpose tests
TEST_CASE("transpose swaps rows and columns and values")
{
    MatD m(2, 3);
    m(0, 0) = 1.0; m(0, 1) = 2.0; m(0, 2) = 3.0;
    m(1, 0) = 4.0; m(1, 1) = 5.0; m(1, 2) = 6.0;

    MatD t = transpose(m);
    CHECK(t.rows == 3);
    CHECK(t.cols == 2);

    CHECK(t(0, 0) == doctest::Approx(1.0));
    CHECK(t(0, 1) == doctest::Approx(4.0));
    CHECK(t(1, 0) == doctest::Approx(2.0));
    CHECK(t(1, 1) == doctest::Approx(5.0));
    CHECK(t(2, 0) == doctest::Approx(3.0));
    CHECK(t(2, 1) == doctest::Approx(6.0));
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

    CHECK_THROWS_AS(generate_spiral_data(1, 3, X, y), runtime_error);
    CHECK_THROWS_AS(generate_spiral_data(10, 0, X, y), runtime_error);
}

// layer_forward_batch tests
TEST_CASE("layer_forward_batch matches known example")
{
    // inputs: 3 samples x 4 features
    MatD inputs(3, 4);
    inputs(0, 0) = 1.0;  inputs(0, 1) = 2.0;  inputs(0, 2) = 3.0;  inputs(0, 3) = 2.5;
    inputs(1, 0) = 2.0;  inputs(1, 1) = 5.0;  inputs(1, 2) = -1.0; inputs(1, 3) = 2.0;
    inputs(2, 0) = -1.5; inputs(2, 1) = 2.7;  inputs(2, 2) = 3.3;  inputs(2, 3) = -0.8;

    // weights: 3 neurons x 4 inputs
    MatD weights(3, 4);
    weights(0, 0) = 0.2;   weights(0, 1) = 0.8;   weights(0, 2) = -0.5;  weights(0, 3) = 1.0;
    weights(1, 0) = 0.5;   weights(1, 1) = -0.91; weights(1, 2) = 0.26;  weights(1, 3) = -0.5;
    weights(2, 0) = -0.26; weights(2, 1) = -0.27; weights(2, 2) = 0.17;  weights(2, 3) = 0.87;

    VecD biases = {2.0, 3.0, 0.5};

    MatD out = layer_forward_batch(inputs, weights, biases);
    CHECK(out.rows == 3);
    CHECK(out.cols == 3);

    CHECK(out(0, 0) == doctest::Approx(4.8));
    CHECK(out(0, 1) == doctest::Approx(1.21));
    CHECK(out(0, 2) == doctest::Approx(2.385));

    CHECK(out(1, 0) == doctest::Approx(8.9));
    CHECK(out(1, 1) == doctest::Approx(-1.81));
    CHECK(out(1, 2) == doctest::Approx(0.2));

    CHECK(out(2, 0) == doctest::Approx(1.41));
    CHECK(out(2, 1) == doctest::Approx(1.051));
    CHECK(out(2, 2) == doctest::Approx(0.026));
}

TEST_CASE("layer_forward_batch throws on weights/bias mismatch")
{
    MatD inputs(1, 2);
    MatD weights(3, 2);
    VecD biases = {1.0, 2.0}; // size 2, but weights.rows = 3
    CHECK_THROWS_AS(layer_forward_batch(inputs, weights, biases), runtime_error);
}

TEST_CASE("layer_forward_batch throws on input/weight shape mismatch")
{
    MatD inputs(1, 3);       // 1 x 3
    MatD weights(2, 2);      // 2 x 2 (n_inputs = 2, mismatch)
    VecD biases = {1.0, 2.0};
    CHECK_THROWS_AS(layer_forward_batch(inputs, weights, biases), runtime_error);
}

// LayerDense tests
TEST_CASE("LayerDense forward with custom weights and biases")
{
    // prepare inputs as above example
    MatD inputs(3, 4);
    inputs(0, 0) = 1.0;  inputs(0, 1) = 2.0;  inputs(0, 2) = 3.0;  inputs(0, 3) = 2.5;
    inputs(1, 0) = 2.0;  inputs(1, 1) = 5.0;  inputs(1, 2) = -1.0; inputs(1, 3) = 2.0;
    inputs(2, 0) = -1.5; inputs(2, 1) = 2.7;  inputs(2, 2) = 3.3;  inputs(2, 3) = -0.8;

    LayerDense layer(4, 3);        // will override weights/biases
    layer.weights.assign(3, 4);    // 3 neurons x 4 inputs

    layer.weights(0, 0) = 0.2;   layer.weights(0, 1) = 0.8;   layer.weights(0, 2) = -0.5;  layer.weights(0, 3) = 1.0;
    layer.weights(1, 0) = 0.5;   layer.weights(1, 1) = -0.91; layer.weights(1, 2) = 0.26;  layer.weights(1, 3) = -0.5;
    layer.weights(2, 0) = -0.26; layer.weights(2, 1) = -0.27; layer.weights(2, 2) = 0.17;  layer.weights(2, 3) = 0.87;

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
