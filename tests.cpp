#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include <sstream>
#include <iterator>
#include <limits>
#include <fstream>
#include <cstdio>
#include <string>

#define NNFS_NO_MAIN
#include "NNFS_Diploma.cpp"

// === helpers ===
static LayerDense make_dense_with_grads(double w, double b, double dw, double db)
{
    LayerDense layer(1, 1);
    layer.weights(0, 0) = w;
    layer.biases(0, 0) = b;
    layer.dweights.assign(1, 1);
    layer.dbiases.assign(1, 1);
    layer.dweights(0, 0) = dw;
    layer.dbiases(0, 0) = db;
    return layer;
}

// === is_whole_number / multiplication_overflow_check ===
TEST_CASE("is_whole_number detects integer-like values")
{
    CHECK(is_whole_number(5.0));
    CHECK(is_whole_number(5.0 + 1e-8));
    CHECK_FALSE(is_whole_number(5.1));
}

TEST_CASE("multiplication_overflow_check throws on overflow and allows safe values")
{
    CHECK_NOTHROW(multiplication_overflow_check(100, 200, "overflow"));
    const size_t max = std::numeric_limits<size_t>::max();
    CHECK_THROWS_WITH_AS(multiplication_overflow_check(max, 2, "overflow"),
                         "overflow",
                         runtime_error);
}

// === random_gaussian / random_uniform ===
TEST_CASE("random_gaussian produces finite values")
{
    g_rng.seed(0);
    double v1 = random_gaussian();
    double v2 = random_gaussian();
    CHECK(isfinite(v1));
    CHECK(isfinite(v2));
}

TEST_CASE("random_uniform produces values in [0,1) and is deterministic with seed")
{
    g_rng.seed(42);
    double v1 = random_uniform();
    CHECK(v1 >= 0.0);
    CHECK(v1 < 1.0);
    g_rng.seed(42);
    double v2 = random_uniform();
    CHECK(v1 == doctest::Approx(v2));
}

// === Matrix ===
TEST_CASE("Matrix construction, assignment, and access")
{
    Matrix m(2, 3, 1.5);
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

TEST_CASE("Matrix operator() throws on out-of-bounds access")
{
    Matrix m(1, 1, 0.0);
    CHECK_THROWS_WITH_AS(m(1, 0),
                         "Matrix::operator(): index out of bounds",
                         runtime_error);

    const Matrix mc(1, 1, 0.0);
    CHECK_THROWS_WITH_AS(mc(0, 1),
                         "Matrix::operator() const: index out of bounds",
                         runtime_error);
}

TEST_CASE("Matrix check helpers detect empty matrices and row/column vectors")
{
    Matrix empty;
    CHECK(empty.is_empty());

    Matrix row(1, 3, 0.0);
    CHECK_FALSE(row.is_empty());
    CHECK(row.is_row_vector());
    CHECK_FALSE(row.is_col_vector());
    CHECK(row.is_vector());

    Matrix col(3, 1, 0.0);
    CHECK(col.is_col_vector());
    CHECK_FALSE(col.is_row_vector());
    CHECK(col.is_vector());

    Matrix box(2, 2, 0.0);
    CHECK_FALSE(box.is_vector());
}

TEST_CASE("Matrix require_* helpers validate shape and emptiness")
{
    Matrix empty;
    CHECK_THROWS_WITH_AS(empty.require_non_empty("empty"),
                         "empty",
                         runtime_error);

    Matrix m(2, 3, 0.0);
    CHECK_THROWS_WITH_AS(m.require_rows(1, "rows"),
                         "rows",
                         runtime_error);
    CHECK_NOTHROW(m.require_rows(2, "rows"));
    CHECK_THROWS_WITH_AS(m.require_cols(4, "cols"),
                         "cols",
                         runtime_error);
    CHECK_THROWS_WITH_AS(m.require_shape(2, 2, "shape"),
                         "shape",
                         runtime_error);
}

TEST_CASE("Matrix print writes expected output")
{
    std::ostringstream oss;
    auto* old_buf = cout.rdbuf(oss.rdbuf());

    Matrix m(2, 2);
    m(0, 0) = 1.0; m(0, 1) = 2.0;
    m(1, 0) = 3.0; m(1, 1) = 4.5;
    m.print();

    cout.rdbuf(old_buf);
    CHECK(oss.str() == "1 2\n3 4.5\n");
}

TEST_CASE("Matrix scale_by_scalar scales values and rejects a value of zero")
{
    Matrix m(1, 2);
    m(0, 0) = 2.0;
    m(0, 1) = 4.0;
    m.scale_by_scalar(2);
    CHECK(m(0, 0) == doctest::Approx(1.0));
    CHECK(m(0, 1) == doctest::Approx(2.0));

    Matrix bad(1, 1, 2.0);
    CHECK_THROWS_WITH_AS(bad.scale_by_scalar(0),
                         "Matrix::scale_by_scalar: value cannot be zero 0",
                         runtime_error);
}

TEST_CASE("Matrix transpose and argmax handle empty and non-empty cases")
{
    Matrix empty;
    CHECK(empty.transpose().is_empty());
    CHECK(empty.argmax().is_empty());

    Matrix m(2, 3);
    m(0, 0) = 1.0; m(0, 1) = 2.0; m(0, 2) = 3.0;
    m(1, 0) = 4.0; m(1, 1) = 5.0; m(1, 2) = 0.5;

    Matrix t = m.transpose();
    CHECK(t.rows == 3);
    CHECK(t.cols == 2);
    CHECK(t(0, 0) == doctest::Approx(1.0));
    CHECK(t(2, 1) == doctest::Approx(0.5));

    Matrix arg = m.argmax();
    CHECK(arg.rows == 2);
    CHECK(arg.cols == 1);
    CHECK(arg(0, 0) == doctest::Approx(2.0));
    CHECK(arg(1, 0) == doctest::Approx(1.0));
}

TEST_CASE("Matrix slice_rows and slice_cols work and validate bounds")
{
    Matrix m(3, 4);
    double v = 1.0;
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            m(i, j) = v++;
        }
    }

    Matrix rows = m.slice_rows(1, 3);
    CHECK(rows.rows == 2);
    CHECK(rows.cols == 4);
    CHECK(rows(0, 0) == doctest::Approx(m(1, 0)));
    CHECK(rows(1, 3) == doctest::Approx(m(2, 3)));

    Matrix cols = m.slice_cols(1, 3);
    CHECK(cols.rows == 3);
    CHECK(cols.cols == 2);
    CHECK(cols(0, 0) == doctest::Approx(m(0, 1)));
    CHECK(cols(2, 1) == doctest::Approx(m(2, 2)));

    CHECK_THROWS_WITH_AS(m.slice_rows(2, 1),
                         "Matrix::slice_rows: invalid slice bounds",
                         runtime_error);
    CHECK_THROWS_WITH_AS(m.slice_rows(0, 4),
                         "Matrix::slice_rows: invalid slice bounds",
                         runtime_error);
    CHECK_THROWS_WITH_AS(m.slice_cols(2, 1),
                         "Matrix::slice_cols: invalid slice bounds",
                         runtime_error);
    CHECK_THROWS_WITH_AS(m.slice_cols(0, 5),
                         "Matrix::slice_cols: invalid slice bounds",
                         runtime_error);
}

TEST_CASE("Matrix as_size_t validates integer-like values")
{
    Matrix m(1, 2);
    m(0, 0) = 2.0;
    m(0, 1) = 2.5;
    CHECK(m.as_size_t(0, 0) == static_cast<size_t>(2));

    CHECK_THROWS_WITH_AS(m.as_size_t(0, 1),
                         "Matrix::as_size_t: value is not integer-like",
                         runtime_error);

    Matrix n(1, 1);
    n(0, 0) = -1.0;
    CHECK_THROWS_WITH_AS(n.as_size_t(0, 0),
                         "Matrix::as_size_t: integer out of range",
                         runtime_error);
}

TEST_CASE("Matrix scalar_mean computes mean and rejects empty matrix")
{
    Matrix m(2, 2);
    m(0, 0) = 1.0; m(0, 1) = 3.0;
    m(1, 0) = 5.0; m(1, 1) = 7.0;
    CHECK(m.scalar_mean() == doctest::Approx(4.0));

    Matrix empty;
    CHECK_THROWS_WITH_AS(empty.scalar_mean(),
                         "Matrix::scalar_mean: cannot find mean of empty matrix",
                         runtime_error);
}

TEST_CASE("Matrix dot multiplies matrices and validates shapes")
{
    Matrix a(2, 3);
    a(0, 0) = 1.0; a(0, 1) = 2.0; a(0, 2) = 3.0;
    a(1, 0) = 4.0; a(1, 1) = 5.0; a(1, 2) = 6.0;

    Matrix b(3, 2);
    b(0, 0) = 7.0;  b(0, 1) = 8.0;
    b(1, 0) = 9.0;  b(1, 1) = 10.0;
    b(2, 0) = 11.0; b(2, 1) = 12.0;

    Matrix c = Matrix::dot(a, b);
    CHECK(c.rows == 2);
    CHECK(c.cols == 2);
    CHECK(c(0, 0) == doctest::Approx(58.0));
    CHECK(c(0, 1) == doctest::Approx(64.0));
    CHECK(c(1, 0) == doctest::Approx(139.0));
    CHECK(c(1, 1) == doctest::Approx(154.0));

    Matrix empty;
    CHECK_THROWS_WITH_AS(Matrix::dot(empty, b),
                         "Matrix::dot: matrices must not be empty",
                         runtime_error);
    CHECK_THROWS_WITH_AS(Matrix::dot(a, empty),
                         "Matrix::dot: matrices must not be empty",
                         runtime_error);

    Matrix bad(4, 1, 0.0);
    CHECK_THROWS_WITH_AS(Matrix::dot(a, bad),
                         "Matrix::dot: matrices have incompatible shapes",
                         runtime_error);
}

TEST_CASE("Matrix shuffle_rows_with validates inputs and shuffles with row/col labels")
{
    Matrix empty;
    Matrix y_empty(1, 1, 0.0);
    CHECK_THROWS_WITH_AS(empty.shuffle_rows_with(y_empty),
                         "shuffle_rows_with: base matrix must be non-empty",
                         runtime_error);

    Matrix X(3, 2);
    X(0, 0) = 10.0; X(0, 1) = 11.0;
    X(1, 0) = 20.0; X(1, 1) = 21.0;
    X(2, 0) = 30.0; X(2, 1) = 31.0;

    Matrix bad_y(2, 2, 0.0);
    CHECK_THROWS_WITH_AS(X.shuffle_rows_with(bad_y),
                         "shuffle_rows_with: y must be shape (1,N) or (N,1), where N = base matrix rows",
                         runtime_error);

    Matrix X_row = X;
    Matrix y_row(1, 3);
    y_row(0, 0) = 0.0; y_row(0, 1) = 1.0; y_row(0, 2) = 2.0;
    g_rng.seed(0);
    X_row.shuffle_rows_with(y_row);
    for (size_t i = 0; i < X_row.rows; ++i) {
        const size_t label = y_row.as_size_t(0, i);
        CHECK(X_row(i, 0) == doctest::Approx((label + 1) * 10.0));
        CHECK(X_row(i, 1) == doctest::Approx((label + 1) * 10.0 + 1.0));
    }

    Matrix X_col = X;
    Matrix y_col(3, 1);
    y_col(0, 0) = 0.0; y_col(1, 0) = 1.0; y_col(2, 0) = 2.0;
    g_rng.seed(0);
    X_col.shuffle_rows_with(y_col);
    for (size_t i = 0; i < X_col.rows; ++i) {
        const size_t label = y_col.as_size_t(i, 0);
        CHECK(X_col(i, 0) == doctest::Approx((label + 1) * 10.0));
        CHECK(X_col(i, 1) == doctest::Approx((label + 1) * 10.0 + 1.0));
    }

    // rows < 2 early return branch
    Matrix X_one(1, 2);
    X_one(0, 0) = 7.0; X_one(0, 1) = 8.0;
    Matrix y_one(1, 1, 3.0);
    X_one.shuffle_rows_with(y_one);
    CHECK(X_one(0, 0) == doctest::Approx(7.0));
    CHECK(y_one(0, 0) == doctest::Approx(3.0));
}

// === fashion_mnist_create ===
TEST_CASE("fashion_mnist_create loads and normalizes data")
{
    Matrix X_train;
    Matrix y_train;
    Matrix X_test;
    Matrix y_test;

    fashion_mnist_create(X_train, y_train, X_test, y_test);

    CHECK(X_train.rows > 0);
    CHECK(X_test.rows > 0);
    CHECK(X_train.cols == 784);
    CHECK(X_test.cols == 784);
    CHECK(y_train.rows == X_train.rows);
    CHECK(y_test.rows == X_test.rows);
    CHECK(y_train.cols == 1);
    CHECK(y_test.cols == 1);

    const size_t max_train_rows = min<size_t>(X_train.rows, 5);
    for (size_t i = 0; i < max_train_rows; ++i) {
        for (size_t j = 0; j < 10; ++j) {
            CHECK(isfinite(X_train(i, j)));
            CHECK(X_train(i, j) >= -1.0);
            CHECK(X_train(i, j) <= 1.0);
        }
        const size_t label = y_train.as_size_t(i, 0);
        CHECK(label < 10);
    }
}

// === generate_spiral_data ===
TEST_CASE("generate_spiral_data validates arguments and overflow")
{
    Matrix X;
    Matrix y;
    CHECK_THROWS_WITH_AS(generate_spiral_data(1, 3, X, y),
                         "generate_spiral_data: invalid arguments",
                         runtime_error);
    CHECK_THROWS_WITH_AS(generate_spiral_data(10, 0, X, y),
                         "generate_spiral_data: invalid arguments",
                         runtime_error);

    const size_t max = std::numeric_limits<size_t>::max();
    CHECK_THROWS_WITH_AS(generate_spiral_data(max, 2, X, y),
                         "generate_spiral_data: total_samples overflow",
                         runtime_error);
}

TEST_CASE("generate_spiral_data produces expected shapes and labels")
{
    Matrix X;
    Matrix y;
    g_rng.seed(0);
    generate_spiral_data(5, 3, X, y);

    CHECK(X.rows == 15);
    CHECK(X.cols == 2);
    CHECK(y.rows == 15);
    CHECK(y.cols == 1);

    for (size_t i = 0; i < y.rows; ++i) {
        size_t label = y.as_size_t(i, 0);
        CHECK(label < 3);
    }
    for (size_t i = 0; i < X.rows; ++i) {
        for (size_t j = 0; j < X.cols; ++j) {
            CHECK(isfinite(X(i, j)));
        }
    }
}

// === generate_vertical_data ===
TEST_CASE("generate_vertical_data validates arguments and overflow")
{
    Matrix X;
    Matrix y;
    CHECK_THROWS_WITH_AS(generate_vertical_data(0, 3, X, y),
                         "generate_vertical_data: invalid arguments",
                         runtime_error);
    CHECK_THROWS_WITH_AS(generate_vertical_data(5, 0, X, y),
                         "generate_vertical_data: invalid arguments",
                         runtime_error);

    const size_t max = std::numeric_limits<size_t>::max();
    CHECK_THROWS_WITH_AS(generate_vertical_data(max, 2, X, y),
                         "generate_vertical_data: total_samples overflow",
                         runtime_error);
}

TEST_CASE("generate_vertical_data produces expected shapes and labels")
{
    Matrix X;
    Matrix y;
    g_rng.seed(0);
    generate_vertical_data(4, 3, X, y);

    CHECK(X.rows == 12);
    CHECK(X.cols == 2);
    CHECK(y.rows == 12);
    CHECK(y.cols == 1);

    for (size_t i = 0; i < y.rows; ++i) {
        size_t label = y.as_size_t(i, 0);
        CHECK(label < 3);
        CHECK(isfinite(X(i, 0)));
        CHECK(isfinite(X(i, 1)));
    }
}

// === generate_sine_data ===
TEST_CASE("generate_sine_data validates arguments and produces sine pairs")
{
    Matrix X;
    Matrix y;
    CHECK_THROWS_WITH_AS(generate_sine_data(1, X, y),
                         "generate_sine_data: invalid arguments",
                         runtime_error);

    generate_sine_data(5, X, y);
    CHECK(X.rows == 5);
    CHECK(X.cols == 1);
    CHECK(y.rows == 5);
    CHECK(y.cols == 1);

    for (size_t i = 1; i < X.rows; ++i) {
        CHECK(X(i, 0) > X(i - 1, 0));
    }
    for (size_t i = 0; i < y.rows; ++i) {
        CHECK(y(i, 0) <= 1.0);
        CHECK(y(i, 0) >= -1.0);
    }
}

// === plot_scatter_svg ===
TEST_CASE("plot_scatter_svg validates inputs and labels")
{
    Matrix empty;
    CHECK_THROWS_WITH_AS(plot_scatter_svg("unused.svg", empty),
                         "plot_scatter_svg: points must be non-empty",
                         runtime_error);

    Matrix wrong_cols(1, 1, 0.0);
    CHECK_THROWS_WITH_AS(plot_scatter_svg("unused.svg", wrong_cols),
                         "plot_scatter_svg: invalid input data",
                         runtime_error);

    Matrix points(2, 2, 0.5);
    Matrix labels(2, 2, 0.0);
    CHECK_THROWS_WITH_AS(plot_scatter_svg("unused.svg", points, labels),
                         "plot_scatter_svg: labels must be shape (N,1) or (1,N) where N = points.rows",
                         runtime_error);
}

TEST_CASE("plot_scatter_svg rejects zero distance between values and invalid path")
{
    Matrix points(2, 2);
    points(0, 0) = 1.0; points(0, 1) = 0.0;
    points(1, 0) = 1.0; points(1, 1) = 1.0;
    CHECK_THROWS_WITH_AS(plot_scatter_svg("unused.svg", points),
                         "plot_scatter_svg: x and y must have non-zero distance between values",
                         runtime_error);

    Matrix ok(2, 2);
    ok(0, 0) = 0.0; ok(0, 1) = 0.0;
    ok(1, 0) = 1.0; ok(1, 1) = 1.0;
    CHECK_THROWS_WITH_AS(plot_scatter_svg("/nonexistent_dir/plot.svg", ok),
                         "plot_scatter_svg: given path is invalid",
                         runtime_error);
}

TEST_CASE("plot_scatter_svg writes expected SVG with and without labels")
{
    Matrix points(3, 2);
    points(0, 0) = 0.0; points(0, 1) = 0.0;
    points(1, 0) = 1.0; points(1, 1) = 0.5;
    points(2, 0) = 0.5; points(2, 1) = 1.0;

    Matrix labels(3, 1);
    labels(0, 0) = 0.0;
    labels(1, 0) = 1.0;
    labels(2, 0) = 2.0;

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

    const std::string path2 = "test_plot_unlabeled.svg";
    plot_scatter_svg(path2, points);
    std::ifstream file2(path2);
    REQUIRE(file2.good());
    std::string contents((std::istreambuf_iterator<char>(file2)),
                         std::istreambuf_iterator<char>());
    CHECK(contents.find("<circle") != std::string::npos);
    std::remove(path2.c_str());
}

TEST_CASE("plot_scatter_svg accepts row-vector labels")
{
    Matrix points(2, 2);
    points(0, 0) = 0.0; points(0, 1) = 0.0;
    points(1, 0) = 1.0; points(1, 1) = 1.0;

    Matrix labels(1, 2);
    labels(0, 0) = 0.0;
    labels(0, 1) = 1.0;

    const std::string path = "test_plot_row_labels.svg";
    plot_scatter_svg(path, points, labels);
    std::remove(path.c_str());
}

// === LayerDense ===
TEST_CASE("LayerDense constructor validates inputs")
{
    CHECK_THROWS_WITH_AS(LayerDense(0, 1),
                         "LayerDense:  n_inputs and n_neurons must be > 0",
                         runtime_error);
    CHECK_THROWS_WITH_AS(LayerDense(1, 0),
                         "LayerDense:  n_inputs and n_neurons must be > 0",
                         runtime_error);
    CHECK_THROWS_WITH_AS(LayerDense(1, 1, -0.1),
                         "LayerDense: regularizers must be non-negative",
                         runtime_error);
}

TEST_CASE("LayerDense forward matches known example and stores inputs")
{
    Matrix inputs(3, 4);
    inputs(0, 0) = 1.0;  inputs(0, 1) = 2.0;  inputs(0, 2) = 3.0;  inputs(0, 3) = 2.5;
    inputs(1, 0) = 2.0;  inputs(1, 1) = 5.0;  inputs(1, 2) = -1.0; inputs(1, 3) = 2.0;
    inputs(2, 0) = -1.5; inputs(2, 1) = 2.7;  inputs(2, 2) = 3.3;  inputs(2, 3) = -0.8;

    LayerDense layer(4, 3);
    layer.weights.assign(4, 3);
    layer.weights(0, 0) = 0.2;  layer.weights(0, 1) = 0.5;   layer.weights(0, 2) = -0.26;
    layer.weights(1, 0) = 0.8;  layer.weights(1, 1) = -0.91; layer.weights(1, 2) = -0.27;
    layer.weights(2, 0) = -0.5; layer.weights(2, 1) = 0.26;  layer.weights(2, 2) = 0.17;
    layer.weights(3, 0) = 1.0;  layer.weights(3, 1) = -0.5;  layer.weights(3, 2) = 0.87;
    layer.biases.assign(1, 3);
    layer.biases(0, 0) = 2.0; layer.biases(0, 1) = 3.0; layer.biases(0, 2) = 0.5;

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

    CHECK(layer.inputs.rows == inputs.rows);
    CHECK(layer.inputs.cols == inputs.cols);
}

TEST_CASE("LayerDense forward overload accepts training flag")
{
    Matrix inputs(1, 2, 1.0);
    LayerDense layer(2, 1);
    layer.forward(inputs, true);
    CHECK(layer.output.rows == 1);
    CHECK(layer.output.cols == 1);
}

TEST_CASE("LayerDense forward validates inputs and shapes")
{
    Matrix inputs(1, 2, 1.0);
    LayerDense empty_weights(2, 1);
    empty_weights.weights.assign(0, 0);
    CHECK_THROWS_WITH_AS(empty_weights.forward(inputs),
                         "LayerDense::forward: weights must be initialized",
                         runtime_error);

    Matrix empty;
    LayerDense layer(2, 1);
    CHECK_THROWS_WITH_AS(layer.forward(empty),
                         "LayerDense::forward: inputs must be non-empty",
                         runtime_error);

    Matrix bad_inputs(1, 3, 1.0);
    CHECK_THROWS_WITH_AS(layer.forward(bad_inputs),
                         "LayerDense::forward: inputs.cols must match weights.rows",
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

    LayerDense layer(2, 2);
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

    CHECK(layer.dweights(0, 0) == doctest::Approx(10.0));
    CHECK(layer.dweights(0, 1) == doctest::Approx(14.0));
    CHECK(layer.dweights(1, 0) == doctest::Approx(14.0));
    CHECK(layer.dweights(1, 1) == doctest::Approx(20.0));

    CHECK(layer.dbiases(0, 0) == doctest::Approx(4.0));
    CHECK(layer.dbiases(0, 1) == doctest::Approx(6.0));

    CHECK(layer.dinputs(0, 0) == doctest::Approx(1.0));
    CHECK(layer.dinputs(0, 1) == doctest::Approx(2.0));
    CHECK(layer.dinputs(1, 0) == doctest::Approx(3.0));
    CHECK(layer.dinputs(1, 1) == doctest::Approx(4.0));

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
    LayerDense layer(2, 2, 0.3, 0.7, 0.5, 0.9);
    layer.weights(0, 0) = -1.0; layer.weights(0, 1) = 2.0;
    layer.weights(1, 0) = -3.0; layer.weights(1, 1) = 4.0;
    layer.biases(0, 0) = -5.0; layer.biases(0, 1) = 6.0;

    Matrix inputs(1, 2);
    inputs(0, 0) = 1.0; inputs(0, 1) = 1.0;
    layer.forward(inputs);

    Matrix dvalues(1, 2);
    dvalues(0, 0) = 0.1; dvalues(0, 1) = -0.2;
    layer.backward(dvalues);

    CHECK(layer.dweights(0, 0) == doctest::Approx(-1.6));
    CHECK(layer.dweights(0, 1) == doctest::Approx(2.9));
    CHECK(layer.dweights(1, 0) == doctest::Approx(-4.4));
    CHECK(layer.dweights(1, 1) == doctest::Approx(5.7));

    CHECK(layer.dbiases(0, 0) == doctest::Approx(-9.4));
    CHECK(layer.dbiases(0, 1) == doctest::Approx(11.1));
}

// === LayerDropout ===
TEST_CASE("LayerDropout constructor validates rate")
{
    CHECK_THROWS_WITH_AS(LayerDropout(1.0),
                         "LayerDropout: rate must be in (0,1]",
                         runtime_error);
    CHECK_THROWS_WITH_AS(LayerDropout(-0.1),
                         "LayerDropout: rate must be in (0,1]",
                         runtime_error);
}

TEST_CASE("LayerDropout forward scales activations with training flag")
{
    Matrix inputs(1, 3);
    inputs(0, 0) = 2.0;
    inputs(0, 1) = -3.0;
    inputs(0, 2) = 4.0;

    LayerDropout dropout(0.2); // keep rate 0.8

    dropout.forward(inputs, false);
    CHECK(dropout.scaled_binary_mask(0, 0) == doctest::Approx(1.0));
    CHECK(dropout.scaled_binary_mask(0, 1) == doctest::Approx(1.0));
    CHECK(dropout.scaled_binary_mask(0, 2) == doctest::Approx(1.0));
    CHECK(dropout.output(0, 0) == doctest::Approx(inputs(0, 0)));
    CHECK(dropout.output(0, 1) == doctest::Approx(inputs(0, 1)));
    CHECK(dropout.output(0, 2) == doctest::Approx(inputs(0, 2)));

    g_rng.seed(0);
    double expected[3];
    const double keep_rate = 0.8;
    for (double& m : expected) {
        double r = random_uniform();
        m = (r < keep_rate) ? (1.0 / keep_rate) : 0.0;
    }

    g_rng.seed(0);
    dropout.forward(inputs, true);
    CHECK(dropout.scaled_binary_mask(0, 0) == doctest::Approx(expected[0]));
    CHECK(dropout.scaled_binary_mask(0, 1) == doctest::Approx(expected[1]));
    CHECK(dropout.scaled_binary_mask(0, 2) == doctest::Approx(expected[2]));
}

TEST_CASE("LayerDropout forward/backward validates shapes")
{
    LayerDropout dropout(0.2);
    Matrix empty;
    CHECK_THROWS_WITH_AS(dropout.forward(empty),
                         "LayerDropout::forward: inputs must be non-empty",
                         runtime_error);

    dropout.scaled_binary_mask.assign(1, 2);
    dropout.scaled_binary_mask(0, 0) = 5.0;
    dropout.scaled_binary_mask(0, 1) = 0.0;

    Matrix dvalues(1, 2);
    dvalues(0, 0) = 3.0; dvalues(0, 1) = 4.0;
    dropout.backward(dvalues);
    CHECK(dropout.dinputs(0, 0) == doctest::Approx(15.0));
    CHECK(dropout.dinputs(0, 1) == doctest::Approx(0.0));

    Matrix bad_dvalues(2, 2, 0.0);
    CHECK_THROWS_WITH_AS(dropout.backward(bad_dvalues),
                         "LayerDropout::backward: dvalues shape mismatch",
                         runtime_error);

    Matrix empty_dvalues;
    CHECK_THROWS_WITH_AS(dropout.backward(empty_dvalues),
                         "LayerDropout::backward: dvalues must be non-empty",
                         runtime_error);
}

// === LayerInput ===
TEST_CASE("LayerInput forwards inputs and validates emptiness")
{
    Matrix inputs(2, 2);
    inputs(0, 0) = 1.0; inputs(0, 1) = 2.0;
    inputs(1, 0) = 3.0; inputs(1, 1) = 4.0;

    LayerInput input_layer;
    input_layer.forward(inputs);
    CHECK(input_layer.output.rows == 2);
    CHECK(input_layer.output.cols == 2);
    CHECK(input_layer.output(1, 1) == doctest::Approx(4.0));

    input_layer.forward(inputs, true);
    CHECK(input_layer.output(0, 1) == doctest::Approx(2.0));

    Matrix empty;
    CHECK_THROWS_WITH_AS(input_layer.forward(empty),
                         "LayerInput::forward: inputs must be non-empty",
                         runtime_error);
}

// === ActivationReLU ===
TEST_CASE("ActivationReLU forward and backward")
{
    Matrix inputs(2, 3);
    inputs(0, 0) = -1.0; inputs(0, 1) = 0.0; inputs(0, 2) = 2.5;
    inputs(1, 0) = 3.0;  inputs(1, 1) = -0.1; inputs(1, 2) = 0.0;

    ActivationReLU activation;
    activation.forward(inputs);
    CHECK(activation.output(0, 0) == doctest::Approx(0.0));
    CHECK(activation.output(0, 2) == doctest::Approx(2.5));
    CHECK(activation.output(1, 0) == doctest::Approx(3.0));

    Matrix dvalues(2, 2);
    dvalues(0, 0) = 5.0; dvalues(0, 1) = 6.0;
    dvalues(1, 0) = 7.0; dvalues(1, 1) = 8.0;

    Matrix inputs2(2, 2);
    inputs2(0, 0) = -1.0; inputs2(0, 1) = 1.0;
    inputs2(1, 0) = 0.0;  inputs2(1, 1) = 2.0;
    activation.forward(inputs2);
    activation.backward(dvalues);
    CHECK(activation.dinputs(0, 0) == doctest::Approx(0.0));
    CHECK(activation.dinputs(0, 1) == doctest::Approx(6.0));
    CHECK(activation.dinputs(1, 0) == doctest::Approx(0.0));
    CHECK(activation.dinputs(1, 1) == doctest::Approx(8.0));

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

TEST_CASE("ActivationReLU overload and predictions passthrough")
{
    Matrix inputs(1, 2);
    inputs(0, 0) = -1.0;
    inputs(0, 1) = 2.0;

    ActivationReLU activation;
    activation.forward(inputs, true);
    CHECK(activation.output(0, 0) == doctest::Approx(0.0));
    CHECK(activation.output(0, 1) == doctest::Approx(2.0));

    Matrix preds = activation.predictions(activation.output);
    CHECK(preds.rows == activation.output.rows);
    CHECK(preds.cols == activation.output.cols);
    CHECK(preds(0, 1) == doctest::Approx(2.0));
}

// === ActivationSoftmax ===
TEST_CASE("ActivationSoftmax computes probabilities and predictions")
{
    Matrix inputs(2, 3);
    inputs(0, 0) = 0.0; inputs(0, 1) = 1.0; inputs(0, 2) = 2.0;
    inputs(1, 0) = 0.0; inputs(1, 1) = 0.0; inputs(1, 2) = 0.0;

    ActivationSoftmax activation;
    activation.forward(inputs);

    CHECK(activation.output(0, 0) == doctest::Approx(0.0900306));
    CHECK(activation.output(0, 1) == doctest::Approx(0.2447285));
    CHECK(activation.output(0, 2) == doctest::Approx(0.6652409));
    CHECK(activation.output(1, 0) == doctest::Approx(1.0 / 3.0));

    Matrix preds = activation.predictions(activation.output);
    CHECK(preds.rows == 2);
    CHECK(preds.cols == 1);
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
                         "ActivationSoftmax::predictions: computation of softmax predictions requires outputs.cols >= 2",
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
    CHECK(activation.dinputs(0, 0) == doctest::Approx(0.5));
    CHECK(activation.dinputs(0, 1) == doctest::Approx(-0.5));

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
    inputs(0, 0) = -std::numeric_limits<double>::infinity();
    ActivationSoftmax activation;
    CHECK_THROWS_WITH_AS(activation.forward(inputs),
                         "ActivationSoftmax: invalid sum of exponentials",
                         runtime_error);

    Matrix empty;
    CHECK_THROWS_WITH_AS(activation.predictions(empty),
                         "ActivationSoftmax::predictions: outputs must be non-empty",
                         runtime_error);
}

// === ActivationSigmoid ===
TEST_CASE("ActivationSigmoid forward/backward and predictions")
{
    Matrix inputs(2, 2);
    inputs(0, 0) = 0.0; inputs(0, 1) = 1.0;
    inputs(1, 0) = -1.0; inputs(1, 1) = 2.0;

    ActivationSigmoid activation;
    activation.forward(inputs);
    CHECK(activation.output(0, 0) == doctest::Approx(0.5));
    CHECK(activation.output(0, 1) == doctest::Approx(1.0 / (1.0 + exp(-1.0))));

    Matrix upstream(1, 2, 1.0);
    Matrix inputs2(1, 2);
    inputs2(0, 0) = 0.0; inputs2(0, 1) = -2.0;
    activation.forward(inputs2);
    activation.backward(upstream);
    const double s0 = 1.0 / (1.0 + exp(-inputs2(0, 0)));
    const double s1 = 1.0 / (1.0 + exp(-inputs2(0, 1)));
    CHECK(activation.dinputs(0, 0) == doctest::Approx((1.0 - s0) * s0));
    CHECK(activation.dinputs(0, 1) == doctest::Approx((1.0 - s1) * s1));

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

// === ActivationLinear ===
TEST_CASE("ActivationLinear forward/backward")
{
    Matrix inputs(1, 3);
    inputs(0, 0) = -1.0; inputs(0, 1) = 0.5; inputs(0, 2) = 2.0;

    ActivationLinear activation;
    activation.forward(inputs);
    CHECK(activation.output(0, 0) == doctest::Approx(-1.0));
    CHECK(activation.output(0, 2) == doctest::Approx(2.0));

    Matrix upstream(1, 2);
    upstream(0, 0) = 3.0; upstream(0, 1) = -4.0;
    Matrix inputs2(1, 2);
    inputs2(0, 0) = 0.1; inputs2(0, 1) = -0.2;
    activation.forward(inputs2);
    activation.backward(upstream);
    CHECK(activation.dinputs(0, 0) == doctest::Approx(3.0));
    CHECK(activation.dinputs(0, 1) == doctest::Approx(-4.0));

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

// === Loss base ===
TEST_CASE("Loss::calculate validates non-empty inputs")
{
    struct DummyLoss : Loss {
        Matrix forward(const Matrix& output, const Matrix&) const override
        {
            Matrix losses(1, output.rows, 1.0);
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
            Matrix losses(1, output.rows, 2.0);
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
    CHECK(loss.calculate(out, y, reg) == doctest::Approx(2.0));
    CHECK(reg == doctest::Approx(0.0));

    double reg_accum = 0.0;
    CHECK(loss.calculate_accumulated(reg_accum) == doctest::Approx(2.0));
    CHECK(reg_accum == doctest::Approx(0.0));

    loss.new_pass();
    CHECK_THROWS_WITH_AS(loss.calculate_accumulated(),
                         "Loss::calculate_accumulated: accumulated_count must be > 0",
                         runtime_error);
}

TEST_CASE("Loss::regularization_loss_layer validates and computes sums")
{
    LayerDense layer(2, 2, 0.3, 0.7, 0.5, 0.9);
    layer.weights(0, 0) = 1.0;  layer.weights(0, 1) = -2.0;
    layer.weights(1, 0) = -3.0; layer.weights(1, 1) = 4.0;
    layer.biases(0, 0) = 3.0; layer.biases(0, 1) = -4.0;

    const double reg = Loss::regularization_loss_layer(layer);
    CHECK(reg == doctest::Approx(50.0));

    LayerDense bad(1, 1);
    bad.weight_regularizer_l1 = -0.1;
    CHECK_THROWS_WITH_AS(Loss::regularization_loss_layer(bad),
                         "Loss::regularization_loss_layer: regularizer coefficients must be non-negative",
                         runtime_error);

    LayerDense no_weights(1, 1, 0.1, 0.0, 0.0, 0.0);
    no_weights.weights.assign(0, 0);
    CHECK_THROWS_WITH_AS(Loss::regularization_loss_layer(no_weights),
                         "Loss::regularization_loss_layer: weights must be non-empty",
                         runtime_error);

    LayerDense bad_bias_shape(2, 2, 0.0, 0.0, 0.1, 0.0);
    bad_bias_shape.biases.assign(1, 1);
    CHECK_THROWS_WITH_AS(Loss::regularization_loss_layer(bad_bias_shape),
                         "Loss::regularization_loss_layer: biases must have shape (1, n_neurons)",
                         runtime_error);
}

// === LossCategoricalCrossEntropy ===
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
                         "LossCategoricalCrossEntropy::forward: y_pred.cols must be >= 2",
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
    CHECK(loss.dinputs(0, 0) == doctest::Approx(-0.50000005).epsilon(1e-6));
    CHECK(loss.dinputs(1, 1) == doctest::Approx(-0.625));

    Matrix one_hot(2, 2, 0.0);
    one_hot(0, 0) = 1.0;
    one_hot(1, 1) = 1.0;
    preds(0, 0) = 0.0; preds(0, 1) = 1.0;
    preds(1, 0) = 0.6; preds(1, 1) = 0.4;
    loss.backward(preds, one_hot);
    CHECK(loss.dinputs(0, 0) == doctest::Approx(-5000000.0).epsilon(1e-6));
    CHECK(loss.dinputs(1, 1) == doctest::Approx(-1.25).epsilon(1e-9));

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
                         "LossCategoricalCrossEntropy::backward: y_pred.cols must be >= 2",
                         runtime_error);
}

// === LossBinaryCrossentropy ===
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
    CHECK(loss.dinputs(0, 0) == doctest::Approx(-0.27777778));
    CHECK(loss.dinputs(0, 1) == doctest::Approx(0.3125));
    CHECK(loss.dinputs(1, 0) == doctest::Approx(0.35714286));
    CHECK(loss.dinputs(1, 1) == doctest::Approx(-0.41666667));
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

// === LossMeanSquaredError ===
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
    CHECK(loss.dinputs(0, 0) == doctest::Approx(-1.0));
    CHECK(loss.dinputs(1, 0) == doctest::Approx(1.0));
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

// === LossMeanAbsoluteError ===
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
    CHECK(loss.dinputs(0, 0) == doctest::Approx(-0.5));
    CHECK(loss.dinputs(1, 0) == doctest::Approx(0.5));
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

// === ActivationSoftmaxLossCategoricalCrossEntropy ===
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
    for (size_t i = 0; i < expected.rows; ++i) {
        for (size_t j = 0; j < expected.cols; ++j) {
            expected(i, j) /= static_cast<double>(expected.rows);
            CHECK(combo.dinputs(i, j) == doctest::Approx(expected(i, j)));
        }
    }

    Matrix one_hot(2, 3, 0.0);
    one_hot(0, 2) = 1.0;
    one_hot(1, 0) = 1.0;
    combo.backward(preds, one_hot);
    expected = preds;
    expected(0, 2) -= 1.0;
    expected(1, 0) -= 1.0;
    for (size_t i = 0; i < expected.rows; ++i) {
        for (size_t j = 0; j < expected.cols; ++j) {
            expected(i, j) /= static_cast<double>(expected.rows);
            CHECK(combo.dinputs(i, j) == doctest::Approx(expected(i, j)));
        }
    }

    Matrix sparse_row(1, 2);
    sparse_row(0, 0) = 0.0;
    sparse_row(0, 1) = 1.0;
    combo.backward(preds, sparse_row);
    expected = preds;
    expected(0, 0) -= 1.0;
    expected(1, 1) -= 1.0;
    for (size_t i = 0; i < expected.rows; ++i) {
        for (size_t j = 0; j < expected.cols; ++j) {
            expected(i, j) /= static_cast<double>(expected.rows);
            CHECK(combo.dinputs(i, j) == doctest::Approx(expected(i, j)));
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
                         "ActivationSoftmaxLossCategoricalCrossEntropy::backward: y_pred.cols must be >= 2",
                         runtime_error);
}

// === Optimizer base ===
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
    CHECK(opt.current_learning_rate == doctest::Approx(1.0));
    CHECK(opt.iterations == 0);

    opt.pre_update_params();
    CHECK(opt.current_learning_rate == doctest::Approx(1.0));
    opt.post_update_params();
    CHECK(opt.iterations == 1);

    opt.pre_update_params();
    CHECK(opt.current_learning_rate == doctest::Approx(1.0 / (1.0 + 0.5 * 1.0)));
    opt.post_update_params();
    CHECK(opt.iterations == 2);

    LayerDense dummy(1, 1);
    opt.update_params(dummy);
}

// === OptimizerSGD ===
TEST_CASE("OptimizerSGD updates weights with and without momentum")
{
    LayerDense layer(1, 1);
    layer.weights(0, 0) = 1.0;
    layer.biases(0, 0) = 0.5;
    layer.dweights.assign(1, 1);
    layer.dbiases.assign(1, 1);
    layer.dweights(0, 0) = 2.0;
    layer.dbiases(0, 0) = 1.0;

    OptimizerSGD sgd_no_momentum(0.1, 0.0, 0.0);
    sgd_no_momentum.update_params(layer);
    CHECK(layer.weights(0, 0) == doctest::Approx(0.8));
    CHECK(layer.biases(0, 0) == doctest::Approx(0.4));

    LayerDense layer_m = make_dense_with_grads(1.0, 0.0, 2.0, 1.0);
    OptimizerSGD sgd_momentum(0.1, 0.0, 0.9);
    sgd_momentum.update_params(layer_m);

    CHECK(layer_m.weight_momentums.rows == 1);
    CHECK(layer_m.weight_momentums.cols == 1);
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
    LayerDense layer(1, 1);
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
    layer.dweights.assign(1, 2);
    CHECK_THROWS_WITH_AS(sgd.update_params(layer),
                         "OptimizerSGD::update_params: dweights must match weights shape",
                         runtime_error);

    layer.dweights.assign(1, 1);
    layer.dbiases.assign(1, 2);
    CHECK_THROWS_WITH_AS(sgd.update_params(layer),
                         "OptimizerSGD::update_params: dbiases must match biases shape",
                         runtime_error);
}

// === OptimizerAdagrad ===
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
    LayerDense layer(1, 1);
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
    layer.dweights.assign(1, 2);
    CHECK_THROWS_WITH_AS(opt.update_params(layer),
                         "OptimizerAdagrad::update_params: dweights must match weights shape",
                         runtime_error);

    layer.dweights.assign(1, 1);
    layer.dbiases.assign(1, 2);
    CHECK_THROWS_WITH_AS(opt.update_params(layer),
                         "OptimizerAdagrad::update_params: dbiases must match biases shape",
                         runtime_error);
}

// === OptimizerRMSprop ===
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
    LayerDense layer(1, 1);
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
    layer.dweights.assign(1, 2);
    CHECK_THROWS_WITH_AS(opt.update_params(layer),
                         "OptimizerRMSprop::update_params: dweights must match weights shape",
                         runtime_error);

    layer.dweights.assign(1, 1);
    layer.dbiases.assign(1, 2);
    CHECK_THROWS_WITH_AS(opt.update_params(layer),
                         "OptimizerRMSprop::update_params: dbiases must match biases shape",
                         runtime_error);
}

// === OptimizerAdam ===
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
    LayerDense layer(1, 1);
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
    layer.dweights.assign(1, 2);
    CHECK_THROWS_WITH_AS(opt.update_params(layer),
                         "OptimizerAdam::update_params: dweights must match weights shape",
                         runtime_error);

    layer.dweights.assign(1, 1);
    layer.dbiases.assign(1, 2);
    CHECK_THROWS_WITH_AS(opt.update_params(layer),
                         "OptimizerAdam::update_params: dbiases must match biases shape",
                         runtime_error);
}

// === Accuracy ===
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
    Matrix preds(2, 2);
    preds(0, 0) = 1.0; preds(0, 1) = 0.0;
    preds(1, 0) = 1.0; preds(1, 1) = 0.0;

    Matrix targets(2, 2);
    targets(0, 0) = 1.0; targets(0, 1) = 0.0;
    targets(1, 0) = 1.0; targets(1, 1) = 0.0;
    CHECK(binary_acc.calculate(preds, targets) == doctest::Approx(1.0));

    AccuracyCategorical acc;
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

// === Model ===
TEST_CASE("Model add and set configure layers and trainables")
{
    Model model;
    LayerDense dense(2, 3);
    ActivationReLU relu;
    ActivationSoftmax softmax;

    model.add(dense);
    model.add(relu);
    model.add(softmax);

    CHECK(model.layers.size() == 3);
    CHECK(model.trainable_layers.size() == 1);

    LossCategoricalCrossEntropy loss;
    OptimizerSGD opt(0.1, 0.0, 0.0);
    AccuracyCategorical acc;
    model.set(loss, opt, acc);
}

TEST_CASE("Model finalize validates setup and sets trainable layers")
{
    Model model;
    CHECK_THROWS_WITH_AS(model.finalize(),
                         "Model::finalize: loss, optimizer, and accuracy must be set",
                         runtime_error);

    LayerDense dense(2, 2);
    model.add(dense);
    CHECK_THROWS_WITH_AS(model.finalize(),
                         "Model::finalize: loss, optimizer, and accuracy must be set",
                         runtime_error);

    LossMeanSquaredError loss;
    OptimizerSGD opt(0.1, 0.0, 0.0);
    AccuracyRegression acc;
    {
        Model no_layers;
        no_layers.set(loss, opt, acc);
        CHECK_THROWS_WITH_AS(no_layers.finalize(),
                             "Model::finalize: no layers added",
                             runtime_error);
    }

    model.set(loss, opt, acc);
    model.finalize();
    CHECK(model.trainable_layers.size() == 1);
}

TEST_CASE("Model train validates setup and data")
{
    Model model;
    Matrix X(1, 2, 0.0);
    Matrix y(1, 1, 0.0);

    CHECK_THROWS_WITH_AS(model.train(X, y, 0),
                         "Model::train: loss, optimizer, and accuracy must be set",
                         runtime_error);

    LossMeanSquaredError loss;
    OptimizerSGD opt(0.1, 0.0, 0.0);
    AccuracyRegression acc;
    model.set(loss, opt, acc);

    CHECK_THROWS_WITH_AS(model.train(X, y, 0),
                         "Model::train: no layers added",
                         runtime_error);

    LayerDense dense(2, 1);
    ActivationLinear linear;
    model.add(dense);
    model.add(linear);

    Matrix empty;
    CHECK_THROWS_WITH_AS(model.train(empty, y, 0),
                         "Model::train: X must be non-empty",
                         runtime_error);
    CHECK_THROWS_WITH_AS(model.train(X, empty, 0),
                         "Model::train: y must be non-empty",
                         runtime_error);
}

TEST_CASE("Model train uses separate loss backward path")
{
    Matrix X(2, 1);
    X(0, 0) = 0.0;
    X(1, 0) = 1.0;

    Matrix y(2, 1);
    y(0, 0) = 0.0;
    y(1, 0) = 1.0;

    LayerDense dense(1, 1);
    ActivationLinear linear;

    LossMeanSquaredError loss;
    OptimizerSGD opt(0.1, 0.0, 0.0);
    AccuracyRegression acc;

    Model model;
    model.add(dense);
    model.add(linear);
    model.set(loss, opt, acc);

    model.train(X, y, 1, 0, 0);
    CHECK(model.output.rows == 2);
    CHECK(model.output.cols == 1);
}

TEST_CASE("Model train uses combined softmax + CCE path")
{
    Matrix X(2, 2);
    X(0, 0) = 1.0; X(0, 1) = 0.0;
    X(1, 0) = 0.0; X(1, 1) = 1.0;

    Matrix y(2, 1);
    y(0, 0) = 0.0;
    y(1, 0) = 1.0;

    LayerDense dense(2, 2);
    ActivationSoftmax softmax;

    LossCategoricalCrossEntropy loss;
    OptimizerSGD opt(0.1, 0.0, 0.0);
    AccuracyCategorical acc;

    Model model;
    model.add(dense);
    model.add(softmax);
    model.set(loss, opt, acc);

    model.train(X, y, 1, 0, 0);
    CHECK(model.output.rows == 2);
    CHECK(model.output.cols == 2);
}

TEST_CASE("Model train prints when print_every is non-zero")
{
    Matrix X(1, 2);
    X(0, 0) = 1.0; X(0, 1) = 0.0;

    Matrix y(1, 1);
    y(0, 0) = 0.0;

    LayerDense dense(2, 2);
    ActivationSoftmax softmax;

    LossCategoricalCrossEntropy loss;
    OptimizerSGD opt(0.1, 0.0, 0.0);
    AccuracyCategorical acc;

    Model model;
    model.add(dense);
    model.add(softmax);
    model.set(loss, opt, acc);

    std::ostringstream oss;
    auto* old_buf = cout.rdbuf(oss.rdbuf());
    model.train(X, y, 1, 0, 1);
    cout.rdbuf(old_buf);
    const std::string out = oss.str();
    CHECK(out.find("epoch: 1") != std::string::npos);
    CHECK(out.find("step:") != std::string::npos);
}

TEST_CASE("Model train uses batch slicing when batch_size > 0")
{
    Matrix X(3, 2);
    X(0, 0) = 1.0; X(0, 1) = 0.0;
    X(1, 0) = 0.0; X(1, 1) = 1.0;
    X(2, 0) = 1.0; X(2, 1) = 1.0;

    Matrix y(3, 1);
    y(0, 0) = 0.0;
    y(1, 0) = 1.0;
    y(2, 0) = 0.0;

    LayerDense dense(2, 2);
    ActivationSoftmax softmax;

    LossCategoricalCrossEntropy loss;
    OptimizerSGD opt(0.1, 0.0, 0.0);
    AccuracyCategorical acc;

    Model model;
    model.add(dense);
    model.add(softmax);
    model.set(loss, opt, acc);

    model.train(X, y, 1, 2, 0);
    CHECK(model.output.rows == 1);
    CHECK(model.output.cols == 2);
}

TEST_CASE("Model evaluate validates setup and runs")
{
    Model model;
    Matrix X(1, 1, 0.0);
    Matrix y(1, 1, 0.0);

    LossMeanSquaredError loss;
    OptimizerSGD opt(0.1, 0.0, 0.0);
    AccuracyRegression acc;

    CHECK_THROWS_WITH_AS(model.evaluate(X, y),
                         "Model::evaluate: loss, and accuracy must be set",
                         runtime_error);

    model.set(loss, opt, acc);
    CHECK_THROWS_WITH_AS(model.evaluate(X, y),
                         "Model::evaluate: no layers added",
                         runtime_error);

    LayerDense dense(1, 1);
    ActivationLinear linear;
    model.add(dense);
    model.add(linear);

    Matrix empty;
    CHECK_THROWS_WITH_AS(model.evaluate(empty, y),
                         "Model::evaluate: X must be non-empty",
                         runtime_error);
    CHECK_THROWS_WITH_AS(model.evaluate(X, empty),
                         "Model::evaluate: y must be non-empty",
                         runtime_error);

    model.evaluate(X, y);
}

TEST_CASE("Model evaluate uses batch slicing when batch_size > 0")
{
    Matrix X(3, 1);
    X(0, 0) = 0.0;
    X(1, 0) = 1.0;
    X(2, 0) = 2.0;

    Matrix y(3, 1);
    y(0, 0) = 0.0;
    y(1, 0) = 1.0;
    y(2, 0) = 0.0;

    LayerDense dense(1, 1);
    ActivationLinear linear;

    LossMeanSquaredError loss;
    OptimizerSGD opt(0.1, 0.0, 0.0);
    AccuracyRegression acc;

    Model model;
    model.add(dense);
    model.add(linear);
    model.set(loss, opt, acc);

    model.evaluate(X, y, 2, false);
}
