#include "tests/test_common.hpp"

#ifdef ENABLE_MATPLOT
#include <filesystem>

struct PlotFileCleanup
{
    vector<string> paths;

    void track(const string& path)
    {
        paths.push_back(path);
    }

    ~PlotFileCleanup()
    {
        for (const auto& path : paths) {
            remove(path.c_str());
        }
    }
};

static bool can_write_plot_file(const string& path)
{
    std::error_code ec;
    const bool exists = std::filesystem::exists(path, ec);
    const auto bytes = (exists && !ec) ? std::filesystem::file_size(path, ec) : 0;
    const bool ok = !ec && exists && bytes > 0;
    return ok;
}
#endif

TEST_CASE("fashion_mnist_create throws when dataset files are missing")
{
    Matrix X_train;
    Matrix y_train;
    Matrix X_test;
    Matrix y_test;

    CHECK_THROWS_WITH_AS(
        fashion_mnist_create(X_train, y_train, X_test, y_test, "definitely_missing_fashion_mnist_dir"),
        "fashion_mnist_create: dataset files not found under: definitely_missing_fashion_mnist_dir",
        runtime_error);
}

TEST_CASE("fashion_mnist_create loads and normalizes data")
{
    Matrix X_train;
    Matrix y_train;
    Matrix X_test;
    Matrix y_test;

    fashion_mnist_create(X_train, y_train, X_test, y_test);

    CHECK(X_train.get_rows() > 0);
    CHECK(X_test.get_rows() > 0);
    CHECK(X_train.get_cols() == 784);
    CHECK(X_test.get_cols() == 784);
    CHECK(y_train.get_rows() == X_train.get_rows());
    CHECK(y_test.get_rows() == X_test.get_rows());
    CHECK(y_train.get_cols() == 1);
    CHECK(y_test.get_cols() == 1);

    const size_t max_train_rows = min<size_t>(X_train.get_rows(), 5);
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

    const size_t max = numeric_limits<size_t>::max();
    CHECK_THROWS_WITH_AS(generate_spiral_data(max, 2, X, y),
                         "generate_spiral_data: total_samples overflow",
                         runtime_error);
}

TEST_CASE("generate_spiral_data produces expected shapes and labels")
{
    Matrix X;
    Matrix y;
    reset_deterministic_rng(0);
    generate_spiral_data(5, 3, X, y);

    CHECK(X.get_rows() == 15);
    CHECK(X.get_cols() == 2);
    CHECK(y.get_rows() == 15);
    CHECK(y.get_cols() == 1);

    for (size_t i = 0; i < y.get_rows(); ++i) {
        size_t label = y.as_size_t(i, 0);
        CHECK(label < 3);
    }
    for (size_t i = 0; i < X.get_rows(); ++i) {
        for (size_t j = 0; j < X.get_cols(); ++j) {
            CHECK(isfinite(X(i, j)));
        }
    }
}

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

    const size_t max = numeric_limits<size_t>::max();
    CHECK_THROWS_WITH_AS(generate_vertical_data(max, 2, X, y),
                         "generate_vertical_data: total_samples overflow",
                         runtime_error);
}

TEST_CASE("generate_vertical_data produces expected shapes and labels")
{
    Matrix X;
    Matrix y;
    reset_deterministic_rng(0);
    generate_vertical_data(4, 3, X, y);

    CHECK(X.get_rows() == 12);
    CHECK(X.get_cols() == 2);
    CHECK(y.get_rows() == 12);
    CHECK(y.get_cols() == 1);

    for (size_t i = 0; i < y.get_rows(); ++i) {
        size_t label = y.as_size_t(i, 0);
        CHECK(label < 3);
    }
    for (size_t i = 0; i < X.get_rows(); ++i) {
        for (size_t j = 0; j < X.get_cols(); ++j) {
            CHECK(isfinite(X(i, j)));
        }
    }
}

TEST_CASE("generate_sine_data validates arguments and produces sine pairs")
{
    Matrix X;
    Matrix y;
    CHECK_THROWS_WITH_AS(generate_sine_data(1, X, y),
                         "generate_sine_data: invalid arguments",
                         runtime_error);

    generate_sine_data(5, X, y);
    CHECK(X.get_rows() == 5);
    CHECK(X.get_cols() == 1);
    CHECK(y.get_rows() == 5);
    CHECK(y.get_cols() == 1);

    for (size_t i = 1; i < X.get_rows(); ++i) {
        CHECK(X(i, 0) > X(i - 1, 0));
    }
    for (size_t i = 0; i < y.get_rows(); ++i) {
        CHECK(y(i, 0) <= 1.0);
        CHECK(y(i, 0) >= -1.0);
    }
}

TEST_CASE("scatter_plot validates inputs and labels")
{
#ifdef ENABLE_MATPLOT
    Matrix empty;
    CHECK_THROWS_WITH_AS(scatter_plot("unused.png", empty),
                         "scatter_plot: points must be non-empty",
                         runtime_error);

    Matrix wrong_cols(1, 1, 0.0);
    CHECK_THROWS_WITH_AS(scatter_plot("unused.png", wrong_cols),
                         "scatter_plot: points must have at least 2 columns",
                         runtime_error);

    Matrix points(2, 2, 0.5);
    Matrix labels(2, 2, 0.0);
    CHECK_THROWS_WITH_AS(scatter_plot("unused.png", points, labels),
                         "scatter_plot: labels must be a 1D vector with shape (N,1) or (1,N), where N = points.get_rows()",
                         runtime_error);
#else
    Matrix points(1, 2, 0.0);
    CHECK_THROWS_WITH_AS(scatter_plot("unused.png", points),
                         "scatter_plot: built without Matplot++ (ENABLE_MATPLOT=OFF)",
                         runtime_error);
#endif
}

TEST_CASE("scatter_plot validates path")
{
#ifdef ENABLE_MATPLOT
    Matrix points(2, 2);
    points(0, 0) = 0.0; points(0, 1) = 0.0;
    points(1, 0) = 1.0; points(1, 1) = 1.0;

    CHECK_THROWS_WITH_AS(scatter_plot("", points),
                         "scatter_plot: given path is invalid",
                         runtime_error);
#else
    Matrix points(1, 2, 0.0);
    CHECK_THROWS_WITH_AS(scatter_plot("", points),
                         "scatter_plot: built without Matplot++ (ENABLE_MATPLOT=OFF)",
                         runtime_error);
#endif
}

TEST_CASE("scatter_plot writes output with and without labels")
{
#ifdef ENABLE_MATPLOT
    PlotFileCleanup cleanup;

    Matrix points(3, 2);
    points(0, 0) = 0.0; points(0, 1) = 0.0;
    points(1, 0) = 1.0; points(1, 1) = 0.5;
    points(2, 0) = 0.5; points(2, 1) = 1.0;

    Matrix labels(3, 1);
    labels(0, 0) = 0.0;
    labels(1, 0) = 1.0;
    labels(2, 0) = 2.0;

    const string path = "test_plot.png";
    cleanup.track(path);
    REQUIRE_NOTHROW(scatter_plot(path, points, labels));
    CHECK(can_write_plot_file(path));

    const string path2 = "test_plot_unlabeled.png";
    cleanup.track(path2);
    REQUIRE_NOTHROW(scatter_plot(path2, points));
    CHECK(can_write_plot_file(path2));
#endif
}

TEST_CASE("scatter_plot accepts row-vector labels")
{
#ifdef ENABLE_MATPLOT
    PlotFileCleanup cleanup;

    Matrix points(2, 2);
    points(0, 0) = 0.0; points(0, 1) = 0.0;
    points(1, 0) = 1.0; points(1, 1) = 1.0;

    Matrix labels(1, 2);
    labels(0, 0) = 0.0;
    labels(0, 1) = 1.0;

    const string path = "test_plot_row_labels.png";
    cleanup.track(path);
    REQUIRE_NOTHROW(scatter_plot(path, points, labels));
    CHECK(can_write_plot_file(path));
#endif
}
