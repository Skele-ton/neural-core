#include "neural_core/scatter_plot.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef ENABLE_MATPLOT
#include <matplot/matplot.h>
#endif

using std::max;
using std::min;
using std::runtime_error;
using std::size_t;
using std::string;
using std::vector;

void scatter_plot(const string& path, const Matrix& points, const Matrix& labels)
{
#ifndef ENABLE_MATPLOT
    (void)path;
    (void)points;
    (void)labels;
    throw runtime_error("scatter_plot: built without Matplot++ (ENABLE_MATPLOT=OFF)");
#else
    if (path.empty()) {
        throw runtime_error("scatter_plot: given path is invalid");
    }

    points.require_non_empty("scatter_plot: points must be non-empty");
    if (points.get_cols() < 2) {
        throw runtime_error("scatter_plot: points must have at least 2 columns");
    }

    const size_t num_points = points.get_rows();

    const bool has_labels = !labels.is_empty();
    Matrix labels_sparse;
    if (has_labels) {
        if (labels.is_col_vector() && labels.get_rows() == num_points) {
            labels_sparse = labels;
        } else if (labels.is_row_vector() && labels.get_cols() == num_points) {
            labels_sparse = labels.transpose();
        } else {
            throw runtime_error("scatter_plot: labels must be a 1D vector with shape (N,1) or (1,N), where N = points.get_rows()");
        }

        labels_sparse.require_shape(num_points, 1,
            "scatter_plot: normalized labels must have shape (N,1)");
    }

    double xmin = points(0, 0);
    double xmax = points(0, 0);
    double ymin = points(0, 1);
    double ymax = points(0, 1);
    for (size_t point_index = 1; point_index < num_points; ++point_index) {
        const double x = points(point_index, 0);
        const double y = points(point_index, 1);
        xmin = min(xmin, x);
        xmax = max(xmax, x);
        ymin = min(ymin, y);
        ymax = max(ymax, y);
    }

    const double x_span = xmax - xmin;
    const double y_span = ymax - ymin;
    const double pad_ratio = 0.05;
    const double min_pad = 1e-6;
    const double x_pad = max(x_span * pad_ratio, min_pad);
    const double y_pad = max(y_span * pad_ratio, min_pad);

    vector<vector<double>> xs(1);
    vector<vector<double>> ys(1);
    for (size_t point_index = 0; point_index < num_points; ++point_index) {
        size_t class_id = 0;
        if (has_labels) {
            class_id = labels_sparse.as_size_t(point_index, 0);
        }

        if (class_id >= xs.size()) {
            xs.resize(class_id + 1);
            ys.resize(class_id + 1);
        }

        xs[class_id].push_back(points(point_index, 0));
        ys[class_id].push_back(points(point_index, 1));
    }

    matplot::figure(true);
    matplot::hold(matplot::on);

    for (size_t class_idx = 0; class_idx < xs.size(); ++class_idx) {
        if (xs[class_idx].empty()) continue;
        auto p = matplot::plot(xs[class_idx], ys[class_idx], ".");
        p->marker_size(3.0);
    }

    matplot::title("Scatter Plot");
    matplot::xlabel("x");
    matplot::ylabel("y");
    matplot::axis(matplot::equal);
    matplot::xlim({xmin - x_pad, xmax + x_pad});
    matplot::ylim({ymin - y_pad, ymax + y_pad});
    matplot::grid(matplot::on);

    (void)matplot::save(path);

    // some Matplot++/backend combinations complete file output asynchronously
    // poll briefly to avoid false negatives from immediate existence checks
    constexpr int max_attempts = 100;
    constexpr auto retry_delay = std::chrono::milliseconds(10);
    bool wrote_file = false;
    for (int attempt = 0; attempt < max_attempts; ++attempt) {
        std::error_code ec;
        const bool exists = std::filesystem::exists(path, ec);
        const auto bytes = (exists && !ec) ? std::filesystem::file_size(path, ec) : 0;
        wrote_file = !ec && exists && bytes > 0;
        if (wrote_file) {
            break;
        }
        std::this_thread::sleep_for(retry_delay);
    }

    if (!wrote_file) {
        throw runtime_error("scatter_plot: failed to write output file: " + path);
    }
#endif
}
