#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "doctest.h"

#include "neural_core/accuracy.hpp"
#include "neural_core/activations.hpp"
#include "neural_core/core_utils.hpp"
#include "neural_core/data.hpp"
#include "neural_core/layers.hpp"
#include "neural_core/losses.hpp"
#include "neural_core/matrix.hpp"
#include "neural_core/model.hpp"
#include "neural_core/optimizers.hpp"
#include "neural_core/rng.hpp"
#include "neural_core/scatter_plot.hpp"

using std::cout;
using std::exception;
using std::exp;
using std::ifstream;
using std::ios;
using std::isfinite;
using std::log;
using std::min;
using std::numeric_limits;
using std::ofstream;
using std::ostringstream;
using std::remove;
using std::runtime_error;
using std::size_t;
using std::streambuf;
using std::string;
using std::thread;
using std::vector;

struct CoutSilencer {
    ostringstream buffer;
    streambuf* old = nullptr;

    CoutSilencer()
        : old(cout.rdbuf(buffer.rdbuf()))
    {
    }

    ~CoutSilencer()
    {
        cout.rdbuf(old);
    }
};

inline void reset_deterministic_rng(uint32_t seed)
{
    set_global_seed(seed);
    set_thread_stream_id(0);
}

inline LayerDense make_dense_with_grads(double w, double b, double dw, double db)
{
    LayerDense layer(1, "linear");
    layer.weights.assign(1, 1);
    layer.weights(0, 0) = w;
    layer.biases.assign(1, 1);
    layer.biases(0, 0) = b;

    const double input_value = dw / db;
    Matrix inputs(1, 1);
    inputs(0, 0) = input_value;
    layer.forward(inputs);

    Matrix dvalues(1, 1);
    dvalues(0, 0) = db;
    layer.backward(dvalues);
    return layer;
}

template <typename T>
inline void write_trivial_raw(ofstream& out, const T& v)
{
    out.write(reinterpret_cast<const char*>(&v), static_cast<std::streamsize>(sizeof(T)));
}

inline void write_matrix_raw(ofstream& out, const Matrix& m)
{
    const uint64_t r = static_cast<uint64_t>(m.get_rows());
    const uint64_t c = static_cast<uint64_t>(m.get_cols());
    write_trivial_raw(out, r);
    write_trivial_raw(out, c);
    for (size_t i = 0; i < m.get_rows(); ++i) {
        for (size_t j = 0; j < m.get_cols(); ++j) {
            const double x = m(i, j);
            write_trivial_raw(out, x);
        }
    }
}
