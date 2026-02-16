#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <ios>
#include <istream>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include "neural_core/accuracy.hpp"
#include "neural_core/core_utils.hpp"
#include "neural_core/layers.hpp"
#include "neural_core/losses.hpp"
#include "neural_core/matrix.hpp"
#include "neural_core/optimizers.hpp"

class Model
{
    using istream = std::istream;
    using ifstream = std::ifstream;
    using ostream = std::ostream;
    using ofstream = std::ofstream;
    using streamsize = std::streamsize;
    using ios = std::ios;
    using size_t = std::size_t;
    using string = std::string;
    template <class T> using vector = std::vector<T>;
    template <class T> using unique_ptr = std::unique_ptr<T>;
    using runtime_error = std::runtime_error;

public:
    Model() = default;

    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    Model(Model&&) noexcept = default;
    Model& operator=(Model&&) noexcept = default;

    void clear();

    LayerDense& add_dense(size_t n_neurons, const string& activation_name,
                          double w_l1 = 0.0, double w_l2 = 0.0,
                          double b_l1 = 0.0, double b_l2 = 0.0);

    LayerDropout& add_dropout(double dropout_rate);

    void configure(Loss& loss_obj, Accuracy& accuracy_obj, Optimizer* optimizer_obj = nullptr);
    void configure(Loss& loss_obj, Accuracy& accuracy_obj, Optimizer& optimizer_obj);

    void finalize();

    void train(const Matrix& X, const Matrix& y, size_t epochs = 1,
               size_t batch_size = 0, size_t print_every = 100,
               const Matrix* X_val = nullptr, const Matrix* y_val = nullptr);

    void evaluate(const Matrix& X, const Matrix& y,
                  size_t batch_size = 0, bool reinit = true);

    Matrix predict(const Matrix& X, size_t batch_size = 0);

    void save_params(const string& path);
    void load_params(const string& path);

    void save(const string& path);
    static Model load(const string& path);

    vector<Matrix> get_params() const;
    void set_params(const vector<Matrix>& params);

    const Matrix& get_output() const;
    const Matrix& get_last_predictions() const;

private:
    enum class LayerKind { Dense, Dropout };
    enum class ActivationKind { ReLU, Softmax, Sigmoid, Linear };

    struct DenseDesc
    {
        size_t n_neurons = 0;
        ActivationKind activation = ActivationKind::ReLU;
        double w_l1 = 0.0;
        double w_l2 = 0.0;
        double b_l1 = 0.0;
        double b_l2 = 0.0;
    };

    struct DropoutDesc
    {
        double dropout_rate = 0.0;
    };

    struct LayerDesc
    {
        LayerKind kind = LayerKind::Dense;
        DenseDesc dense;
        DropoutDesc dropout;
    };

    LayerInput input_layer;
    vector<unique_ptr<Layer>> layers;
    vector<LayerDense*> trainable_layers;

    vector<LayerDesc> architecture;

    Loss* loss = nullptr;
    bool loss_is_cce = false;
    Accuracy* accuracy = nullptr;
    Optimizer* optimizer = nullptr;

    bool finalized = false;

    const Activation* last_activation = nullptr;
    ActivationKind last_activation_kind = ActivationKind::Linear;
    bool use_combined_softmax_ce = false;

    Matrix output;
    Matrix last_predictions;

    ActivationSoftmaxLossCategoricalCrossEntropy combined_softmax_ce;

    static ActivationKind string_to_activation_kind(string name);
    static string activation_kind_to_string(ActivationKind k);

    void require_dense_params_initialized() const;
    size_t dense_layers_count() const;

    void invalidate_compiled_state();

    LayerDense& add_dense_to_model(size_t n_neurons, ActivationKind activation_kind,
                                   double w_l1 = 0.0, double w_l2 = 0.0,
                                   double b_l1 = 0.0, double b_l2 = 0.0);

    void set_architecture(const vector<LayerDesc>& arch);

    void rebuild_caches();
    void update_combined_flag();

    static size_t calc_steps(size_t samples, size_t batch_size);

    static void slice_batch(const Matrix& X, const Matrix* Y, size_t step,
                            size_t batch_size, Matrix& batch_X, Matrix* batch_Y);

    void forward_pass(const Matrix& X, bool training);

    static void write_exact(ostream& os, const void* data, size_t bytes);

    template <class T>
    static void write_trivial(ostream& os, const T& v);

    static void read_exact(istream& is, void* data, size_t bytes);

    template <class T>
    static T read_trivial(istream& is);

    static void write_matrix(ostream& os, const Matrix& m);
    static Matrix read_matrix(istream& is);
};
