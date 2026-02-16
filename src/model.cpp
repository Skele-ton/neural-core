#include "neural_core/model.hpp"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <iostream>
#include <limits>
#include <type_traits>
#include <utility>

void Model::clear()
{
    input_layer = LayerInput();
    layers.clear();
    architecture.clear();

    loss = nullptr;
    accuracy = nullptr;
    optimizer = nullptr;
    loss_is_cce = false;

    invalidate_compiled_state();
}

LayerDense& Model::add_dense(size_t n_neurons, const string& activation_name,
                             double w_l1, double w_l2, double b_l1, double b_l2)
{
    const ActivationKind k = string_to_activation_kind(activation_name);
    return add_dense_to_model(n_neurons, k, w_l1, w_l2, b_l1, b_l2);
}

LayerDropout& Model::add_dropout(double dropout_rate)
{
    LayerDesc d;
    d.kind = LayerKind::Dropout;
    d.dropout.dropout_rate = dropout_rate;

    architecture.push_back(d);

    auto layer = std::make_unique<LayerDropout>(dropout_rate);
    LayerDropout& ref = *layer;
    layers.push_back(std::move(layer));

    invalidate_compiled_state();

    return ref;
}

void Model::configure(Loss& loss_obj, Accuracy& accuracy_obj, Optimizer* optimizer_obj)
{
    loss = &loss_obj;
    accuracy = &accuracy_obj;
    optimizer = optimizer_obj;
    loss_is_cce = (dynamic_cast<LossCategoricalCrossEntropy*>(loss) != nullptr);

    if (finalized) {
        update_combined_flag();
    }
}

void Model::configure(Loss& loss_obj, Accuracy& accuracy_obj, Optimizer& optimizer_obj)
{
    configure(loss_obj, accuracy_obj, &optimizer_obj);
}

void Model::finalize()
{
    if (layers.empty()) {
        throw runtime_error("Model::finalize: no layers added");
    }
    if (architecture.empty()) {
        throw runtime_error("Model::finalize: internal error (empty architecture)");
    }
    if (layers.size() != architecture.size()) {
        throw runtime_error("Model::finalize: internal error (layers/spec size mismatch)");
    }
    if (architecture.back().kind == LayerKind::Dropout) {
        throw runtime_error("Model::finalize: final layer cannot be dropout");
    }

    rebuild_caches();
    update_combined_flag();

    finalized = true;
}

void Model::train(const Matrix& X, const Matrix& y, size_t epochs, size_t batch_size,
                  size_t print_every, const Matrix* X_val, const Matrix* y_val)
{
    if (!finalized) {
        finalize();
    }

    if (!loss || !accuracy || !optimizer) {
        throw runtime_error("Model::train: loss, accuracy and optimizer must be set");
    }

    X.require_non_empty("Model::train: X must be non-empty");
    y.require_non_empty("Model::train: y must be non-empty");

    if (batch_size > X.get_rows()) {
        throw runtime_error("Model::train: batch_size cannot exceed number of samples aka X.get_rows()");
    }

    const size_t samples = X.get_rows();

    Matrix Y;
    if (y.get_rows() == 1 && y.get_cols() == samples) {
        Y = y.transpose();
    } else {
        y.require_rows(samples, "Model::train: non-row-vector y must have same number of rows as X");
        Y = y;
    }

    accuracy->init(Y);

    const size_t steps = calc_steps(samples, batch_size);

    for (size_t epoch = 1; epoch <= epochs; ++epoch) {
        std::cout << "epoch: " << epoch << '\n';

        loss->new_pass();
        accuracy->new_pass();

        for (size_t step = 0; step < steps; ++step) {
            Matrix batch_X;
            Matrix batch_y;
            slice_batch(X, &Y, step, batch_size, batch_X, &batch_y);

            forward_pass(batch_X, true);

            double reg_loss = 0.0;
            const double data_loss = loss->calculate(output, batch_y, reg_loss, trainable_layers);
            const double total_loss = data_loss + reg_loss;

            last_predictions = last_activation->predictions(output);
            const double acc = accuracy->calculate(last_predictions, batch_y);

            const Matrix* dvalues = nullptr;
            auto iter = layers.rbegin();

            if (use_combined_softmax_ce) {
                combined_softmax_ce.backward(output, batch_y);
                dvalues = &combined_softmax_ce.get_dinputs();

                Layer* last_layer = iter->get();
                last_layer->backward(*dvalues, false);
                dvalues = &last_layer->get_dinputs();
                ++iter;
            } else {
                loss->backward(output, batch_y);
                dvalues = &loss->get_dinputs();
            }

            for (; iter != layers.rend(); ++iter) {
                (*iter)->backward(*dvalues, true);
                dvalues = &(*iter)->get_dinputs();
            }

            optimizer->pre_update_params();
            for (LayerDense* dense : trainable_layers) {
                optimizer->update_params(*dense);
            }
            optimizer->post_update_params();

            if (print_every != 0 && ((step % print_every) == 0 || step == steps - 1)) {
                std::cout << "  step: " << step
                          << ", accuracy: " << acc
                          << ", loss: " << total_loss
                          << " (data loss: " << data_loss
                          << ", regularization loss: " << reg_loss
                          << ")"
                          << ", learning rate: " << optimizer->get_current_learning_rate()
                          << '\n';
            }
        }

        double epoch_reg_loss = 0.0;
        const double epoch_data_loss = loss->calculate_accumulated(epoch_reg_loss, trainable_layers);
        const double epoch_loss = epoch_data_loss + epoch_reg_loss;
        const double epoch_accuracy = accuracy->calculate_accumulated();

        std::cout << "training - accuracy: " << epoch_accuracy
                  << ", loss: " << epoch_loss
                  << " (data loss: " << epoch_data_loss
                  << ", regularization loss: " << epoch_reg_loss
                  << ")"
                  << ", learning rate: " << optimizer->get_current_learning_rate()
                  << '\n';

        if (X_val && y_val) {
            evaluate(*X_val, *y_val, batch_size, false);
        }
    }
}

void Model::evaluate(const Matrix& X, const Matrix& y, size_t batch_size, bool reinit)
{
    if (!finalized) {
        finalize();
    }

    if (!loss || !accuracy) {
        throw runtime_error("Model::evaluate: loss and accuracy must be set");
    }

    X.require_non_empty("Model::evaluate: X must be non-empty");
    y.require_non_empty("Model::evaluate: y must be non-empty");

    if (batch_size > X.get_rows()) {
        throw runtime_error("Model::evaluate: batch_size cannot exceed number of samples aka X.get_rows()");
    }

    const size_t samples = X.get_rows();

    Matrix Y;
    if (y.get_rows() == 1 && y.get_cols() == samples) {
        Y = y.transpose();
    } else {
        y.require_rows(samples, "Model::evaluate: non-row-vector y must have same number of rows as X");
        Y = y;
    }

    if (reinit) {
        accuracy->init(Y);
    }

    loss->new_pass();
    accuracy->new_pass();

    const size_t steps = calc_steps(samples, batch_size);

    for (size_t step = 0; step < steps; ++step) {
        Matrix batch_X;
        Matrix batch_y;
        slice_batch(X, &Y, step, batch_size, batch_X, &batch_y);

        forward_pass(batch_X, false);
        loss->calculate(output, batch_y);
        last_predictions = last_activation->predictions(output);
        accuracy->calculate(last_predictions, batch_y);
    }

    const double val_loss = loss->calculate_accumulated();
    const double val_acc = accuracy->calculate_accumulated();

    std::cout << "validation - accuracy: " << val_acc
              << ", loss: " << val_loss
              << '\n';
}

Matrix Model::predict(const Matrix& X, size_t batch_size)
{
    if (!finalized) {
        finalize();
    }

    X.require_non_empty("Model::predict: X must be non-empty");

    if (batch_size > X.get_rows()) {
        throw runtime_error("Model::predict: batch_size cannot exceed number of samples aka X.get_rows()");
    }

    const size_t samples = X.get_rows();
    const size_t steps = calc_steps(samples, batch_size);

    Matrix predictions;
    bool preds_init = false;

    for (size_t step = 0; step < steps; ++step) {
        Matrix batch_X;
        slice_batch(X, nullptr, step, batch_size, batch_X, nullptr);

        forward_pass(batch_X, false);
        last_predictions = last_activation->predictions(output);

        last_predictions.require_rows(batch_X.get_rows(), "Model::predict: predictions must have same rows as batch_X");

        if (!preds_init) {
            predictions.assign(samples, last_predictions.get_cols());
            preds_init = true;
        } else {
            last_predictions.require_cols(predictions.get_cols(), "Model::predict: predictions cols mismatch across batches");
        }

        const size_t start = step * batch_size;
        for (size_t i = 0; i < last_predictions.get_rows(); ++i) {
            for (size_t j = 0; j < last_predictions.get_cols(); ++j) {
                predictions(start + i, j) = last_predictions(i, j);
            }
        }
    }

    return predictions;
}

void Model::save_params(const string& path)
{
    if (path.empty()) {
        throw runtime_error("Model::save_params: invalid path");
    }
    if (!finalized) {
        finalize();
    }

    require_dense_params_initialized();

    ofstream f(path, ios::binary);
    if (!f) {
        throw runtime_error("Model::save_params: failed to open file");
    }

    constexpr char magic[11] = {'M', 'O', 'D', 'E', 'L', 'P', 'A', 'R', 'A', 'M', 'S'};
    write_exact(f, magic, sizeof(magic));

    const uint64_t dense_count = static_cast<uint64_t>(dense_layers_count());
    write_trivial(f, dense_count);

    for (const LayerDense* layer : trainable_layers) {
        write_matrix(f, layer->weights);
        write_matrix(f, layer->biases);
    }
}

void Model::load_params(const string& path)
{
    if (path.empty()) {
        throw runtime_error("Model::load_params: invalid path");
    }
    if (!finalized) {
        finalize();
    }

    ifstream f(path, ios::binary);
    if (!f) {
        throw runtime_error("Model::load_params: failed to open file");
    }

    constexpr char magic_expected[11] = {'M', 'O', 'D', 'E', 'L', 'P', 'A', 'R', 'A', 'M', 'S'};
    char magic[11]{};
    read_exact(f, magic, sizeof(magic));

    if (std::memcmp(magic, magic_expected, sizeof(magic)) != 0) {
        throw runtime_error("Model::load_params: invalid file (magic mismatch)");
    }

    const uint64_t dense_count_file = read_trivial<uint64_t>(f);
    if (dense_count_file > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        throw runtime_error("Model::load_params: dense layer count exceeds size_t range");
    }

    const size_t dense_count_model = dense_layers_count();
    if (dense_count_file != static_cast<uint64_t>(dense_count_model)) {
        throw runtime_error("Model::load_params: dense layer count mismatch");
    }

    vector<Matrix> params;
    multiplication_overflow_check(dense_count_model, 2, "Model::load_params: params reserve size overflow");
    params.reserve(dense_count_model * 2);

    for (size_t i = 0; i < dense_count_model; ++i) {
        params.push_back(read_matrix(f));
        params.push_back(read_matrix(f));
    }

    set_params(params);
}

void Model::save(const string& path)
{
    if (path.empty()) {
        throw runtime_error("Model::save: invalid path");
    }
    if (!finalized) {
        finalize();
    }

    require_dense_params_initialized();

    ofstream f(path, ios::binary);
    if (!f) {
        throw runtime_error("Model::save: failed to open file");
    }

    constexpr char magic[11] = {'M', 'O', 'D', 'E', 'L', 'O', 'B', 'J', 'E', 'C', 'T'};
    write_exact(f, magic, sizeof(magic));

    const uint64_t layers_count = static_cast<uint64_t>(architecture.size());
    write_trivial(f, layers_count);

    for (const LayerDesc& d : architecture) {
        write_trivial<uint8_t>(f, static_cast<uint8_t>(d.kind));

        if (d.kind == LayerKind::Dense) {
            write_trivial<uint64_t>(f, static_cast<uint64_t>(d.dense.n_neurons));
            write_trivial<uint8_t>(f, static_cast<uint8_t>(d.dense.activation));
            write_trivial<double>(f, d.dense.w_l1);
            write_trivial<double>(f, d.dense.w_l2);
            write_trivial<double>(f, d.dense.b_l1);
            write_trivial<double>(f, d.dense.b_l2);
        } else if (d.kind == LayerKind::Dropout) {
            write_trivial<double>(f, d.dropout.dropout_rate);
        } else {
            throw runtime_error("Model::save: unsupported layer kind");
        }
    }

    const uint64_t dense_count = static_cast<uint64_t>(dense_layers_count());
    write_trivial(f, dense_count);

    for (const LayerDense* layer : trainable_layers) {
        write_matrix(f, layer->weights);
        write_matrix(f, layer->biases);
    }
}

Model Model::load(const string& path)
{
    if (path.empty()) {
        throw runtime_error("Model::load: invalid path");
    }

    ifstream f(path, ios::binary);
    if (!f) {
        throw runtime_error("Model::load: failed to open file");
    }

    constexpr char magic_expected[11] = {'M', 'O', 'D', 'E', 'L', 'O', 'B', 'J', 'E', 'C', 'T'};
    char magic[11]{};
    read_exact(f, magic, sizeof(magic));

    if (std::memcmp(magic, magic_expected, sizeof(magic)) != 0) {
        throw runtime_error("Model::load: invalid file (magic mismatch)");
    }

    const uint64_t layers_count = read_trivial<uint64_t>(f);
    if (layers_count == 0) {
        throw runtime_error("Model::load: empty architecture");
    }
    if (layers_count > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        throw runtime_error("Model::load: layer count exceeds size_t range");
    }

    vector<LayerDesc> arch;
    arch.reserve(static_cast<size_t>(layers_count));

    for (uint64_t i = 0; i < layers_count; ++i) {
        LayerDesc d;

        const uint8_t kind_u8 = read_trivial<uint8_t>(f);
        if (kind_u8 > static_cast<uint8_t>(LayerKind::Dropout)) {
            throw runtime_error("Model serialization: invalid LayerKind");
        }
        d.kind = static_cast<LayerKind>(kind_u8);

        if (d.kind == LayerKind::Dense) {
            const uint64_t n_neurons_u64 = read_trivial<uint64_t>(f);
            if (n_neurons_u64 == 0) {
                throw runtime_error("Model::load: dense layer n_neurons must be > 0");
            }
            if (n_neurons_u64 > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
                throw runtime_error("Model::load: dense layer n_neurons exceeds size_t range");
            }
            d.dense.n_neurons = static_cast<size_t>(n_neurons_u64);

            const uint8_t activation_u8 = read_trivial<uint8_t>(f);
            if (activation_u8 > static_cast<uint8_t>(ActivationKind::Linear)) {
                throw runtime_error("Model serialization: invalid ActivationKind");
            }
            d.dense.activation = static_cast<ActivationKind>(activation_u8);

            d.dense.w_l1 = read_trivial<double>(f);
            d.dense.w_l2 = read_trivial<double>(f);
            d.dense.b_l1 = read_trivial<double>(f);
            d.dense.b_l2 = read_trivial<double>(f);
        } else if (d.kind == LayerKind::Dropout) {
            d.dropout.dropout_rate = read_trivial<double>(f);
        } else {
            throw runtime_error("Model::load: unsupported layer kind");
        }

        arch.push_back(d);
    }

    const uint64_t dense_count_file = read_trivial<uint64_t>(f);
    if (dense_count_file > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        throw runtime_error("Model::load: dense layer count exceeds size_t range");
    }

    size_t dense_count_arch = 0;
    for (const LayerDesc& d : arch) {
        if (d.kind == LayerKind::Dense) {
            ++dense_count_arch;
        }
    }

    if (dense_count_file != static_cast<uint64_t>(dense_count_arch)) {
        throw runtime_error("Model::load: dense layer count mismatch");
    }

    vector<Matrix> params;
    multiplication_overflow_check(dense_count_arch, 2, "Model::load: params reserve size overflow");
    params.reserve(dense_count_arch * 2);

    for (size_t i = 0; i < dense_count_arch; ++i) {
        params.push_back(read_matrix(f));
        params.push_back(read_matrix(f));
    }

    Model m;
    m.set_architecture(arch);
    m.set_params(params);
    return m;
}

Model::vector<Matrix> Model::get_params() const
{
    vector<Matrix> params;

    for (const auto& p : layers) {
        const auto* dense = dynamic_cast<const LayerDense*>(p.get());
        if (!dense) {
            continue;
        }

        if (dense->weights.is_empty() || dense->biases.is_empty()) {
            throw runtime_error("Model::get_params: weights/biases not initialized (run predict/train or load params first)");
        }

        params.push_back(dense->weights);
        params.push_back(dense->biases);
    }

    return params;
}

void Model::set_params(const vector<Matrix>& params)
{
    if (!finalized) {
        finalize();
    }

    size_t dense_count = dense_layers_count();
    if (dense_count != trainable_layers.size()) {
        throw runtime_error("Model::set_params: internal error (dense_count != trainable_layers.size())");
    }

    if (params.size() != dense_count * 2) {
        throw runtime_error("Model::set_params: params size must be 2 * number_of_dense_layers");
    }

    size_t dense_idx = 0;
    for (const LayerDesc& d : architecture) {
        if (d.kind != LayerKind::Dense) {
            continue;
        }

        const Matrix& w = params[dense_idx * 2];
        const Matrix& b = params[dense_idx * 2 + 1];

        w.require_non_empty("Model::set_params: weights must be non-empty");
        b.require_non_empty("Model::set_params: biases must be non-empty");

        if (w.get_cols() != d.dense.n_neurons) {
            throw runtime_error("Model::set_params: weights cols must match architecture n_neurons");
        }
        b.require_shape(1, d.dense.n_neurons,
                        "Model::set_params: biases must have shape (1, n_neurons)");

        if (dense_idx > 0) {
            const Matrix& prev_w = params[(dense_idx - 1) * 2];
            prev_w.require_cols(w.get_rows(),
                                "Model::set_params: consecutive dense weight matrices must be shape-compatible");
        }

        LayerDense* layer = trainable_layers[dense_idx];
        if (!layer) {
            throw runtime_error("Model::set_params: internal error (null trainable layer)");
        }

        layer->weights = w;
        layer->biases = b;

        ++dense_idx;
    }
}

const Matrix& Model::get_output() const
{
    return output;
}

const Matrix& Model::get_last_predictions() const
{
    return last_predictions;
}

Model::ActivationKind Model::string_to_activation_kind(string name)
{
    std::transform(name.begin(), name.end(), name.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    if (name == "relu") {
        return ActivationKind::ReLU;
    }
    if (name == "softmax") {
        return ActivationKind::Softmax;
    }
    if (name == "sigmoid") {
        return ActivationKind::Sigmoid;
    }
    if (name == "linear") {
        return ActivationKind::Linear;
    }

    throw runtime_error("Model unknown activation. use relu, softmax, sigmoid or linear");
}

Model::string Model::activation_kind_to_string(ActivationKind k)
{
    switch (k) {
        case ActivationKind::ReLU:
            return "relu";
        case ActivationKind::Softmax:
            return "softmax";
        case ActivationKind::Sigmoid:
            return "sigmoid";
        case ActivationKind::Linear:
            return "linear";
        default:
            throw runtime_error("Model: unknown ActivationKind");
    }
}

void Model::require_dense_params_initialized() const
{
    for (const LayerDense* layer : trainable_layers) {
        if (!layer) {
            throw runtime_error("Model: internal error (null dense layer)");
        }
        if (layer->weights.is_empty() || layer->biases.is_empty()) {
            throw runtime_error("Model: dense weights/biases are not initialized (run train/predict or load params first)");
        }
    }
}

Model::size_t Model::dense_layers_count() const
{
    size_t n = 0;
    for (const LayerDesc& d : architecture) {
        if (d.kind == LayerKind::Dense) {
            ++n;
        }
    }
    return n;
}

void Model::invalidate_compiled_state()
{
    finalized = false;

    trainable_layers.clear();

    last_activation = nullptr;
    last_activation_kind = ActivationKind::Linear;
    use_combined_softmax_ce = false;

    output = Matrix();
    last_predictions = Matrix();
}

LayerDense& Model::add_dense_to_model(size_t n_neurons, ActivationKind activation_kind,
                                      double w_l1, double w_l2, double b_l1, double b_l2)
{
    LayerDesc d;
    d.kind = LayerKind::Dense;
    d.dense.n_neurons = n_neurons;
    d.dense.activation = activation_kind;
    d.dense.w_l1 = w_l1;
    d.dense.w_l2 = w_l2;
    d.dense.b_l1 = b_l1;
    d.dense.b_l2 = b_l2;

    architecture.push_back(d);

    auto layer = std::make_unique<LayerDense>(
        n_neurons,
        activation_kind_to_string(activation_kind),
        w_l1,
        w_l2,
        b_l1,
        b_l2);

    LayerDense& ref = *layer;
    layers.push_back(std::move(layer));

    invalidate_compiled_state();

    return ref;
}

void Model::set_architecture(const vector<LayerDesc>& arch)
{
    if (arch.empty()) {
        throw runtime_error("Model::set_architecture: empty architecture");
    }
    if (arch.back().kind == LayerKind::Dropout) {
        throw runtime_error("Model::set_architecture: final layer cannot be dropout");
    }

    vector<unique_ptr<Layer>> new_layers;
    new_layers.reserve(arch.size());

    for (const LayerDesc& d : arch) {
        if (d.kind == LayerKind::Dense) {
            new_layers.push_back(std::make_unique<LayerDense>(
                d.dense.n_neurons,
                activation_kind_to_string(d.dense.activation),
                d.dense.w_l1,
                d.dense.w_l2,
                d.dense.b_l1,
                d.dense.b_l2));
        } else if (d.kind == LayerKind::Dropout) {
            new_layers.push_back(std::make_unique<LayerDropout>(d.dropout.dropout_rate));
        } else {
            throw runtime_error("Model::set_architecture: unsupported layer kind");
        }
    }

    clear();
    layers.swap(new_layers);
    architecture = arch;
}

void Model::rebuild_caches()
{
    if (layers.size() != architecture.size()) {
        throw runtime_error("Model: internal error (layers/spec size mismatch)");
    }

    trainable_layers.clear();

    for (auto& p : layers) {
        if (auto* dense = dynamic_cast<LayerDense*>(p.get())) {
            trainable_layers.push_back(dense);
        }
    }

    last_activation = layers.back()->get_activation();
    if (!last_activation) {
        throw runtime_error("Model: last layer has null activation");
    }

    const LayerDesc& last = architecture.back();
    if (last.kind != LayerKind::Dense) {
        throw runtime_error("Model: internal error (last layer kind is not Dense)");
    }
    last_activation_kind = last.dense.activation;
}

void Model::update_combined_flag()
{
    use_combined_softmax_ce = false;
    if (!loss) {
        return;
    }
    use_combined_softmax_ce = (last_activation_kind == ActivationKind::Softmax) && loss_is_cce;
}

Model::size_t Model::calc_steps(size_t samples, size_t batch_size)
{
    if (batch_size == 0) {
        return 1;
    }

    size_t steps = samples / batch_size;
    if (steps * batch_size < samples) {
        ++steps;
    }
    return steps;
}

void Model::slice_batch(const Matrix& X, const Matrix* Y, size_t step,
                        size_t batch_size, Matrix& batch_X, Matrix* batch_Y)
{
    if (batch_size == 0) {
        batch_X = X;
        if (Y && batch_Y) {
            *batch_Y = *Y;
        }
        return;
    }

    const size_t start = step * batch_size;
    const size_t end = std::min(start + batch_size, X.get_rows());
    batch_X = X.slice_rows(start, end);
    if (Y && batch_Y) {
        *batch_Y = Y->slice_rows(start, end);
    }
}

void Model::forward_pass(const Matrix& X, bool training)
{
    input_layer.forward(X);

    const Matrix* current = &input_layer.get_output();
    for (auto& p : layers) {
        p->forward(*current, training);
        current = &p->get_output();
    }

    output = *current;
}

void Model::write_exact(ostream& os, const void* data, size_t bytes)
{
    os.write(static_cast<const char*>(data), static_cast<streamsize>(bytes));
    if (!os) {
        throw runtime_error("Model serialization: write failed");
    }
}

template <class T>
void Model::write_trivial(ostream& os, const T& v)
{
    static_assert(std::is_trivially_copyable_v<T>);
    write_exact(os, &v, sizeof(T));
}

void Model::read_exact(istream& is, void* data, size_t bytes)
{
    is.read(static_cast<char*>(data), static_cast<streamsize>(bytes));
    if (!is) {
        throw runtime_error("Model serialization: read failed");
    }
}

template <class T>
T Model::read_trivial(istream& is)
{
    static_assert(std::is_trivially_copyable_v<T>);
    T v{};
    read_exact(is, &v, sizeof(T));
    return v;
}

void Model::write_matrix(ostream& os, const Matrix& m)
{
    const uint64_t r = static_cast<uint64_t>(m.get_rows());
    const uint64_t c = static_cast<uint64_t>(m.get_cols());
    write_trivial(os, r);
    write_trivial(os, c);

    for (size_t i = 0; i < m.get_rows(); ++i) {
        for (size_t j = 0; j < m.get_cols(); ++j) {
            const double x = m(i, j);
            write_trivial(os, x);
        }
    }
}

Matrix Model::read_matrix(istream& is)
{
    const uint64_t r64 = read_trivial<uint64_t>(is);
    const uint64_t c64 = read_trivial<uint64_t>(is);

    if (r64 > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
        c64 > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        throw runtime_error("Model serialization: matrix shape exceeds size_t range");
    }

    const size_t r = static_cast<size_t>(r64);
    const size_t c = static_cast<size_t>(c64);

    Matrix m;
    m.assign(r, c);

    for (size_t i = 0; i < r; ++i) {
        for (size_t j = 0; j < c; ++j) {
            m(i, j) = read_trivial<double>(is);
        }
    }

    return m;
}
