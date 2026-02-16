#include "tests/test_common.hpp"

TEST_CASE("Model add_dense validates activation names")
{
    Model model;
    CHECK_THROWS_WITH_AS(model.add_dense(4, "not_real_activation"),
                         "Model unknown activation. use relu, softmax, sigmoid or linear",
                         runtime_error);
}

TEST_CASE("Model finalize validates layers")
{
    Model model;
    CHECK_THROWS_WITH_AS(model.finalize(),
                         "Model::finalize: no layers added",
                         runtime_error);

    model.add_dropout(0.1);
    CHECK_THROWS_WITH_AS(model.finalize(),
                         "Model::finalize: final layer cannot be dropout",
                         runtime_error);

    Model model2;
    model2.add_dense(1, "linear");
    CHECK_NOTHROW(model2.finalize());
}

TEST_CASE("Model train validates setup and data")
{
    Model model;
    model.add_dense(1, "linear");

    Matrix X(1, 1, 0.0);
    Matrix y(1, 1, 0.0);

    CHECK_THROWS_WITH_AS(model.train(X, y, 0),
                         "Model::train: loss, accuracy and optimizer must be set",
                         runtime_error);

    LossMeanSquaredError loss;
    AccuracyRegression acc;
    OptimizerSGD opt(0.1, 0.0, 0.0);
    model.configure(loss, acc, &opt);

    Matrix empty;
    CHECK_THROWS_WITH_AS(model.train(empty, y, 0),
                         "Model::train: X must be non-empty",
                         runtime_error);
    CHECK_THROWS_WITH_AS(model.train(X, empty, 0),
                         "Model::train: y must be non-empty",
                         runtime_error);

    CHECK_THROWS_WITH_AS(model.train(X, y, 1, 5, 0),
                         "Model::train: batch_size cannot exceed number of samples aka X.get_rows()",
                         runtime_error);
}

TEST_CASE("Model train runs with softmax + CCE and validation")
{
    CoutSilencer silence;
    reset_deterministic_rng(0);

    Model model;
    model.add_dense(3, "relu");
    model.add_dense(2, "softmax");

    LossCategoricalCrossEntropy loss;
    AccuracyCategorical acc;
    OptimizerSGD opt(0.1, 0.0, 0.0);
    model.configure(loss, acc, opt);

    Matrix X(3, 2);
    X(0, 0) = 1.0; X(0, 1) = 0.0;
    X(1, 0) = 0.0; X(1, 1) = 1.0;
    X(2, 0) = 1.0; X(2, 1) = 1.0;

    Matrix y_row(1, 3);
    y_row(0, 0) = 0.0;
    y_row(0, 1) = 1.0;
    y_row(0, 2) = 1.0;

    Matrix y_col = y_row.transpose();

    model.train(X, y_row, 1, 1, 1, &X, &y_col);
}

TEST_CASE("Model train runs without combined softmax path")
{
    CoutSilencer silence;
    reset_deterministic_rng(1);

    Model model;
    model.add_dense(1, "linear");

    LossMeanSquaredError loss;
    AccuracyRegression acc;
    OptimizerSGD opt(0.1, 0.0, 0.0);
    model.configure(loss, acc, opt);

    Matrix X(2, 1);
    X(0, 0) = 1.0;
    X(1, 0) = 2.0;
    Matrix y(2, 1);
    y(0, 0) = 1.0;
    y(1, 0) = 2.0;

    model.train(X, y, 1, 0, 0);
}

TEST_CASE("Model evaluate validates setup and data")
{
    Model model;
    model.add_dense(1, "linear");

    Matrix X(1, 1, 0.0);
    Matrix y(1, 1, 0.0);

    CHECK_THROWS_WITH_AS(model.evaluate(X, y),
                         "Model::evaluate: loss and accuracy must be set",
                         runtime_error);

    LossMeanSquaredError loss;
    AccuracyRegression acc;
    model.configure(loss, acc, nullptr);

    Matrix empty;
    CHECK_THROWS_WITH_AS(model.evaluate(empty, y),
                         "Model::evaluate: X must be non-empty",
                         runtime_error);
    CHECK_THROWS_WITH_AS(model.evaluate(X, empty),
                         "Model::evaluate: y must be non-empty",
                         runtime_error);

    CHECK_THROWS_WITH_AS(model.evaluate(X, y, 5),
                         "Model::evaluate: batch_size cannot exceed number of samples aka X.get_rows()",
                         runtime_error);
}

TEST_CASE("Model evaluate runs with row-vector labels")
{
    CoutSilencer silence;
    reset_deterministic_rng(0);

    Model model;
    model.add_dense(1, "linear");

    LossMeanSquaredError loss;
    AccuracyRegression acc;
    model.configure(loss, acc, nullptr);

    Matrix X(2, 1);
    X(0, 0) = 1.0;
    X(1, 0) = 2.0;
    Matrix y_row(1, 2);
    y_row(0, 0) = 1.0;
    y_row(0, 1) = 2.0;

    model.evaluate(X, y_row, 0, true);
}

TEST_CASE("Model predict returns expected shape and validates batch size")
{
    Model model;
    model.add_dense(1, "linear");

    Matrix X(3, 2, 1.0);
    Matrix preds = model.predict(X, 0);
    CHECK(preds.get_rows() == 3);
    CHECK(preds.get_cols() == 1);

    const Matrix& last_preds = model.get_last_predictions();
    const Matrix& last_output = model.get_output();
    CHECK(last_preds.get_rows() == preds.get_rows());
    CHECK(last_preds.get_cols() == preds.get_cols());
    CHECK(last_output.get_rows() == preds.get_rows());
    CHECK(last_output.get_cols() == preds.get_cols());
    for (size_t i = 0; i < preds.get_rows(); ++i) {
        CHECK(last_preds(i, 0) == doctest::Approx(preds(i, 0)));
        CHECK(last_output(i, 0) == doctest::Approx(preds(i, 0)));
    }

    Matrix preds_batched = model.predict(X, 2);
    CHECK(preds_batched.get_rows() == 3);
    CHECK(preds_batched.get_cols() == 1);

    CHECK_THROWS_WITH_AS(model.predict(X, 5),
                         "Model::predict: batch_size cannot exceed number of samples aka X.get_rows()",
                         runtime_error);
}

TEST_CASE("Model clear resets architecture and compiled state")
{
    Model model;
    model.add_dense(4, "relu");
    model.add_dropout(0.1);
    model.add_dense(2, "softmax");
    CHECK_NOTHROW(model.finalize());

    model.clear();
    CHECK_THROWS_WITH_AS(model.finalize(),
                         "Model::finalize: no layers added",
                         runtime_error);
}

TEST_CASE("Model get_params and set_params validate consistency")
{
    Model model;
    model.add_dense(2, "linear");
    model.add_dense(1, "linear");

    Matrix X(2, 2, 1.0);
    CHECK_THROWS_WITH_AS(model.get_params(),
                         "Model::get_params: weights/biases not initialized (run predict/train or load params first)",
                         runtime_error);
    model.predict(X, 0); // initialize weights

    vector<Matrix> params = model.get_params();
    CHECK(params.size() == 4);

    CHECK_NOTHROW(model.set_params(params));

    vector<Matrix> bad_params = params;
    bad_params[2] = Matrix(3, 1, 0.0);
    CHECK_THROWS_WITH_AS(model.set_params(bad_params),
                         "Model::set_params: consecutive dense weight matrices must be shape-compatible",
                         runtime_error);

    bad_params = params;
    bad_params[0] = Matrix(2, 3, 0.0);
    CHECK_THROWS_WITH_AS(model.set_params(bad_params),
                         "Model::set_params: weights cols must match architecture n_neurons",
                         runtime_error);

    bad_params = params;
    bad_params[1] = Matrix(2, 2, 0.0);
    CHECK_THROWS_WITH_AS(model.set_params(bad_params),
                         "Model::set_params: biases must have shape (1, n_neurons)",
                         runtime_error);
}

TEST_CASE("Model set_params validates empty and size mismatch cases")
{
    Model empty_model;
    vector<Matrix> empty_params;
    CHECK_THROWS_WITH_AS(empty_model.set_params(empty_params),
                         "Model::finalize: no layers added",
                         runtime_error);

    vector<Matrix> some_params = { Matrix(1, 1, 0.0) };
    CHECK_THROWS_WITH_AS(empty_model.set_params(some_params),
                         "Model::finalize: no layers added",
                         runtime_error);

    Model model;
    model.add_dense(1, "linear");

    CHECK_THROWS_WITH_AS(model.set_params(some_params),
                         "Model::set_params: params size must be 2 * number_of_dense_layers",
                         runtime_error);
}

TEST_CASE("Model save/save_params require initialized dense parameters")
{
    Model model;
    model.add_dense(2, "relu");

    CHECK_THROWS_WITH_AS(model.save_params("tmp_params.bin"),
                         "Model: dense weights/biases are not initialized (run train/predict or load params first)",
                         runtime_error);
    CHECK_THROWS_WITH_AS(model.save("tmp_model.bin"),
                         "Model: dense weights/biases are not initialized (run train/predict or load params first)",
                         runtime_error);

    remove("tmp_params.bin");
    remove("tmp_model.bin");
}

TEST_CASE("Model save_params and load_params roundtrip and validation")
{
    reset_deterministic_rng(11);

    const string params_path = "test_model_params.bin";
    remove(params_path.c_str());

    Matrix X(4, 2);
    X(0, 0) = 1.0; X(0, 1) = 0.0;
    X(1, 0) = 0.0; X(1, 1) = 1.0;
    X(2, 0) = 1.0; X(2, 1) = 1.0;
    X(3, 0) = 0.5; X(3, 1) = 0.25;

    Model m1;
    m1.add_dense(3, "relu");
    m1.add_dense(2, "softmax");
    Matrix ref_preds = m1.predict(X, 2); // initialize weights
    m1.save_params(params_path);

    Model m2;
    m2.add_dense(3, "relu");
    m2.add_dense(2, "softmax");
    m2.load_params(params_path);
    Matrix loaded_preds = m2.predict(X, 2);
    CHECK(Matrix::max_absolute_difference(ref_preds, loaded_preds) == doctest::Approx(0.0));

    CHECK_THROWS_WITH_AS(m2.load_params(""),
                         "Model::load_params: invalid path",
                         runtime_error);
    CHECK_THROWS_WITH_AS(m2.load_params("no_such_params_file.bin"),
                         "Model::load_params: failed to open file",
                         runtime_error);

    Model wrong_arch;
    wrong_arch.add_dense(5, "relu");
    CHECK_THROWS_WITH_AS(wrong_arch.load_params(params_path),
                         "Model::load_params: dense layer count mismatch",
                         runtime_error);

    const string bad_magic_path = "test_bad_magic_params.bin";
    {
        ofstream out(bad_magic_path, ios::binary);
        const char bad_magic[11] = {'B','A','D','P','A','R','A','M','S','X','X'};
        out.write(bad_magic, sizeof(bad_magic));
    }
    CHECK_THROWS_WITH_AS(m2.load_params(bad_magic_path),
                         "Model::load_params: invalid file (magic mismatch)",
                         runtime_error);

    remove(params_path.c_str());
    remove(bad_magic_path.c_str());
}

TEST_CASE("Model save and load roundtrip and validation")
{
    reset_deterministic_rng(17);

    const string model_path = "test_full_model.bin";
    remove(model_path.c_str());

    Matrix X(5, 2);
    X(0, 0) = 1.0; X(0, 1) = 0.0;
    X(1, 0) = 0.0; X(1, 1) = 1.0;
    X(2, 0) = 1.0; X(2, 1) = 1.0;
    X(3, 0) = 0.5; X(3, 1) = 0.2;
    X(4, 0) = 0.2; X(4, 1) = 0.5;

    Model m1;
    m1.add_dense(4, "relu", 0.0, 1e-4, 0.0, 1e-4);
    m1.add_dropout(0.1);
    m1.add_dense(3, "sigmoid");
    m1.add_dense(2, "softmax");

    Matrix ref_preds = m1.predict(X, 2); // initialize weights
    m1.save(model_path);

    Model m2 = Model::load(model_path);
    Matrix loaded_preds = m2.predict(X, 2);
    CHECK(Matrix::max_absolute_difference(ref_preds, loaded_preds) == doctest::Approx(0.0));

    CHECK_THROWS_WITH_AS(m1.save(""),
                         "Model::save: invalid path",
                         runtime_error);
    CHECK_THROWS_WITH_AS(Model::load(""),
                         "Model::load: invalid path",
                         runtime_error);
    CHECK_THROWS_WITH_AS(Model::load("no_such_model_file.bin"),
                         "Model::load: failed to open file",
                         runtime_error);

    const string bad_magic_path = "test_bad_magic_model.bin";
    {
        ofstream out(bad_magic_path, ios::binary);
        const char bad_magic[11] = {'B','A','D','O','B','J','E','C','T','X','X'};
        out.write(bad_magic, sizeof(bad_magic));
    }
    CHECK_THROWS_WITH_AS(Model::load(bad_magic_path),
                         "Model::load: invalid file (magic mismatch)",
                         runtime_error);

    remove(model_path.c_str());
    remove(bad_magic_path.c_str());
}

TEST_CASE("Model::load validates corrupted layer metadata")
{
    const string p_bad_kind = "test_model_bad_kind.bin";
    const string p_zero_neurons = "test_model_zero_neurons.bin";
    const string p_bad_activation = "test_model_bad_activation.bin";
    const string p_count_mismatch = "test_model_count_mismatch.bin";
    const string p_last_dropout = "test_model_last_dropout.bin";

    remove(p_bad_kind.c_str());
    remove(p_zero_neurons.c_str());
    remove(p_bad_activation.c_str());
    remove(p_count_mismatch.c_str());
    remove(p_last_dropout.c_str());

    const char magic[11] = {'M','O','D','E','L','O','B','J','E','C','T'};
    const uint8_t dense_kind = 0;
    const uint8_t dropout_kind = 1;
    const uint8_t relu_activation = 0;

    // invalid LayerKind
    {
        ofstream out(p_bad_kind, ios::binary);
        out.write(magic, sizeof(magic));
        write_trivial_raw<uint64_t>(out, 1);      // layers_count
        write_trivial_raw<uint8_t>(out, 99);      // invalid kind
        write_trivial_raw<uint64_t>(out, 0);      // dense_count
    }
    CHECK_THROWS_WITH_AS(Model::load(p_bad_kind),
                         "Model serialization: invalid LayerKind",
                         runtime_error);

    // dense n_neurons == 0
    {
        ofstream out(p_zero_neurons, ios::binary);
        out.write(magic, sizeof(magic));
        write_trivial_raw<uint64_t>(out, 1);      // layers_count
        write_trivial_raw<uint8_t>(out, dense_kind);
        write_trivial_raw<uint64_t>(out, 0);      // n_neurons invalid
        write_trivial_raw<uint8_t>(out, relu_activation);
        write_trivial_raw<double>(out, 0.0);
        write_trivial_raw<double>(out, 0.0);
        write_trivial_raw<double>(out, 0.0);
        write_trivial_raw<double>(out, 0.0);
        write_trivial_raw<uint64_t>(out, 0);      // dense_count
    }
    CHECK_THROWS_WITH_AS(Model::load(p_zero_neurons),
                         "Model::load: dense layer n_neurons must be > 0",
                         runtime_error);

    // invalid ActivationKind
    {
        ofstream out(p_bad_activation, ios::binary);
        out.write(magic, sizeof(magic));
        write_trivial_raw<uint64_t>(out, 1);      // layers_count
        write_trivial_raw<uint8_t>(out, dense_kind);
        write_trivial_raw<uint64_t>(out, 2);      // n_neurons
        write_trivial_raw<uint8_t>(out, 99);      // invalid activation kind
        write_trivial_raw<double>(out, 0.0);
        write_trivial_raw<double>(out, 0.0);
        write_trivial_raw<double>(out, 0.0);
        write_trivial_raw<double>(out, 0.0);
        write_trivial_raw<uint64_t>(out, 0);      // dense_count
    }
    CHECK_THROWS_WITH_AS(Model::load(p_bad_activation),
                         "Model serialization: invalid ActivationKind",
                         runtime_error);

    // dense count mismatch
    {
        ofstream out(p_count_mismatch, ios::binary);
        out.write(magic, sizeof(magic));
        write_trivial_raw<uint64_t>(out, 1);      // layers_count
        write_trivial_raw<uint8_t>(out, dense_kind);
        write_trivial_raw<uint64_t>(out, 2);      // n_neurons
        write_trivial_raw<uint8_t>(out, relu_activation);
        write_trivial_raw<double>(out, 0.0);
        write_trivial_raw<double>(out, 0.0);
        write_trivial_raw<double>(out, 0.0);
        write_trivial_raw<double>(out, 0.0);
        write_trivial_raw<uint64_t>(out, 2);      // wrong dense_count
        Matrix w(1, 1, 0.0), b(1, 1, 0.0);
        write_matrix_raw(out, w);
        write_matrix_raw(out, b);
    }
    CHECK_THROWS_WITH_AS(Model::load(p_count_mismatch),
                         "Model::load: dense layer count mismatch",
                         runtime_error);

    // last layer dropout (rejected in set_architecture)
    {
        ofstream out(p_last_dropout, ios::binary);
        out.write(magic, sizeof(magic));
        write_trivial_raw<uint64_t>(out, 1);      // layers_count
        write_trivial_raw<uint8_t>(out, dropout_kind);
        write_trivial_raw<double>(out, 0.1);
        write_trivial_raw<uint64_t>(out, 0);      // dense_count
    }
    CHECK_THROWS_WITH_AS(Model::load(p_last_dropout),
                         "Model::set_architecture: final layer cannot be dropout",
                         runtime_error);

    remove(p_bad_kind.c_str());
    remove(p_zero_neurons.c_str());
    remove(p_bad_activation.c_str());
    remove(p_count_mismatch.c_str());
    remove(p_last_dropout.c_str());
}
