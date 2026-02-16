#include <algorithm>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include "neural_core/accuracy.hpp"
#include "neural_core/data.hpp"
#include "neural_core/losses.hpp"
#include "neural_core/matrix.hpp"
#include "neural_core/model.hpp"
#include "neural_core/optimizers.hpp"
#include "neural_core/rng.hpp"
#include "neural_core/scatter_plot.hpp"

#ifndef NEURAL_CORE_NO_MAIN
int main()
{
    try {
        const std::string params_path = "model_params.bin";
        const std::string model_path = "full_model.bin";

        set_global_seed(0);
        set_thread_stream_id(0);

#ifdef ENABLE_MATPLOT
        Matrix X_generated;
        Matrix y_generated;
        generate_spiral_data(1000, 3, X_generated, y_generated);

        const std::string path = "plot.png";
        try {
            scatter_plot(path, X_generated, y_generated);
            std::cout << "data plotting complete. generated data plot is in file: "
                      << path << "\n";
        } catch (const std::runtime_error& e) {
            std::cout << "plotting skipped: " << e.what() << '\n';
        }
#else
        std::cout << "plotting skipped (built with ENABLE_MATPLOT=OFF)\n";
#endif

        Matrix X;
        Matrix y;
        Matrix X_test;
        Matrix y_test;
        fashion_mnist_create(X, y, X_test, y_test);

        Model model;
        model.add_dense(128, "relu", 0.0, 5e-4, 0.0, 5e-4);
        model.add_dense(128, "relu", 0.0, 5e-4, 0.0, 5e-4);
        model.add_dropout(0.1);
        model.add_dense(10, "softmax");

        LossCategoricalCrossEntropy loss;
        AccuracyCategorical accuracy;
        OptimizerAdam optimizer(1e-3);
        model.configure(loss, accuracy, optimizer);

        model.train(X, y, 1, 128, 100, &X_test, &y_test);

        const Matrix preds_model = model.predict(X_test, 128);
        const std::size_t pred_rows = preds_model.get_rows();

        std::cout << "predict output shape: " << pred_rows
                  << "x" << preds_model.get_cols() << '\n';

        const std::vector<std::string> label_names = {
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"};

        std::cout << "sample predictions:\n";
        const std::size_t samples_to_check = std::min<std::size_t>(20, pred_rows);
        for (std::size_t i = 0; i < samples_to_check; ++i) {
            const std::size_t pred_id = (pred_rows == 1) ? preds_model.as_size_t(0, i) : preds_model.as_size_t(i, 0);
            const std::size_t true_id = (y_test.get_rows() == 1) ? y_test.as_size_t(0, i) : y_test.as_size_t(i, 0);

            std::cout << "  sample " << i
                      << " - predicted: " << label_names[pred_id] << " (" << pred_id << ")"
                      << ", actual: " << label_names[true_id] << " (" << true_id << ")"
                      << '\n';
        }

        model.evaluate(X_test, y_test, 128);

        std::vector<Matrix> params = model.get_params();
        model.set_params(params);

        Matrix preds_model_set_params = model.predict(X_test, 128);
        std::cout << "\ndifference in predictions after saving the model parameters to a variable and setting them back: "
                  << Matrix::max_absolute_difference(preds_model, preds_model_set_params) << '\n'
                  << "evaluation after setting model parameters:\n";

        model.evaluate(X_test, y_test, 128);

        model.save_params(params_path);

        Model m2;
        m2.add_dense(128, "relu", 0.0, 5e-4, 0.0, 5e-4);
        m2.add_dense(128, "relu", 0.0, 5e-4, 0.0, 5e-4);
        m2.add_dropout(0.1);
        m2.add_dense(10, "softmax");

        m2.load_params(params_path);

        const Matrix preds_model_loaded_params = m2.predict(X_test, 128);
        std::cout << "\ndifference in predictions after saving the model parameters to a file and loading them: "
                  << Matrix::max_absolute_difference(preds_model, preds_model_loaded_params) << '\n'
                  << "evaluation after loading in model parameters:\n";

        m2.configure(loss, accuracy);
        m2.evaluate(X_test, y_test, 128);

        model.save(model_path);
        Model m3 = Model::load(model_path);

        const Matrix preds_model_loaded_full = m3.predict(X_test, 128);
        std::cout << "\ndifference in predictions after saving the full model to a file and loading it: "
                  << Matrix::max_absolute_difference(preds_model, preds_model_loaded_full) << '\n'
                  << "evaluation after loading in full model:\n";

        m3.configure(loss, accuracy);
        m3.evaluate(X_test, y_test, 128);

        return 0;
    } catch (const std::exception& e) {
        std::cout << "ERROR: " << e.what() << '\n';
        return 1;
    }
}
#endif
