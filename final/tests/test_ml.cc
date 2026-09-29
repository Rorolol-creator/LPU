#include "../src/ml.hh"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace
{

    int g_failures = 0;
    int g_checks = 0;

    void check(bool condition, const std::string& name)
    {
        g_checks++;
        if (!condition)
        {
            g_failures++;
            std::cerr << "[FAIL] " << name << "\n";
        }
        else
        {
            std::cout << "[ OK ] " << name << "\n";
        }
    }

    void build_separable_dataset(std::vector<std::vector<float>>& X,
                                 std::vector<std::string>& y)
    {
        for (int i = 0; i < 20; ++i)
        {
            X.push_back({ 0.0f + 0.01f * i, 0.5f });
            y.push_back("low");
        }
        for (int i = 0; i < 20; ++i)
        {
            X.push_back({ 1.0f + 0.01f * i, 0.5f });
            y.push_back("mid");
        }
        for (int i = 0; i < 20; ++i)
        {
            X.push_back({ 2.0f + 0.01f * i, 0.5f });
            y.push_back("high");
        }
    }

    void test_decision_tree_fits_separable_data()
    {
        std::vector<std::vector<float>> X;
        std::vector<std::string> y_str;
        build_separable_dataset(X, y_str);

        std::vector<int> y(y_str.size());
        for (size_t i = 0; i < y_str.size(); ++i)
        {
            y[i] = y_str[i] == "low" ? 0 : (y_str[i] == "mid" ? 1 : 2);
        }

        std::vector<int> indices(X.size());
        for (size_t i = 0; i < X.size(); ++i)
            indices[i] = static_cast<int>(i);

        std::mt19937 rng(0);
        DecisionTree tree(/*max_depth=*/6, /*min_samples_leaf=*/1,
                          /*max_features=*/2);
        tree.fit(X, y, indices, /*n_classes=*/3, rng);

        int correct = 0;
        for (size_t i = 0; i < X.size(); ++i)
        {
            if (tree.predict(X[i]) == y[i])
                correct++;
        }
        check(correct == static_cast<int>(X.size()),
              "DecisionTree perfectly fits a linearly separable dataset");

        check(tree.predict({ 0.05f, 0.5f }) == 0,
              "DecisionTree predicts 'low' class correctly");
        check(tree.predict({ 1.05f, 0.5f }) == 1,
              "DecisionTree predicts 'mid' class correctly");
        check(tree.predict({ 2.05f, 0.5f }) == 2,
              "DecisionTree predicts 'high' class correctly");
    }

    void test_random_forest_fits_separable_data()
    {
        std::vector<std::vector<float>> X;
        std::vector<std::string> y;
        build_separable_dataset(X, y);

        RandomForest forest(/*n_trees=*/20, /*max_depth=*/6,
                            /*min_samples_leaf=*/1, /*max_features=*/2,
                            /*seed=*/42);
        forest.fit(X, y);

        check(forest.classes().size() == 3,
              "RandomForest discovers exactly 3 classes");

        int correct = 0;
        for (size_t i = 0; i < X.size(); ++i)
        {
            if (forest.predict(X[i]) == y[i])
                correct++;
        }
        check(correct == static_cast<int>(X.size()),
              "RandomForest perfectly fits a linearly separable dataset");

        check(forest.predict({ 0.05f, 0.5f }) == "low",
              "RandomForest predicts 'low' class correctly");
        check(forest.predict({ 2.05f, 0.5f }) == "high",
              "RandomForest predicts 'high' class correctly");
    }

    void test_predict_top_k_invariants()
    {
        std::vector<std::vector<float>> X;
        std::vector<std::string> y;
        build_separable_dataset(X, y);

        RandomForest forest(20, 6, 1, 2, 42);
        forest.fit(X, y);

        auto top1 = forest.predict_top_k({ 0.05f, 0.5f }, 1);
        check(top1.size() == 1, "predict_top_k(k=1) returns exactly one label");
        check(top1[0] == forest.predict({ 0.05f, 0.5f }),
              "predict_top_k(k=1)[0] matches predict()");

        auto top10 = forest.predict_top_k({ 0.05f, 0.5f }, 10);
        check(
            top10.size() == 3,
            "predict_top_k(k > n_classes) is clamped to the number of classes");

        auto top3 = forest.predict_top_k({ 0.05f, 0.5f }, 3);
        check(top3[0] == "low", "predict_top_k ranks the correct class first");
    }

    void test_random_forest_is_deterministic_given_a_seed()
    {
        std::vector<std::vector<float>> X;
        std::vector<std::string> y;
        build_separable_dataset(X, y);

        RandomForest forest_a(15, 5, 1, 2, 7);
        forest_a.fit(X, y);
        RandomForest forest_b(15, 5, 1, 2, 7);
        forest_b.fit(X, y);

        bool all_match = true;
        for (const auto& x : X)
        {
            if (forest_a.predict(x) != forest_b.predict(x))
            {
                all_match = false;
                break;
            }
        }
        check(all_match, "RandomForest is deterministic for a fixed seed");
    }

    void test_random_forest_save_load_roundtrip()
    {
        std::vector<std::vector<float>> X;
        std::vector<std::string> y;
        build_separable_dataset(X, y);

        RandomForest original(15, 5, 1, 2, 7);
        original.fit(X, y);

        const std::string path = "/tmp/test_ml_model.rf";
        original.save(path);

        RandomForest reloaded;
        reloaded.load(path);
        std::remove(path.c_str());

        check(reloaded.classes() == original.classes(),
              "Reloaded RandomForest has the same classes");

        bool all_match = true;
        for (const auto& x : X)
        {
            if (reloaded.predict(x) != original.predict(x))
            {
                all_match = false;
                break;
            }
        }
        check(all_match,
              "Reloaded RandomForest makes identical predictions to the "
              "original");

        auto top3_original = original.predict_top_k({ 0.05f, 0.5f }, 3);
        auto top3_reloaded = reloaded.predict_top_k({ 0.05f, 0.5f }, 3);
        check(top3_original == top3_reloaded,
              "Reloaded RandomForest predict_top_k matches the original");
    }

    void test_random_forest_save_load_handles_nan_and_inf_thresholds()
    {
        std::vector<std::vector<float>> X = {
            { 1.0f, 0.0f },
            { std::numeric_limits<float>::quiet_NaN(), 0.0f },
            { std::numeric_limits<float>::infinity(), 0.0f },
            { 3.0f, 0.0f },
        };
        std::vector<std::string> y = { "a", "b", "a", "b" };

        RandomForest forest(5, 3, 1, 2, 1);
        forest.fit(X, y);

        const std::string path = "/tmp/test_ml_model_nan.rf";
        forest.save(path);

        RandomForest reloaded;
        bool loaded_ok = true;
        try
        {
            reloaded.load(path);
        }
        catch (const std::exception&)
        {
            loaded_ok = false;
        }
        std::remove(path.c_str());

        check(loaded_ok,
              "RandomForest with NaN/Inf-valued thresholds can be saved and "
              "reloaded without throwing");
        if (!loaded_ok)
            return;

        bool all_match = true;
        for (const auto& x : X)
        {
            if (reloaded.predict(x) != forest.predict(x))
            {
                all_match = false;
                break;
            }
        }
        check(all_match,
              "Reloaded RandomForest with NaN/Inf thresholds predicts "
              "identically to the original");
    }

    void test_random_forest_load_throws_on_missing_file()
    {
        RandomForest forest;
        bool threw = false;
        try
        {
            forest.load("/tmp/this_model_file_does_not_exist_9f3a7c.rf");
        }
        catch (const std::exception&)
        {
            threw = true;
        }
        check(threw,
              "RandomForest::load throws when the model file does not "
              "exist");
    }

    void test_random_forest_save_throws_on_unwritable_path()
    {
        RandomForest forest(5, 3, 1, 2, 1);
        std::vector<std::vector<float>> X = { { 1.0f }, { 2.0f } };
        std::vector<std::string> y = { "a", "b" };
        forest.fit(X, y);

        bool threw = false;
        try
        {
            forest.save("/tmp/no_such_directory_9f3a7c/model.rf");
        }
        catch (const std::exception&)
        {
            threw = true;
        }
        check(threw,
              "RandomForest::save throws when the target path cannot be "
              "opened for writing");
    }

    void test_random_forest_load_throws_on_empty_file()
    {
        const std::string path = "/tmp/test_ml_model_empty.rf";
        std::ofstream(path).close();

        RandomForest forest;
        bool threw = false;
        try
        {
            forest.load(path);
        }
        catch (const std::exception&)
        {
            threw = true;
        }
        std::remove(path.c_str());

        check(threw, "RandomForest::load throws on an empty model file");
    }

    void test_random_forest_load_throws_on_truncated_tree_data()
    {
        std::vector<std::vector<float>> X;
        std::vector<std::string> y;
        build_separable_dataset(X, y);

        RandomForest original(10, 5, 1, 2, 7);
        original.fit(X, y);

        const std::string path = "/tmp/test_ml_model_truncated.rf";
        original.save(path);

        std::ifstream in(path);
        std::string content((std::istreambuf_iterator<char>(in)),
                            std::istreambuf_iterator<char>());
        in.close();

        std::ofstream out(path);
        out << content.substr(0, content.size() / 2);
        out.close();

        RandomForest reloaded;
        bool threw = false;
        try
        {
            reloaded.load(path);
        }
        catch (const std::exception&)
        {
            threw = true;
        }
        std::remove(path.c_str());

        check(threw,
              "RandomForest::load throws on a truncated/corrupted model "
              "file");
    }

} // namespace

int main()
{
    test_decision_tree_fits_separable_data();
    test_random_forest_fits_separable_data();
    test_predict_top_k_invariants();
    test_random_forest_is_deterministic_given_a_seed();
    test_random_forest_save_load_roundtrip();
    test_random_forest_save_load_handles_nan_and_inf_thresholds();
    test_random_forest_load_throws_on_missing_file();
    test_random_forest_save_throws_on_unwritable_path();
    test_random_forest_load_throws_on_empty_file();
    test_random_forest_load_throws_on_truncated_tree_data();

    std::cout << "\n"
              << (g_checks - g_failures) << "/" << g_checks
              << " checks passed\n";
    return g_failures == 0 ? 0 : 1;
}
