#include <iostream>
#include <vector>
#include <filesystem>
#include <fstream>
#include <cmath>
#include <chrono>

#include <opencv2/opencv.hpp>
#include "src/pretreatment.hh"
#include "src/ml.hh"

namespace fs = std::filesystem;

namespace
{

    const std::string kDefaultModelPath = "model.rf";

    void configureFeatureExtraction(PyramidConfig& pyrCfg,
                                    ColorConfig& colorCfg)
    {
        pyrCfg.levels = 2;
        pyrCfg.scale = 0.75f;
        pyrCfg.baseSize = { 64, 64 };
        colorCfg.bins = 32;
    }

} // namespace

void saveFeaturesToCSV(const std::string& path,
                       const std::vector<std::vector<float>>& features,
                       const std::vector<std::string>& labels)
{
    std::ofstream file(path);
    for (size_t i = 0; i < features.size(); i++)
    {
        file << labels[i];
        for (float v : features[i])
            file << "," << v;
        file << "\n";
    }
    std::cout << "Saved " << features.size() << " feature vectors to " << path
              << "\n";
}

void loadDataset(const std::string& root, std::vector<cv::Mat>& images,
                 std::vector<std::string>& labels)
{
    for (const auto& classDir : fs::directory_iterator(root))
    {
        if (!classDir.is_directory())
            continue;
        std::string label = classDir.path().filename().string();

        for (const auto& imgFile : fs::directory_iterator(classDir))
        {
            cv::Mat img = cv::imread(imgFile.path().string());
            if (img.empty())
                continue;
            images.push_back(img);
            labels.push_back(label);
        }
    }
    std::cout << "Loaded " << images.size() << " images\n";
}

std::vector<std::vector<float>>
extractDatasetFeatures(const std::string& root,
                       std::vector<std::string>& labels,
                       const HOGConfig& hogCfg, const PyramidConfig& pyrCfg,
                       const ColorConfig& colorCfg)
{
    std::vector<cv::Mat> images;
    loadDataset(root, images, labels);

    std::vector<std::vector<float>> features;
    features.reserve(images.size());

    for (size_t i = 0; i < images.size(); i++)
    {
        try
        {
            features.push_back(
                extractFeatures(images[i], hogCfg, pyrCfg, colorCfg));

            if (i % 100 == 0)
                std::cout << i << "/" << images.size() << "\r" << std::flush;
        }
        catch (const std::exception& e)
        {
            std::cerr << "Error on image " << i << ": " << e.what() << "\n";
        }
    }
    std::cout << "Extracted " << features.size() << " feature vectors from "
              << root << "\n";
    return features;
}

int train(const std::string& train_root, const std::string& model_path)
{
    HOGConfig hogCfg;
    PyramidConfig pyrCfg;
    ColorConfig colorCfg;
    configureFeatureExtraction(pyrCfg, colorCfg);

    auto extraction_start = std::chrono::steady_clock::now();
    std::vector<std::string> train_labels;
    auto train_features = extractDatasetFeatures(train_root, train_labels,
                                                 hogCfg, pyrCfg, colorCfg);
    auto extraction_end = std::chrono::steady_clock::now();
    if (train_features.empty())
    {
        std::cerr << "No training images found under " << train_root << "\n";
        return 1;
    }

    int n_trees = 50;
    int max_depth = 14;
    int min_samples_leaf = 2;
    int max_features = static_cast<int>(
        std::sqrt(static_cast<double>(train_features[0].size())));

    std::cout << "Training random forest on " << train_features.size()
              << " samples, " << train_features[0].size()
              << " features, max_features=" << max_features << "\n";

    RandomForest forest(n_trees, max_depth, min_samples_leaf, max_features, 42);

    auto train_start = std::chrono::steady_clock::now();
    forest.fit(train_features, train_labels);
    auto train_end = std::chrono::steady_clock::now();

    forest.save(model_path);
    std::cout << "Model saved to " << model_path << "\n";

    double extraction_seconds =
        std::chrono::duration<double>(extraction_end - extraction_start)
            .count();
    double train_seconds =
        std::chrono::duration<double>(train_end - train_start).count();
    std::cout << "extraction_seconds=" << extraction_seconds << "\n";
    std::cout << "train_seconds=" << train_seconds << "\n";

    return 0;
}

int test(const std::string& test_root, const std::string& model_path,
         const std::string& output_path = "predictions.csv")
{
    HOGConfig hogCfg;
    PyramidConfig pyrCfg;
    ColorConfig colorCfg;
    configureFeatureExtraction(pyrCfg, colorCfg);

    auto extraction_start = std::chrono::steady_clock::now();
    std::vector<std::string> test_labels;
    auto test_features = extractDatasetFeatures(test_root, test_labels, hogCfg,
                                                pyrCfg, colorCfg);
    auto extraction_end = std::chrono::steady_clock::now();
    if (test_features.empty())
    {
        std::cerr << "No testing images found under " << test_root << "\n";
        return 1;
    }

    RandomForest forest;
    try
    {
        forest.load(model_path);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Could not load model " << model_path << ": " << e.what()
                  << "\n";
        std::cerr << "Train one first with: " << "train <train_dataset_root> ["
                  << model_path << "]\n";
        return 1;
    }

    // Prediction
    std::ofstream out(output_path);
    out << "true_label,top1,top2,top3,top4,top5\n";

    int correct = 0;

    auto predict_start = std::chrono::steady_clock::now();
    for (size_t i = 0; i < test_features.size(); ++i)
    {
        auto top5 = forest.predict_top_k(test_features[i], 5);
        if (!top5.empty() && top5[0] == test_labels[i])
            correct++;

        out << test_labels[i];
        for (const auto& label : top5)
            out << "," << label;
        out << "\n";
    }
    auto predict_end = std::chrono::steady_clock::now();

    out.close();

    double extraction_seconds =
        std::chrono::duration<double>(extraction_end - extraction_start)
            .count();
    double predict_seconds =
        std::chrono::duration<double>(predict_end - predict_start).count();

    double accuracy = static_cast<double>(correct) / test_features.size();
    std::cout << "top1_accuracy=" << accuracy << "\n";
    std::cout << "extraction_seconds=" << extraction_seconds << "\n";
    std::cout << "predict_seconds=" << predict_seconds << "\n";
    std::cout << "Predictions written to " << output_path << "\n";

    return 0;
}

int predictImage(const std::string& image_path, const std::string& model_path)
{
    RandomForest forest;
    try
    {
        forest.load(model_path);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Could not load model " << model_path << ": " << e.what()
                  << "\n";
        std::cerr << "Train one first with: " << "train <train_dataset_root> ["
                  << model_path << "]\n";
        return 1;
    }

    cv::Mat image = cv::imread(image_path);
    if (image.empty())
    {
        std::cerr << "Could not read image: " << image_path << "\n";
        return 1;
    }

    HOGConfig hogCfg;
    PyramidConfig pyrCfg;
    ColorConfig colorCfg;
    configureFeatureExtraction(pyrCfg, colorCfg);

    auto features = extractFeatures(image, hogCfg, pyrCfg, colorCfg);
    auto top5 = forest.predict_top_k(features, 5);

    std::cout << "Prediction for " << image_path << ":\n";
    for (size_t i = 0; i < top5.size(); i++)
        std::cout << "  " << (i + 1) << ". " << top5[i] << "\n";

    return 0;
}

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::cerr << "Usage:\n"
                  << "  " << argv[0]
                  << " train <train_dataset_root> [model_path]\n"
                  << "  " << argv[0]
                  << " test <test_dataset_root> [model_path]\n"
                  << "  " << argv[0] << " <image_path> <model_path>\n";
        return 1;
    }

    std::string first_arg = argv[1];

    if (first_arg == "train")
    {
        if (argc < 3)
        {
            std::cerr << "Usage: " << argv[0]
                      << " train <train_dataset_root> [model_path]\n";
            return 1;
        }
        std::string model_path = argc > 3 ? argv[3] : kDefaultModelPath;
        return train(argv[2], model_path);
    }

    if (first_arg == "test")
    {
        if (argc < 3)
        {
            std::cerr << "Usage: " << argv[0]
                      << " test <test_dataset_root> <model_path>\n";
            return 1;
        }
        std::string model_path = argc > 3 ? argv[3] : kDefaultModelPath;
        return test(argv[2], model_path);
    }

    std::string model_path = argc > 2 ? argv[2] : kDefaultModelPath;
    return predictImage(first_arg, model_path);
}
