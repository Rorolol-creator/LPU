#pragma once

#include <iosfwd>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

class DecisionTree
{
public:
    DecisionTree() = default;
    DecisionTree(int max_depth, int min_samples_leaf, int max_features);

    void fit(const std::vector<std::vector<float>>& X,
             const std::vector<int>& y, const std::vector<int>& sample_indices,
             int n_classes, std::mt19937& rng);

    int predict(const std::vector<float>& x) const;
    void accumulate_votes(const std::vector<float>& x,
                          std::vector<float>& votes) const;

    void save(std::ostream& os) const;
    void load(std::istream& is);

private:
    struct Node
    {
        bool is_leaf = false;
        int feature_index = -1;
        float threshold = 0.0f;
        int left = -1;
        int right = -1;
        std::vector<int> class_counts;
        int majority_class = -1;
    };

    int build_node(const std::vector<std::vector<float>>& X,
                   const std::vector<int>& y, const std::vector<int>& indices,
                   int depth, int n_classes, std::mt19937& rng);

    int max_depth_ = 0;
    int min_samples_leaf_ = 0;
    int max_features_ = 0;
    int n_classes_ = 0;
    std::vector<Node> nodes_;
    int root_ = -1;
};

class RandomForest
{
public:
    RandomForest() = default;
    RandomForest(int n_trees, int max_depth, int min_samples_leaf,
                 int max_features, unsigned seed);

    void fit(const std::vector<std::vector<float>>& X,
             const std::vector<std::string>& y);

    std::string predict(const std::vector<float>& x) const;
    std::vector<std::string> predict_top_k(const std::vector<float>& x,
                                           int k) const;

    const std::vector<std::string>& classes() const
    {
        return classes_;
    }

    void save(const std::string& path) const;
    void load(const std::string& path);

private:
    int n_trees_ = 0;
    int max_depth_ = 0;
    int min_samples_leaf_ = 0;
    int max_features_ = 0;
    unsigned seed_ = 0;

    std::vector<DecisionTree> trees_;
    std::vector<std::string> classes_;
    std::unordered_map<std::string, int> class_to_index_;
};
