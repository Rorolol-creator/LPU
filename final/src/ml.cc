#include "ml.hh"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <numeric>
#include <set>
#include <stdexcept>

namespace
{

    void writeFloatBits(std::ostream& os, float value)
    {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        os << bits;
    }

    float readFloatBits(std::istream& is)
    {
        std::uint32_t bits = 0;
        is >> bits;
        float value = 0.0f;
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }

    double gini(const std::vector<int>& counts, int total)
    {
        if (total == 0)
        {
            return 0.0;
        }
        double impurity = 1.0;
        for (int c : counts)
        {
            double p = static_cast<double>(c) / total;
            impurity -= p * p;
        }
        return impurity;
    }

} // namespace

DecisionTree::DecisionTree(int max_depth, int min_samples_leaf,
                           int max_features)
    : max_depth_(max_depth)
    , min_samples_leaf_(min_samples_leaf)
    , max_features_(max_features)
{}

void DecisionTree::fit(const std::vector<std::vector<float>>& X,
                       const std::vector<int>& y,
                       const std::vector<int>& sample_indices, int n_classes,
                       std::mt19937& rng)
{
    n_classes_ = n_classes;
    nodes_.clear();
    root_ = build_node(X, y, sample_indices, 0, n_classes, rng);
}

int DecisionTree::build_node(const std::vector<std::vector<float>>& X,
                             const std::vector<int>& y,
                             const std::vector<int>& indices, int depth,
                             int n_classes, std::mt19937& rng)
{
    Node node;
    node.class_counts.assign(n_classes, 0);
    for (int idx : indices)
    {
        node.class_counts[y[idx]]++;
    }
    node.majority_class = static_cast<int>(
        std::max_element(node.class_counts.begin(), node.class_counts.end())
        - node.class_counts.begin());

    bool is_pure = node.class_counts[node.majority_class]
        == static_cast<int>(indices.size());
    bool too_deep = depth >= max_depth_;
    bool too_small = static_cast<int>(indices.size()) < 2 * min_samples_leaf_;

    if (is_pure || too_deep || too_small)
    {
        node.is_leaf = true;
        nodes_.push_back(std::move(node));
        return static_cast<int>(nodes_.size()) - 1;
    }

    size_t n_features = X[indices[0]].size();
    std::vector<int> feature_pool(n_features);
    std::iota(feature_pool.begin(), feature_pool.end(), 0);
    std::shuffle(feature_pool.begin(), feature_pool.end(), rng);
    int n_candidates = std::min(static_cast<int>(n_features), max_features_);

    double best_impurity =
        gini(node.class_counts, static_cast<int>(indices.size()));
    int best_feature = -1;
    float best_threshold = 0.0f;

    for (int f = 0; f < n_candidates; ++f)
    {
        int feature_index = feature_pool[f];

        std::vector<int> sorted_indices = indices;
        std::sort(sorted_indices.begin(), sorted_indices.end(),
                  [&](int a, int b) {
                      return X[a][feature_index] < X[b][feature_index];
                  });

        std::vector<int> left_counts(n_classes, 0);
        std::vector<int> right_counts = node.class_counts;
        int left_total = 0;
        int right_total = static_cast<int>(indices.size());

        for (size_t i = 0; i + 1 < sorted_indices.size(); ++i)
        {
            int idx = sorted_indices[i];
            int label = y[idx];
            left_counts[label]++;
            right_counts[label]--;
            left_total++;
            right_total--;

            float value = X[idx][feature_index];
            float next_value = X[sorted_indices[i + 1]][feature_index];
            if (value == next_value)
            {
                continue;
            }
            if (left_total < min_samples_leaf_
                || right_total < min_samples_leaf_)
            {
                continue;
            }

            double weighted_impurity =
                (static_cast<double>(left_total) / indices.size())
                    * gini(left_counts, left_total)
                + (static_cast<double>(right_total) / indices.size())
                    * gini(right_counts, right_total);

            if (weighted_impurity < best_impurity)
            {
                best_impurity = weighted_impurity;
                best_feature = feature_index;
                best_threshold = (value + next_value) / 2.0f;
            }
        }
    }

    if (best_feature == -1)
    {
        node.is_leaf = true;
        nodes_.push_back(std::move(node));
        return static_cast<int>(nodes_.size()) - 1;
    }

    std::vector<int> left_indices;
    std::vector<int> right_indices;
    for (int idx : indices)
    {
        if (X[idx][best_feature] <= best_threshold)
        {
            left_indices.push_back(idx);
        }
        else
        {
            right_indices.push_back(idx);
        }
    }

    node.feature_index = best_feature;
    node.threshold = best_threshold;
    nodes_.push_back(std::move(node));
    int node_index = static_cast<int>(nodes_.size()) - 1;

    int left = build_node(X, y, left_indices, depth + 1, n_classes, rng);
    int right = build_node(X, y, right_indices, depth + 1, n_classes, rng);
    nodes_[node_index].left = left;
    nodes_[node_index].right = right;

    return node_index;
}

int DecisionTree::predict(const std::vector<float>& x) const
{
    int current = root_;
    while (!nodes_[current].is_leaf)
    {
        const Node& node = nodes_[current];
        current =
            (x[node.feature_index] <= node.threshold) ? node.left : node.right;
    }
    return nodes_[current].majority_class;
}

void DecisionTree::accumulate_votes(const std::vector<float>& x,
                                    std::vector<float>& votes) const
{
    int current = root_;
    while (!nodes_[current].is_leaf)
    {
        const Node& node = nodes_[current];
        current =
            (x[node.feature_index] <= node.threshold) ? node.left : node.right;
    }
    const Node& leaf = nodes_[current];
    int total =
        std::accumulate(leaf.class_counts.begin(), leaf.class_counts.end(), 0);
    if (total == 0)
    {
        return;
    }
    for (size_t c = 0; c < leaf.class_counts.size(); ++c)
    {
        votes[c] += static_cast<float>(leaf.class_counts[c])
            / static_cast<float>(total);
    }
}

void DecisionTree::save(std::ostream& os) const
{
    os << max_depth_ << ' ' << min_samples_leaf_ << ' ' << max_features_ << ' '
       << n_classes_ << ' ' << root_ << '\n';
    os << nodes_.size() << '\n';
    for (const auto& node : nodes_)
    {
        os << node.is_leaf << ' ' << node.feature_index << ' ';
        writeFloatBits(os, node.threshold);
        os << ' ' << node.left << ' ' << node.right << ' '
           << node.majority_class << ' ' << node.class_counts.size();
        for (int count : node.class_counts)
            os << ' ' << count;
        os << '\n';
    }
}

void DecisionTree::load(std::istream& is)
{
    is >> max_depth_ >> min_samples_leaf_ >> max_features_ >> n_classes_
        >> root_;

    size_t n_nodes = 0;
    is >> n_nodes;
    nodes_.assign(n_nodes, Node{});
    for (Node& node : nodes_)
    {
        int is_leaf = 0;
        size_t n_counts = 0;
        is >> is_leaf >> node.feature_index;
        node.threshold = readFloatBits(is);
        is >> node.left >> node.right >> node.majority_class >> n_counts;
        node.is_leaf = is_leaf != 0;
        node.class_counts.resize(n_counts);
        for (int& count : node.class_counts)
            is >> count;
    }

    if (!is)
        throw std::runtime_error("Malformed decision tree in model file");
}

RandomForest::RandomForest(int n_trees, int max_depth, int min_samples_leaf,
                           int max_features, unsigned seed)
    : n_trees_(n_trees)
    , max_depth_(max_depth)
    , min_samples_leaf_(min_samples_leaf)
    , max_features_(max_features)
    , seed_(seed)
{}

void RandomForest::fit(const std::vector<std::vector<float>>& X,
                       const std::vector<std::string>& y)
{
    std::set<std::string> unique_labels(y.begin(), y.end());
    classes_.assign(unique_labels.begin(), unique_labels.end());
    for (size_t i = 0; i < classes_.size(); ++i)
    {
        class_to_index_[classes_[i]] = static_cast<int>(i);
    }

    std::vector<int> y_encoded(y.size());
    for (size_t i = 0; i < y.size(); ++i)
    {
        y_encoded[i] = class_to_index_[y[i]];
    }

    int n_classes = static_cast<int>(classes_.size());
    int n_samples = static_cast<int>(X.size());

    std::mt19937 rng(seed_);
    std::uniform_int_distribution<int> sample_dist(0, n_samples - 1);

    trees_.clear();
    trees_.reserve(n_trees_);
    for (int t = 0; t < n_trees_; ++t)
    {
        std::vector<int> bootstrap(n_samples);
        for (int i = 0; i < n_samples; ++i)
        {
            bootstrap[i] = sample_dist(rng);
        }
        DecisionTree tree(max_depth_, min_samples_leaf_, max_features_);
        tree.fit(X, y_encoded, bootstrap, n_classes, rng);
        trees_.push_back(std::move(tree));
    }
}

std::string RandomForest::predict(const std::vector<float>& x) const
{
    std::vector<float> votes(classes_.size(), 0.0f);
    for (const auto& tree : trees_)
    {
        tree.accumulate_votes(x, votes);
    }
    int best = static_cast<int>(std::max_element(votes.begin(), votes.end())
                                - votes.begin());
    return classes_[best];
}

std::vector<std::string>
RandomForest::predict_top_k(const std::vector<float>& x, int k) const
{
    std::vector<float> votes(classes_.size(), 0.0f);
    for (const auto& tree : trees_)
    {
        tree.accumulate_votes(x, votes);
    }

    std::vector<int> order(classes_.size());
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(
        order.begin(), order.begin() + std::min<size_t>(k, order.size()),
        order.end(), [&](int a, int b) { return votes[a] > votes[b]; });

    std::vector<std::string> top_k;
    for (int i = 0; i < std::min<int>(k, static_cast<int>(order.size())); ++i)
    {
        top_k.push_back(classes_[order[i]]);
    }
    return top_k;
}

void RandomForest::save(const std::string& path) const
{
    std::ofstream os(path);
    if (!os)
        throw std::runtime_error("Cannot open " + path + " for writing");

    os << n_trees_ << ' ' << max_depth_ << ' ' << min_samples_leaf_ << ' '
       << max_features_ << ' ' << seed_ << '\n';
    os << classes_.size() << '\n';
    for (const std::string& label : classes_)
        os << label << '\n';
    for (const DecisionTree& tree : trees_)
        tree.save(os);
}

void RandomForest::load(const std::string& path)
{
    std::ifstream is(path);
    if (!is)
        throw std::runtime_error("Cannot open " + path + " for reading");

    is >> n_trees_ >> max_depth_ >> min_samples_leaf_ >> max_features_ >> seed_;

    size_t n_classes = 0;
    is >> n_classes;
    is.ignore();

    classes_.resize(n_classes);
    for (std::string& label : classes_)
        std::getline(is, label);

    class_to_index_.clear();
    for (size_t i = 0; i < classes_.size(); ++i)
        class_to_index_[classes_[i]] = static_cast<int>(i);

    trees_.assign(n_trees_, DecisionTree{});
    for (DecisionTree& tree : trees_)
        tree.load(is);

    if (!is)
        throw std::runtime_error("Malformed model file: " + path);
}
