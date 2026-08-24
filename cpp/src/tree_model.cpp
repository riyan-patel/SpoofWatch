#include "spoofwatch/tree_model.hpp"

#include <cmath>
#include <fstream>
#include <stdexcept>

namespace spoofwatch {

namespace {

constexpr uint32_t kMagic = 0x53574654; // "SWFT", matches export_model.py

template <typename T>
T read_value(std::ifstream& file, const std::string& what) {
    T value{};
    file.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!file) {
        throw std::runtime_error("Failed to read " + what + " from tree model file");
    }
    return value;
}

} // namespace

TreeModel::TreeModel(const std::string& binary_model_path) {
    std::ifstream file(binary_model_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open tree model file: " + binary_model_path);
    }

    uint32_t magic = read_value<uint32_t>(file, "magic");
    if (magic != kMagic) {
        throw std::runtime_error("Bad magic in tree model file: " + binary_model_path);
    }
    uint32_t num_trees = read_value<uint32_t>(file, "num_trees");
    uint32_t num_features = read_value<uint32_t>(file, "num_features");
    uint32_t max_depth_used = read_value<uint32_t>(file, "max_depth_used");
    if (max_depth_used > kMaxDepth) {
        throw std::runtime_error(
            "Tree model requires depth " + std::to_string(max_depth_used) +
            " but TreeModel::kMaxDepth is " + std::to_string(kMaxDepth));
    }

    num_features_ = num_features;
    trees_.resize(num_trees);
    for (Tree& tree : trees_) {
        uint32_t num_nodes = read_value<uint32_t>(file, "num_nodes");
        tree.nodes.resize(num_nodes);
        for (Node& node : tree.nodes) {
            node.feature = read_value<int32_t>(file, "node.feature");
            node.threshold = read_value<double>(file, "node.threshold");
            node.left = read_value<int32_t>(file, "node.left");
            node.right = read_value<int32_t>(file, "node.right");
            node.leaf_value = read_value<double>(file, "node.leaf_value");
        }
    }
}

double TreeModel::predict_proba(const double* features, size_t num_features) const {
    (void)num_features; // only checked by the caller in debug builds; hot path trusts it

    double raw = 0.0;
    for (const Tree& tree : trees_) {
        const Node* nodes = tree.nodes.data();
        int32_t idx = 0;
        for (size_t depth = 0; depth < kMaxDepth; ++depth) {
            const Node& n = nodes[idx];
            int go_right = static_cast<int>(features[n.feature] > n.threshold);
            idx = n.left + go_right * (n.right - n.left);
        }
        raw += nodes[idx].leaf_value;
    }
    return 1.0 / (1.0 + std::exp(-raw));
}

} // namespace spoofwatch
