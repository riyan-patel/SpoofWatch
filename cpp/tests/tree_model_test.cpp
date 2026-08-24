#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include "spoofwatch/tree_model.hpp"

using spoofwatch::TreeModel;

namespace {

std::string temp_model_path(const std::string& test_name) {
    return (std::filesystem::temp_directory_path() / ("spoofwatch_" + test_name + ".model")).string();
}

struct RawNode {
    int32_t feature;
    double threshold;
    int32_t left;
    int32_t right;
    double leaf_value;
};

// Mirrors python/training/export_model.py's binary layout exactly, so
// these tests exercise the real file format, not a simplified stand-in.
void write_test_model(const std::string& path, uint32_t num_features,
                       const std::vector<std::vector<RawNode>>& trees) {
    std::ofstream out(path, std::ios::binary);
    uint32_t magic = 0x53574654;
    uint32_t num_trees = static_cast<uint32_t>(trees.size());
    uint32_t max_depth = 0; // unused by TreeModel beyond the <= kMaxDepth check
    out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    out.write(reinterpret_cast<const char*>(&num_trees), sizeof(num_trees));
    out.write(reinterpret_cast<const char*>(&num_features), sizeof(num_features));
    out.write(reinterpret_cast<const char*>(&max_depth), sizeof(max_depth));
    for (const auto& nodes : trees) {
        uint32_t num_nodes = static_cast<uint32_t>(nodes.size());
        out.write(reinterpret_cast<const char*>(&num_nodes), sizeof(num_nodes));
        for (const RawNode& n : nodes) {
            out.write(reinterpret_cast<const char*>(&n.feature), sizeof(n.feature));
            out.write(reinterpret_cast<const char*>(&n.threshold), sizeof(n.threshold));
            out.write(reinterpret_cast<const char*>(&n.left), sizeof(n.left));
            out.write(reinterpret_cast<const char*>(&n.right), sizeof(n.right));
            out.write(reinterpret_cast<const char*>(&n.leaf_value), sizeof(n.leaf_value));
        }
    }
}

double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

// leaf nodes self-loop, matching export_model.py's convention.
RawNode leaf(int32_t self_index, double value) {
    return RawNode{0, std::numeric_limits<double>::infinity(), self_index, self_index, value};
}

} // namespace

TEST(TreeModel, SingleTreeSingleSplitMatchesHandComputedSigmoid) {
    std::string path = temp_model_path("single_tree");
    // root: feature0 <= 0.5 ? left(leaf=-1.0) : right(leaf=2.0)
    std::vector<RawNode> nodes = {
        RawNode{0, 0.5, 1, 2, 0.0},
        leaf(1, -1.0),
        leaf(2, 2.0),
    };
    write_test_model(path, /*num_features=*/1, {nodes});

    TreeModel model(path);
    double f_low = 0.0, f_high = 1.0;
    EXPECT_DOUBLE_EQ(model.predict_proba(&f_low, 1), sigmoid(-1.0));
    EXPECT_DOUBLE_EQ(model.predict_proba(&f_high, 1), sigmoid(2.0));
    std::remove(path.c_str());
}

TEST(TreeModel, EnsembleSumsLeafValuesAcrossTrees) {
    std::string path = temp_model_path("ensemble");
    std::vector<RawNode> tree1 = {
        RawNode{0, 0.5, 1, 2, 0.0},
        leaf(1, -1.0),
        leaf(2, 2.0),
    };
    std::vector<RawNode> tree2 = {
        RawNode{1, 1.0, 1, 2, 0.0},
        leaf(1, 0.5),
        leaf(2, -0.5),
    };
    write_test_model(path, /*num_features=*/2, {tree1, tree2});

    TreeModel model(path);
    EXPECT_EQ(model.num_trees(), 2u);
    // feature0=1.0 (>0.5, tree1 right=2.0), feature1=0.0 (<=1.0, tree2 left=0.5)
    double features[2] = {1.0, 0.0};
    EXPECT_DOUBLE_EQ(model.predict_proba(features, 2), sigmoid(2.0 + 0.5));
    std::remove(path.c_str());
}

TEST(TreeModel, AsymmetricDepthConvergesCorrectlyViaLeafSelfLoop) {
    std::string path = temp_model_path("asymmetric");
    // root: feature0 <= 0.0 ? leaf(depth 1, value=5.0)
    //                       : (feature1 <= 0.0 ? leaf(depth 2, value=1.0)
    //                                          : leaf(depth 2, value=-1.0))
    std::vector<RawNode> nodes = {
        RawNode{0, 0.0, 1, 2, 0.0},
        leaf(1, 5.0),
        RawNode{1, 0.0, 3, 4, 0.0},
        leaf(3, 1.0),
        leaf(4, -1.0),
    };
    write_test_model(path, /*num_features=*/2, {nodes});

    TreeModel model(path);
    double shallow[2] = {-1.0, 100.0}; // hits the depth-1 leaf; feature1 must be ignored
    double deep_left[2] = {1.0, -1.0};
    double deep_right[2] = {1.0, 1.0};
    EXPECT_DOUBLE_EQ(model.predict_proba(shallow, 2), sigmoid(5.0));
    EXPECT_DOUBLE_EQ(model.predict_proba(deep_left, 2), sigmoid(1.0));
    EXPECT_DOUBLE_EQ(model.predict_proba(deep_right, 2), sigmoid(-1.0));
    std::remove(path.c_str());
}

TEST(TreeModel, ThrowsOnBadMagic) {
    std::string path = temp_model_path("bad_magic");
    std::ofstream out(path, std::ios::binary);
    uint32_t bad_magic = 0xDEADBEEF;
    out.write(reinterpret_cast<const char*>(&bad_magic), sizeof(bad_magic));
    out.close();

    EXPECT_THROW(TreeModel model(path), std::runtime_error);
    std::remove(path.c_str());
}
