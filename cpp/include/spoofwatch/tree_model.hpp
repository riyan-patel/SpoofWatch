#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace spoofwatch {

// Loads and evaluates a LightGBM binary classifier exported by
// python/training/export_model.py, without LightGBM or ONNX Runtime in
// the hot path.
//
// predict_proba() is branchless in the sense that matters for a real-time
// engine: no data-dependent conditional jump. Each tree is walked for a
// fixed kMaxDepth iterations regardless of its actual depth, and the next
// node index is chosen with arithmetic (`left + go_right * (right -
// left)`) rather than an if/else, so the compiler has no branch to
// mispredict on the split outcome. Leaf nodes self-loop (left == right ==
// their own index, threshold == +inf) so a path that reaches its leaf
// before kMaxDepth iterations just stays there — no separate "have I hit
// a leaf yet?" branch is needed either. All tree storage is loaded once
// at construction; predict_proba() itself never allocates.
class TreeModel {
public:
    // Deepest leaf depth export_model.py will produce before erroring out
    // and asking for this constant to be raised (see export_model.py's
    // MAX_ALLOWED_DEPTH, which must match).
    static constexpr size_t kMaxDepth = 16;

    explicit TreeModel(const std::string& binary_model_path);

    // `features` must be indexed exactly as at export time — the column
    // order the model was trained on (see python/training/dataset.py's
    // FEATURE_COLUMNS). Caller owns the storage; this never allocates.
    double predict_proba(const double* features, size_t num_features) const;

    size_t num_trees() const { return trees_.size(); }
    size_t num_features() const { return num_features_; }

private:
    struct Node {
        int32_t feature;
        double threshold;
        int32_t left;
        int32_t right;
        double leaf_value;
    };

    struct Tree {
        std::vector<Node> nodes; // node 0 is always the root
    };

    std::vector<Tree> trees_;
    size_t num_features_;
};

} // namespace spoofwatch
