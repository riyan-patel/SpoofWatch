"""Exports a trained LightGBM binary classifier to the flat format
cpp/src/tree_model.cpp reads for branchless inference.

Verified empirically (see commit history): for this model configuration,
LightGBM's predicted probability is exactly sigmoid(sum of each tree's
leaf value for the input) — no extra bias/init_score term. That's the
whole evaluation this format needs to support.

File layout (little-endian, no padding — every field read individually
on the C++ side rather than memcpy'd as a struct, so there's no risk of
compiler struct-packing mismatches between the two languages):

    uint32   magic (0x53574654 = "SWFT")
    uint32   num_trees
    uint32   num_features
    uint32   max_depth_used
    for each tree:
        uint32  num_nodes
        for each node (node 0 is the root):
            int32   feature index (unused for leaves, always valid)
            double  threshold
            int32   left child index
            int32   right child index
            double  leaf_value (only meaningful for leaf nodes)

Leaf nodes are encoded with left == right == their own index and
threshold = +inf, so a fixed-iteration traversal that keeps comparing
past a leaf just stays there — see tree_model.hpp for why that matters.
"""

from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path

import lightgbm as lgb

MAGIC = 0x53574654
MAX_ALLOWED_DEPTH = 16  # must match spoofwatch::TreeModel::kMaxDepth


def _flatten_tree(node_json: dict, nodes: list[tuple], depth: int) -> tuple[int, int]:
    """Appends this subtree's nodes (post-order-ish, children before the
    parent index is finalized) to `nodes`, returns (this node's index,
    deepest leaf depth beneath it).
    """
    if "leaf_value" in node_json:
        idx = len(nodes)
        nodes.append([0, math.inf, idx, idx, float(node_json["leaf_value"])])
        return idx, depth

    if node_json["decision_type"] != "<=":
        raise ValueError(
            f"unsupported decision_type {node_json['decision_type']!r} — "
            "only numerical <= splits are supported (no categorical features expected)"
        )

    idx = len(nodes)
    nodes.append(None)  # placeholder, patched in below
    left_idx, left_depth = _flatten_tree(node_json["left_child"], nodes, depth + 1)
    right_idx, right_depth = _flatten_tree(node_json["right_child"], nodes, depth + 1)
    nodes[idx] = [
        int(node_json["split_feature"]), float(node_json["threshold"]),
        left_idx, right_idx, 0.0,
    ]
    return idx, max(left_depth, right_depth)


def export_lightgbm_model(booster: lgb.Booster, feature_columns: list[str], out_path: Path) -> int:
    """Returns the max tree depth actually used, for the caller to sanity-check."""
    dump = booster.dump_model()
    if not dump["objective"].startswith("binary"):
        raise ValueError(f"expected a binary classifier, got objective={dump['objective']!r}")
    if dump["num_class"] != 1:
        raise ValueError(f"expected num_class=1, got {dump['num_class']}")
    if dump["feature_names"] != feature_columns:
        raise ValueError(
            "feature order mismatch between the trained model and the caller's "
            f"expected column order: model={dump['feature_names']!r} "
            f"expected={feature_columns!r}"
        )

    trees: list[list[list]] = []
    max_depth = 0
    for tree_info in dump["tree_info"]:
        nodes: list = []
        _, depth = _flatten_tree(tree_info["tree_structure"], nodes, 0)
        trees.append(nodes)
        max_depth = max(max_depth, depth)

    if max_depth > MAX_ALLOWED_DEPTH:
        raise ValueError(
            f"tree depth {max_depth} exceeds TreeModel::kMaxDepth={MAX_ALLOWED_DEPTH}; "
            "raise the constant in cpp/include/spoofwatch/tree_model.hpp (and re-verify "
            "predict_proba's fixed-iteration traversal is still correct) before retraining "
            "with settings that grow deeper trees"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(struct.pack("<IIII", MAGIC, len(trees), len(feature_columns), max_depth))
        for nodes in trees:
            f.write(struct.pack("<I", len(nodes)))
            for feature, threshold, left, right, leaf_value in nodes:
                f.write(struct.pack("<idiid", feature, threshold, left, right, leaf_value))

    return max_depth


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_txt", type=Path, help="LightGBM model saved via booster.save_model()")
    parser.add_argument("out_path", type=Path)
    args = parser.parse_args()

    booster = lgb.Booster(model_file=str(args.model_txt))
    depth = export_lightgbm_model(booster, booster.dump_model()["feature_names"], args.out_path)
    print(f"Exported {args.out_path} (max tree depth used: {depth})")


if __name__ == "__main__":
    main()
