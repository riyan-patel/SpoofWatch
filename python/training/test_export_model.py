import math
import struct

import numpy as np
import pandas as pd
import lightgbm as lgb
import pytest

from python.training.export_model import MAGIC, export_lightgbm_model


def _read_binary_model(path):
    """Minimal from-scratch parser mirroring cpp/src/tree_model.cpp's
    reader, kept independent of that C++ code so this test can catch a
    format regression without needing the C++ binary built.
    """
    with open(path, "rb") as f:
        data = f.read()
    magic, num_trees, num_features, max_depth = struct.unpack_from("<IIII", data, 0)
    assert magic == MAGIC
    offset = 16
    trees = []
    for _ in range(num_trees):
        (num_nodes,) = struct.unpack_from("<I", data, offset)
        offset += 4
        nodes = []
        for _ in range(num_nodes):
            feature, threshold, left, right, leaf_value = struct.unpack_from("<idiid", data, offset)
            offset += struct.calcsize("<idiid")
            nodes.append((feature, threshold, left, right, leaf_value))
        trees.append(nodes)
    assert offset == len(data)
    return num_features, max_depth, trees


def _predict_via_parsed_trees(trees, x):
    raw = 0.0
    for nodes in trees:
        idx = 0
        for _ in range(64):  # generous fixed bound, mirrors kMaxDepth's role
            feature, threshold, left, right, leaf_value = nodes[idx]
            next_idx = left if x[feature] <= threshold else right
            if next_idx == idx:
                raw += leaf_value
                break
            idx = next_idx
        else:
            raise AssertionError("tree traversal did not terminate")
    return 1.0 / (1.0 + math.exp(-raw))


def test_export_round_trips_to_same_probabilities_as_booster(tmp_path):
    rng = np.random.default_rng(0)
    feature_columns = ["a", "b", "c"]
    X = pd.DataFrame(rng.normal(size=(300, 3)), columns=feature_columns)
    y = (X["a"] + X["b"] > 0.3).astype(int)

    model = lgb.LGBMClassifier(n_estimators=15, num_leaves=7, random_state=0, verbosity=-1)
    model.fit(X, y)

    out_path = tmp_path / "model.bin"
    export_lightgbm_model(model.booster_, feature_columns, out_path)

    num_features, max_depth, trees = _read_binary_model(out_path)
    assert num_features == 3
    assert max_depth <= 16

    expected = model.predict_proba(X)[:, 1]
    actual = np.array([_predict_via_parsed_trees(trees, x) for x in X.to_numpy()])
    np.testing.assert_allclose(actual, expected, atol=1e-9)


def test_export_rejects_feature_order_mismatch(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 2))
    y = (X[:, 0] > 0).astype(int)
    model = lgb.LGBMClassifier(n_estimators=3, num_leaves=3, random_state=0, verbosity=-1)
    model.fit(X, y, feature_name=["x0", "x1"])

    with pytest.raises(ValueError, match="feature order mismatch"):
        export_lightgbm_model(model.booster_, ["x1", "x0"], tmp_path / "model.bin")


def test_export_rejects_non_binary_objective(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 2))
    y = rng.integers(0, 3, size=60)
    model = lgb.LGBMClassifier(n_estimators=3, num_leaves=3, random_state=0, verbosity=-1)
    model.fit(X, y, feature_name=["x0", "x1"])

    with pytest.raises(ValueError, match="binary classifier"):
        export_lightgbm_model(model.booster_, ["x0", "x1"], tmp_path / "model.bin")
