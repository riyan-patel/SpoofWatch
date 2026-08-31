"""Shared model-fitting logic used by both `train.py` (Phase 4 evaluation)
and `python/eval/case_study.py` (Phase 7 qualitative sanity check), so the
regularization choices documented in `train.py` — capped scale_pos_weight,
depth/leaf/reg_lambda limits, see docs/PHASES.md Phase 4's correction note
— live in exactly one place.
"""

from __future__ import annotations

import lightgbm as lgb
import pandas as pd

from python.training.dataset import FEATURE_COLUMNS


def scale_pos_weight_for(train: pd.DataFrame, cap: float = 50.0) -> float:
    """Negative:positive ratio, capped so it can't blow up raw scores under
    extreme class imbalance (see train.py's comment on the is_unbalance=True
    regression this replaced).
    """
    positives = max(int((train["label"] == 1).sum()), 1)
    negatives = int((train["label"] == 0).sum())
    return min(negatives / positives, cap)


def fit_model(train: pd.DataFrame, seed: int = 0) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        n_estimators=200,
        num_leaves=15,
        max_depth=6,
        min_child_samples=20,
        reg_lambda=1.0,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight_for(train),
        random_state=seed,
        verbosity=-1,
    )
    model.fit(train[FEATURE_COLUMNS], train["label"])
    return model
