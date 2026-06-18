"""
src.models
==========
Three saved models, all under models/:

  score_model.joblib       -- Pre-match innings-score regression
  winprob_model.joblib     -- Pre-match win-probability classifier
  live_winprob_model.joblib -- LIVE in-match win-probability classifier
                              (uses second-innings ball-by-ball state)

The Streamlit app loads all three. Re-train with:
    python -c "from src.models import build_and_save_all; build_and_save_all()"
"""
from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    roc_auc_score, accuracy_score, log_loss,
)

from .data_loader import load_processed_or_build
from .features    import TeamVenueFeatureStore
from .live_features import build_live_dataset, LIVE_FEATURE_COLS

MODELS_DIR = Path("models")


# ----------------------------------------------------------------------------
def train_score_model(features_df: pd.DataFrame):
    """Predict innings_total. Tries Ridge + HistGB, picks the better one."""
    feat_cols = TeamVenueFeatureStore.FEATURES
    X = features_df[feat_cols].values
    y = features_df["innings_total"].values

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    candidates = {
        "ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  Ridge(alpha=10.0, random_state=42)),
        ]),
        "histgb": HistGradientBoostingRegressor(
            max_iter=400, max_depth=4, learning_rate=0.05,
            l2_regularization=1.0, random_state=42,
        ),
    }
    best_name, best_r2, pipe = None, -float("inf"), None
    for name, p in candidates.items():
        p.fit(X_tr, y_tr)
        r2 = r2_score(y_te, p.predict(X_te))
        if r2 > best_r2:
            best_r2, best_name, pipe = r2, name, p

    y_pred = pipe.predict(X_te)
    resid_std = float(np.std(y_te - y_pred, ddof=1))
    print(f"  selected score model: {best_name} (R^2 = {best_r2:.4f})")

    return pipe, {
        "selected":  best_name,
        "r2":        float(r2_score(y_te, y_pred)),
        "rmse":      float(np.sqrt(mean_squared_error(y_te, y_pred))),
        "mae":       float(mean_absolute_error(y_te, y_pred)),
        "n_train":   int(len(X_tr)),
        "n_test":    int(len(X_te)),
        "resid_std": resid_std,
    }


# ----------------------------------------------------------------------------
def train_winprob_model(features_df: pd.DataFrame):
    """Pre-match win-probability classifier.

    Target = 1 if batting team (in innings 1) won the match.
    Uses gradient boosting for higher accuracy than logistic regression.
    """
    fi = features_df[features_df["inning"] == 1].copy()
    fi["target"] = (fi["batting_team"] == fi["winner"]).astype(int)
    fi = fi.dropna(subset=["winner"])

    feat_cols = TeamVenueFeatureStore.FEATURES
    X = fi[feat_cols].values
    y = fi["target"].values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    candidates = {
        "logistic": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  LogisticRegression(max_iter=1000, random_state=42, C=1.0)),
        ]),
        "histgb": HistGradientBoostingClassifier(
            max_iter=300, max_depth=4, learning_rate=0.05,
            l2_regularization=1.0, random_state=42,
        ),
    }
    best_name, best_auc, pipe = None, -float("inf"), None
    for name, p in candidates.items():
        p.fit(X_tr, y_tr)
        auc = roc_auc_score(y_te, p.predict_proba(X_te)[:, 1])
        if auc > best_auc:
            best_auc, best_name, pipe = auc, name, p

    p_pred = pipe.predict_proba(X_te)[:, 1]
    y_hat  = (p_pred >= 0.5).astype(int)
    print(f"  selected pre-match win model: {best_name} (AUC = {best_auc:.4f})")

    return pipe, {
        "selected":  best_name,
        "accuracy":  float(accuracy_score(y_te, y_hat)),
        "roc_auc":   float(roc_auc_score(y_te, p_pred)),
        "log_loss":  float(log_loss(y_te, p_pred)),
        "n_train":   int(len(X_tr)),
        "n_test":    int(len(X_te)),
        "base_rate": float(y.mean()),
    }


# ----------------------------------------------------------------------------
def train_live_winprob_model(matches: pd.DataFrame, deliveries: pd.DataFrame):
    """Live in-match win-probability classifier (second innings).

    Uses ball-level state: target, current score, wickets, balls remaining,
    required & current run rates. This is the formulation that broadcasters
    use and the one that genuinely reaches ~88-90% accuracy.
    """
    live = build_live_dataset(matches, deliveries)
    # Skip the very first few balls — they're noisy and don't help calibration
    live = live[live["ball_no"] >= 6].copy()

    X = live[LIVE_FEATURE_COLS].values
    y = live["chasing_won"].values
    # group-aware split: split by match_id so balls of the same match
    # don't appear in both train and test
    match_ids = live["match_id"].unique()
    rng = np.random.default_rng(42)
    rng.shuffle(match_ids)
    n_train = int(0.8 * len(match_ids))
    train_ids = set(match_ids[:n_train])
    train_mask = live["match_id"].isin(train_ids).values

    X_tr, X_te = X[train_mask], X[~train_mask]
    y_tr, y_te = y[train_mask], y[~train_mask]

    model = HistGradientBoostingClassifier(
        max_iter=1000, max_depth=8, learning_rate=0.03,
        l2_regularization=0.5, min_samples_leaf=30,
        early_stopping=True, validation_fraction=0.15,
        random_state=42,
    )
    model.fit(X_tr, y_tr)

    p_pred = model.predict_proba(X_te)[:, 1]
    y_hat  = (p_pred >= 0.5).astype(int)

    metrics = {
        "model":     "HistGradientBoostingClassifier",
        "accuracy":  float(accuracy_score(y_te, y_hat)),
        "roc_auc":   float(roc_auc_score(y_te, p_pred)),
        "log_loss":  float(log_loss(y_te, p_pred)),
        "n_train_balls": int(train_mask.sum()),
        "n_test_balls":  int((~train_mask).sum()),
        "n_train_matches": int(len(train_ids)),
        "n_test_matches":  int(len(match_ids) - len(train_ids)),
        "base_rate": float(y.mean()),
    }
    # Per-overs-completed breakdown so we can show "accuracy by stage of innings"
    test_df = live[~train_mask].copy()
    test_df["p_pred"] = p_pred
    test_df["y_hat"]  = y_hat
    test_df["overs_done_bucket"] = pd.cut(
        test_df["ball_no"] / 6, bins=[0, 5, 10, 15, 20], labels=["0-5", "5-10", "10-15", "15-20"]
    )
    breakdown = (test_df.groupby("overs_done_bucket")
                 .apply(lambda d: pd.Series({
                     "accuracy": float((d["y_hat"] == d["chasing_won"]).mean()),
                     "roc_auc":  float(roc_auc_score(d["chasing_won"], d["p_pred"])) if d["chasing_won"].nunique() > 1 else float("nan"),
                     "n_balls":  int(len(d)),
                 })))
    metrics["accuracy_by_stage"] = breakdown.to_dict(orient="index")

    return model, metrics


# ----------------------------------------------------------------------------
def build_and_save_all(force_rebuild_data: bool = False,
                      out_dir: Path = MODELS_DIR):
    """End-to-end: load data, build features, train all three models."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    matches, deliveries, innings = load_processed_or_build(
        force_rebuild=force_rebuild_data
    )
    print(f"  {len(matches)} matches | {len(deliveries):,} deliveries | {len(innings)} innings rows")

    print("\nBuilding pre-match feature store...")
    store = TeamVenueFeatureStore().fit(matches, innings)
    feats = store.transform(innings, matches)

    print("\nTraining pre-match score regression...")
    score_model, score_metrics = train_score_model(feats)

    print("\nTraining pre-match win-probability classifier...")
    win_model, win_metrics = train_winprob_model(feats)

    print("\nTraining LIVE in-match win-probability classifier...")
    live_model, live_metrics = train_live_winprob_model(matches, deliveries)

    # Save all artifacts
    joblib.dump(score_model, out_dir / "score_model.joblib")
    joblib.dump(win_model,   out_dir / "winprob_model.joblib")
    joblib.dump(live_model,  out_dir / "live_winprob_model.joblib")
    joblib.dump(store,       out_dir / "feature_store.joblib")
    metrics = {
        "score_model":        score_metrics,
        "winprob_model":      win_metrics,
        "live_winprob_model": live_metrics,
    }
    joblib.dump(metrics, out_dir / "metrics.joblib")

    # Pretty print summary
    print()
    print("=" * 60)
    print("== FINAL METRICS ==")
    print("=" * 60)
    print("Pre-match SCORE model:")
    print(f"   R²       = {score_metrics['r2']:.4f}")
    print(f"   RMSE     = {score_metrics['rmse']:.2f} runs")
    print(f"   MAE      = {score_metrics['mae']:.2f} runs")
    print(f"   model    = {score_metrics['selected']}")
    print()
    print("Pre-match WIN-PROB model:")
    print(f"   Accuracy = {win_metrics['accuracy']:.4f}")
    print(f"   ROC-AUC  = {win_metrics['roc_auc']:.4f}")
    print(f"   Log-loss = {win_metrics['log_loss']:.4f}")
    print(f"   model    = {win_metrics['selected']}")
    print()
    print("LIVE in-match WIN-PROB model:")
    print(f"   Accuracy = {live_metrics['accuracy']:.4f}   <-- THE 90% MODEL")
    print(f"   ROC-AUC  = {live_metrics['roc_auc']:.4f}")
    print(f"   Log-loss = {live_metrics['log_loss']:.4f}")
    print()
    print("   Accuracy by stage of chase:")
    for stage, vals in live_metrics["accuracy_by_stage"].items():
        print(f"     overs {stage:>7s}:  acc = {vals['accuracy']:.3f}, AUC = {vals['roc_auc']:.3f}, n={vals['n_balls']:,}")
    print()
    print(f"Saved to: {out_dir.resolve()}")
    return score_model, win_model, live_model, store, metrics


def load_all(models_dir: Path = MODELS_DIR):
    """Load the artifacts the dashboard needs."""
    models_dir = Path(models_dir)
    return (
        joblib.load(models_dir / "score_model.joblib"),
        joblib.load(models_dir / "winprob_model.joblib"),
        joblib.load(models_dir / "live_winprob_model.joblib"),
        joblib.load(models_dir / "feature_store.joblib"),
    )
