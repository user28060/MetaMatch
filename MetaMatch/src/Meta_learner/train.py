# Meta_learner/train.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import click
import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Core classifiers (always available if scikit-learn is installed)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# Optional: CatBoost / XGBoost if installed
try:
    from catboost import CatBoostClassifier  # type: ignore
    _HAS_CATBOOST = True
except Exception:
    _HAS_CATBOOST = False

try:
    from xgboost import XGBClassifier  # type: ignore
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False


META_COLS = {
    "true_match",
    "Attribute1",
    "Attribute2",
    "Category",
    "Relation",
    "Dataset",
    "Model",
    "ExecutionTime",
    "FilePath",
    "Run",
    "Scenario",
}


def load_features(csv_path: str | Path) -> pd.DataFrame:
    """Load the feature table produced by MetaMatch."""
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")
    return pd.read_csv(p)


def make_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Select X (features) and y=true_match from a MetaMatch table."""
    if "true_match" not in df.columns:
        raise ValueError("Column 'true_match' not found in features CSV.")
    y = df["true_match"].astype(int)

    keep = [c for c in df.columns if c not in META_COLS]
    X = df[keep].copy()

    # Drop all-NA columns
    all_na = X.columns[X.isna().all()].tolist()
    if all_na:
        X = X.drop(columns=all_na)

    return X, y


def split_by_dataset(
    df: pd.DataFrame, train_fraction: float, seed: int
) -> np.ndarray:
    """Return a boolean mask of rows for training, stratified by Dataset IDs."""
    if "Dataset" not in df.columns:
        # fallback: random split handled by train_test_split later
        return np.array([True] * len(df))
    dsets = df["Dataset"].astype(str).to_numpy()
    uniq = np.unique(dsets)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    if len(uniq) <= 1:
        train_set = set(uniq.tolist())
    else:
        cut = max(1, min(int(train_fraction * len(uniq)), len(uniq) - 1))
        train_set = set(uniq[:cut].tolist())
    return np.isin(dsets, list(train_set))


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Preprocess numeric and categorical features."""
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    num_tf = Pipeline(
        steps=[
            ("imp", SimpleImputer(strategy="median")),
            ("std", StandardScaler(with_mean=True, with_std=True)),
        ]
    )
    cat_tf = Pipeline(
        steps=[
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[("num", num_tf, num_cols), ("cat", cat_tf, cat_cols)],
        remainder="drop",
        sparse_threshold=0.0,
    )


def list_classifiers() -> Dict[str, object]:
    """Return the available classifiers dictionary."""
    clf: Dict[str, object] = {
        "LogReg": LogisticRegression(max_iter=5000, class_weight="balanced"),
        "RF": RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", random_state=42),
        "GBT": GradientBoostingClassifier(random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5, weights="distance"),
        "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42),
        "SVMlin": SVC(kernel="linear", probability=True, class_weight="balanced", random_state=42),
    }
    if _HAS_CATBOOST:
        clf["CatBoost"] = CatBoostClassifier(verbose=0, auto_class_weights="Balanced", random_seed=42)
    if _HAS_XGB:
        clf["XGBoost"] = XGBClassifier(
            use_label_encoder=False, eval_metric="logloss", random_state=42, n_estimators=300, n_jobs=1
        )
    return clf



def evaluate(
    y_true: np.ndarray, y_prob: Optional[np.ndarray], y_pred: np.ndarray
) -> dict:
    """Compute a compact set of metrics."""
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": p,
        "recall": r,
        "f1": f1,
    }
    if y_prob is not None:
        try:
            out["roc_auc"] = roc_auc_score(y_true, y_prob)
        except Exception:
            out["roc_auc"] = np.nan
    return out

# --- keep all your imports and helper functions above ---

def get_available_classifiers() -> Dict[str, object]:
    """Return the available classifiers dictionary."""
    clf: Dict[str, object] = {
        "LogReg": LogisticRegression(max_iter=5000, class_weight="balanced"),
        "RF": RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", random_state=42),
        "GBT": GradientBoostingClassifier(random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5, weights="distance"),
        "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42),
        "SVMlin": SVC(kernel="linear", probability=True, class_weight="balanced", random_state=42),
    }
    if _HAS_CATBOOST:
        clf["CatBoost"] = CatBoostClassifier(verbose=0, auto_class_weights="Balanced", random_seed=42)
    if _HAS_XGB:
        clf["XGBoost"] = XGBClassifier(
            use_label_encoder=False, eval_metric="logloss", random_state=42, n_estimators=300, n_jobs=1
        )
    return clf


@click.command(name="MetaLearn", context_settings={"show_default": True})
@click.option("--features-csv", required=False, type=click.Path(exists=True, dir_okay=False),
              help="Path to MetaMatch features CSV.")
@click.option("--classifier", "classifiers", multiple=True,
              help="One or more classifiers (default: RF). Choices printed by --list-classifiers.")
@click.option("--list-classifiers", "list_clfs", is_flag=True,
              help="List available classifier names and exit.")
@click.option("--split", type=click.Choice(["random", "by-dataset"]), default="by-dataset",
              help="Train/test split strategy.")
@click.option("--test-size", default=0.2, type=float,
              help="Hold-out split size if --split=random or by-dataset.")
@click.option("--seed", default=42, type=int, help="Random seed.")
@click.option("--out-dir", default="MetaMatch/tests/results_meta_learner", type=click.Path(file_okay=False),
              help="Output directory.")
def meta_learn(
    features_csv: Optional[str],
    classifiers: tuple[str, ...],
    list_clfs: bool,
    split: str,
    test_size: float,
    seed: int,
    out_dir: str,
) -> None:
    """Train/test a binary classifier to predict true_match from MetaMatch features."""
    available = get_available_classifiers()

    # Fast path: just list classifiers and exit
    if list_clfs:
        click.echo("Available classifiers:")
        for k in available:
            click.echo(f" - {k}")
        return

    # From here on we actually need the features CSV
    if not features_csv:
        raise click.BadParameter("Missing --features-csv (required unless --list-classifiers is used).")

    # Defaults and validation
    if not classifiers:
        classifiers = ("RF",)
    for name in classifiers:
        if name not in available:
            raise click.BadParameter(
                f"Unknown classifier '{name}'. Use --list-classifiers to see choices."
            )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_features(features_csv)
    X, y = make_xy(df)

    # Split
    if split == "by-dataset":
        mask_train = split_by_dataset(df, train_fraction=1.0 - test_size, seed=seed)
        mask_test = ~mask_train
        if not mask_test.any():
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=seed, stratify=y
            )
        else:
            X_train, X_test = X[mask_train], X[mask_test]
            y_train, y_test = y[mask_train], y[mask_test]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )

    pre = build_preprocessor(X)

    results_rows: List[dict] = []
    percat_rows: List[dict] = []

    cats_test = (
        df.loc[y_test.index, "Category"].astype(str).values
        if "Category" in df.columns
        else np.array(["NA"] * len(y_test))
    )

    for name in classifiers:
        base = available[name]
        model = clone(base)
        pipe = Pipeline(steps=[("pre", pre), ("clf", model)])

        pipe.fit(X_train, y_train)

        # Probabilities if available
        if hasattr(pipe[-1], "predict_proba"):
            y_prob: Optional[np.ndarray] = pipe.predict_proba(X_test)[:, 1]
        elif hasattr(pipe[-1], "decision_function"):
            scores = pipe.decision_function(X_test)
            y_prob = pd.Series(scores).rank(pct=True).to_numpy()
        else:
            y_prob = None

        y_pred = pipe.predict(X_test)

        met = evaluate(y_test.to_numpy(), y_prob, y_pred)
        results_rows.append(
            {
                "Classifier": name,
                "n_train": int(len(y_train)),
                "n_test": int(len(y_test)),
                **{k: float(v) if v is not None else np.nan for k, v in met.items()},
            }
        )

        # Per-category metrics
        for cat in np.unique(cats_test):
            m = cats_test == cat
            yt = y_test.to_numpy()[m]
            yp = y_pred[m]
            if len(yt) == 0:
                continue
            percat_rows.append(
                {
                    "Classifier": name,
                    "Category": str(cat),
                    "n_test": int(len(yt)),
                    "precision": precision_score(yt, yp, zero_division=0),
                    "recall": recall_score(yt, yp, zero_division=0),
                    "f1": f1_score(yt, yp, zero_division=0),
                }
            )

        # Save predictions and model
        pred_df = pd.DataFrame(
            {"y_true": y_test.to_numpy(), "y_pred": y_pred, "y_prob": y_prob if y_prob is not None else np.nan}
        )
        pred_csv = out / f"predictions__{name}.csv"
        pred_df.to_csv(pred_csv, index=False)
        joblib.dump(pipe, out / f"model__{name}.joblib")

        click.echo(f"[OK] {name}: saved predictions -> {pred_csv}")

    # Write metrics
    pd.DataFrame(results_rows).to_csv(out / "metrics_global.csv", index=False)
    if percat_rows:
        pd.DataFrame(percat_rows).to_csv(out / "metrics_per_category.csv", index=False)
    click.echo(f"[OK] Wrote metrics to: {out}")
