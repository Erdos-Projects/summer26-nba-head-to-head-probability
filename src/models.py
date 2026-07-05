"""
Model training and evaluation for the Court Intelligence NBA project.

walk_forward_cv() replaces what used to be five near-identical copy-pasted
loops (RF raw, ET raw, RF ewm, ET ewm, XGB ewm) in baselines_rf_et_xgb.ipynb.
Each of those is now a single call with a different model_factory/feature set.
"""

import pandas as pd
from sklearn.metrics import accuracy_score


def walk_forward_cv(
    df: pd.DataFrame,
    feature_cols,
    model_factory,
    season_col: str = "season",
    target_col: str = "win",
    train_window: int = 5,
    start_year: int = 2015,
    end_year: int = 2026,
) -> pd.DataFrame:
    """Walk-forward validation: for each season i in [start_year, end_year),
    train on seasons (i - train_window) .. (i - 1) and validate on season i.

    model_factory: a zero-arg callable returning a *fresh, unfitted* model
        instance (e.g. `lambda: RandomForestClassifier(...)`), so every fold
        trains an independent model rather than reusing/refitting one object.

    Returns a DataFrame indexed by season with an 'accuracy' column, plus one
    column per feature with that fold's feature_importances_ if the model
    exposes that attribute (tree-based models do; e.g. LogisticRegression
    does not, in which case only 'accuracy' is returned).
    """
    accuracy_by_season = {}
    feature_importance_rows = []
    seasons_with_importance = []

    for i in range(start_year, end_year):
        train = df[df[season_col].isin(range(i - train_window, i))]
        val = df[df[season_col] == i]

        X_tt, y_tt = train[feature_cols], train[target_col]
        X_val, y_val = val[feature_cols], val[target_col]

        model = model_factory()
        model.fit(X_tt, y_tt)
        y_pred = model.predict(X_val)
        accuracy_by_season[i] = accuracy_score(y_val, y_pred)

        if hasattr(model, "feature_importances_"):
            feature_importance_rows.append(model.feature_importances_)
            seasons_with_importance.append(i)

    results = pd.DataFrame.from_dict(accuracy_by_season, orient="index", columns=["accuracy"])
    results.index.name = "season"

    if feature_importance_rows:
        fi_df = pd.DataFrame(
            feature_importance_rows, index=seasons_with_importance, columns=feature_cols
        )
        fi_df.index.name = "season"
        results = results.join(fi_df)

    return results


def home_team_always_wins_accuracy(
    df: pd.DataFrame,
    season_col: str = "season",
    target_col: str = "win",
    start_year: int = 2015,
    end_year: int = 2026,
) -> pd.Series:
    """Baseline: predict the home team always wins. Per-season accuracy is
    just the home team's win rate that season."""
    rows = []
    for year in range(start_year, end_year):
        rows.append({
            "season": year,
            "accuracy": df[df[season_col] == year][target_col].mean(),
        })
    return pd.DataFrame(rows).set_index("season")["accuracy"].rename("home_team_win")


def better_recent_record_accuracy(
    df: pd.DataFrame,
    diff_col: str,
    season_col: str = "season",
    target_col: str = "win",
    start_year: int = 2015,
    end_year: int = 2026,
) -> pd.Series:
    """Baseline: predict the team with the better recent record wins (ties go
    to the home team). `diff_col` should be a home-minus-away rolling win-rate
    diff column (e.g. 'win_roll10_diff'); prediction is diff_col >= 0."""
    rows = []
    for year in range(start_year, end_year):
        season_df = df[df[season_col] == year]
        y_true = season_df[target_col]
        y_pred = (season_df[diff_col] >= 0).astype(int)
        rows.append({"season": year, "accuracy": accuracy_score(y_true, y_pred)})
    return pd.DataFrame(rows).set_index("season")["accuracy"].rename("better_last10_win")
