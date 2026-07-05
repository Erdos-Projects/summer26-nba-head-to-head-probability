"""
Feature engineering for the Court Intelligence NBA project.

Two feature families, both computed per-team with a shift(1) to prevent
lookahead (a game's own result never leaks into its own feature):
  - Roll10: trailing 10-game rolling average ("form" features)
  - EWM3:   exponentially weighted mean, halflife=3 ("form" features)

Plus one game-context feature family that is NOT smoothed, since it describes
the specific game rather than a trend: rest_days, b2b.
"""

import pandas as pd

WINDOW = 10
HALF_LIFE = 3

# Features that need smoothing -- a single game's value is noisy, so we track
# recent form via rolling/EWM average instead.
FORM_STAT_COLS = [
    "teamScore", "fieldGoalsPercentage", "threePointersPercentage",
    "freeThrowsPercentage", "reboundsOffensive", "reboundsDefensive",
    "assists", "turnovers", "steals", "blocks", "plusMinusPoints", "win",
]

# Features that describe game-specific context -- already exactly right as a
# single value; smoothing them would destroy the signal (e.g. averaging rest
# days over the last 10 games would erase how rested a team is *tonight*).
CONTEXT_COLS = ["rest_days", "b2b"]


def add_rolling_features(df: pd.DataFrame, cols=FORM_STAT_COLS, window: int = WINDOW) -> pd.DataFrame:
    """Trailing N-game rolling average per team, shifted by 1 (no lookahead)."""
    df = df.copy()
    for col in cols:
        df[f"{col}_roll{window}"] = (
            df.groupby("teamId")[col]
            .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
        )
    return df


def add_ewm_features(df: pd.DataFrame, cols=FORM_STAT_COLS, half_life: int = HALF_LIFE) -> pd.DataFrame:
    """Exponentially weighted mean per team (halflife games), shifted by 1 (no lookahead)."""
    df = df.copy()
    for col in cols:
        df[f"{col}_ewm{half_life}"] = (
            df.groupby("teamId")[col]
            .transform(lambda x: x.shift(1).ewm(halflife=half_life).mean())
        )
    return df


def roll10_cols(cols=FORM_STAT_COLS, window: int = WINDOW):
    return [f"{col}_roll{window}" for col in cols]


def ewm_cols(cols=FORM_STAT_COLS, half_life: int = HALF_LIFE):
    return [f"{col}_ewm{half_life}" for col in cols]


def build_matchups(df: pd.DataFrame, feature_cols, extra_cols=tuple(CONTEXT_COLS)):
    """Pair each game's home and away rows together, then take home-minus-away
    for every feature_col and extra_col. Returns (matchups_df, diff_cols).
    """
    id_cols = ["teamId", "teamName"] + list(extra_cols)

    home = df[df["home"] == 1][
        ["gameId", "gameDate", "season", "win"] + id_cols + list(feature_cols)
    ].rename(columns={c: f"home_{c}" for c in id_cols + list(feature_cols)})

    away = df[df["home"] == 0][
        ["gameId"] + id_cols + list(feature_cols)
    ].rename(columns={c: f"away_{c}" for c in id_cols + list(feature_cols)})

    matchups = home.merge(away, on="gameId")

    diff_cols = []
    for c in list(feature_cols) + list(extra_cols):
        diff_name = f"{c}_diff"
        matchups[diff_name] = matchups[f"home_{c}"] - matchups[f"away_{c}"]
        diff_cols.append(diff_name)

    return matchups, diff_cols


def build_matchups_clean(df: pd.DataFrame, feature_cols, extra_cols=tuple(CONTEXT_COLS)):
    """Same as build_matchups, but also drops rows with NaN diff features
    (early-season games where the rolling/EWM window isn't warmed up yet)."""
    matchups, diff_cols = build_matchups(df, feature_cols, extra_cols)
    return matchups.dropna(subset=diff_cols), diff_cols
