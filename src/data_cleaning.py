"""
Data loading and cleaning for the NBA Head-to-Head Prediction project.

Pipeline: raw per-team-game rows -> season-boundary filtering (RP scope) ->
gameLabel cleanup -> column trim/sort -> rest_days/b2b -> Reg scope subset.
"""

import pandas as pd

KEEP_COLS = [
    "gameId", "gameDate", "season", "teamId", "teamName",
    "opponentTeamId", "opponentTeamName", "home", "win",
    "teamScore", "opponentScore", "fieldGoalsPercentage",
    "threePointersPercentage", "freeThrowsPercentage",
    "reboundsOffensive", "reboundsDefensive", "assists",
    "turnovers", "steals", "blocks", "plusMinusPoints",
]

EXCLUDE_LABELS = [
    "All-Star", "All-Star Championship",
    "Rising Stars Semifinal", "Rising Stars Final",
]

MAX_REST_DAYS = 5


def load_raw_team_stats(prefix: str) -> pd.DataFrame:
    """Load the raw per-team-game CSV and parse gameDate."""
    team_stats = pd.read_csv(
        prefix + "raw/TeamStatisticsFrom2010.csv",
        dtype={"gameLabel": str, "gameSubLabel": str},
    )
    team_stats["gameDate"] = pd.to_datetime(team_stats["gameDate"])
    return team_stats


def attach_season_boundaries(team_stats: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Merge season_start/season_end/playoffs_start/playoffs_end onto each row."""
    season_dates = pd.read_csv(
        prefix + "processed/nba_season_dates.csv",
        parse_dates=["season_start", "season_end", "playoffs_start", "playoffs_end"],
    )
    season_dates["season_year"] = season_dates["season"].str[:4].astype(int)

    return team_stats.merge(
        season_dates[
            ["season_year", "season_start", "season_end", "playoffs_start", "playoffs_end"]
        ],
        left_on="season", right_on="season_year", how="left",
    )


def filter_to_rp_scope(team_stats: pd.DataFrame) -> pd.DataFrame:
    """Regular season + playoffs only (drops preseason/offseason rows)."""
    return team_stats[
        (team_stats["gameDate"] >= team_stats["season_start"])
        & (team_stats["gameDate"] <= team_stats["playoffs_end"])
    ]


def remove_noncompetitive_games(team_stats: pd.DataFrame) -> pd.DataFrame:
    """Drop All-Star weekend / Rising Stars games. International regular-season
    games (London, Paris, Mexico City, etc.) are legitimate and are kept."""
    return team_stats[~team_stats["gameLabel"].isin(EXCLUDE_LABELS)]


def trim_and_sort(team_stats: pd.DataFrame, extra_cols=("season_end",)) -> pd.DataFrame:
    """Keep only modeling-relevant columns (plus any extras needed downstream),
    sorted by team and date so later groupby('teamId') operations are in order."""
    cols = KEEP_COLS + list(extra_cols)
    return (
        team_stats[cols]
        .copy()
        .sort_values(["teamId", "gameDate"])
        .reset_index(drop=True)
    )


def add_rest_days_and_b2b(df: pd.DataFrame, max_rest_days: int = MAX_REST_DAYS) -> pd.DataFrame:
    """Add rest_days (days off before this game, capped) and b2b (back-to-back flag)."""
    df = df.copy()
    df["rest_days"] = (
        df.groupby("teamId")["gameDate"].diff().dt.days.sub(1).clip(upper=max_rest_days)
    )
    df["rest_days"] = df["rest_days"].fillna(max_rest_days)
    df["b2b"] = (df["rest_days"] == 0).astype(int)
    return df


def split_reg_scope(team_stats_rp: pd.DataFrame) -> pd.DataFrame:
    """Regular-season-only subset of an RP-scope frame."""
    return (
        team_stats_rp[team_stats_rp["gameDate"] <= team_stats_rp["season_end"]]
        .drop(columns=["season_end"])
        .copy()
    )


def build_rp_and_reg_datasets(prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full pipeline: raw CSV -> cleaned team_stats_rp and team_stats_reg, both carrying rest_days/b2b.
    """
    team_stats = load_raw_team_stats(prefix)
    team_stats = attach_season_boundaries(team_stats, prefix)
    team_stats = filter_to_rp_scope(team_stats)
    team_stats = remove_noncompetitive_games(team_stats)

    team_stats_rp = trim_and_sort(team_stats, extra_cols=("season_end",))
    team_stats_rp = add_rest_days_and_b2b(team_stats_rp)

    team_stats_reg = split_reg_scope(team_stats_rp)
    team_stats_rp = team_stats_rp.drop(columns=["season_end"])

    return team_stats_rp, team_stats_reg
