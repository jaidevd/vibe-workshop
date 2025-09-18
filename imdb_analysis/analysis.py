"""Core data preparation and analysis helpers for IMDb data."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class MovieDataset:
    """Container for the cleaned movie-level information used in the study."""

    basics: pd.DataFrame
    ratings: pd.DataFrame
    _merged_cache: pd.DataFrame | None = field(default=None, init=False, repr=False)

    @property
    def merged(self) -> pd.DataFrame:
        """Return the join between basics and ratings with cached result."""

        if self._merged_cache is None:
            if {"averageRating", "numVotes"}.issubset(self.basics.columns):
                merged = self.basics.copy()
            else:
                merged = self.basics.merge(
                    self.ratings,
                    on="tconst",
                    how="left",
                    validate="one_to_one",
                )
            self._merged_cache = merged
        return self._merged_cache


def load_movie_basics(path: Path, start_year_min: int = 1980) -> pd.DataFrame:
    """Load the title basics dataset and keep mainstream feature films."""

    usecols = [
        "tconst",
        "titleType",
        "primaryTitle",
        "startYear",
        "runtimeMinutes",
        "genres",
        "isAdult",
    ]
    dtype = {
        "tconst": "string",
        "titleType": "category",
        "primaryTitle": "string",
        "startYear": "string",
        "runtimeMinutes": "string",
        "genres": "string",
        "isAdult": "Int8",
    }

    chunks = pd.read_csv(
        path,
        sep="\t",
        compression="infer",
        usecols=usecols,
        dtype=dtype,
        na_values="\\N",
        chunksize=250_000,
        low_memory=False,
    )

    frames = []
    for chunk in chunks:
        movies = chunk[chunk["titleType"] == "movie"].copy()
        movies = movies[movies["isAdult"].fillna(0) == 0]
        movies["startYear"] = pd.to_numeric(movies["startYear"], errors="coerce")
        movies = movies[movies["startYear"].notna()]
        movies = movies[movies["startYear"] >= start_year_min]
        movies["runtimeMinutes"] = pd.to_numeric(movies["runtimeMinutes"], errors="coerce")
        movies.loc[movies["runtimeMinutes"] <= 0, "runtimeMinutes"] = pd.NA
        frames.append(movies[["tconst", "primaryTitle", "startYear", "runtimeMinutes", "genres"]])

    basics = pd.concat(frames, ignore_index=True)
    basics["startYear"] = basics["startYear"].astype("int64")
    return basics


def compute_release_trend(basics: pd.DataFrame, end_year: int | None = None) -> pd.DataFrame:
    """Summarise the growth in the number of releases over time."""

    if end_year is None:
        end_year = datetime.now().year - 1

    counts = (
        basics.groupby("startYear")
        .size()
        .rename("movie_count")
        .reset_index()
        .sort_values("startYear")
    )
    counts = counts[counts["startYear"] <= end_year]
    if counts.empty:
        return pd.DataFrame(columns=["period", "total_movies", "annual_avg"])

    full_years = pd.RangeIndex(counts["startYear"].min(), end_year + 1)
    counts = (
        counts.set_index("startYear")
        .reindex(full_years, fill_value=0)
        .rename_axis("startYear")
        .reset_index()
    )
    counts["movie_count"] = counts["movie_count"].astype(int)
    counts["period_start"] = (counts["startYear"] // 5) * 5
    grouped = counts.groupby("period_start").agg(
        total_movies=("movie_count", "sum"),
        years=("startYear", "nunique"),
    )
    grouped["period_end"] = grouped.index + grouped["years"] - 1
    grouped["annual_avg"] = grouped["total_movies"] / grouped["years"]
    grouped["period"] = grouped.index.astype(int).astype(str) + "-" + grouped["period_end"].astype(int).astype(str)
    release_trend = grouped[["period", "total_movies", "annual_avg"]].reset_index(drop=True)
    release_trend["annual_avg"] = release_trend["annual_avg"].round(1)
    release_trend["total_movies"] = release_trend["total_movies"].astype(int)
    return release_trend


def prepare_recent_movies(dataset: MovieDataset) -> pd.DataFrame:
    """Return recent movies enriched with ratings data."""

    merged = dataset.basics.merge(dataset.ratings, on="tconst", how="inner", validate="one_to_one")
    merged = merged[merged["numVotes"] >= 1]
    return merged


def compute_genre_scores(movies: pd.DataFrame, start_year: int = 2010, min_votes: int = 25_000, min_titles: int = 100) -> pd.DataFrame:
    """Compute aggregated rating information by genre."""

    recent = movies[(movies["startYear"] >= start_year) & (movies["numVotes"] >= min_votes)].copy()
    recent = recent.dropna(subset=["genres", "averageRating"])
    recent["genre"] = recent["genres"].str.split(",")
    exploded = recent.explode("genre")
    exploded = exploded[exploded["genre"].notna()]

    grouped = exploded.groupby("genre").agg(
        movie_count=("tconst", "count"),
        average_rating=("averageRating", "mean"),
        median_rating=("averageRating", "median"),
        high_share=("averageRating", lambda s: (s >= 7.5).mean()),
        median_votes=("numVotes", "median"),
    )

    grouped = grouped[grouped["movie_count"] >= min_titles]
    grouped = grouped.sort_values(["average_rating", "movie_count"], ascending=[False, False])
    grouped = grouped.reset_index()
    grouped["average_rating"] = grouped["average_rating"].round(2)
    grouped["median_rating"] = grouped["median_rating"].round(2)
    grouped["high_share"] = (grouped["high_share"] * 100).round(1)
    grouped["median_votes"] = grouped["median_votes"].astype(int)
    return grouped


def runtime_rating_relationship(movies: pd.DataFrame, start_year: int = 1990, min_votes: int = 5_000) -> Tuple[float, pd.DataFrame]:
    """Evaluate how runtime relates to audience ratings."""

    runtime_data = movies[(movies["startYear"] >= start_year) & (movies["numVotes"] >= min_votes)].copy()
    runtime_data = runtime_data.dropna(subset=["runtimeMinutes", "averageRating"])
    runtime_data = runtime_data[runtime_data["runtimeMinutes"] >= 40]

    correlation = float(runtime_data["runtimeMinutes"].corr(runtime_data["averageRating"]))

    bins = pd.IntervalIndex.from_tuples(
        [(0, 90), (90, 120), (120, 150), (150, 1000)],
        closed="left",
    )
    labels = ["<90 min", "90-119 min", "120-149 min", "150+ min"]
    runtime_data["runtime_band"] = pd.cut(runtime_data["runtimeMinutes"], bins=bins, labels=labels)

    summary = runtime_data.groupby("runtime_band", observed=False).agg(
        movie_count=("tconst", "count"),
        avg_rating=("averageRating", "mean"),
        avg_runtime=("runtimeMinutes", "mean"),
    )
    summary = summary.reset_index()
    summary["avg_rating"] = summary["avg_rating"].round(2)
    summary["avg_runtime"] = summary["avg_runtime"].round(1)
    summary["movie_count"] = summary["movie_count"].astype(int)
    return correlation, summary


def compute_rating_band_distribution(movies: pd.DataFrame) -> pd.DataFrame:
    """Summarise how movies are distributed across IMDb rating bands."""

    rated = movies.dropna(subset=["averageRating", "numVotes", "tconst"]).copy()
    rated = rated[rated["averageRating"] > 0]

    bands = [f"{i}.0-{i}.9" for i in range(1, 10)] + ["10.0"]
    floor_values = np.floor(rated["averageRating"].to_numpy())
    floor_values = np.clip(floor_values, 1, 10).astype(int)

    def _label(value: int) -> str:
        return "10.0" if value >= 10 else f"{value}.0-{value}.9"

    rated["rating_band"] = [_label(val) for val in floor_values]

    grouped = rated.groupby("rating_band", observed=False).agg(
        title_count=("tconst", "count"),
        vote_total=("numVotes", "sum"),
    )

    grouped = grouped.reindex(bands, fill_value=0)
    total_titles = grouped["title_count"].sum()
    total_votes = grouped["vote_total"].sum()

    counts = grouped["title_count"].to_numpy(dtype=float)
    votes = grouped["vote_total"].to_numpy(dtype=float)

    share = counts / total_titles if total_titles > 0 else np.zeros_like(counts)
    vote_share = votes / total_votes if total_votes > 0 else np.zeros_like(votes)
    avg_votes = np.divide(votes, counts, out=np.zeros_like(votes), where=counts > 0)

    grouped["share"] = share
    grouped["vote_share"] = vote_share
    grouped["avg_votes"] = avg_votes

    grouped = grouped.reset_index().rename(columns={"rating_band": "band"})
    grouped["share"] = (grouped["share"] * 100).round(1)
    grouped["vote_share"] = (grouped["vote_share"] * 100).round(1)
    grouped["avg_votes"] = grouped["avg_votes"].round(0).astype(int)
    grouped["title_count"] = grouped["title_count"].astype(int)
    grouped["vote_total"] = grouped["vote_total"].astype(int)
    return grouped


def compute_lockdown_genre_peaks(
    basics: pd.DataFrame,
    lockdown_years: Sequence[int] = (2020, 2021),
    context_years: int = 3,
    min_lockdown_titles: int = 25,
    min_ratio: float = 1.2,
) -> pd.DataFrame:
    """Identify genres whose release counts peaked during lockdown years.

    Parameters
    ----------
    basics:
        Cleaned movie basics dataset.
    lockdown_years:
        Years considered part of the COVID-19 lockdown window.
    context_years:
        Number of years before and after the lockdown window used for baseline
        comparisons.
    min_lockdown_titles:
        Minimum number of movies required in the lockdown year to be
        considered.
    min_ratio:
        Minimum factor by which the lockdown-year volume must exceed the
        average of the preceding context window.
    """

    if not lockdown_years:
        raise ValueError("lockdown_years must contain at least one year")

    lockdown_years = tuple(sorted(set(int(year) for year in lockdown_years)))
    lockdown_start, lockdown_end = lockdown_years[0], lockdown_years[-1]

    pre_start = lockdown_start - context_years
    pre_end = lockdown_start - 1
    post_start = lockdown_end + 1
    post_end = lockdown_end + context_years

    scoped = basics[(basics["startYear"] >= pre_start) & (basics["startYear"] <= post_end)].copy()
    scoped = scoped.dropna(subset=["genres"])
    scoped["genre"] = scoped["genres"].str.split(",")
    scoped = scoped.explode("genre")
    scoped = scoped[scoped["genre"].notna()]
    scoped["genre"] = scoped["genre"].str.strip()
    scoped = scoped[scoped["genre"] != ""]

    counts = (
        scoped.groupby(["genre", "startYear"], observed=False)
        .size()
        .rename("movie_count")
        .reset_index()
    )

    records = []
    full_years = pd.RangeIndex(pre_start, post_end + 1)
    for genre, group in counts.groupby("genre"):
        timeline = (
            group.set_index("startYear")["movie_count"]
            .reindex(full_years, fill_value=0)
            .rename_axis("startYear")
            .reset_index()
        )
        timeline["movie_count"] = timeline["movie_count"].astype(int)

        peak_idx = timeline["movie_count"].idxmax()
        peak_row = timeline.loc[peak_idx]
        peak_year = int(peak_row["startYear"])

        if peak_year not in lockdown_years:
            continue

        peak_count = int(peak_row["movie_count"])
        if peak_count < min_lockdown_titles:
            continue

        pre_window = timeline[(timeline["startYear"] >= pre_start) & (timeline["startYear"] <= pre_end)]
        if pre_window.empty:
            continue

        pre_avg = float(pre_window["movie_count"].mean())
        if pre_avg <= 0:
            continue

        ratio_pre = peak_count / pre_avg
        if ratio_pre < min_ratio:
            continue

        post_window = timeline[(timeline["startYear"] >= post_start) & (timeline["startYear"] <= post_end)]
        post_avg = float(post_window["movie_count"].mean()) if not post_window.empty else float("nan")
        ratio_post = peak_count / post_avg if post_avg and not pd.isna(post_avg) else float("nan")

        records.append(
            {
                "genre": genre,
                "lockdown_year": peak_year,
                "lockdown_releases": peak_count,
                "pre_lockdown_avg": round(pre_avg, 1),
                "post_lockdown_avg": round(post_avg, 1) if not pd.isna(post_avg) else float("nan"),
                "pre_ratio": round(ratio_pre, 2),
                "post_ratio": round(ratio_post, 2) if not pd.isna(ratio_post) else float("nan"),
            }
        )

    peaks = pd.DataFrame.from_records(records)
    if peaks.empty:
        return peaks

    peaks = peaks.sort_values(["pre_ratio", "lockdown_releases"], ascending=[False, False]).reset_index(drop=True)
    return peaks
