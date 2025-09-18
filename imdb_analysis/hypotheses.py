"""Helpers for evaluating published hypotheses against IMDb aggregates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RatingHypothesis:
    """Expected distribution metrics for one or more rating bands."""

    claim: str
    bands: tuple[str, ...]
    expected_titles: int | None = None
    expected_share: float | None = None
    expected_votes: int | None = None


@dataclass(frozen=True)
class LockdownHypothesis:
    """Expected lockdown spike metrics for a single genre."""

    genre: str
    expected_year: int | None = None
    expected_releases: int | None = None
    expected_pre_avg: float | None = None
    expected_post_avg: float | None = None
    expected_pre_ratio: float | None = None
    expected_post_ratio: float | None = None


RATING_BAND_HYPOTHESES: tuple[RatingHypothesis, ...] = (
    RatingHypothesis(
        claim="7.0–7.9 titles dominate",
        bands=("7.0-7.9",),
        expected_titles=534_788,
        expected_share=33.1,
        expected_votes=588_000_000,
    ),
    RatingHypothesis(
        claim="6.0–6.9 ranks second",
        bands=("6.0-6.9",),
        expected_titles=379_310,
        expected_share=23.5,
    ),
    RatingHypothesis(
        claim="Mid-5 scores remain modest",
        bands=("5.0-5.9",),
        expected_titles=193_466,
        expected_share=12.0,
    ),
    RatingHypothesis(
        claim="Low-end 1.x ratings are rare",
        bands=("1.0-1.9",),
        expected_titles=6_509,
        expected_share=0.4,
    ),
    RatingHypothesis(
        claim="High-end 9.0–10.0 acclaim is scarce",
        bands=("9.0-9.9", "10.0"),
        expected_titles=77_073,
        expected_share=4.8,
        expected_votes=64_000_000,
    ),
)


LOCKDOWN_PEAK_HYPOTHESES: tuple[LockdownHypothesis, ...] = (
    LockdownHypothesis(
        genre="Reality-TV",
        expected_year=2020,
        expected_releases=78,
        expected_pre_avg=37.7,
        expected_post_avg=25.0,
        expected_pre_ratio=2.07,
        expected_post_ratio=3.12,
    ),
    LockdownHypothesis(
        genre="Talk-Show",
        expected_year=2020,
        expected_releases=26,
        expected_pre_avg=13.0,
        expected_post_avg=11.7,
        expected_pre_ratio=2.00,
        expected_post_ratio=2.23,
    ),
)


def evaluate_rating_band_hypotheses(
    distribution: pd.DataFrame,
    hypotheses: Sequence[RatingHypothesis] = RATING_BAND_HYPOTHESES,
) -> pd.DataFrame:
    """Compare observed rating distribution metrics with published expectations."""

    if distribution.empty:
        return pd.DataFrame(columns=["claim", "metric", "expected", "actual", "difference"])

    indexed = distribution.set_index("band")
    results: list[dict[str, object]] = []

    for hypothesis in hypotheses:
        subset = indexed.reindex(hypothesis.bands)
        titles = subset["title_count"].sum(skipna=True)
        share = subset["share"].sum(skipna=True)
        votes = subset["vote_total"].sum(skipna=True)

        if hypothesis.expected_titles is not None:
            results.append(
                {
                    "claim": hypothesis.claim,
                    "metric": "Titles",
                    "expected": float(hypothesis.expected_titles),
                    "actual": float(titles),
                    "difference": float(titles - hypothesis.expected_titles),
                }
            )
        if hypothesis.expected_share is not None:
            results.append(
                {
                    "claim": hypothesis.claim,
                    "metric": "Share (%)",
                    "expected": float(hypothesis.expected_share),
                    "actual": float(share),
                    "difference": float(share - hypothesis.expected_share),
                }
            )
        if hypothesis.expected_votes is not None:
            results.append(
                {
                    "claim": hypothesis.claim,
                    "metric": "Votes",
                    "expected": float(hypothesis.expected_votes),
                    "actual": float(votes),
                    "difference": float(votes - hypothesis.expected_votes),
                }
            )

    return pd.DataFrame(results)


def evaluate_lockdown_hypotheses(
    lockdown_peaks: pd.DataFrame,
    hypotheses: Sequence[LockdownHypothesis] = LOCKDOWN_PEAK_HYPOTHESES,
) -> pd.DataFrame:
    """Compare observed lockdown spikes with expected genre performance."""

    if lockdown_peaks.empty:
        return pd.DataFrame(columns=["genre", "metric", "expected", "actual", "difference"])

    indexed = lockdown_peaks.set_index("genre")
    results: list[dict[str, object]] = []

    for hypothesis in hypotheses:
        row = indexed.loc[hypothesis.genre] if hypothesis.genre in indexed.index else None

        if row is not None:
            year = int(row["lockdown_year"])
            releases = int(row["lockdown_releases"])
            pre_avg = float(row["pre_lockdown_avg"])
            post_avg = float(row["post_lockdown_avg"]) if not pd.isna(row["post_lockdown_avg"]) else np.nan
            pre_ratio = float(row["pre_ratio"])
            post_ratio = float(row["post_ratio"]) if not pd.isna(row["post_ratio"]) else np.nan
        else:
            year = np.nan
            releases = np.nan
            pre_avg = np.nan
            post_avg = np.nan
            pre_ratio = np.nan
            post_ratio = np.nan

        def _append(metric: str, expected: float | None, actual: float | int) -> None:
            if expected is None:
                return
            results.append(
                {
                    "genre": hypothesis.genre,
                    "metric": metric,
                    "expected": float(expected),
                    "actual": float(actual),
                    "difference": float(actual) - float(expected),
                }
            )

        _append("Peak year", hypothesis.expected_year, year)
        _append("Lockdown releases", hypothesis.expected_releases, releases)
        _append("Avg pre-lockdown", hypothesis.expected_pre_avg, pre_avg)
        _append("Avg post-lockdown", hypothesis.expected_post_avg, post_avg)
        _append("× vs pre-lockdown", hypothesis.expected_pre_ratio, pre_ratio)
        _append("× vs post-lockdown", hypothesis.expected_post_ratio, post_ratio)

    return pd.DataFrame(results)


def determine_modal_band(distribution: pd.DataFrame) -> str | None:
    """Return the rating band with the highest title count."""

    if distribution.empty:
        return None
    top = distribution.sort_values("title_count", ascending=False).iloc[0]
    return str(top["band"])


def summarise_extreme_share(
    distribution: pd.DataFrame,
    high_bands: Iterable[str] = ("9.0-9.9", "10.0"),
    low_bands: Iterable[str] = ("1.0-1.9",),
) -> dict[str, float]:
    """Return combined shares for the highest and lowest rating bands."""

    if distribution.empty:
        return {"high_share": np.nan, "low_share": np.nan}

    indexed = distribution.set_index("band")
    high = indexed.reindex(tuple(high_bands))["share"].sum(skipna=True)
    low = indexed.reindex(tuple(low_bands))["share"].sum(skipna=True)
    return {"high_share": float(high), "low_share": float(low)}
