"""Run the IMDb data pipeline and write a Markdown report."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.patches as mpatches

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from imdb_analysis.analysis import (
    MovieDataset,
    compute_genre_scores,
    compute_lockdown_genre_peaks,
    compute_rating_band_distribution,
    compute_release_trend,
    load_movie_basics,
    runtime_rating_relationship,
)
from imdb_analysis.hypotheses import (
    LOCKDOWN_PEAK_HYPOTHESES,
    RATING_BAND_HYPOTHESES,
    determine_modal_band,
    evaluate_lockdown_hypotheses,
    evaluate_rating_band_hypotheses,
)
from imdb_analysis.data_access import ensure_datasets, load_ratings


LOGGER = logging.getLogger(__name__)


def render_table(df: pd.DataFrame) -> str:
    """Render a dataframe as GitHub-flavoured Markdown."""

    return df.to_markdown(index=False)


def build_report(sections: Dict[str, str]) -> str:
    """Combine named sections into a Markdown document."""

    lines = ["# IMDb Movie Trends Report", ""]
    for title, body in sections.items():
        lines.append(f"## {title}")
        lines.append("")
        lines.append(body)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _format_count(value: float | int) -> str:
    if pd.isna(value):
        return "—"
    return f"{int(round(value)):,}"


def _format_signed_count(value: float | int) -> str:
    if pd.isna(value):
        return "—"
    return f"{int(round(value)):+,}"


def _format_percent(value: float | int, decimals: int = 1) -> str:
    if pd.isna(value):
        return "—"
    return f"{value:.{decimals}f}%"


def _format_signed_percent(value: float | int, decimals: int = 1) -> str:
    if pd.isna(value):
        return "—"
    return f"{value:+.{decimals}f}pp"


def _format_votes(value: float | int) -> str:
    if pd.isna(value):
        return "—"
    return f"{value / 1_000_000:.1f}M"


def _format_signed_votes(value: float | int) -> str:
    if pd.isna(value):
        return "—"
    return f"{value / 1_000_000:+.1f}M"


def _format_year(value: float | int) -> str:
    if pd.isna(value):
        return "—"
    return f"{int(round(value))}"


def _format_signed_year(value: float | int) -> str:
    if pd.isna(value):
        return "—"
    return f"{int(round(value)):+d}"


def _format_float1(value: float | int) -> str:
    if pd.isna(value):
        return "—"
    return f"{value:.1f}"


def _format_signed_float1(value: float | int) -> str:
    if pd.isna(value):
        return "—"
    return f"{value:+.1f}"


def _format_ratio(value: float | int) -> str:
    if pd.isna(value):
        return "—"
    return f"{value:.2f}×"


def _format_signed_ratio(value: float | int) -> str:
    if pd.isna(value):
        return "—"
    return f"{value:+.2f}"


RatingFormatter = dict[str, tuple[Callable[[float | int], str], Callable[[float | int], str]]]


RATING_FORMATS: RatingFormatter = {
    "Titles": (_format_count, _format_signed_count),
    "Share (%)": (_format_percent, _format_signed_percent),
    "Votes": (_format_votes, _format_signed_votes),
}


RATING_TOLERANCES: dict[str, float] = {
    "Titles": 0.5,
    "Share (%)": 0.05,
    "Votes": 0.5,
}


LOCKDOWN_FORMATS: RatingFormatter = {
    "Peak year": (_format_year, _format_signed_year),
    "Lockdown releases": (_format_count, _format_signed_count),
    "Avg pre-lockdown": (_format_float1, _format_signed_float1),
    "Avg post-lockdown": (_format_float1, _format_signed_float1),
    "× vs pre-lockdown": (_format_ratio, _format_signed_ratio),
    "× vs post-lockdown": (_format_ratio, _format_signed_ratio),
}


LOCKDOWN_TOLERANCES: dict[str, float] = {
    "Peak year": 0.0,
    "Lockdown releases": 0.5,
    "Avg pre-lockdown": 0.05,
    "Avg post-lockdown": 0.05,
    "× vs pre-lockdown": 0.02,
    "× vs post-lockdown": 0.02,
}


def _format_metric_value(
    metric: str,
    value: float | int,
    formats: RatingFormatter,
    *,
    signed: bool = False,
) -> str:
    formatter, signed_formatter = formats.get(metric, (_format_float1, _format_signed_float1))
    return signed_formatter(value) if signed else formatter(value)


def _rating_verdict(metric: str, actual: float | int, diff: float | int) -> str:
    if pd.isna(actual):
        return "Not observed"
    tolerance = RATING_TOLERANCES.get(metric, 0.0)
    return "Matches" if abs(float(diff)) <= tolerance else "Differs"


def _lockdown_verdict(metric: str, actual: float | int, diff: float | int) -> str:
    if pd.isna(actual):
        return "Not observed"
    tolerance = LOCKDOWN_TOLERANCES.get(metric, 0.0)
    return "Matches" if abs(float(diff)) <= tolerance else "Differs"


def build_rating_hypothesis_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    rows = []
    for _, row in df.iterrows():
        metric = row["metric"]
        rows.append(
            {
                "Claim": row["claim"],
                "Metric": metric,
                "Expected": _format_metric_value(metric, row["expected"], RATING_FORMATS),
                "Actual": _format_metric_value(metric, row["actual"], RATING_FORMATS),
                "Difference": _format_metric_value(metric, row["difference"], RATING_FORMATS, signed=True),
                "Verdict": _rating_verdict(metric, row["actual"], row["difference"]),
            }
        )
    return pd.DataFrame(rows, columns=["Claim", "Metric", "Expected", "Actual", "Difference", "Verdict"])


def build_lockdown_hypothesis_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    rows = []
    for _, row in df.iterrows():
        metric = row["metric"]
        rows.append(
            {
                "Genre": row["genre"],
                "Metric": metric,
                "Expected": _format_metric_value(metric, row["expected"], LOCKDOWN_FORMATS),
                "Actual": _format_metric_value(metric, row["actual"], LOCKDOWN_FORMATS),
                "Difference": _format_metric_value(metric, row["difference"], LOCKDOWN_FORMATS, signed=True),
                "Verdict": _lockdown_verdict(metric, row["actual"], row["difference"]),
            }
        )
    return pd.DataFrame(rows, columns=["Genre", "Metric", "Expected", "Actual", "Difference", "Verdict"])


def summarise_rating_mismatches(df: pd.DataFrame) -> list[str]:
    summaries: list[str] = []
    if df.empty:
        return summaries

    for claim in df["claim"].unique():
        subset = df[df["claim"] == claim]
        parts = []
        for _, row in subset.iterrows():
            metric = row["metric"]
            diff = row["difference"]
            if pd.isna(row["actual"]):
                continue
            tolerance = RATING_TOLERANCES.get(metric, 0.0)
            if abs(float(diff)) <= tolerance:
                continue
            expected_str = _format_metric_value(metric, row["expected"], RATING_FORMATS)
            actual_str = _format_metric_value(metric, row["actual"], RATING_FORMATS)
            if metric == "Share (%)":
                label = f"share {actual_str} vs {expected_str}"
            elif metric == "Votes":
                label = f"votes {actual_str} vs {expected_str}"
            else:
                label = f"{actual_str} vs {expected_str}"
            parts.append(label)
        if parts:
            summaries.append(f"{claim}: " + "; ".join(parts))
    return summaries


def summarise_lockdown_mismatches(df: pd.DataFrame) -> list[str]:
    summaries: list[str] = []
    if df.empty:
        return summaries

    for genre in df["genre"].unique():
        subset = df[df["genre"] == genre]
        parts = []
        for _, row in subset.iterrows():
            metric = row["metric"]
            diff = row["difference"]
            if pd.isna(row["actual"]):
                continue
            tolerance = LOCKDOWN_TOLERANCES.get(metric, 0.0)
            if abs(float(diff)) <= tolerance:
                continue
            expected_str = _format_metric_value(metric, row["expected"], LOCKDOWN_FORMATS)
            actual_str = _format_metric_value(metric, row["actual"], LOCKDOWN_FORMATS)
            label = f"{metric.lower()} {actual_str} vs {expected_str}"
            parts.append(label)
        if parts:
            summaries.append(f"{genre}: " + "; ".join(parts))
    return summaries


def _autolabel(
    bars: Iterable[mpatches.Rectangle], *, fmt: str = "{:.0f}", rotation: int | None = None
) -> None:
    """Annotate bars with their value."""

    for bar in bars:
        height = bar.get_height()
        if np.isnan(height):
            continue
        ax = bar.axes
        ax.annotate(
            fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=rotation if rotation is not None else 0,
        )


def save_release_growth_chart(df: pd.DataFrame, output_path: Path) -> None:
    """Plot the five-year release trend and save it to ``output_path``."""

    if df.empty:
        LOGGER.warning("Release growth dataframe is empty; skipping chart")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    periods = df["period"].astype(str)
    values = df["annual_avg"].astype(float)
    bars = ax.bar(periods, values, color="#4C78A8")
    ax.set_title("Feature film output accelerated in the streaming era")
    ax.set_xlabel("Five-year period")
    ax.set_ylabel("Average annual releases")
    ax.set_ylim(0, values.max() * 1.15)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.tick_params(axis="x", rotation=45)
    plt.setp(ax.get_xticklabels(), ha="right")
    _autolabel(bars)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    LOGGER.info("Saved release growth chart to %s", output_path)


def save_genre_scores_chart(df: pd.DataFrame, output_path: Path) -> None:
    """Plot the top genres by audience rating."""

    if df.empty:
        LOGGER.warning("Genre scores dataframe is empty; skipping chart")
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    data = df.sort_values("average_rating")
    ratings = data["average_rating"].astype(float)
    genres = data["genre"].astype(str)
    bars = ax.barh(genres, ratings, color="#F58518")
    ax.set_title("Top genres by average IMDb rating (2010+, ≥25k votes)")
    ax.set_xlabel("Average rating")
    ax.set_xlim(ratings.min() - 0.2, ratings.max() + 0.3)
    ax.xaxis.grid(True, linestyle="--", alpha=0.3)
    _autolabel(bars, fmt="{:.2f}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    LOGGER.info("Saved genre scores chart to %s", output_path)


def save_rating_distribution_chart(df: pd.DataFrame, output_path: Path) -> None:
    """Plot how titles distribute across rating bands."""

    if df.empty:
        LOGGER.warning("Rating distribution dataframe is empty; skipping chart")
        return

    fig, ax = plt.subplots(figsize=(11, 6))
    bands = df["band"].astype(str)
    share = df["share"].astype(float)
    bars = ax.bar(bands, share, color="#54A24B")
    ax.set_title("IMDb ratings cluster around the 6–7 range")
    ax.set_xlabel("IMDb rating band")
    ax.set_ylabel("Share of rated features (%)")
    ax.set_ylim(0, share.max() * 1.2)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.tick_params(axis="x", rotation=45)
    plt.setp(ax.get_xticklabels(), ha="right")
    _autolabel(bars, fmt="{:.1f}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    LOGGER.info("Saved rating distribution chart to %s", output_path)


def save_lockdown_peaks_chart(
    df: pd.DataFrame,
    output_path: Path,
    pre_window_label: str,
    post_window_label: str,
) -> None:
    """Plot lockdown-era genre spikes compared with surrounding windows."""

    if df.empty:
        LOGGER.warning("Lockdown peaks dataframe is empty; skipping chart")
        return

    subset = df.copy()
    subset = subset.sort_values("lockdown_releases", ascending=False)
    genres = subset["genre"].astype(str).tolist()
    x = np.arange(len(genres))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    pre_values = subset["pre_lockdown_avg"].astype(float)
    lockdown_values = subset["lockdown_releases"].astype(float)
    post_values = subset["post_lockdown_avg"].astype(float)

    pre_bars = ax.bar(x - width, pre_values, width, label=f"Avg {pre_window_label}", color="#4C78A8")
    lockdown_bars = ax.bar(x, lockdown_values, width, label="Lockdown peak", color="#F58518")
    post_bar_values = post_values.fillna(0)
    post_bars = ax.bar(x + width, post_bar_values, width, label=f"Avg {post_window_label}", color="#72B7B2")

    for idx, bar in enumerate(lockdown_bars):
        year = int(subset.iloc[idx]["lockdown_year"])
        count = int(lockdown_values.iloc[idx])
        ax.annotate(
            f"{count}\n({year})",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    for idx, value in enumerate(post_values):
        if np.isnan(value):
            ax.annotate(
                "N/A",
                xy=(x[idx] + width, 1),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#555555",
            )

    _autolabel(pre_bars)
    mask = ~post_values.isna()
    _autolabel([bar for idx, bar in enumerate(post_bars) if mask.iloc[idx]])

    ax.set_title("Lockdown-era release spikes versus baseline output")
    ax.set_ylabel("Number of feature releases")
    ax.set_xticks(x)
    ax.set_xticklabels(genres, rotation=20, ha="right")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    LOGGER.info("Saved lockdown peaks chart to %s", output_path)


def save_runtime_rating_chart(df: pd.DataFrame, output_path: Path) -> None:
    """Plot the relationship between runtime bands and ratings."""

    if df.empty:
        LOGGER.warning("Runtime summary dataframe is empty; skipping chart")
        return

    fig, ax1 = plt.subplots(figsize=(9, 6))
    bands = df["runtime_band"].astype(str).tolist()
    x = np.arange(len(bands))

    ratings = df["avg_rating"].astype(float)
    counts = df["movie_count"].astype(int)

    ax1.plot(x, ratings, marker="o", color="#F58518", label="Average rating")
    ax1.set_xlabel("Runtime band")
    ax1.set_ylabel("Average IMDb rating", color="#F58518")
    ax1.tick_params(axis="y", labelcolor="#F58518")
    ax1.set_xticks(x)
    ax1.set_xticklabels(bands, rotation=20, ha="right")
    ax1.set_title("Runtime versus audience ratings (1990+, ≥5k votes)")
    ax1.set_ylim(ratings.min() - 0.3, ratings.max() + 0.3)
    ax1.set_axisbelow(True)
    ax1.yaxis.grid(True, linestyle="--", alpha=0.3)

    ax2 = ax1.twinx()
    bars = ax2.bar(x, counts, width=0.5, alpha=0.25, color="#4C78A8", label="Films in band")
    ax2.set_ylabel("Number of films", color="#4C78A8")
    ax2.tick_params(axis="y", labelcolor="#4C78A8")

    for bar in bars:
        height = bar.get_height()
        ax2.annotate(
            f"{int(height)}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    lines, labels = ax1.get_legend_handles_labels()
    bars_handles, bars_labels = ax2.get_legend_handles_labels()
    ax1.legend(lines + bars_handles, labels + bars_labels, loc="upper left")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    LOGGER.info("Saved runtime versus ratings chart to %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Where datasets should be stored")
    parser.add_argument("--report", type=Path, default=Path("reports/imdb_report.md"), help="Output report path")
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("reports/figures"),
        help="Directory where generated charts will be written",
    )
    parser.add_argument("--start-year", type=int, default=1980, help="Lower bound for startYear filtering")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    LOGGER.info("Ensuring datasets are available in %s", args.data_dir)
    dataset_paths = ensure_datasets(args.data_dir)

    LOGGER.info("Loading basics dataset")
    basics = load_movie_basics(dataset_paths["title.basics.tsv.gz"], start_year_min=args.start_year)

    LOGGER.info("Loading ratings dataset")
    ratings = load_ratings(dataset_paths["title.ratings.tsv.gz"])

    dataset = MovieDataset(basics=basics, ratings=ratings)
    movies_with_ratings = dataset.merged.dropna(subset=["averageRating", "numVotes"])

    analysis_end_year = datetime.now(timezone.utc).year - 1
    release_trend = compute_release_trend(dataset.basics, end_year=analysis_end_year)
    release_trend_recent = release_trend.tail(9).copy()
    genre_scores = compute_genre_scores(movies_with_ratings)
    runtime_corr, runtime_summary = runtime_rating_relationship(movies_with_ratings)
    rating_distribution = compute_rating_band_distribution(movies_with_ratings)
    modal_band = determine_modal_band(rating_distribution)
    min_rating_value = (
        float(movies_with_ratings["averageRating"].min()) if not movies_with_ratings.empty else float("nan")
    )
    rating_hypothesis_raw = evaluate_rating_band_hypotheses(rating_distribution, RATING_BAND_HYPOTHESES)
    rating_hypothesis_table = build_rating_hypothesis_table(rating_hypothesis_raw)
    rating_mismatch_lines = summarise_rating_mismatches(rating_hypothesis_raw)
    lockdown_years = (2020, 2021)
    context_years = 3
    min_lockdown_titles = 25
    min_ratio = 1.2
    lockdown_peaks = compute_lockdown_genre_peaks(
        dataset.basics,
        lockdown_years=lockdown_years,
        context_years=context_years,
        min_lockdown_titles=min_lockdown_titles,
        min_ratio=min_ratio,
    )
    lockdown_hypothesis_raw = evaluate_lockdown_hypotheses(lockdown_peaks, LOCKDOWN_PEAK_HYPOTHESES)
    lockdown_hypothesis_table = build_lockdown_hypothesis_table(lockdown_hypothesis_raw)
    lockdown_mismatch_lines = summarise_lockdown_mismatches(lockdown_hypothesis_raw)

    total_movies = len(dataset.basics)
    rated_movies = len(movies_with_ratings)
    rating_coverage = rated_movies / total_movies * 100
    median_runtime = movies_with_ratings["runtimeMinutes"].median()
    median_rating = movies_with_ratings["averageRating"].median()

    now = datetime.now(timezone.utc)

    sections: Dict[str, str] = {}

    sections["Data Snapshot"] = (
        "* Downloaded from [datasets.imdbws.com](https://datasets.imdbws.com/) on "
        f"{now:%Y-%m-%d %H:%M UTC}.\n"
        f"* Focused on mainstream feature films released from {args.start_year} onwards (counts capped at {analysis_end_year}).\n"
        f"* {total_movies:,} feature films in scope; {rating_coverage:.1f}% have audience rating data.\n"
        f"* Median runtime {median_runtime:.0f} minutes and median IMDb rating {median_rating:.1f}."
    )

    sections["Release Growth"] = (
        "Mainstream film production has accelerated sharply since the 2000s."
        " The five-year view below (1980–"
        f"{analysis_end_year}) shows streaming-era years (2015 onwards) averaging"
        " more than twice as many releases per year as the 1980s.\n\n"
        + render_table(release_trend_recent)
    )

    top_genres = genre_scores.head(10)
    highlight_genres = ", ".join(top_genres.head(3)["genre"].tolist())
    sections["Genres audiences love"] = (
        "Recent big-audience hits (2010 onwards, 25k+ votes) show clear genre patterns."
        f" {highlight_genres} top the list by average rating, and all top genres have"
        " more than 100 major releases in the last decade.\n\n"
        + render_table(
            top_genres.rename(
                columns={
                    "genre": "Genre",
                    "movie_count": "Films",
                    "average_rating": "Avg rating",
                    "median_rating": "Median rating",
                    "high_share": "% ≥ 7.5",
                    "median_votes": "Median votes",
                }
            )
        )
    )

    rating_by_band = rating_distribution.set_index("band")
    dominant_bands = rating_distribution.sort_values("title_count", ascending=False).head(2).to_dict("records")
    mid_fives = rating_by_band.loc["5.0-5.9"]
    low_end = rating_by_band.loc["1.0-1.9"]
    high_nines = rating_by_band.loc["9.0-9.9"]
    perfect_tens = rating_by_band.loc["10.0"]
    high_total = int(high_nines["title_count"] + perfect_tens["title_count"])
    high_share = float(high_nines["share"] + perfect_tens["share"])
    high_votes = int(high_nines["vote_total"] + perfect_tens["vote_total"])
    mid_fives_titles = int(mid_fives["title_count"])
    low_end_titles = int(low_end["title_count"])
    mid_fives_share = float(mid_fives["share"])
    low_end_share = float(low_end["share"])

    rating_sentence = ""
    if dominant_bands:
        lead = dominant_bands[0]
        runner = dominant_bands[1] if len(dominant_bands) > 1 else None
        rating_sentence = (
            f"{lead['band']} ratings lead the pack with {lead['title_count']:,} titles "
            f"({lead['share']:.1f}% of rated features)"
        )
        if runner:
            rating_sentence += (
                f", edging past {runner['band']} at {runner['title_count']:,} titles"
                f" ({runner['share']:.1f}%)."
            )
        else:
            rating_sentence += "."

    sections["IMDb rating distribution"] = (
        "Audience scores skew toward the middle of the 10-point scale, leaving both"
        " extremes sparsely populated. "
        + rating_sentence
        + f" Mid-5 scores account for {mid_fives_titles:,} titles"
        + f" ({mid_fives_share:.1f}%), while the lowest band (1.0-1.9) only captures"
        + f" {low_end_titles:,} titles ({low_end_share:.1f}%)."
        + f" High-end acclaim remains rare: {high_total:,} titles sit at 9.0 or above"
        + f" ({high_share:.1f}% of the set) yet those releases still attract"
        + f" {high_votes/1_000_000:.1f}M cumulative votes."
        + " No rated feature in the dataset falls below 1.0.\n\n"
        + render_table(
            rating_distribution.rename(
                columns={
                    "band": "Rating band",
                    "title_count": "Titles",
                    "share": "% of titles",
                    "vote_total": "Total votes",
                    "vote_share": "% of votes",
                    "avg_votes": "Avg votes",
                }
            )
        )
    )

    hypothesis_intro = (
        "We compared the published hypotheses with the 1980+ feature-film aggregates used throughout this report."
    )
    context_sentences: list[str] = []
    if modal_band is not None:
        context_sentences.append(f"The modal rating band in this sample is {modal_band}.")
    if not np.isnan(min_rating_value):
        context_sentences.append(f"The lowest recorded IMDb score is {min_rating_value:.1f}.")
    if context_sentences:
        hypothesis_intro += " " + " ".join(context_sentences)
    hypothesis_intro += " Tables below compare the stated expectations with observed values (Δ = actual − expected)."

    mismatch_sections: list[str] = []
    if rating_mismatch_lines:
        mismatch_sections.append("**Ratings:** " + "; ".join(rating_mismatch_lines))
    if lockdown_mismatch_lines:
        mismatch_sections.append("**Lockdown genres:** " + "; ".join(lockdown_mismatch_lines))
    if mismatch_sections:
        hypothesis_intro += "\n\nKey discrepancies:\n" + "\n".join(f"- {line}" for line in mismatch_sections)
    else:
        hypothesis_intro += " All reviewed claims align with the computed aggregates."

    table_blocks: list[str] = []
    if not rating_hypothesis_table.empty:
        table_blocks.append("**Rating distribution claims**\n\n" + render_table(rating_hypothesis_table))
    if not lockdown_hypothesis_table.empty:
        table_blocks.append("**Lockdown spike claims**\n\n" + render_table(lockdown_hypothesis_table))
    if table_blocks:
        hypothesis_intro += "\n\n" + "\n\n".join(table_blocks)

    sections["Hypothesis fact-checks"] = hypothesis_intro

    pre_window = f"{lockdown_years[0] - context_years}-{lockdown_years[0] - 1}"
    post_window = f"{lockdown_years[-1] + 1}-{lockdown_years[-1] + context_years}"
    lockdown_table_raw = lockdown_peaks.head(8)

    if not lockdown_table_raw.empty:
        lockdown_table = lockdown_table_raw.rename(
            columns={
                "genre": "Genre",
                "lockdown_year": "Peak year",
                "lockdown_releases": "Lockdown releases",
                "pre_lockdown_avg": f"Avg {pre_window}",
                "post_lockdown_avg": f"Avg {post_window}",
                "pre_ratio": "× vs pre-lockdown",
                "post_ratio": "× vs post-lockdown",
            }
        )
        ratio_pct = int(round((min_ratio - 1) * 100))
        highlights = lockdown_peaks.head(2).to_dict("records")
        highlight_sentence = ""
        if highlights:
            bullet_points = "; ".join(
                f"{row['genre']} ({row['lockdown_releases']} films in {row['lockdown_year']}, {row['pre_ratio']:.1f}× pre-lockdown)"
                for row in highlights
            )
            highlight_sentence = f" Leading spikes include {bullet_points}."

        sections["Lockdown-era genre spikes"] = (
            "A handful of genres hit their highest release counts during the 2020–2021 lockdown period."
            f" Each listed genre released at least {min_lockdown_titles} feature films in its peak lockdown year"
            f" and outpaced the {pre_window} average by {ratio_pct}% or more."
            + highlight_sentence
            + f" The table compares the lockdown surge with the {pre_window} baseline"
            f" and the {post_window} recovery window.\n\n"
            + render_table(lockdown_table)
        )

    sections["Runtime and ratings"] = (
        "Longer movies are not dramatically better rated, but the correlation is mildly"
        f" positive ({runtime_corr:.2f}). Movies over 150 minutes gain the highest"
        " average scores, while sub-90-minute releases trail.\n\n"
        + render_table(
            runtime_summary.rename(
                columns={
                    "runtime_band": "Runtime band",
                    "movie_count": "Films",
                    "avg_runtime": "Avg minutes",
                    "avg_rating": "Avg rating",
                }
            )
        )
    )

    figures_dir = args.figures_dir
    figures_dir.mkdir(parents=True, exist_ok=True)

    save_release_growth_chart(release_trend_recent, figures_dir / "release_growth.png")
    save_genre_scores_chart(top_genres, figures_dir / "top_genres.png")
    save_rating_distribution_chart(rating_distribution, figures_dir / "rating_distribution.png")
    if not lockdown_table_raw.empty:
        save_lockdown_peaks_chart(
            lockdown_table_raw,
            figures_dir / "lockdown_spikes.png",
            pre_window,
            post_window,
        )
    save_runtime_rating_chart(runtime_summary, figures_dir / "runtime_vs_rating.png")

    report_text = build_report(sections)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report_text, encoding="utf-8")
    LOGGER.info("Report written to %s", args.report)


if __name__ == "__main__":
    main()
