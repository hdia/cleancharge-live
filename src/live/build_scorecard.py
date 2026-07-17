"""
python -m src.live.build_scorecard

Build the CleanCharge Live rolling operational scorecard.

This module aggregates all completed daily evaluation summaries and creates:

1. rolling_scorecard.json
   Scientific and decision-performance statistics.

2. forecast_history.json
   A compact chronological record of evaluated forecasts.

3. system_status.json
   Current operational health, data freshness and publication status.

Run from the repository root:

    python -m src.live.build_scorecard
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Constants and paths
# ---------------------------------------------------------------------

TZ_LOCAL = "Australia/Melbourne"

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_VALIDATION_DIR = (
    ROOT
    / "data"
    / "live"
    / "validation"
)

DEFAULT_FORECAST_ARCHIVE_DIR = (
    ROOT
    / "data"
    / "live"
    / "forecasts"
    / "archive"
)

DEFAULT_UPDATE_STATUS_FILE = (
    ROOT
    / "data"
    / "live"
    / "update_status.json"
)

DEFAULT_FORECAST_STATUS_FILE = (
    ROOT
    / "data"
    / "live"
    / "status"
    / "forecast_status.json"
)

DEFAULT_SCORECARD_DIR = (
    ROOT
    / "data"
    / "live"
    / "scorecard"
)

DEFAULT_ROLLING_SCORECARD_FILE = (
    DEFAULT_SCORECARD_DIR
    / "rolling_scorecard.json"
)

DEFAULT_FORECAST_HISTORY_FILE = (
    DEFAULT_SCORECARD_DIR
    / "forecast_history.json"
)

DEFAULT_SYSTEM_STATUS_FILE = (
    DEFAULT_SCORECARD_DIR
    / "system_status.json"
)


# ---------------------------------------------------------------------
# JSON utilities
# ---------------------------------------------------------------------

def load_json(
    path: Path,
) -> dict[str, Any]:
    """
    Load a JSON file.

    Returns an empty dictionary if the file does not exist or cannot
    be decoded.
    """
    if not path.exists():
        return {}

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    except (
        json.JSONDecodeError,
        OSError,
    ):
        return {}


def save_json(
    payload: Any,
    path: Path,
) -> None:
    """
    Save JSON using readable indentation.
    """
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
        )


# ---------------------------------------------------------------------
# Safe statistics
# ---------------------------------------------------------------------

def numeric_values(
    records: list[dict[str, Any]],
    field: str,
) -> list[float]:
    """
    Extract finite numeric values from a list of dictionaries.
    """
    values: list[float] = []

    for record in records:
        value = record.get(field)

        if value is None:
            continue

        try:
            numeric = float(value)
        except (
            TypeError,
            ValueError,
        ):
            continue

        if np.isfinite(numeric):
            values.append(numeric)

    return values


def summarise_numeric(
    values: list[float],
) -> dict[str, float | int | None]:
    """
    Return standard descriptive statistics.
    """
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "minimum": None,
            "maximum": None,
        }

    array = np.asarray(
        values,
        dtype=float,
    )

    return {
        "count": int(len(array)),
        "mean": round(
            float(np.mean(array)),
            3,
        ),
        "median": round(
            float(np.median(array)),
            3,
        ),
        "minimum": round(
            float(np.min(array)),
            3,
        ),
        "maximum": round(
            float(np.max(array)),
            3,
        ),
    }


def percentage_true(
    records: list[dict[str, Any]],
    field: str,
) -> float | None:
    """
    Calculate the percentage of non-null Boolean values that are true.
    """
    values = [
        record.get(field)
        for record in records
        if isinstance(
            record.get(field),
            bool,
        )
    ]

    if not values:
        return None

    return round(
        100.0
        * sum(values)
        / len(values),
        2,
    )


# ---------------------------------------------------------------------
# Evaluation discovery
# ---------------------------------------------------------------------

def find_evaluation_summaries(
    validation_dir: Path,
) -> list[Path]:
    """
    Find all compact evaluation_summary.json files.
    """
    if not validation_dir.exists():
        return []

    return sorted(
        validation_dir.glob(
            "*/evaluation_summary.json"
        )
    )


def load_evaluation_records(
    validation_dir: Path,
) -> list[dict[str, Any]]:
    """
    Load valid evaluation summaries and sort them by target date.
    """
    records: list[dict[str, Any]] = []

    for path in find_evaluation_summaries(
        validation_dir
    ):
        payload = load_json(path)

        if not payload:
            continue

        target_date = payload.get(
            "target_date"
        )

        forecast_id = payload.get(
            "forecast_id"
        )

        if not target_date or not forecast_id:
            continue

        record = payload.copy()
        record["source_file"] = str(path)

        records.append(record)

    records.sort(
        key=lambda item: item.get(
            "target_date",
            "",
        )
    )

    return records


# ---------------------------------------------------------------------
# Forecast archive statistics
# ---------------------------------------------------------------------

def archived_forecast_dates(
    archive_dir: Path,
) -> list[str]:
    """
    Return forecast archive dates from YYYY-MM-DD.csv filenames.
    """
    if not archive_dir.exists():
        return []

    dates: list[str] = []

    for path in archive_dir.glob(
        "*.csv"
    ):
        try:
            date = pd.Timestamp(
                path.stem
            ).strftime(
                "%Y-%m-%d"
            )

            dates.append(date)

        except Exception:
            continue

    return sorted(
        set(dates)
    )


def consecutive_date_streaks(
    dates: list[str],
) -> dict[str, int]:
    """
    Calculate current and longest uninterrupted daily streaks.
    """
    if not dates:
        return {
            "current_streak_days": 0,
            "longest_streak_days": 0,
        }

    parsed = sorted(
        {
            pd.Timestamp(date).normalize()
            for date in dates
        }
    )

    longest = 1
    running = 1

    for previous, current in zip(
        parsed,
        parsed[1:],
    ):
        if (
            current - previous
            == pd.Timedelta(days=1)
        ):
            running += 1
            longest = max(
                longest,
                running,
            )
        else:
            running = 1

    latest = parsed[-1]
    current_streak = 1

    for index in range(
        len(parsed) - 1,
        0,
        -1,
    ):
        if (
            parsed[index]
            - parsed[index - 1]
            == pd.Timedelta(days=1)
        ):
            current_streak += 1
        else:
            break

    return {
        "current_streak_days": int(
            current_streak
        ),
        "longest_streak_days": int(
            longest
        ),
    }


# ---------------------------------------------------------------------
# Rolling scorecard
# ---------------------------------------------------------------------

def build_rolling_scorecard(
    records: list[dict[str, Any]],
    archived_dates: list[str],
) -> dict[str, Any]:
    """
    Build cumulative scientific and decision-performance statistics.
    """
    generated_at = pd.Timestamp.now(
        tz=TZ_LOCAL
    )

    evaluated_dates = [
        record["target_date"]
        for record in records
    ]

    grade_counts = Counter(
        str(
            record.get(
                "grade",
                "Not available",
            )
        )
        for record in records
    )

    star_values = numeric_values(
        records,
        "stars",
    )

    scientific = {
        "mae_g_per_kwh": summarise_numeric(
            numeric_values(
                records,
                "scientific_mae_g_per_kwh",
            )
        ),
        "rmse_g_per_kwh": summarise_numeric(
            numeric_values(
                records,
                "scientific_rmse_g_per_kwh",
            )
        ),
        "smape_percent": summarise_numeric(
            numeric_values(
                records,
                "scientific_smape_percent",
            )
        ),
        "r2": summarise_numeric(
            numeric_values(
                records,
                "scientific_r2",
            )
        ),
        "bias_g_per_kwh": summarise_numeric(
            numeric_values(
                records,
                "scientific_bias_g_per_kwh",
            )
        ),
        "data_completeness_percent": summarise_numeric(
            numeric_values(
                records,
                "data_completeness_percent",
            )
        ),
    }

    decision = {
        "carbon_savings_capture_percent": summarise_numeric(
            numeric_values(
                records,
                "carbon_savings_capture_percent",
            )
        ),
        "timing_error_hours": summarise_numeric(
            numeric_values(
                records,
                "timing_error_hours",
            )
        ),
        "window_overlap_percent": summarise_numeric(
            numeric_values(
                records,
                "window_overlap_percent",
            )
        ),
        "actual_emissions_following_recommendation_kg": summarise_numeric(
            numeric_values(
                records,
                "actual_emissions_following_recommendation_kg",
            )
        ),
        "perfect_hindsight_emissions_kg": summarise_numeric(
            numeric_values(
                records,
                "perfect_hindsight_emissions_kg",
            )
        ),
        "exact_window_match_percent": percentage_true(
            records,
            "exact_window_match",
        ),
        "within_one_hour_percent": percentage_true(
            records,
            "within_one_hour",
        ),
        "average_stars": (
            round(
                float(
                    np.mean(star_values)
                ),
                3,
            )
            if star_values
            else None
        ),
        "grade_distribution": dict(
            sorted(
                grade_counts.items()
            )
        ),
    }

    operational = {
        "forecasts_published": int(
            len(archived_dates)
        ),
        "forecasts_evaluated": int(
            len(records)
        ),
        "forecasts_pending_evaluation": int(
            max(
                0,
                len(archived_dates)
                - len(records),
            )
        ),
        "first_published_forecast_date": (
            archived_dates[0]
            if archived_dates
            else None
        ),
        "latest_published_forecast_date": (
            archived_dates[-1]
            if archived_dates
            else None
        ),
        "first_evaluated_forecast_date": (
            evaluated_dates[0]
            if evaluated_dates
            else None
        ),
        "latest_evaluated_forecast_date": (
            evaluated_dates[-1]
            if evaluated_dates
            else None
        ),
    }

    operational.update(
        consecutive_date_streaks(
            archived_dates
        )
    )

    return {
        "status": "success",
        "generated_at_local": (
            generated_at.isoformat()
        ),
        "location": (
            "Melbourne, Victoria, Australia"
        ),
        "network": "NEM",
        "network_region": "VIC1",
        "operational_summary": operational,
        "scientific_performance": scientific,
        "decision_performance": decision,
        "interpretation_note": (
            "Scientific accuracy and decision performance are "
            "reported separately. A forecast may have imperfect "
            "hourly accuracy while still identifying a useful "
            "low-carbon charging window."
        ),
    }


# ---------------------------------------------------------------------
# Forecast history
# ---------------------------------------------------------------------

def build_forecast_history(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Create a compact chronological history for dashboard charts.
    """
    generated_at = pd.Timestamp.now(
        tz=TZ_LOCAL
    )

    history: list[dict[str, Any]] = []

    for record in records:
        history.append(
            {
                "forecast_id": record.get(
                    "forecast_id"
                ),
                "target_date": record.get(
                    "target_date"
                ),
                "grade": record.get(
                    "grade"
                ),
                "stars": record.get(
                    "stars"
                ),
                "decision_quality_label": record.get(
                    "decision_quality_label"
                ),
                "carbon_savings_capture_percent": record.get(
                    "carbon_savings_capture_percent"
                ),
                "timing_error_hours": record.get(
                    "timing_error_hours"
                ),
                "window_overlap_percent": record.get(
                    "window_overlap_percent"
                ),
                "exact_window_match": record.get(
                    "exact_window_match"
                ),
                "within_one_hour": record.get(
                    "within_one_hour"
                ),
                "scientific_mae_g_per_kwh": record.get(
                    "scientific_mae_g_per_kwh"
                ),
                "scientific_rmse_g_per_kwh": record.get(
                    "scientific_rmse_g_per_kwh"
                ),
                "scientific_smape_percent": record.get(
                    "scientific_smape_percent"
                ),
                "scientific_r2": record.get(
                    "scientific_r2"
                ),
                "scientific_bias_g_per_kwh": record.get(
                    "scientific_bias_g_per_kwh"
                ),
                "data_completeness_percent": record.get(
                    "data_completeness_percent"
                ),
                "actual_emissions_following_recommendation_kg": record.get(
                    "actual_emissions_following_recommendation_kg"
                ),
                "perfect_hindsight_emissions_kg": record.get(
                    "perfect_hindsight_emissions_kg"
                ),
            }
        )

    return {
        "status": "success",
        "generated_at_local": (
            generated_at.isoformat()
        ),
        "forecast_count": int(
            len(history)
        ),
        "forecasts": history,
    }


# ---------------------------------------------------------------------
# System status
# ---------------------------------------------------------------------

def parse_timestamp(
    value: Any,
) -> pd.Timestamp | None:
    """
    Parse a timestamp and convert it to Melbourne time.
    """
    if not value:
        return None

    try:
        timestamp = pd.Timestamp(
            value
        )

        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize(
                TZ_LOCAL
            )
        else:
            timestamp = timestamp.tz_convert(
                TZ_LOCAL
            )

        return timestamp

    except Exception:
        return None


def build_system_status(
    update_status: dict[str, Any],
    forecast_status: dict[str, Any],
    archived_dates: list[str],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Build current operational-health metadata.
    """
    now_local = pd.Timestamp.now(
        tz=TZ_LOCAL
    )

    history_status = update_status.get(
        "history",
        {}
    )

    latest_observation = parse_timestamp(
        history_status.get(
            "last_timestamp_local"
        )
    )

    data_age_hours = None

    if latest_observation is not None:
        data_age_hours = round(
            float(
                (
                    now_local
                    - latest_observation
                ).total_seconds()
                / 3600.0
            ),
            2,
        )

    update_success = (
        update_status.get(
            "status"
        )
        == "success"
    )

    forecast_success = (
        forecast_status.get(
            "status"
        )
        == "success"
    )

    data_complete = (
        history_status.get(
            "coverage_percent"
        )
        == 100.0
    )

    if (
        update_success
        and forecast_success
        and data_complete
        and (
            data_age_hours is None
            or data_age_hours <= 6
        )
    ):
        health = "healthy"
        health_label = "Operational"

    elif (
        update_success
        and forecast_success
    ):
        health = "degraded"
        health_label = "Operational with warnings"

    else:
        health = "unavailable"
        health_label = "Operational attention required"

    latest_published = (
        archived_dates[-1]
        if archived_dates
        else None
    )

    latest_evaluated = (
        records[-1].get(
            "target_date"
        )
        if records
        else None
    )

    return {
        "status": "success",
        "generated_at_local": (
            now_local.isoformat()
        ),
        "system_health": health,
        "system_health_label": health_label,
        "data_update_status": (
            update_status.get(
                "status"
            )
        ),
        "forecast_publication_status": (
            forecast_status.get(
                "status"
            )
        ),
        "latest_observation_local": (
            latest_observation.isoformat()
            if latest_observation is not None
            else None
        ),
        "data_age_hours": data_age_hours,
        "history_coverage_percent": (
            history_status.get(
                "coverage_percent"
            )
        ),
        "missing_hourly_observations": (
            history_status.get(
                "missing_hourly_rows"
            )
        ),
        "latest_published_forecast_date": (
            latest_published
        ),
        "latest_evaluated_forecast_date": (
            latest_evaluated
        ),
        "forecast_id": (
            forecast_status.get(
                "forecast_id"
            )
        ),
        "forecast_target_date": (
            forecast_status.get(
                "target_date"
            )
        ),
        "forecast_published_at_local": (
            forecast_status.get(
                "published_at_local"
            )
        ),
        "notes": (
            "System health reports operational status and data "
            "freshness. It is separate from forecast accuracy and "
            "decision quality."
        ),
    }


# ---------------------------------------------------------------------
# Public orchestration
# ---------------------------------------------------------------------

def build_scorecard(
    validation_dir: Path = DEFAULT_VALIDATION_DIR,
    forecast_archive_dir: Path = DEFAULT_FORECAST_ARCHIVE_DIR,
    update_status_file: Path = DEFAULT_UPDATE_STATUS_FILE,
    forecast_status_file: Path = DEFAULT_FORECAST_STATUS_FILE,
    rolling_scorecard_file: Path = DEFAULT_ROLLING_SCORECARD_FILE,
    forecast_history_file: Path = DEFAULT_FORECAST_HISTORY_FILE,
    system_status_file: Path = DEFAULT_SYSTEM_STATUS_FILE,
) -> None:
    """
    Build all CleanCharge Live scorecard outputs.
    """
    print(
        "\n=============================================="
    )
    print(
        " CleanCharge Live scorecard builder"
    )
    print(
        "==============================================\n"
    )

    records = load_evaluation_records(
        validation_dir
    )

    archived_dates = archived_forecast_dates(
        forecast_archive_dir
    )

    update_status = load_json(
        update_status_file
    )

    forecast_status = load_json(
        forecast_status_file
    )

    print(
        f">> Published forecast archives: "
        f"{len(archived_dates)}"
    )

    print(
        f">> Completed evaluation summaries: "
        f"{len(records)}"
    )

    rolling_scorecard = build_rolling_scorecard(
        records=records,
        archived_dates=archived_dates,
    )

    forecast_history = build_forecast_history(
        records=records,
    )

    system_status = build_system_status(
        update_status=update_status,
        forecast_status=forecast_status,
        archived_dates=archived_dates,
        records=records,
    )

    save_json(
        rolling_scorecard,
        rolling_scorecard_file,
    )

    save_json(
        forecast_history,
        forecast_history_file,
    )

    save_json(
        system_status,
        system_status_file,
    )

    operational = rolling_scorecard[
        "operational_summary"
    ]

    scientific = rolling_scorecard[
        "scientific_performance"
    ]

    decision = rolling_scorecard[
        "decision_performance"
    ]

    print(
        "\n=============================================="
    )
    print(
        " Scorecard build completed"
    )
    print(
        "=============================================="
    )

    print(
        f"\nForecasts published:\n  "
        f"{operational['forecasts_published']}"
    )

    print(
        f"\nForecasts evaluated:\n  "
        f"{operational['forecasts_evaluated']}"
    )

    mae_mean = scientific[
        "mae_g_per_kwh"
    ][
        "mean"
    ]

    if mae_mean is not None:
        print(
            f"\nAverage MAE:\n  "
            f"{mae_mean:.2f} gCO2/kWh"
        )

    csc_mean = decision[
        "carbon_savings_capture_percent"
    ][
        "mean"
    ]

    if csc_mean is not None:
        print(
            "\nAverage Carbon Savings Capture:"
        )

        print(
            f"  {csc_mean:.1f}%"
        )

    timing_mean = decision[
        "timing_error_hours"
    ][
        "mean"
    ]

    if timing_mean is not None:
        print(
            "\nAverage start-time error:"
        )

        print(
            f"  {timing_mean:.2f} hours"
        )

    print(
        "\nSystem health:"
    )

    print(
        "  "
        f"{system_status['system_health']} | "
        f"{system_status['system_health_label']}"
    )

    print(
        f"\nRolling scorecard:\n  "
        f"{rolling_scorecard_file}"
    )

    print(
        f"\nForecast history:\n  "
        f"{forecast_history_file}"
    )

    print(
        f"\nSystem status:\n  "
        f"{system_status_file}"
    )

    print()


# ---------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate CleanCharge Live forecast evaluations and "
            "build operational scorecard outputs."
        )
    )

    parser.add_argument(
        "--validation-dir",
        type=Path,
        default=DEFAULT_VALIDATION_DIR,
        help="Directory containing daily evaluation summaries.",
    )

    parser.add_argument(
        "--forecast-archive-dir",
        type=Path,
        default=DEFAULT_FORECAST_ARCHIVE_DIR,
        help="Directory containing official daily forecast archives.",
    )

    parser.add_argument(
        "--update-status-file",
        type=Path,
        default=DEFAULT_UPDATE_STATUS_FILE,
        help="History update-status JSON file.",
    )

    parser.add_argument(
        "--forecast-status-file",
        type=Path,
        default=DEFAULT_FORECAST_STATUS_FILE,
        help="Forecast publication-status JSON file.",
    )

    parser.add_argument(
        "--rolling-scorecard-file",
        type=Path,
        default=DEFAULT_ROLLING_SCORECARD_FILE,
        help="Rolling scorecard output JSON file.",
    )

    parser.add_argument(
        "--forecast-history-file",
        type=Path,
        default=DEFAULT_FORECAST_HISTORY_FILE,
        help="Forecast history output JSON file.",
    )

    parser.add_argument(
        "--system-status-file",
        type=Path,
        default=DEFAULT_SYSTEM_STATUS_FILE,
        help="System status output JSON file.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    build_scorecard(
        validation_dir=args.validation_dir,
        forecast_archive_dir=(
            args.forecast_archive_dir
        ),
        update_status_file=(
            args.update_status_file
        ),
        forecast_status_file=(
            args.forecast_status_file
        ),
        rolling_scorecard_file=(
            args.rolling_scorecard_file
        ),
        forecast_history_file=(
            args.forecast_history_file
        ),
        system_status_file=(
            args.system_status_file
        ),
    )


if __name__ == "__main__":
    main()