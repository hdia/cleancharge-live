"""
python -m src.live.evaluate_decision --target-date 2026-07-11

Decision evaluation for an official CleanCharge Live daily forecast.

This module evaluates the practical charging recommendation rather than
only the accuracy of individual hourly forecasts.

It compares:

1. The published forecast-recommended charging window.
2. The actual cleanest contiguous window, known with hindsight.
3. The actual highest-intensity contiguous window.
4. Charging immediately from the start of the day.

It calculates:

- charging-window timing error
- window overlap
- actual emissions from following the recommendation
- maximum possible emissions saving
- realised emissions saving
- Carbon Savings Capture
- decision-quality classification

Example
-------

Run from the repository root:

    python -m src.live.evaluate_decision --target-date 2026-07-11
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Constants and paths
# ---------------------------------------------------------------------

TZ_LOCAL = "Australia/Melbourne"
TIME_COL = "local_time"

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_HISTORY_FILE = (
    ROOT
    / "data"
    / "live"
    / "vic_intensity_history.csv"
)

DEFAULT_FORECAST_ARCHIVE_DIR = (
    ROOT
    / "data"
    / "live"
    / "forecasts"
    / "archive"
)

DEFAULT_VALIDATION_DIR = (
    ROOT
    / "data"
    / "live"
    / "validation"
)

DEFAULT_RECOMMENDATION_FILE = (
    ROOT
    / "data"
    / "live"
    / "status"
    / "today_recommendation.json"
)


# ---------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------

@dataclass
class DecisionEvaluation:
    hourly_actual: pd.DataFrame
    metrics: dict[str, Any]


# ---------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------

def save_json(
    payload: dict[str, Any],
    path: Path,
) -> None:
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

def load_json(
    path: Path,
) -> dict[str, Any]:
    """
    Load a JSON file and return an empty dictionary if it does not exist.
    """
    if not path.exists():
        return {}

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)        

def resolve_target_date(
    target_date_text: str,
) -> pd.Timestamp:
    """
    Return the requested Melbourne calendar date at local midnight.
    """
    target = pd.Timestamp(
        target_date_text
    )

    if target.tzinfo is None:
        target = target.tz_localize(
            TZ_LOCAL
        )
    else:
        target = target.tz_convert(
            TZ_LOCAL
        )

    return target.normalize()


def parse_local_time(
    series: pd.Series,
) -> pd.Series:
    """
    Parse timestamps and convert them to Melbourne local time.
    """
    parsed = pd.to_datetime(
        series,
        errors="coerce",
    )

    if parsed.dt.tz is None:
        parsed = parsed.dt.tz_localize(
            TZ_LOCAL,
            ambiguous="infer",
            nonexistent="shift_forward",
        )
    else:
        parsed = parsed.dt.tz_convert(
            TZ_LOCAL
        )

    return parsed


def circular_hour_difference(
    hour_a: float,
    hour_b: float,
) -> float:
    """
    Return the shortest clock-time difference between two local hours.

    Example:
        Difference between 23:00 and 01:00 is 2 hours, not 22.
    """
    raw_difference = abs(
        hour_a - hour_b
    )

    return float(
        min(
            raw_difference,
            24.0 - raw_difference,
        )
    )


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def load_archived_forecast(
    forecast_file: Path,
    target_start: pd.Timestamp,
) -> pd.DataFrame:
    """
    Load and validate one official archived daily forecast.
    """
    if not forecast_file.exists():
        raise FileNotFoundError(
            f"Archived forecast not found: {forecast_file}"
        )

    forecast = pd.read_csv(
        forecast_file
    )

    required = {
        TIME_COL,
        "intensity_hat",
    }

    missing = required.difference(
        forecast.columns
    )

    if missing:
        raise ValueError(
            "Forecast archive is missing required columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    forecast[TIME_COL] = parse_local_time(
        forecast[TIME_COL]
    )

    forecast["intensity_hat"] = pd.to_numeric(
        forecast["intensity_hat"],
        errors="coerce",
    )

    target_end = (
        target_start
        + pd.Timedelta(days=1)
    )

    forecast = forecast[
        (forecast[TIME_COL] >= target_start)
        & (forecast[TIME_COL] < target_end)
    ].copy()

    forecast = (
        forecast.dropna(
            subset=[
                TIME_COL,
                "intensity_hat",
            ]
        )
        .sort_values(TIME_COL)
        .drop_duplicates(
            subset=[TIME_COL],
            keep="last",
        )
        .reset_index(drop=True)
    )

    if len(forecast) != 24:
        raise ValueError(
            "Decision evaluation requires 24 archived forecast rows. "
            f"Found {len(forecast)}."
        )

    return forecast


def load_actual_day(
    history_file: Path,
    target_start: pd.Timestamp,
) -> pd.DataFrame:
    """
    Load the 24 actual Victoria carbon-intensity observations.
    """
    if not history_file.exists():
        raise FileNotFoundError(
            f"History file not found: {history_file}"
        )

    actual = pd.read_csv(
        history_file
    )

    if "intensity" not in actual.columns:
        raise ValueError(
            "History file has no 'intensity' column."
        )

    if "ts" in actual.columns:
        actual[TIME_COL] = (
            pd.to_datetime(
                actual["ts"],
                utc=True,
                errors="coerce",
            )
            .dt.tz_convert(TZ_LOCAL)
        )
    elif TIME_COL in actual.columns:
        actual[TIME_COL] = parse_local_time(
            actual[TIME_COL]
        )
    else:
        raise ValueError(
            "History file requires either 'ts' or "
            f"'{TIME_COL}'."
        )

    actual["intensity"] = pd.to_numeric(
        actual["intensity"],
        errors="coerce",
    )

    target_end = (
        target_start
        + pd.Timedelta(days=1)
    )

    actual = actual[
        (actual[TIME_COL] >= target_start)
        & (actual[TIME_COL] < target_end)
    ].copy()

    actual = (
        actual.dropna(
            subset=[
                TIME_COL,
                "intensity",
            ]
        )
        .sort_values(TIME_COL)
        .drop_duplicates(
            subset=[TIME_COL],
            keep="last",
        )
        .reset_index(drop=True)
    )

    if len(actual) != 24:
        raise ValueError(
            "Decision evaluation requires 24 actual hourly rows. "
            f"Found {len(actual)}."
        )

    return actual


# ---------------------------------------------------------------------
# Charging-window calculations
# ---------------------------------------------------------------------

def identify_contiguous_window(
    dataframe: pd.DataFrame,
    value_col: str,
    window_hours: int,
    lowest: bool,
) -> tuple[pd.DataFrame, float]:
    """
    Identify the lowest- or highest-intensity contiguous window.
    """
    if window_hours < 1:
        raise ValueError(
            "window_hours must be at least one."
        )

    if len(dataframe) < window_hours:
        raise ValueError(
            "The daily series is shorter than the charging window."
        )

    working = dataframe[
        [
            TIME_COL,
            value_col,
        ]
    ].copy()

    working = working.sort_values(
        TIME_COL
    ).reset_index(drop=True)

    working["window_mean"] = (
        working[value_col]
        .rolling(
            window=window_hours,
            min_periods=window_hours,
        )
        .mean()
    )

    valid = working.dropna(
        subset=["window_mean"]
    )

    if valid.empty:
        raise ValueError(
            "No valid contiguous charging window was found."
        )

    end_index = (
        valid["window_mean"].idxmin()
        if lowest
        else valid["window_mean"].idxmax()
    )

    start_index = (
        int(end_index)
        - window_hours
        + 1
    )

    block = working.iloc[
        start_index:int(end_index) + 1
    ].copy()

    mean_intensity = float(
        block[value_col].mean()
    )

    return block, mean_intensity


def extract_actual_window(
    actual: pd.DataFrame,
    window_start: pd.Timestamp,
    window_hours: int,
) -> pd.DataFrame:
    """
    Extract actual observations corresponding to a published window.
    """
    window_end = (
        window_start
        + pd.Timedelta(
            hours=window_hours
        )
    )

    block = actual[
        (actual[TIME_COL] >= window_start)
        & (actual[TIME_COL] < window_end)
    ].copy()

    block = block.sort_values(
        TIME_COL
    ).reset_index(drop=True)

    if len(block) != window_hours:
        raise ValueError(
            "The forecast-recommended window could not be matched "
            f"to {window_hours} actual hourly observations."
        )

    return block


def window_overlap_hours(
    first_start: pd.Timestamp,
    first_end: pd.Timestamp,
    second_start: pd.Timestamp,
    second_end: pd.Timestamp,
) -> float:
    """
    Calculate overlap between two time intervals in hours.
    """
    overlap_start = max(
        first_start,
        second_start,
    )

    overlap_end = min(
        first_end,
        second_end,
    )

    if overlap_end <= overlap_start:
        return 0.0

    return float(
        (
            overlap_end
            - overlap_start
        ).total_seconds()
        / 3600.0
    )


# ---------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------

def classify_decision_quality(
    carbon_savings_capture_percent: float | None,
    timing_error_hours: float,
    overlap_percent: float,
) -> dict[str, Any]:
    """
    Assign a transparent descriptive rating.

    The rating is primarily based on Carbon Savings Capture, with
    timing and overlap used to distinguish borderline cases.
    """
    if carbon_savings_capture_percent is None:
        return {
            "grade": "Not available",
            "stars": None,
            "label": "Insufficient saving opportunity",
            "explanation": (
                "The day did not provide enough difference between "
                "clean and high-intensity windows to calculate a "
                "meaningful Carbon Savings Capture value."
            ),
        }

    csc = carbon_savings_capture_percent

    if (
        csc >= 95.0
        and timing_error_hours <= 1.0
    ):
        grade = "A+"
        stars = 5
        label = "Excellent"

    elif (
        csc >= 90.0
        and timing_error_hours <= 2.0
    ):
        grade = "A"
        stars = 5
        label = "Excellent"

    elif csc >= 80.0:
        grade = "B"
        stars = 4
        label = "Very good"

    elif csc >= 65.0:
        grade = "C"
        stars = 3
        label = "Useful"

    elif csc >= 40.0:
        grade = "D"
        stars = 2
        label = "Limited"

    else:
        grade = "E"
        stars = 1
        label = "Poor"

    explanation = (
        f"The published recommendation captured {csc:.1f}% "
        f"of the maximum possible carbon saving, with a "
        f"{timing_error_hours:.1f}-hour start-time error and "
        f"{overlap_percent:.1f}% overlap with the actual cleanest "
        "window."
    )

    return {
        "grade": grade,
        "stars": stars,
        "label": label,
        "explanation": explanation,
    }


# ---------------------------------------------------------------------
# Main evaluation logic
# ---------------------------------------------------------------------

def calculate_decision_metrics(
    forecast: pd.DataFrame,
    actual: pd.DataFrame,
    target_start: pd.Timestamp,
    need_kwh: float,
    charger_kw: float,
) -> dict[str, Any]:
    """
    Evaluate the charging decision produced by the official forecast.
    """
    if need_kwh <= 0:
        raise ValueError(
            "need_kwh must be greater than zero."
        )

    if charger_kw <= 0:
        raise ValueError(
            "charger_kw must be greater than zero."
        )

    exact_duration_hours = (
        need_kwh / charger_kw
    )

    window_hours = max(
        1,
        int(
            np.ceil(
                exact_duration_hours
            )
        ),
    )

    # Forecast-recommended low-intensity window.
    forecast_best_block, forecast_best_mean = (
        identify_contiguous_window(
            dataframe=forecast,
            value_col="intensity_hat",
            window_hours=window_hours,
            lowest=True,
        )
    )

    forecast_start = (
        forecast_best_block[TIME_COL]
        .iloc[0]
    )

    forecast_end = (
        forecast_best_block[TIME_COL]
        .iloc[-1]
        + pd.Timedelta(hours=1)
    )

    # Actual outcomes under perfect hindsight.
    actual_best_block, actual_best_mean = (
        identify_contiguous_window(
            dataframe=actual,
            value_col="intensity",
            window_hours=window_hours,
            lowest=True,
        )
    )

    actual_worst_block, actual_worst_mean = (
        identify_contiguous_window(
            dataframe=actual,
            value_col="intensity",
            window_hours=window_hours,
            lowest=False,
        )
    )

    actual_best_start = (
        actual_best_block[TIME_COL]
        .iloc[0]
    )

    actual_best_end = (
        actual_best_block[TIME_COL]
        .iloc[-1]
        + pd.Timedelta(hours=1)
    )

    actual_worst_start = (
        actual_worst_block[TIME_COL]
        .iloc[0]
    )

    actual_worst_end = (
        actual_worst_block[TIME_COL]
        .iloc[-1]
        + pd.Timedelta(hours=1)
    )

    # Actual intensity during the published recommendation.
    followed_block = extract_actual_window(
        actual=actual,
        window_start=forecast_start,
        window_hours=window_hours,
    )

    followed_mean = float(
        followed_block["intensity"].mean()
    )

    # Immediate charging baseline, beginning at midnight.
    immediate_block = actual.head(
        window_hours
    ).copy()

    if len(immediate_block) != window_hours:
        raise ValueError(
            "Insufficient observations for the immediate-charging "
            "baseline."
        )

    immediate_mean = float(
        immediate_block["intensity"].mean()
    )

    immediate_start = (
        immediate_block[TIME_COL]
        .iloc[0]
    )

    immediate_end = (
        immediate_block[TIME_COL]
        .iloc[-1]
        + pd.Timedelta(hours=1)
    )

    # Emissions for the selected charging energy.
    forecast_expected_emissions_kg = (
        forecast_best_mean
        * need_kwh
        / 1000.0
    )

    followed_actual_emissions_kg = (
        followed_mean
        * need_kwh
        / 1000.0
    )

    actual_best_emissions_kg = (
        actual_best_mean
        * need_kwh
        / 1000.0
    )

    actual_worst_emissions_kg = (
        actual_worst_mean
        * need_kwh
        / 1000.0
    )

    immediate_emissions_kg = (
        immediate_mean
        * need_kwh
        / 1000.0
    )

    # Savings relative to the actual highest-intensity window.
    realised_saving_vs_worst_kg = (
        actual_worst_emissions_kg
        - followed_actual_emissions_kg
    )

    maximum_saving_vs_worst_kg = (
        actual_worst_emissions_kg
        - actual_best_emissions_kg
    )

    # Savings relative to charging immediately.
    realised_saving_vs_immediate_kg = (
        immediate_emissions_kg
        - followed_actual_emissions_kg
    )

    perfect_saving_vs_immediate_kg = (
        immediate_emissions_kg
        - actual_best_emissions_kg
    )

    # Carbon Savings Capture.
    if maximum_saving_vs_worst_kg > 0:
        carbon_savings_capture_percent = (
            100.0
            * realised_saving_vs_worst_kg
            / maximum_saving_vs_worst_kg
        )

        carbon_savings_capture_percent = float(
            np.clip(
                carbon_savings_capture_percent,
                0.0,
                100.0,
            )
        )
    else:
        carbon_savings_capture_percent = None

    # Regret, or the additional emissions compared with hindsight.
    carbon_regret_kg = (
        followed_actual_emissions_kg
        - actual_best_emissions_kg
    )

    carbon_regret_percent = (
        100.0
        * carbon_regret_kg
        / actual_best_emissions_kg
        if actual_best_emissions_kg > 0
        else None
    )

    # Window timing and overlap.
    forecast_start_hour = (
        forecast_start.hour
        + forecast_start.minute / 60.0
    )

    actual_start_hour = (
        actual_best_start.hour
        + actual_best_start.minute / 60.0
    )

    timing_error_hours = (
        circular_hour_difference(
            forecast_start_hour,
            actual_start_hour,
        )
    )

    overlap_hours = window_overlap_hours(
        first_start=forecast_start,
        first_end=forecast_end,
        second_start=actual_best_start,
        second_end=actual_best_end,
    )

    overlap_percent = (
        100.0
        * overlap_hours
        / window_hours
    )

    exact_window_match = bool(
        forecast_start == actual_best_start
    )

    within_one_hour = bool(
        timing_error_hours <= 1.0
    )

    within_two_hours = bool(
        timing_error_hours <= 2.0
    )

    # Did following the recommendation beat key baselines?
    cleaner_than_immediate = bool(
        followed_actual_emissions_kg
        < immediate_emissions_kg
    )

    cleaner_than_daily_average = bool(
        followed_mean
        < actual["intensity"].mean()
    )

    decision_quality = classify_decision_quality(
        carbon_savings_capture_percent=(
            carbon_savings_capture_percent
        ),
        timing_error_hours=timing_error_hours,
        overlap_percent=overlap_percent,
    )

    forecast_id = (
        "CCL-"
        + target_start.strftime(
            "%Y%m%d"
        )
    )

    if "forecast_id" in forecast.columns:
        values = (
            forecast["forecast_id"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        if values:
            forecast_id = values[0]

    return {
        "status": "success",
        "forecast_id": forecast_id,
        "target_date": target_start.strftime(
            "%Y-%m-%d"
        ),
        "evaluation_type": "decision_quality",
        "location": (
            "Melbourne, Victoria, Australia"
        ),
        "need_kwh": float(need_kwh),
        "charger_kw": float(charger_kw),
        "exact_duration_hours": round(
            float(exact_duration_hours),
            3,
        ),
        "evaluation_window_hours": int(
            window_hours
        ),

        "forecast_recommended_window": {
            "start_local": (
                forecast_start.isoformat()
            ),
            "end_local": (
                forecast_end.isoformat()
            ),
            "forecast_mean_intensity_g_per_kwh": round(
                forecast_best_mean,
                3,
            ),
            "forecast_expected_emissions_kg": round(
                forecast_expected_emissions_kg,
                3,
            ),
            "actual_mean_intensity_g_per_kwh": round(
                followed_mean,
                3,
            ),
            "actual_emissions_kg": round(
                followed_actual_emissions_kg,
                3,
            ),
        },

        "actual_cleanest_window": {
            "start_local": (
                actual_best_start.isoformat()
            ),
            "end_local": (
                actual_best_end.isoformat()
            ),
            "mean_intensity_g_per_kwh": round(
                actual_best_mean,
                3,
            ),
            "emissions_kg": round(
                actual_best_emissions_kg,
                3,
            ),
        },

        "actual_highest_intensity_window": {
            "start_local": (
                actual_worst_start.isoformat()
            ),
            "end_local": (
                actual_worst_end.isoformat()
            ),
            "mean_intensity_g_per_kwh": round(
                actual_worst_mean,
                3,
            ),
            "emissions_kg": round(
                actual_worst_emissions_kg,
                3,
            ),
        },

        "immediate_charging_baseline": {
            "start_local": (
                immediate_start.isoformat()
            ),
            "end_local": (
                immediate_end.isoformat()
            ),
            "mean_intensity_g_per_kwh": round(
                immediate_mean,
                3,
            ),
            "emissions_kg": round(
                immediate_emissions_kg,
                3,
            ),
        },

        "window_accuracy": {
            "forecast_start_hour_local": round(
                forecast_start_hour,
                2,
            ),
            "actual_cleanest_start_hour_local": round(
                actual_start_hour,
                2,
            ),
            "start_time_error_hours": round(
                timing_error_hours,
                3,
            ),
            "overlap_hours": round(
                overlap_hours,
                3,
            ),
            "overlap_percent": round(
                overlap_percent,
                2,
            ),
            "exact_window_match": exact_window_match,
            "within_one_hour": within_one_hour,
            "within_two_hours": within_two_hours,
        },

        "environmental_outcome": {
            "realised_saving_vs_actual_worst_kg": round(
                realised_saving_vs_worst_kg,
                3,
            ),
            "maximum_possible_saving_vs_actual_worst_kg": round(
                maximum_saving_vs_worst_kg,
                3,
            ),
            "carbon_savings_capture_percent": (
                round(
                    carbon_savings_capture_percent,
                    2,
                )
                if carbon_savings_capture_percent
                is not None
                else None
            ),
            "carbon_regret_vs_perfect_hindsight_kg": round(
                carbon_regret_kg,
                3,
            ),
            "carbon_regret_vs_perfect_hindsight_percent": (
                round(
                    carbon_regret_percent,
                    2,
                )
                if carbon_regret_percent
                is not None
                else None
            ),
            "realised_saving_vs_immediate_kg": round(
                realised_saving_vs_immediate_kg,
                3,
            ),
            "perfect_saving_vs_immediate_kg": round(
                perfect_saving_vs_immediate_kg,
                3,
            ),
            "cleaner_than_immediate_charging": (
                cleaner_than_immediate
            ),
            "cleaner_than_daily_average": (
                cleaner_than_daily_average
            ),
        },

        "decision_quality": decision_quality,
    }


# ---------------------------------------------------------------------
# Public orchestration function
# ---------------------------------------------------------------------

def evaluate_decision_quality(
    target_date_text: str,
    history_file: Path = DEFAULT_HISTORY_FILE,
    forecast_archive_dir: Path = DEFAULT_FORECAST_ARCHIVE_DIR,
    validation_dir: Path = DEFAULT_VALIDATION_DIR,
    need_kwh: float = 20.0,
    charger_kw: float = 7.0,
) -> DecisionEvaluation:
    """
    Run the complete decision evaluation for one official forecast.
    """
    print(
        "\n=============================================="
    )
    print(
        " CleanCharge Live decision evaluation"
    )
    print(
        "==============================================\n"
    )

    target_start = resolve_target_date(
        target_date_text
    )

    target_date = target_start.strftime(
        "%Y-%m-%d"
    )

    forecast_file = (
        forecast_archive_dir
        / f"{target_date}.csv"
    )

    print(
        f">> Target date: {target_date}"
    )

    print(
        f">> Forecast archive: {forecast_file}"
    )

    forecast = load_archived_forecast(
        forecast_file=forecast_file,
        target_start=target_start,
    )

    actual = load_actual_day(
        history_file=history_file,
        target_start=target_start,
    )

    print(
        f">> Forecast rows available: {len(forecast)}"
    )

    print(
        f">> Actual rows available: {len(actual)}"
    )

    metrics = calculate_decision_metrics(
        forecast=forecast,
        actual=actual,
        target_start=target_start,
        need_kwh=need_kwh,
        charger_kw=charger_kw,
    )

    output_dir = (
        validation_dir
        / target_date
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_file = (
        output_dir
        / "decision_metrics.json"
    )

    actual_file = (
        output_dir
        / "actual_day.csv"
    )

    scientific_metrics_file = (
        output_dir
        / "scientific_metrics.json"
    )

    evaluation_summary_file = (
        output_dir
        / "evaluation_summary.json"
    )

    actual_output = actual.copy()

    actual_output[TIME_COL] = actual_output[
        TIME_COL
    ].map(
        lambda value: (
            value.isoformat()
            if pd.notna(value)
            else None
        )
    )

    actual_output.to_csv(
        actual_file,
        index=False,
    )

    save_json(
        metrics,
        metrics_file,
    )

    scientific_metrics = load_json(
        scientific_metrics_file
    )

    decision_quality = metrics[
        "decision_quality"
    ]

    window_accuracy = metrics[
        "window_accuracy"
    ]

    environmental_outcome = metrics[
        "environmental_outcome"
    ]

    evaluation_summary = {
        "status": "success",
        "forecast_id": metrics[
            "forecast_id"
        ],
        "target_date": metrics[
            "target_date"
        ],

        "evaluation_window_hours": (
            metrics[
                "evaluation_window_hours"
            ]
        ),

        "window_overlap_hours": (
            metrics[
                "window_accuracy"
            ][
                "overlap_hours"
            ]
        ),

        "grade": decision_quality[
            "grade"
        ],
        "stars": decision_quality[
            "stars"
        ],
        "decision_quality_label": (
            decision_quality["label"]
        ),

        "carbon_savings_capture_percent": (
            environmental_outcome[
                "carbon_savings_capture_percent"
            ]
        ),

        "timing_error_hours": (
            window_accuracy[
                "start_time_error_hours"
            ]
        ),

        "window_overlap_percent": (
            window_accuracy[
                "overlap_percent"
            ]
        ),

        "exact_window_match": (
            window_accuracy[
                "exact_window_match"
            ]
        ),

        "within_one_hour": (
            window_accuracy[
                "within_one_hour"
            ]
        ),

        "actual_emissions_following_recommendation_kg": (
            metrics[
                "forecast_recommended_window"
            ][
                "actual_emissions_kg"
            ]
        ),

        "perfect_hindsight_emissions_kg": (
            metrics[
                "actual_cleanest_window"
            ][
                "emissions_kg"
            ]
        ),

        "scientific_mae_g_per_kwh": (
            scientific_metrics.get(
                "mae_g_per_kwh"
            )
        ),

        "scientific_rmse_g_per_kwh": (
            scientific_metrics.get(
                "rmse_g_per_kwh"
            )
        ),

        "scientific_smape_percent": (
            scientific_metrics.get(
                "smape_percent"
            )
        ),

        "scientific_r2": (
            scientific_metrics.get(
                "r2"
            )
        ),

        "scientific_bias_g_per_kwh": (
            scientific_metrics.get(
                "mean_error_bias_g_per_kwh"
            )
        ),

        "data_completeness_percent": (
            scientific_metrics.get(
                "data_completeness_percent"
            )
        ),
    }

    save_json(
        evaluation_summary,
        evaluation_summary_file,
    )

    forecast_window = metrics[
        "forecast_recommended_window"
    ]

    actual_window = metrics[
        "actual_cleanest_window"
    ]

    accuracy = metrics[
        "window_accuracy"
    ]

    environmental = metrics[
        "environmental_outcome"
    ]

    quality = metrics[
        "decision_quality"
    ]

    print(
        "\n=============================================="
    )
    print(
        " Decision evaluation completed"
    )
    print(
        "=============================================="
    )

    print(
        f"\nForecast ID:\n  "
        f"{metrics['forecast_id']}"
    )

    print(
        "\nPublished recommendation:"
    )

    print(
        "  "
        f"{forecast_window['start_local']} "
        "to "
        f"{forecast_window['end_local']}"
    )

    print(
        "\nActual cleanest window:"
    )

    print(
        "  "
        f"{actual_window['start_local']} "
        "to "
        f"{actual_window['end_local']}"
    )

    print(
        "\nStart-time error:"
    )

    print(
        "  "
        f"{accuracy['start_time_error_hours']:.1f} hours"
    )

    print(
        "\nWindow overlap:"
    )

    print(
        "  "
        f"{accuracy['overlap_hours']:.1f} hours "
        f"({accuracy['overlap_percent']:.1f}%)"
    )

    print(
        "\nActual emissions following recommendation:"
    )

    print(
        "  "
        f"{forecast_window['actual_emissions_kg']:.2f} "
        "kg CO2"
    )

    print(
        "\nPerfect-hindsight emissions:"
    )

    print(
        "  "
        f"{actual_window['emissions_kg']:.2f} "
        "kg CO2"
    )

    print(
        "\nCarbon Savings Capture:"
    )

    if (
        environmental[
            "carbon_savings_capture_percent"
        ]
        is not None
    ):
        print(
            "  "
            f"{environmental['carbon_savings_capture_percent']:.1f}%"
        )
    else:
        print(
            "  Not available"
        )

    print(
        "\nDecision quality:"
    )

    stars = quality["stars"]

    if stars is not None:
        star_text = (
            "*" * stars
            + "-" * (5 - stars)
        )
    else:
        star_text = "n/a"

    print(
        "  "
        f"{quality['grade']} | "
        f"{quality['label']} | "
        f"{star_text}"
    )

    print(
        f"\nMetrics:\n  {metrics_file}"
    )

    print(
        f"\nActual day data:\n  {actual_file}"
    )

    print(
        f"\nCombined evaluation summary:\n  "
        f"{evaluation_summary_file}"
    )

    print()

    return DecisionEvaluation(
        hourly_actual=actual,
        metrics=metrics,
    )


# ---------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the charging-window decision produced by one "
            "official CleanCharge Live forecast."
        )
    )

    parser.add_argument(
        "--target-date",
        type=str,
        required=True,
        help="Target date in YYYY-MM-DD format.",
    )

    parser.add_argument(
        "--history-file",
        type=Path,
        default=DEFAULT_HISTORY_FILE,
        help="Rolling Victoria observations CSV.",
    )

    parser.add_argument(
        "--forecast-archive-dir",
        type=Path,
        default=DEFAULT_FORECAST_ARCHIVE_DIR,
        help="Directory containing official forecast archives.",
    )

    parser.add_argument(
        "--validation-dir",
        type=Path,
        default=DEFAULT_VALIDATION_DIR,
        help="Validation output directory.",
    )

    parser.add_argument(
        "--need-kwh",
        type=float,
        default=20.0,
        help="Charging energy required. Default: 20 kWh.",
    )

    parser.add_argument(
        "--charger-kw",
        type=float,
        default=7.0,
        help="Charging power. Default: 7 kW.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    evaluate_decision_quality(
        target_date_text=args.target_date,
        history_file=args.history_file,
        forecast_archive_dir=(
            args.forecast_archive_dir
        ),
        validation_dir=args.validation_dir,
        need_kwh=args.need_kwh,
        charger_kw=args.charger_kw,
    )


if __name__ == "__main__":
    main()