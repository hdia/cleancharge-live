"""
Scientific evaluation of an official CleanCharge Live daily forecast.

This module compares an immutable archived forecast with the actual
Victoria-region carbon-intensity observations for the same Melbourne
calendar day.

Outputs
-------
1. Hourly comparison CSV
2. Scientific metrics JSON

Example
-------
Run from the repository root:

    python -m src.live.evaluate_scientific --target-date 2026-07-11
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


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


# ---------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------

@dataclass
class ScientificEvaluation:
    comparison: pd.DataFrame
    metrics: dict[str, Any]


# ---------------------------------------------------------------------
# Utilities
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


def parse_local_time(
    series: pd.Series,
) -> pd.Series:
    """
    Parse timestamps and return Melbourne timezone-aware datetimes.
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


def calculate_smape(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """
    Symmetric mean absolute percentage error.
    """
    numerator = np.abs(
        actual - predicted
    )

    denominator = (
        np.abs(actual)
        + np.abs(predicted)
    ) / 2.0

    ratio = np.where(
        denominator == 0,
        0.0,
        numerator / denominator,
    )

    return float(
        np.mean(ratio) * 100.0
    )


def resolve_target_date(
    target_date_text: str,
) -> pd.Timestamp:
    """
    Return the requested Melbourne date at local midnight.
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


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def load_archived_forecast(
    forecast_file: Path,
    target_start: pd.Timestamp,
) -> pd.DataFrame:
    """
    Load and validate one official archived forecast.
    """
    if not forecast_file.exists():
        raise FileNotFoundError(
            f"Archived forecast not found: {forecast_file}"
        )

    forecast = pd.read_csv(
        forecast_file
    )

    if TIME_COL not in forecast.columns:
        raise ValueError(
            f"Forecast file has no '{TIME_COL}' column."
        )

    if "intensity_hat" not in forecast.columns:
        raise ValueError(
            "Forecast file has no 'intensity_hat' column."
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

    if forecast.empty:
        raise ValueError(
            "No forecast observations were found for "
            f"{target_start.date()}."
        )

    return forecast


def load_actual_observations(
    history_file: Path,
    target_start: pd.Timestamp,
) -> pd.DataFrame:
    """
    Load actual intensity observations for the target Melbourne day.
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

    if actual.empty:
        raise ValueError(
            "No actual observations were found for "
            f"{target_start.date()}."
        )

    return actual


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def align_forecast_and_actual(
    forecast: pd.DataFrame,
    actual: pd.DataFrame,
) -> pd.DataFrame:
    """
    Align forecast and actual intensity values by Melbourne hour.
    """
    forecast_columns = [
        TIME_COL,
        "intensity_hat",
    ]

    for optional in [
        "forecast_id",
        "target_date",
        "forecast_step",
        "published_at_local",
    ]:
        if optional in forecast.columns:
            forecast_columns.append(
                optional
            )

    comparison = pd.merge(
        forecast[forecast_columns],
        actual[
            [
                TIME_COL,
                "intensity",
            ]
        ],
        on=TIME_COL,
        how="outer",
        indicator=True,
    )

    comparison = comparison.rename(
        columns={
            "intensity": "intensity_actual",
        }
    )

    comparison = comparison.sort_values(
        TIME_COL
    ).reset_index(drop=True)

    comparison["error"] = (
        comparison["intensity_hat"]
        - comparison["intensity_actual"]
    )

    comparison["absolute_error"] = (
        comparison["error"].abs()
    )

    comparison["squared_error"] = (
        comparison["error"] ** 2
    )

    comparison["absolute_percentage_error"] = np.where(
        comparison["intensity_actual"] != 0,
        (
            comparison["absolute_error"]
            / comparison["intensity_actual"].abs()
        )
        * 100.0,
        np.nan,
    )

    comparison["hour_local"] = (
        comparison[TIME_COL].dt.hour
    )

    return comparison


def calculate_scientific_metrics(
    comparison: pd.DataFrame,
    target_start: pd.Timestamp,
) -> dict[str, Any]:
    """
    Calculate forecast accuracy and completeness statistics.
    """
    valid = comparison.dropna(
        subset=[
            "intensity_hat",
            "intensity_actual",
        ]
    ).copy()

    if valid.empty:
        raise ValueError(
            "No overlapping valid forecast and actual observations."
        )

    actual = valid[
        "intensity_actual"
    ].to_numpy()

    predicted = valid[
        "intensity_hat"
    ].to_numpy()

    mae = float(
        mean_absolute_error(
            actual,
            predicted,
        )
    )

    rmse = float(
        np.sqrt(
            mean_squared_error(
                actual,
                predicted,
            )
        )
    )

    smape = calculate_smape(
        actual,
        predicted,
    )

    bias = float(
        np.mean(
            predicted - actual
        )
    )

    maximum_absolute_error = float(
        np.max(
            np.abs(
                predicted - actual
            )
        )
    )

    if len(valid) >= 2:
        r2 = float(
            r2_score(
                actual,
                predicted,
            )
        )
    else:
        r2 = None

    expected_hours = 24

    forecast_hours = int(
        comparison[
            "intensity_hat"
        ].notna().sum()
    )

    actual_hours = int(
        comparison[
            "intensity_actual"
        ].notna().sum()
    )

    matched_hours = int(
        len(valid)
    )

    completeness_percent = (
        100.0
        * matched_hours
        / expected_hours
    )

    forecast_id = None

    if "forecast_id" in valid.columns:
        values = (
            valid["forecast_id"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        if values:
            forecast_id = values[0]

    if forecast_id is None:
        forecast_id = (
            "CCL-"
            + target_start.strftime(
                "%Y%m%d"
            )
        )

    return {
        "status": "success",
        "forecast_id": forecast_id,
        "target_date": (
            target_start.strftime(
                "%Y-%m-%d"
            )
        ),
        "evaluation_type": (
            "scientific_accuracy"
        ),
        "expected_hours": expected_hours,
        "forecast_hours_available": (
            forecast_hours
        ),
        "actual_hours_available": (
            actual_hours
        ),
        "matched_hours": matched_hours,
        "data_completeness_percent": round(
            completeness_percent,
            2,
        ),
        "mae_g_per_kwh": round(
            mae,
            3,
        ),
        "rmse_g_per_kwh": round(
            rmse,
            3,
        ),
        "smape_percent": round(
            smape,
            3,
        ),
        "r2": (
            round(
                r2,
                4,
            )
            if r2 is not None
            else None
        ),
        "mean_error_bias_g_per_kwh": round(
            bias,
            3,
        ),
        "maximum_absolute_error_g_per_kwh": round(
            maximum_absolute_error,
            3,
        ),
        "actual_mean_intensity_g_per_kwh": round(
            float(
                np.mean(actual)
            ),
            3,
        ),
        "forecast_mean_intensity_g_per_kwh": round(
            float(
                np.mean(predicted)
            ),
            3,
        ),
        "actual_minimum_intensity_g_per_kwh": round(
            float(
                np.min(actual)
            ),
            3,
        ),
        "actual_maximum_intensity_g_per_kwh": round(
            float(
                np.max(actual)
            ),
            3,
        ),
    }


def evaluate_scientific_accuracy(
    target_date_text: str,
    history_file: Path = DEFAULT_HISTORY_FILE,
    forecast_archive_dir: Path = DEFAULT_FORECAST_ARCHIVE_DIR,
    validation_dir: Path = DEFAULT_VALIDATION_DIR,
) -> ScientificEvaluation:
    """
    Run the complete scientific evaluation for one target date.
    """
    print(
        "\n=============================================="
    )
    print(
        " CleanCharge Live scientific evaluation"
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

    actual = load_actual_observations(
        history_file=history_file,
        target_start=target_start,
    )

    print(
        f">> Forecast rows available: {len(forecast)}"
    )

    print(
        f">> Actual rows available: {len(actual)}"
    )

    comparison = align_forecast_and_actual(
        forecast=forecast,
        actual=actual,
    )

    metrics = calculate_scientific_metrics(
        comparison=comparison,
        target_start=target_start,
    )

    output_dir = (
        validation_dir
        / target_date
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison_file = (
        output_dir
        / "scientific_comparison.csv"
    )

    metrics_file = (
        output_dir
        / "scientific_metrics.json"
    )

    csv_output = comparison.copy()

    csv_output[TIME_COL] = csv_output[
        TIME_COL
    ].map(
        lambda value: (
            value.isoformat()
            if pd.notna(value)
            else None
        )
    )

    csv_output.to_csv(
        comparison_file,
        index=False,
    )

    save_json(
        metrics,
        metrics_file,
    )

    print(
        "\n=============================================="
    )
    print(
        " Scientific evaluation completed"
    )
    print(
        "=============================================="
    )

    print(
        f"\nForecast ID:\n  "
        f"{metrics['forecast_id']}"
    )

    print(
        f"\nMatched hours:\n  "
        f"{metrics['matched_hours']} / "
        f"{metrics['expected_hours']}"
    )

    print(
        f"\nData completeness:\n  "
        f"{metrics['data_completeness_percent']:.1f}%"
    )

    print(
        f"\nMAE:\n  "
        f"{metrics['mae_g_per_kwh']:.2f} "
        "gCO2/kWh"
    )

    print(
        f"\nRMSE:\n  "
        f"{metrics['rmse_g_per_kwh']:.2f} "
        "gCO2/kWh"
    )

    print(
        f"\nsMAPE:\n  "
        f"{metrics['smape_percent']:.2f}%"
    )

    if metrics["r2"] is not None:
        print(
            f"\nR2:\n  "
            f"{metrics['r2']:.3f}"
        )

    print(
        f"\nBias:\n  "
        f"{metrics['mean_error_bias_g_per_kwh']:.2f} "
        "gCO2/kWh"
    )

    print(
        f"\nHourly comparison:\n  "
        f"{comparison_file}"
    )

    print(
        f"\nMetrics:\n  "
        f"{metrics_file}"
    )

    print()

    return ScientificEvaluation(
        comparison=comparison,
        metrics=metrics,
    )


# ---------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one official CleanCharge Live forecast "
            "against actual Victoria-region observations."
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

    return parser.parse_args()


def main():
    args = parse_args()

    evaluate_scientific_accuracy(
        target_date_text=args.target_date,
        history_file=args.history_file,
        forecast_archive_dir=(
            args.forecast_archive_dir
        ),
        validation_dir=args.validation_dir,
    )


if __name__ == "__main__":
    main()