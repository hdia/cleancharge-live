"""
CleanCharge Live forecasting module.

Responsibilities
----------------
1. Load the rolling Victoria-region carbon-intensity history.
2. Build calendar and lagged-intensity features.
3. Perform a walk-forward backtest over the most recent seven days.
4. Train a Gradient Boosting model on the available rolling history.
5. Generate a recursive 24-hour carbon-intensity forecast.
6. Identify the lowest- and highest-intensity contiguous charging windows.
7. Save forecast, validation, recommendation and model metadata outputs.
8. Archive each generated forecast for later evaluation.

This module can be:

A. Run directly from the repository root:

    python src/live/forecast_live.py

B. Imported by daily_update.py:

    from src.live.forecast_live import run_live_forecast

    outputs = run_live_forecast(...)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# ---------------------------------------------------------------------
# Constants and default paths
# ---------------------------------------------------------------------

TZ_LOCAL = "Australia/Melbourne"
TIME_COL = "local_time"
TARGET_COL = "intensity"

INTENSITY_LAGS = [1, 2, 3, 6, 12, 24]

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_HISTORY_FILE = (
    ROOT / "data" / "live" / "vic_intensity_history.csv"
)

DEFAULT_LIVE_DIR = ROOT / "data" / "live"

DEFAULT_FORECAST_FILE = (
    DEFAULT_LIVE_DIR / "intensity_forecast_next24.csv"
)

DEFAULT_BACKTEST_FILE = (
    DEFAULT_LIVE_DIR / "intensity_backtest_last7d.csv"
)

DEFAULT_RECOMMENDATION_FILE = (
    DEFAULT_LIVE_DIR / "today_recommendation.json"
)

DEFAULT_MODEL_STATUS_FILE = (
    DEFAULT_LIVE_DIR / "forecast_status.json"
)

DEFAULT_ARCHIVE_DIR = (
    DEFAULT_LIVE_DIR / "forecast_archive"
)


# ---------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------

@dataclass
class ForecastOutputs:
    forecast: pd.DataFrame
    backtest: pd.DataFrame
    recommendation: dict[str, Any]
    model_status: dict[str, Any]


@dataclass
class TrainedModel:
    model: GradientBoostingRegressor
    features: list[str]
    target: str


# ---------------------------------------------------------------------
# Time and data preparation
# ---------------------------------------------------------------------

def ensure_local_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise the input to an hourly Melbourne-time series.

    Accepted time columns:
    - local_time
    - ts
    - timestamp
    - datetime
    - time
    - date

    If 'ts' is used, it is interpreted as UTC.
    Other naive timestamps are interpreted as Melbourne local time.
    """
    if df.empty:
        raise ValueError("Input history is empty.")

    out = df.copy()

    source_col = None

    for candidate in [
        "local_time",
        "ts",
        "timestamp",
        "datetime",
        "time",
        "date",
    ]:
        if candidate in out.columns:
            source_col = candidate
            break

    if source_col is None:
        raise ValueError(
            "Could not identify a time column. Expected one of: "
            "local_time, ts, timestamp, datetime, time or date."
        )

    if source_col == "ts":
        timestamps = pd.to_datetime(
            out[source_col],
            utc=True,
            errors="coerce",
        )

        timestamps = timestamps.dt.tz_convert(
            TZ_LOCAL
        )

    else:
        timestamps = pd.to_datetime(
            out[source_col],
            errors="coerce",
        )

        if timestamps.dt.tz is None:
            timestamps = timestamps.dt.tz_localize(
                TZ_LOCAL,
                ambiguous="infer",
                nonexistent="shift_forward",
            )
        else:
            timestamps = timestamps.dt.tz_convert(
                TZ_LOCAL
            )

    out[TIME_COL] = timestamps

    if TARGET_COL not in out.columns:
        raise ValueError(
            f"Required target column '{TARGET_COL}' was not found."
        )

    out[TARGET_COL] = pd.to_numeric(
        out[TARGET_COL],
        errors="coerce",
    )

    out = out.dropna(
        subset=[
            TIME_COL,
            TARGET_COL,
        ]
    )

    out = (
        out.sort_values(TIME_COL)
        .drop_duplicates(
            subset=[TIME_COL],
            keep="last",
        )
        .reset_index(drop=True)
    )

    if out.empty:
        raise ValueError(
            "No valid observations remain after timestamp and "
            "intensity validation."
        )

    # Create a complete hourly Melbourne-time index.
    hourly_index = pd.date_range(
        start=out[TIME_COL].min().floor("h"),
        end=out[TIME_COL].max().floor("h"),
        freq="h",
        tz=TZ_LOCAL,
    )

    out = (
        out.set_index(TIME_COL)
        .reindex(hourly_index)
        .rename_axis(TIME_COL)
        .reset_index()
    )

    return out


def load_history(history_file: Path) -> pd.DataFrame:
    """
    Load and validate the rolling Victoria history.
    """
    if not history_file.exists():
        raise FileNotFoundError(
            f"History file not found: {history_file}"
        )

    print(f">> Loading history: {history_file}")

    history = pd.read_csv(history_file)

    history = ensure_local_time(history)

    valid_rows = int(
        history[TARGET_COL].notna().sum()
    )

    if valid_rows < 168:
        raise ValueError(
            "At least seven days of valid hourly intensity observations "
            f"are required. Only {valid_rows} valid rows were found."
        )

    print(
        f">> Loaded {valid_rows} valid hourly observations."
    )

    print(
        f">> History period: "
        f"{history[TIME_COL].min()} to "
        f"{history[TIME_COL].max()}"
    )

    return history


# ---------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------

def add_calendar_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add cyclical hour-of-day and day-of-week variables.
    """
    out = df.copy()

    timestamp = out[TIME_COL]

    out["hour"] = timestamp.dt.hour
    out["day_of_week"] = timestamp.dt.dayofweek

    out["hour_sin"] = np.sin(
        2.0 * np.pi * out["hour"] / 24.0
    )

    out["hour_cos"] = np.cos(
        2.0 * np.pi * out["hour"] / 24.0
    )

    out["dow_sin"] = np.sin(
        2.0 * np.pi * out["day_of_week"] / 7.0
    )

    out["dow_cos"] = np.cos(
        2.0 * np.pi * out["day_of_week"] / 7.0
    )

    return out


def build_design_frame(
    history: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Construct the modelling dataset.

    Forecast predictors:
    - cyclical hour-of-day
    - cyclical day-of-week
    - intensity lags at 1, 2, 3, 6, 12 and 24 hours
    """
    out = add_calendar_features(history)

    feature_names = [
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
    ]

    for lag in INTENSITY_LAGS:
        name = f"intensity_lag_{lag}h"

        out[name] = out[TARGET_COL].shift(
            lag
        )

        feature_names.append(name)

    out = out.dropna(
        subset=feature_names + [TARGET_COL]
    ).reset_index(drop=True)

    if len(out) < 100:
        raise ValueError(
            "Insufficient complete observations after feature "
            f"engineering. Only {len(out)} rows remain."
        )

    return out, feature_names


# ---------------------------------------------------------------------
# Model training and validation
# ---------------------------------------------------------------------

def create_model() -> GradientBoostingRegressor:
    """
    Return the standard CleanCharge Live model.
    """
    return GradientBoostingRegressor(
        random_state=42
    )


def train_model(
    design: pd.DataFrame,
    features: list[str],
) -> TrainedModel:
    """
    Train the final model on all available design rows.
    """
    model = create_model()

    model.fit(
        design[features].to_numpy(),
        design[TARGET_COL].to_numpy(),
    )

    return TrainedModel(
        model=model,
        features=features,
        target=TARGET_COL,
    )


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

    ratios = np.where(
        denominator == 0,
        0.0,
        numerator / denominator,
    )

    return float(
        np.mean(ratios) * 100.0
    )


def walk_forward_backtest(
    design: pd.DataFrame,
    features: list[str],
    backtest_days: int = 7,
    minimum_training_rows: int = 168,
) -> pd.DataFrame:
    """
    Run a strict hourly walk-forward validation.

    For every validation hour:
    - train only on observations available before that hour
    - predict that hour
    - retain the forecast and actual value

    This is computationally heavier than a simple held-out prediction,
    but it provides an honest approximation of live forecasting.
    """
    cutoff = (
        design[TIME_COL].max()
        - pd.Timedelta(days=backtest_days)
    )

    validation = design[
        design[TIME_COL] >= cutoff
    ].copy()

    predictions: list[dict[str, Any]] = []

    for row_index in validation.index:
        training = design.loc[
            design.index < row_index
        ]

        training = training.dropna(
            subset=features + [TARGET_COL]
        )

        if len(training) < minimum_training_rows:
            continue

        row = design.loc[[row_index]]

        model = create_model()

        model.fit(
            training[features].to_numpy(),
            training[TARGET_COL].to_numpy(),
        )

        predicted = float(
            model.predict(
                row[features].to_numpy()
            )[0]
        )

        predictions.append(
            {
                TIME_COL: row[TIME_COL].iloc[0],
                "y_true": float(
                    row[TARGET_COL].iloc[0]
                ),
                "y_hat": predicted,
            }
        )

    result = pd.DataFrame(predictions)

    if not result.empty:
        result = result.sort_values(
            TIME_COL
        ).reset_index(drop=True)

    return result


def backtest_metrics(
    backtest: pd.DataFrame,
) -> dict[str, Any]:
    """
    Calculate validation metrics from walk-forward predictions.
    """
    if backtest.empty:
        return {
            "rows": 0,
            "mae_g_per_kwh": None,
            "smape_percent": None,
            "r2": None,
        }

    actual = backtest["y_true"].to_numpy()
    predicted = backtest["y_hat"].to_numpy()

    metrics = {
        "rows": int(len(backtest)),
        "mae_g_per_kwh": round(
            float(
                mean_absolute_error(
                    actual,
                    predicted,
                )
            ),
            3,
        ),
        "smape_percent": round(
            calculate_smape(
                actual,
                predicted,
            ),
            3,
        ),
        "r2": round(
            float(
                r2_score(
                    actual,
                    predicted,
                )
            ),
            4,
        ),
    }

    return metrics


# ---------------------------------------------------------------------
# Recursive 24-hour forecast
# ---------------------------------------------------------------------

def forecast_next_24_hours(
    history: pd.DataFrame,
    trained: TrainedModel,
) -> pd.DataFrame:
    """
    Generate a recursive 24-hour forecast.

    Each predicted value is appended to the working history so that it
    can be used by subsequent lagged predictions.
    """
    working = history[
        [TIME_COL, TARGET_COL]
    ].copy()

    working = (
        working.dropna(
            subset=[TIME_COL, TARGET_COL]
        )
        .sort_values(TIME_COL)
        .reset_index(drop=True)
    )

    if len(working) < max(INTENSITY_LAGS):
        raise ValueError(
            "Insufficient history for recursive forecasting."
        )

    rows: list[dict[str, Any]] = []

    for forecast_step in range(1, 25):
        next_time = (
            working[TIME_COL].iloc[-1]
            + pd.Timedelta(hours=1)
        )

        next_features: dict[str, float] = {}

        hour = next_time.hour
        day_of_week = next_time.dayofweek

        next_features["hour_sin"] = float(
            np.sin(
                2.0 * np.pi * hour / 24.0
            )
        )

        next_features["hour_cos"] = float(
            np.cos(
                2.0 * np.pi * hour / 24.0
            )
        )

        next_features["dow_sin"] = float(
            np.sin(
                2.0
                * np.pi
                * day_of_week
                / 7.0
            )
        )

        next_features["dow_cos"] = float(
            np.cos(
                2.0
                * np.pi
                * day_of_week
                / 7.0
            )
        )

        for lag in INTENSITY_LAGS:
            next_features[
                f"intensity_lag_{lag}h"
            ] = float(
                working[TARGET_COL].iloc[-lag]
            )

        feature_frame = pd.DataFrame(
            [next_features]
        )

        predicted = float(
            trained.model.predict(
                feature_frame[
                    trained.features
                ].to_numpy()
            )[0]
        )

        rows.append(
            {
                TIME_COL: next_time,
                "forecast_step": forecast_step,
                "intensity_hat": predicted,
            }
        )

        working = pd.concat(
            [
                working,
                pd.DataFrame(
                    {
                        TIME_COL: [next_time],
                        TARGET_COL: [predicted],
                    }
                ),
            ],
            ignore_index=True,
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Charging-window calculations
# ---------------------------------------------------------------------

def identify_contiguous_window(
    forecast: pd.DataFrame,
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

    if len(forecast) < window_hours:
        raise ValueError(
            "Forecast horizon is shorter than the charging window."
        )

    working = forecast[
        [
            TIME_COL,
            "intensity_hat",
        ]
    ].copy()

    working["window_mean"] = (
        working["intensity_hat"]
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

    average_intensity = float(
        block["intensity_hat"].mean()
    )

    return block, average_intensity


def build_recommendation(
    forecast: pd.DataFrame,
    need_kwh: float,
    charger_kw: float,
    generated_at_local: pd.Timestamp,
) -> dict[str, Any]:
    """
    Create the dashboard recommendation for one charging scenario.
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

    best_block, best_intensity = (
        identify_contiguous_window(
            forecast=forecast,
            window_hours=window_hours,
            lowest=True,
        )
    )

    worst_block, worst_intensity = (
        identify_contiguous_window(
            forecast=forecast,
            window_hours=window_hours,
            lowest=False,
        )
    )

    forecast_average_intensity = float(
        forecast["intensity_hat"].mean()
    )

    best_emissions_kg = (
        best_intensity
        * need_kwh
        / 1000.0
    )

    average_emissions_kg = (
        forecast_average_intensity
        * need_kwh
        / 1000.0
    )

    worst_emissions_kg = (
        worst_intensity
        * need_kwh
        / 1000.0
    )

    saving_kg = (
        worst_emissions_kg
        - best_emissions_kg
    )

    saving_percent = (
        100.0
        * saving_kg
        / worst_emissions_kg
        if worst_emissions_kg > 0
        else None
    )

    best_start = best_block[
        TIME_COL
    ].iloc[0]

    best_end = (
        best_block[TIME_COL].iloc[-1]
        + pd.Timedelta(hours=1)
    )

    worst_start = worst_block[
        TIME_COL
    ].iloc[0]

    worst_end = (
        worst_block[TIME_COL].iloc[-1]
        + pd.Timedelta(hours=1)
    )

    forecast_start = forecast[
        TIME_COL
    ].min()

    forecast_end = (
        forecast[TIME_COL].max()
        + pd.Timedelta(hours=1)
    )

    recommendation = {
        "status": "success",
        "location": (
            "Melbourne, Victoria, Australia"
        ),
        "network": "NEM",
        "network_region": "VIC1",
        "generated_at_local": (
            generated_at_local.isoformat()
        ),
        "forecast_start_local": (
            forecast_start.isoformat()
        ),
        "forecast_end_local": (
            forecast_end.isoformat()
        ),
        "need_kwh": float(need_kwh),
        "charger_kw": float(charger_kw),
        "exact_duration_hours": round(
            float(exact_duration_hours),
            3,
        ),
        "forecast_window_hours": int(
            window_hours
        ),
        "best_window_start_local": (
            best_start.isoformat()
        ),
        "best_window_end_local": (
            best_end.isoformat()
        ),
        "best_window_mean_intensity_g_per_kwh": round(
            best_intensity,
            3,
        ),
        "best_window_emissions_kg": round(
            best_emissions_kg,
            3,
        ),
        "forecast_average_intensity_g_per_kwh": round(
            forecast_average_intensity,
            3,
        ),
        "average_window_emissions_kg": round(
            average_emissions_kg,
            3,
        ),
        "highest_intensity_window_start_local": (
            worst_start.isoformat()
        ),
        "highest_intensity_window_end_local": (
            worst_end.isoformat()
        ),
        "highest_window_mean_intensity_g_per_kwh": round(
            worst_intensity,
            3,
        ),
        "highest_window_emissions_kg": round(
            worst_emissions_kg,
            3,
        ),
        "emissions_saving_kg": round(
            saving_kg,
            3,
        ),
        "emissions_saving_percent": (
            round(
                float(saving_percent),
                2,
            )
            if saving_percent is not None
            else None
        ),
    }

    return recommendation


# ---------------------------------------------------------------------
# Output handling
# ---------------------------------------------------------------------

def dataframe_for_csv(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert timezone-aware timestamps to ISO strings for stable CSV output.
    """
    out = df.copy()

    if TIME_COL in out.columns:
        out[TIME_COL] = out[
            TIME_COL
        ].map(
            lambda value: (
                value.isoformat()
                if pd.notna(value)
                else None
            )
        )

    return out


def save_json(
    data: dict[str, Any],
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
            data,
            file,
            indent=2,
        )


def archive_forecast(
    forecast: pd.DataFrame,
    archive_dir: Path,
) -> Path:
    """
    Save an immutable daily forecast copy.
    The filename is based on the first forecast date in NEM time.
    Existing archive files are not overwritten.
    """
    archive_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    forecast_date = (
        forecast[TIME_COL]
        .min()
        .strftime("%Y-%m-%d")
    )

    archive_file = (
        archive_dir
        / f"forecast_{forecast_date}.csv"
    )

    if archive_file.exists():
        print(
            f">> Archive already exists and was preserved: "
            f"{archive_file}"
        )

        return archive_file

    dataframe_for_csv(
        forecast
    ).to_csv(
        archive_file,
        index=False,
    )

    print(
        f">> Archived forecast: {archive_file}"
    )

    return archive_file


def save_forecast_outputs(
    outputs: ForecastOutputs,
    forecast_file: Path,
    backtest_file: Path,
    recommendation_file: Path,
    model_status_file: Path,
    archive_dir: Path,
) -> dict[str, Path]:
    """
    Save all live forecasting outputs.
    """
    forecast_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe_for_csv(
        outputs.forecast
    ).to_csv(
        forecast_file,
        index=False,
    )

    dataframe_for_csv(
        outputs.backtest
    ).to_csv(
        backtest_file,
        index=False,
    )

    save_json(
        outputs.recommendation,
        recommendation_file,
    )

    save_json(
        outputs.model_status,
        model_status_file,
    )

    archive_file = archive_forecast(
        forecast=outputs.forecast,
        archive_dir=archive_dir,
    )

    return {
        "forecast_file": forecast_file,
        "backtest_file": backtest_file,
        "recommendation_file": recommendation_file,
        "model_status_file": model_status_file,
        "archive_file": archive_file,
    }


# ---------------------------------------------------------------------
# Public orchestration function
# ---------------------------------------------------------------------

def run_live_forecast(
    history_file: Path = DEFAULT_HISTORY_FILE,
    forecast_file: Path = DEFAULT_FORECAST_FILE,
    backtest_file: Path = DEFAULT_BACKTEST_FILE,
    recommendation_file: Path = DEFAULT_RECOMMENDATION_FILE,
    model_status_file: Path = DEFAULT_MODEL_STATUS_FILE,
    archive_dir: Path = DEFAULT_ARCHIVE_DIR,
    need_kwh: float = 20.0,
    charger_kw: float = 7.0,
    backtest_days: int = 7,
) -> ForecastOutputs:
    """
    Run the complete CleanCharge Live forecasting process.
    """
    print(
        "\n=============================================="
    )
    print(
        " CleanCharge Live forecast update"
    )
    print(
        "==============================================\n"
    )

    generated_at_utc = datetime.now(
        timezone.utc
    )

    generated_at_local = (
        pd.Timestamp(generated_at_utc)
        .tz_convert(TZ_LOCAL)
    )

    history = load_history(
        history_file
    )

    design, features = build_design_frame(
        history
    )

    print(
        f">> Design rows: {len(design)}"
    )

    print(
        f">> Forecast features: "
        f"{', '.join(features)}"
    )

    print(
        f">> Running {backtest_days}-day "
        "walk-forward validation..."
    )

    backtest = walk_forward_backtest(
        design=design,
        features=features,
        backtest_days=backtest_days,
    )

    metrics = backtest_metrics(
        backtest
    )

    if metrics["rows"] > 0:
        print(
            f">> Validation MAE: "
            f"{metrics['mae_g_per_kwh']:.2f} "
            "gCO2/kWh"
        )

        print(
            f">> Validation sMAPE: "
            f"{metrics['smape_percent']:.2f}%"
        )

        print(
            f">> Validation R2: "
            f"{metrics['r2']:.3f}"
        )
    else:
        print(
            "!! No validation predictions were generated."
        )

    print(
        ">> Training final Gradient Boosting model..."
    )

    trained = train_model(
        design=design,
        features=features,
    )

    print(
        ">> Generating recursive 24-hour forecast..."
    )

    forecast = forecast_next_24_hours(
        history=history,
        trained=trained,
    )

    recommendation = build_recommendation(
        forecast=forecast,
        need_kwh=need_kwh,
        charger_kw=charger_kw,
        generated_at_local=generated_at_local,
    )

    model_status = {
        "status": "success",
        "generated_at_utc": (
            generated_at_utc.isoformat()
        ),
        "generated_at_local": (
            generated_at_local.isoformat()
        ),
        "location": (
            "Melbourne, Victoria, Australia"
        ),
        "network": "NEM",
        "network_region": "VIC1",
        "model": "GradientBoostingRegressor",
        "random_state": 42,
        "training_history_start_local": (
            history[TIME_COL]
            .min()
            .isoformat()
        ),
        "training_history_end_local": (
            history[TIME_COL]
            .max()
            .isoformat()
        ),
        "training_observation_rows": int(
            history[TARGET_COL]
            .notna()
            .sum()
        ),
        "design_rows": int(
            len(design)
        ),
        "features": features,
        "intensity_lags_hours": (
            INTENSITY_LAGS
        ),
        "forecast_horizon_hours": 24,
        "backtest_days": int(
            backtest_days
        ),
        "backtest": metrics,
    }

    outputs = ForecastOutputs(
        forecast=forecast,
        backtest=backtest,
        recommendation=recommendation,
        model_status=model_status,
    )

    saved = save_forecast_outputs(
        outputs=outputs,
        forecast_file=forecast_file,
        backtest_file=backtest_file,
        recommendation_file=recommendation_file,
        model_status_file=model_status_file,
        archive_dir=archive_dir,
    )

    print(
        "\n=============================================="
    )
    print(
        " Forecast update completed successfully"
    )
    print(
        "=============================================="
    )

    print(
        f"\nForecast file:\n  "
        f"{saved['forecast_file']}"
    )

    print(
        f"\nBacktest file:\n  "
        f"{saved['backtest_file']}"
    )

    print(
        f"\nRecommendation file:\n  "
        f"{saved['recommendation_file']}"
    )

    print(
        f"\nModel status file:\n  "
        f"{saved['model_status_file']}"
    )

    print(
        f"\nForecast archive:\n  "
        f"{saved['archive_file']}"
    )

    print(
        "\nRecommended low-carbon window:"
    )

    print(
        "  "
        + recommendation[
            "best_window_start_local"
        ]
        + " to "
        + recommendation[
            "best_window_end_local"
        ]
    )

    print(
        "Mean intensity:"
    )

    print(
        "  "
        + str(
            recommendation[
                "best_window_mean_intensity_g_per_kwh"
            ]
        )
        + " gCO2/kWh"
    )

    print(
        "Estimated emissions saving:"
    )

    print(
        "  "
        + str(
            recommendation[
                "emissions_saving_kg"
            ]
        )
        + " kg CO2 ("
        + str(
            recommendation[
                "emissions_saving_percent"
            ]
        )
        + "%)"
    )

    print()

    return outputs


# ---------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train the CleanCharge Live model, run validation, "
            "forecast the next 24 hours and identify the lowest-"
            "intensity charging window."
        )
    )

    parser.add_argument(
        "--history-file",
        type=Path,
        default=DEFAULT_HISTORY_FILE,
        help="Rolling Victoria intensity history CSV.",
    )

    parser.add_argument(
        "--forecast-file",
        type=Path,
        default=DEFAULT_FORECAST_FILE,
        help="Current 24-hour forecast CSV.",
    )

    parser.add_argument(
        "--backtest-file",
        type=Path,
        default=DEFAULT_BACKTEST_FILE,
        help="Latest seven-day validation CSV.",
    )

    parser.add_argument(
        "--recommendation-file",
        type=Path,
        default=DEFAULT_RECOMMENDATION_FILE,
        help="Current recommendation JSON.",
    )

    parser.add_argument(
        "--model-status-file",
        type=Path,
        default=DEFAULT_MODEL_STATUS_FILE,
        help="Forecast model metadata JSON.",
    )

    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=DEFAULT_ARCHIVE_DIR,
        help="Directory for immutable daily forecast archives.",
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

    parser.add_argument(
        "--backtest-days",
        type=int,
        default=7,
        help="Walk-forward validation period. Default: 7 days.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    run_live_forecast(
        history_file=args.history_file,
        forecast_file=args.forecast_file,
        backtest_file=args.backtest_file,
        recommendation_file=(
            args.recommendation_file
        ),
        model_status_file=(
            args.model_status_file
        ),
        archive_dir=args.archive_dir,
        need_kwh=args.need_kwh,
        charger_kw=args.charger_kw,
        backtest_days=args.backtest_days,
    )


if __name__ == "__main__":
    main()