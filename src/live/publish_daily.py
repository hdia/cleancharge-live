"""

Correct usage: python -m src.live.publish_daily --target-date 2026-07-11

Publish one official CleanCharge Live forecast for a Melbourne calendar day.

The published forecast:

1. Covers 00:00 to 23:00 Melbourne time.
2. Uses only observations available before the target day begins.
3. Receives a permanent forecast ID.
4. Is archived without being overwritten.
5. Produces current dashboard files for the selected day.

Examples
--------

Publish today's forecast:

    python src/live/publish_daily.py

Publish or backfill a specific date:

    python src/live/publish_daily.py --target-date 2026-07-11

Use a different charging scenario:

    python src/live/publish_daily.py \
        --target-date 2026-07-11 \
        --need-kwh 40 \
        --charger-kw 11
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from src.live.forecast_live import (
    TIME_COL,
    TARGET_COL,
    TZ_LOCAL,
    ForecastOutputs,
    backtest_metrics,
    build_design_frame,
    build_recommendation,
    dataframe_for_csv,
    forecast_next_24_hours,
    load_history,
    train_model,
    walk_forward_backtest,
)


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_HISTORY_FILE = (
    ROOT / "data" / "live" / "vic_intensity_history.csv"
)

DEFAULT_FORECAST_DIR = (
    ROOT / "data" / "live" / "forecasts"
)

DEFAULT_ARCHIVE_DIR = (
    DEFAULT_FORECAST_DIR / "archive"
)

DEFAULT_TODAY_FORECAST_FILE = (
    DEFAULT_FORECAST_DIR / "today.csv"
)

DEFAULT_VALIDATION_DIR = (
    ROOT / "data" / "live" / "validation"
)

DEFAULT_STATUS_DIR = (
    ROOT / "data" / "live" / "status"
)

DEFAULT_RECOMMENDATION_FILE = (
    DEFAULT_STATUS_DIR / "today_recommendation.json"
)

DEFAULT_FORECAST_STATUS_FILE = (
    DEFAULT_STATUS_DIR / "forecast_status.json"
)

MELBOURNE_TZ = ZoneInfo(TZ_LOCAL)


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


def resolve_target_date(
    target_date_text: str | None,
) -> pd.Timestamp:
    """
    Return the requested Melbourne calendar date at midnight.

    If no date is supplied, use today's Melbourne date.
    """
    if target_date_text:
        parsed = pd.Timestamp(
            target_date_text
        )

        if parsed.tzinfo is None:
            parsed = parsed.tz_localize(
                TZ_LOCAL
            )
        else:
            parsed = parsed.tz_convert(
                TZ_LOCAL
            )

        return parsed.normalize()

    now_local = pd.Timestamp.now(
        tz=TZ_LOCAL
    )

    return now_local.normalize()


def prepare_training_history(
    history: pd.DataFrame,
    target_start: pd.Timestamp,
) -> pd.DataFrame:
    """
    Keep only observations available before the target day begins.
    """
    training_history = history[
        history[TIME_COL] < target_start
    ].copy()

    training_history = (
        training_history
        .dropna(
            subset=[
                TIME_COL,
                TARGET_COL,
            ]
        )
        .sort_values(TIME_COL)
        .reset_index(drop=True)
    )

    if training_history.empty:
        raise ValueError(
            "No observations are available before the target date."
        )

    expected_last_hour = (
        target_start
        - pd.Timedelta(hours=1)
    )

    actual_last_hour = (
        training_history[TIME_COL]
        .max()
        .floor("h")
    )

    if actual_last_hour != expected_last_hour:
        raise ValueError(
            "The training history does not end at 23:00 on the "
            "day before the requested forecast. "
            f"Expected {expected_last_hour}, found {actual_last_hour}."
        )

    if len(training_history) < 168:
        raise ValueError(
            "At least seven days of prior hourly observations "
            "are required."
        )

    return training_history


def validate_calendar_forecast(
    forecast: pd.DataFrame,
    target_start: pd.Timestamp,
) -> None:
    """
    Confirm that the generated forecast covers exactly the target day.
    """
    if len(forecast) != 24:
        raise ValueError(
            f"Expected 24 forecast rows, found {len(forecast)}."
        )

    expected_start = target_start

    expected_last_hour = (
        target_start
        + pd.Timedelta(hours=23)
    )

    actual_start = (
        forecast[TIME_COL]
        .min()
        .floor("h")
    )

    actual_last_hour = (
        forecast[TIME_COL]
        .max()
        .floor("h")
    )

    if actual_start != expected_start:
        raise ValueError(
            "Forecast does not begin at local midnight. "
            f"Expected {expected_start}, found {actual_start}."
        )

    if actual_last_hour != expected_last_hour:
        raise ValueError(
            "Forecast does not end at 23:00 local time. "
            f"Expected {expected_last_hour}, found {actual_last_hour}."
        )


def publish_archive(
    forecast: pd.DataFrame,
    archive_file: Path,
) -> str:
    """
    Publish an immutable archive forecast.

    Returns:
        'created' if written for the first time.
        'preserved' if an archive already exists.
    """
    archive_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if archive_file.exists():
        print(
            f">> Official archive already exists and was preserved: "
            f"{archive_file}"
        )

        return "preserved"

    dataframe_for_csv(
        forecast
    ).to_csv(
        archive_file,
        index=False,
    )

    print(
        f">> Published official forecast archive: "
        f"{archive_file}"
    )

    return "created"


# ---------------------------------------------------------------------
# Main publication workflow
# ---------------------------------------------------------------------

def publish_daily_forecast(
    history_file: Path = DEFAULT_HISTORY_FILE,
    target_date_text: str | None = None,
    need_kwh: float = 20.0,
    charger_kw: float = 7.0,
    backtest_days: int = 7,
    today_forecast_file: Path = DEFAULT_TODAY_FORECAST_FILE,
    archive_dir: Path = DEFAULT_ARCHIVE_DIR,
    recommendation_file: Path = DEFAULT_RECOMMENDATION_FILE,
    forecast_status_file: Path = DEFAULT_FORECAST_STATUS_FILE,
) -> ForecastOutputs:
    """
    Train and publish one official calendar-day forecast.
    """
    print(
        "\n=============================================="
    )
    print(
        " CleanCharge Live daily forecast publication"
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

    forecast_id = (
        "CCL-"
        + target_start.strftime("%Y%m%d")
    )

    published_at_local = pd.Timestamp.now(
        tz=TZ_LOCAL
    )

    print(
        f">> Forecast ID: {forecast_id}"
    )

    print(
        f">> Target date: {target_date}"
    )

    history = load_history(
        history_file
    )

    training_history = prepare_training_history(
        history=history,
        target_start=target_start,
    )

    print(
        f">> Training observations available: "
        f"{len(training_history)}"
    )

    print(
        f">> Training history ends: "
        f"{training_history[TIME_COL].max()}"
    )

    design, features = build_design_frame(
        training_history
    )

    print(
        f">> Design rows: {len(design)}"
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

    if metrics["mae_g_per_kwh"] is not None:
        print(
            f">> Validation MAE: "
            f"{metrics['mae_g_per_kwh']:.2f} gCO2/kWh"
        )

        print(
            f">> Validation sMAPE: "
            f"{metrics['smape_percent']:.2f}%"
        )

        print(
            f">> Validation R2: "
            f"{metrics['r2']:.3f}"
        )

    print(
        ">> Training final daily model..."
    )

    trained = train_model(
        design=design,
        features=features,
    )

    print(
        ">> Generating midnight-to-midnight forecast..."
    )

    forecast = forecast_next_24_hours(
        history=training_history,
        trained=trained,
    )

    validate_calendar_forecast(
        forecast=forecast,
        target_start=target_start,
    )

    forecast.insert(
        0,
        "forecast_id",
        forecast_id,
    )

    forecast.insert(
        1,
        "target_date",
        target_date,
    )

    forecast["published_at_local"] = (
        published_at_local.isoformat()
    )

    recommendation = build_recommendation(
        forecast=forecast,
        need_kwh=need_kwh,
        charger_kw=charger_kw,
        generated_at_local=published_at_local,
    )

    recommendation["forecast_id"] = forecast_id
    recommendation["target_date"] = target_date
    recommendation["publication_type"] = (
        "official_daily_forecast"
    )

    model_status = {
        "status": "success",
        "forecast_id": forecast_id,
        "target_date": target_date,
        "publication_type": (
            "official_daily_forecast"
        ),
        "published_at_local": (
            published_at_local.isoformat()
        ),
        "location": (
            "Melbourne, Victoria, Australia"
        ),
        "network": "NEM",
        "network_region": "VIC1",
        "model": "GradientBoostingRegressor",
        "random_state": 42,
        "training_history_start_local": (
            training_history[TIME_COL]
            .min()
            .isoformat()
        ),
        "training_history_end_local": (
            training_history[TIME_COL]
            .max()
            .isoformat()
        ),
        "training_observation_rows": int(
            training_history[TARGET_COL]
            .notna()
            .sum()
        ),
        "design_rows": int(
            len(design)
        ),
        "features": features,
        "forecast_horizon_hours": 24,
        "forecast_start_local": (
            forecast[TIME_COL]
            .min()
            .isoformat()
        ),
        "forecast_end_local": (
            (
                forecast[TIME_COL].max()
                + pd.Timedelta(hours=1)
            )
            .isoformat()
        ),
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

    today_forecast_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe_for_csv(
        forecast
    ).to_csv(
        today_forecast_file,
        index=False,
    )

    archive_file = (
        archive_dir
        / f"{target_date}.csv"
    )

    archive_action = publish_archive(
        forecast=forecast,
        archive_file=archive_file,
    )

    recommendation["archive_action"] = (
        archive_action
    )

    recommendation["archive_file"] = (
        str(archive_file)
    )

    model_status["archive_action"] = (
        archive_action
    )

    model_status["archive_file"] = (
        str(archive_file)
    )

    save_json(
        recommendation,
        recommendation_file,
    )

    save_json(
        model_status,
        forecast_status_file,
    )

    print(
        "\n=============================================="
    )
    print(
        " Daily forecast published successfully"
    )
    print(
        "=============================================="
    )

    print(
        f"\nForecast ID:\n  {forecast_id}"
    )

    print(
        f"\nCurrent forecast file:\n  "
        f"{today_forecast_file}"
    )

    print(
        f"\nOfficial archive:\n  "
        f"{archive_file}"
    )

    print(
        f"\nRecommendation file:\n  "
        f"{recommendation_file}"
    )

    print(
        f"\nForecast status file:\n  "
        f"{forecast_status_file}"
    )

    print(
        "\nCleanest charging window:"
    )

    print(
        "  "
        f"{recommendation['best_window_start_local']} "
        "to "
        f"{recommendation['best_window_end_local']}"
    )

    print(
        "Mean forecast intensity:"
    )

    print(
        "  "
        f"{recommendation['best_window_mean_intensity_g_per_kwh']:.1f} "
        "gCO2/kWh"
    )

    print(
        "Estimated emissions saving:"
    )

    print(
        "  "
        f"{recommendation['emissions_saving_kg']:.2f} "
        "kg CO2 "
        f"({recommendation['emissions_saving_percent']:.1f}%)"
    )

    print()

    return outputs


# ---------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Publish one official CleanCharge Live forecast "
            "for a Melbourne calendar day."
        )
    )

    parser.add_argument(
        "--history-file",
        type=Path,
        default=DEFAULT_HISTORY_FILE,
        help="Rolling Victoria history CSV.",
    )

    parser.add_argument(
        "--target-date",
        type=str,
        default=None,
        help=(
            "Forecast date in YYYY-MM-DD format. "
            "Default: today's Melbourne date."
        ),
    )

    parser.add_argument(
        "--need-kwh",
        type=float,
        default=20.0,
        help="Representative charging energy. Default: 20 kWh.",
    )

    parser.add_argument(
        "--charger-kw",
        type=float,
        default=7.0,
        help="Representative charger power. Default: 7 kW.",
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

    publish_daily_forecast(
        history_file=args.history_file,
        target_date_text=args.target_date,
        need_kwh=args.need_kwh,
        charger_kw=args.charger_kw,
        backtest_days=args.backtest_days,
    )


if __name__ == "__main__":
    main()