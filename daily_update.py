"""
CleanCharge Live daily history updater.

Initial responsibilities:
1. Fetch recent Victoria-region electricity observations.
2. Create or update a rolling hourly history file.
3. Remove duplicate timestamps.
4. Retain the latest configured history window.
5. Report coverage and missing hourly observations.
6. Save update metadata for the future CleanCharge Live dashboard.

Run from the repository root:

    python daily_update.py

Useful options:

    python daily_update.py --bootstrap-days 90
    python daily_update.py --hours-back 48
    python daily_update.py --history-days 90
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.live.forecast_live import run_live_forecast


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent

DEFAULT_FETCH_SCRIPT = (
    ROOT / "src" / "fetch" / "fetch_openelectricity.py"
)

DEFAULT_LIVE_DIR = ROOT / "data" / "live"

DEFAULT_HISTORY_FILE = (
    DEFAULT_LIVE_DIR / "vic_intensity_history.csv"
)

DEFAULT_LATEST_FILE = (
    DEFAULT_LIVE_DIR / "vic_latest_observations.csv"
)

DEFAULT_STATUS_FILE = (
    DEFAULT_LIVE_DIR / "update_status.json"
)

TZ_LOCAL = "Australia/Melbourne"

REQUIRED_COLUMNS = {
    "ts",
    "local_time",
    "intensity",
    "emissions",
    "energy_mwh",
}


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def load_fetch_module(fetch_script: Path):
    """
    Load the existing confirmed working OpenElectricity fetcher directly
    from its file path.

    This avoids requiring __init__.py files inside src/ and src/fetch/.
    """
    if not fetch_script.exists():
        raise FileNotFoundError(
            f"Fetch script not found: {fetch_script}"
        )

    spec = importlib.util.spec_from_file_location(
        "cleancharge_fetch_openelectricity",
        fetch_script,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load fetch script: {fetch_script}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "normalise_and_intensity"):
        raise AttributeError(
            "The fetch script does not expose "
            "'normalise_and_intensity'."
        )

    return module


def normalise_observations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and standardise fetched or previously stored observations.
    """
    if df.empty:
        raise ValueError("Observation dataframe is empty.")

    missing = REQUIRED_COLUMNS.difference(df.columns)

    if missing:
        raise ValueError(
            "Observation data are missing required columns: "
            + ", ".join(sorted(missing))
        )

    out = df.copy()

    out["ts"] = pd.to_datetime(
        out["ts"],
        utc=True,
        errors="coerce",
    )

    out["local_time"] = (
        out["ts"]
        .dt.tz_convert(TZ_LOCAL)
    )

    for col in [
        "intensity",
        "emissions",
        "energy_mwh",
    ]:
        out[col] = pd.to_numeric(
            out[col],
            errors="coerce",
        )

    out = out.dropna(
        subset=[
            "ts",
            "local_time",
            "intensity",
            "emissions",
            "energy_mwh",
        ]
    )

    out = (
        out.sort_values("ts")
        .drop_duplicates(
            subset=["ts"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    return out


def load_existing_history(
    history_file: Path,
) -> pd.DataFrame:
    """
    Load the existing rolling history if it exists.
    """
    if not history_file.exists():
        return pd.DataFrame(
            columns=sorted(REQUIRED_COLUMNS)
        )

    print(
        f">> Loading existing history: {history_file}"
    )

    existing = pd.read_csv(history_file)

    if existing.empty:
        return pd.DataFrame(
            columns=sorted(REQUIRED_COLUMNS)
        )

    return normalise_observations(existing)


def merge_history(
    existing: pd.DataFrame,
    fetched: pd.DataFrame,
    history_days: int,
) -> pd.DataFrame:
    """
    Merge old and newly fetched observations, remove duplicates,
    and retain only the latest history window.
    """
    frames = []

    if not existing.empty:
        frames.append(existing)

    if not fetched.empty:
        frames.append(fetched)

    if not frames:
        raise ValueError(
            "No existing or newly fetched observations available."
        )

    merged = pd.concat(
        frames,
        ignore_index=True,
    )

    merged = normalise_observations(merged)

    latest_ts = merged["ts"].max()

    cutoff = latest_ts - pd.Timedelta(
        days=history_days
    )

    merged = (
        merged[merged["ts"] >= cutoff]
        .sort_values("ts")
        .drop_duplicates(
            subset=["ts"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    return merged


def coverage_summary(
    history: pd.DataFrame,
) -> dict:
    """
    Calculate basic completeness statistics for the hourly history.
    """
    if history.empty:
        return {
            "rows": 0,
            "first_timestamp_utc": None,
            "last_timestamp_utc": None,
            "first_timestamp_local": None,
            "last_timestamp_local": None,
            "expected_hourly_rows": 0,
            "missing_hourly_rows": 0,
            "coverage_percent": 0.0,
        }

    first_ts = history["ts"].min()
    last_ts = history["ts"].max()

    expected_index = pd.date_range(
        start=first_ts.floor("h"),
        end=last_ts.floor("h"),
        freq="h",
        tz="UTC",
    )

    observed_index = pd.DatetimeIndex(
        history["ts"].dt.floor("h").unique()
    )

    missing_index = expected_index.difference(
        observed_index
    )

    expected_rows = len(expected_index)
    observed_rows = len(observed_index)

    coverage_percent = (
        100.0 * observed_rows / expected_rows
        if expected_rows
        else 0.0
    )

    return {
        "rows": int(len(history)),
        "first_timestamp_utc": first_ts.isoformat(),
        "last_timestamp_utc": last_ts.isoformat(),
        "first_timestamp_local": (
            first_ts
            .tz_convert(TZ_LOCAL)
            .isoformat()
        ),
        "last_timestamp_local": (
            last_ts
            .tz_convert(TZ_LOCAL)
            .isoformat()
        ),
        "expected_hourly_rows": int(expected_rows),
        "observed_hourly_rows": int(observed_rows),
        "missing_hourly_rows": int(len(missing_index)),
        "coverage_percent": round(
            coverage_percent,
            2,
        ),
        "missing_timestamps_utc": [
            ts.isoformat()
            for ts in missing_index[:100]
        ],
    }


def save_outputs(
    fetched: pd.DataFrame,
    history: pd.DataFrame,
    history_file: Path,
    latest_file: Path,
    status_file: Path,
    status: dict,
) -> None:
    """
    Save live observations, rolling history, and update metadata.
    """
    history_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fetched.to_csv(
        latest_file,
        index=False,
    )

    history.to_csv(
        history_file,
        index=False,
    )

    with status_file.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            status,
            f,
            indent=2,
        )


# ---------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fetch Victoria OpenElectricity observations "
            "and maintain a rolling CleanCharge Live history."
        )
    )

    parser.add_argument(
        "--fetch-script",
        type=Path,
        default=DEFAULT_FETCH_SCRIPT,
        help=(
            "Path to the confirmed working "
            "fetch_openelectricity.py script."
        ),
    )

    parser.add_argument(
        "--history-file",
        type=Path,
        default=DEFAULT_HISTORY_FILE,
        help="Rolling history CSV output path.",
    )

    parser.add_argument(
        "--latest-file",
        type=Path,
        default=DEFAULT_LATEST_FILE,
        help="Latest fetched observations CSV path.",
    )

    parser.add_argument(
        "--status-file",
        type=Path,
        default=DEFAULT_STATUS_FILE,
        help="JSON update-status output path.",
    )

    parser.add_argument(
        "--history-days",
        type=int,
        default=90,
        help=(
            "Number of days retained in the rolling history. "
            "Default: 90."
        ),
    )

    parser.add_argument(
        "--bootstrap-days",
        type=int,
        default=90,
        help=(
            "Number of days requested when no history file exists. "
            "Default: 90."
        ),
    )

    parser.add_argument(
        "--hours-back",
        type=int,
        default=48,
        help=(
            "Number of recent hours requested when history already "
            "exists. A 48-hour overlap protects against delayed or "
            "revised API records. Default: 48."
        ),
    )

    parser.add_argument(
        "--interval",
        type=str,
        default="1h",
        help="OpenElectricity interval. Default: 1h.",
    )


    parser.add_argument(
        "--skip-forecast",
        action="store_true",
        help=(
            "Update the rolling history without running model "
            "validation or generating a new forecast."
        ),
    )

    parser.add_argument(
        "--need-kwh",
        type=float,
        default=20.0,
        help=(
            "Representative charging energy used for the daily "
            "recommendation. Default: 20 kWh."
        ),
    )

    parser.add_argument(
        "--charger-kw",
        type=float,
        default=7.0,
        help=(
            "Representative charger power used for the daily "
            "recommendation. Default: 7 kW."
        ),
    )

    parser.add_argument(
        "--backtest-days",
        type=int,
        default=7,
        help=(
            "Number of recent days used for walk-forward model "
            "validation. Default: 7."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print(
        "\n=============================================="
    )
    print(
        " CleanCharge Live daily history update"
    )
    print(
        "==============================================\n"
    )

    history_exists = args.history_file.exists()

    if history_exists:
        fetch_hours = args.hours_back
        mode = "incremental update"
    else:
        fetch_hours = args.bootstrap_days * 24
        mode = "initial bootstrap"

    print(f">> Mode: {mode}")
    print(
        f">> Fetch period requested: "
        f"{fetch_hours} hours"
    )
    print(
        f">> Rolling history retained: "
        f"{args.history_days} days"
    )

    fetch_module = load_fetch_module(
        args.fetch_script
    )

    print(
        "\n>> Fetching Victoria-region observations..."
    )

    fetched = fetch_module.normalise_and_intensity(
        network_code="NEM",
        hours_back=fetch_hours,
        interval=args.interval,
    )

    fetched = normalise_observations(fetched)

    print(
        f">> Retrieved {len(fetched)} valid hourly rows."
    )

    existing = load_existing_history(
        args.history_file
    )

    print(
        f">> Existing history rows: {len(existing)}"
    )

    history = merge_history(
        existing=existing,
        fetched=fetched,
        history_days=args.history_days,
    )

    summary = coverage_summary(history)

    update_time_utc = datetime.now(
        timezone.utc
    )

    update_time_local = (
        pd.Timestamp(update_time_utc)
        .tz_convert(TZ_LOCAL)
    )

    status = {
        "status": "success",
        "update_time_utc": (
            update_time_utc.isoformat()
        ),
        "update_time_local": (
            update_time_local.isoformat()
        ),
        "mode": mode,
        "network": "NEM",
        "network_region": "VIC1",
        "primary_grouping": "network_region",
        "interval": args.interval,
        "fetch_hours_requested": int(
            fetch_hours
        ),
        "history_days_retained": int(
            args.history_days
        ),
        "latest_fetch_rows": int(
            len(fetched)
        ),
        "history": summary,
    }

    save_outputs(
        fetched=fetched,
        history=history,
        history_file=args.history_file,
        latest_file=args.latest_file,
        status_file=args.status_file,
        status=status,
    )

    forecast_outputs = None

    if args.skip_forecast:
        print(
            "\n>> Forecast stage skipped because "
            "--skip-forecast was supplied."
        )
    else:
        print(
            "\n>> Starting CleanCharge Live forecasting stage..."
        )

        try:
            forecast_outputs = run_live_forecast(
                history_file=args.history_file,
                need_kwh=args.need_kwh,
                charger_kw=args.charger_kw,
                backtest_days=args.backtest_days,
            )

            status["forecast_status"] = "success"

            status["forecast_generated_at_local"] = (
                forecast_outputs.model_status[
                    "generated_at_local"
                ]
            )

            status["forecast_start_local"] = (
                forecast_outputs.recommendation[
                    "forecast_start_local"
                ]
            )

            status["forecast_end_local"] = (
                forecast_outputs.recommendation[
                    "forecast_end_local"
                ]
            )

            status["recommended_window_start_local"] = (
                forecast_outputs.recommendation[
                    "best_window_start_local"
                ]
            )

            status["recommended_window_end_local"] = (
                forecast_outputs.recommendation[
                    "best_window_end_local"
                ]
            )

            status["forecast_validation"] = (
                forecast_outputs.model_status[
                    "backtest"
                ]
            )

        except Exception as exc:
            status["forecast_status"] = "failed"
            status["forecast_error"] = str(exc)

            with args.status_file.open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    status,
                    file,
                    indent=2,
                )

            raise RuntimeError(
                "The history update succeeded, but the "
                "forecasting stage failed."
            ) from exc

        with args.status_file.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                status,
                file,
                indent=2,
            )

    print(
        "\n=============================================="
    )
    print(
        " Update completed successfully"
    )
    print(
        "=============================================="
    )

    print(
        f"\nHistory file:\n  {args.history_file}"
    )

    print(
        f"\nLatest observations:\n  {args.latest_file}"
    )

    print(
        f"\nStatus metadata:\n  {args.status_file}"
    )

    print(
        f"\nHistory rows: "
        f"{summary['rows']}"
    )

    print(
        f"Coverage: "
        f"{summary['coverage_percent']:.2f}%"
    )

    print(
        f"Missing hourly observations: "
        f"{summary['missing_hourly_rows']}"
    )

    print(
        f"History starts: "
        f"{summary['first_timestamp_local']}"
    )

    print(
        f"History ends: "
        f"{summary['last_timestamp_local']}"
    )

    if summary["missing_hourly_rows"] > 0:
        print(
            "\n!! The history contains missing hourly "
            "timestamps. These are reported in:"
        )
        print(
            f"   {args.status_file}"
        )

    if forecast_outputs is not None:
        recommendation = forecast_outputs.recommendation
        validation = forecast_outputs.model_status["backtest"]

        print(
            "\nForecasting stage:"
        )

        print(
            "  Status: success"
        )

        print(
            "  Forecast period: "
            f"{recommendation['forecast_start_local']} "
            "to "
            f"{recommendation['forecast_end_local']}"
        )

        print(
            "  Recommended window: "
            f"{recommendation['best_window_start_local']} "
            "to "
            f"{recommendation['best_window_end_local']}"
        )

        print(
            "  Mean window intensity: "
            f"{recommendation['best_window_mean_intensity_g_per_kwh']:.1f} "
            "gCO2/kWh"
        )

        print(
            "  Estimated emissions saving: "
            f"{recommendation['emissions_saving_kg']:.2f} "
            "kg CO2 "
            f"({recommendation['emissions_saving_percent']:.1f}%)"
        )

        if validation["mae_g_per_kwh"] is not None:
            print(
                "  Seven-day validation MAE: "
                f"{validation['mae_g_per_kwh']:.2f} "
                "gCO2/kWh"
            )

            print(
                "  Seven-day validation R2: "
                f"{validation['r2']:.3f}"
            )

    print()


if __name__ == "__main__":
    main()