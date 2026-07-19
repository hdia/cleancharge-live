"""
One-off CleanCharge Live backfill for missed forecast dates.

Run from the repository root:
    python backfill_missing_dates.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from src.live.publish_daily import publish_daily_forecast


ROOT = Path(__file__).resolve().parent
LIVE_DIR = ROOT / "data" / "live"
ARCHIVE_DIR = LIVE_DIR / "forecasts" / "archive"
TEMP_DIR = LIVE_DIR / "_backfill_temp"

DATES_TO_BACKFILL = [
    "2026-07-16",
    "2026-07-17",
    "2026-07-18",
]

TZ_NAME = "Australia/Melbourne"


def run_module(module: str, *args: str) -> None:
    command = [sys.executable, "-m", module, *args]
    print("\n>> " + " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def mark_archive_as_backfill(archive_file: Path) -> None:
    frame = pd.read_csv(archive_file)
    frame["publication_type"] = "retrospective_backfill"
    frame["reconstructed_at_local"] = datetime.now(
        ZoneInfo(TZ_NAME)
    ).isoformat()
    frame.to_csv(archive_file, index=False)


def main() -> None:
    print("\n==============================================")
    print(" CleanCharge Live missing-date backfill")
    print("==============================================\n")

    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    created_dates = []
    preserved_dates = []

    try:
        for target_date in DATES_TO_BACKFILL:
            archive_file = ARCHIVE_DIR / f"{target_date}.csv"
            print(f"\n--- Processing {target_date} ---")

            if archive_file.exists():
                print(">> Archive already exists; preserving it.")
                preserved_dates.append(target_date)
            else:
                date_temp_dir = TEMP_DIR / target_date

                publish_daily_forecast(
                    target_date_text=target_date,
                    today_forecast_file=date_temp_dir / "forecast.csv",
                    archive_dir=ARCHIVE_DIR,
                    recommendation_file=date_temp_dir / "recommendation.json",
                    forecast_status_file=date_temp_dir / "forecast_status.json",
                )

                if not archive_file.exists():
                    raise FileNotFoundError(
                        f"Archive was not created: {archive_file}"
                    )

                mark_archive_as_backfill(archive_file)
                created_dates.append(target_date)

            run_module(
                "src.live.evaluate_scientific",
                "--target-date",
                target_date,
            )

            run_module(
                "src.live.evaluate_decision",
                "--target-date",
                target_date,
            )

        run_module("src.live.build_scorecard")

    finally:
        if TEMP_DIR.exists():
            shutil.rmtree(TEMP_DIR)

    print("\n==============================================")
    print(" Backfill completed successfully")
    print("==============================================")

    print("\nCreated forecast archives:")
    for target_date in created_dates:
        print(f"  - {target_date}")
    if not created_dates:
        print("  None")

    print("\nPreserved existing archives:")
    for target_date in preserved_dates:
        print(f"  - {target_date}")
    if not preserved_dates:
        print("  None")

    print("\nThe current live forecast files were not overwritten.")
    print("\nNext steps:")
    print("  1. Run: git status")
    print("  2. Run: python sync_github.py")
    print()


if __name__ == "__main__":
    main()
