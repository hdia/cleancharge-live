"""
Run the complete CleanCharge Live daily operational pipeline.

Normal daily sequence
---------------------
1. Update the rolling Victoria observation history.
2. Evaluate yesterday's archived forecast, when available.
3. Publish today's official forecast.
4. Rebuild the rolling scorecard and system status.

Run from the repository root:

    python run_daily_pipeline.py

Useful options:

    python run_daily_pipeline.py --dry-run

    python run_daily_pipeline.py --skip-publish

    python run_daily_pipeline.py --allow-late-publication
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent

NEM_TZ_NAME = "NEM time (UTC+10)"
NEM_TZ = timezone(timedelta(hours=10))

LIVE_DIR = ROOT / "data" / "live"

HISTORY_FILE = LIVE_DIR / "vic_intensity_history.csv"

FORECAST_ARCHIVE_DIR = (
    LIVE_DIR
    / "forecasts"
    / "archive"
)

VALIDATION_DIR = (
    LIVE_DIR
    / "validation"
)

PIPELINE_STATUS_FILE = (
    LIVE_DIR
    / "scorecard"
    / "pipeline_status.json"
)

DEFAULT_LATEST_PUBLICATION_HOUR = 3


# ---------------------------------------------------------------------
# Helpers
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


def run_command(
    command: list[str],
    label: str,
    dry_run: bool = False,
) -> None:
    """
    Run one pipeline command and stop immediately if it fails.
    """
    display_command = " ".join(command)

    print()
    print("==============================================")
    print(f" {label}")
    print("==============================================")
    print()
    print(f">> Command: {display_command}")

    if dry_run:
        print(">> Dry-run mode. Command was not executed.")
        return

    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
    )

    if completed.returncode != 0:
        raise RuntimeError(
            f"{label} failed with exit code "
            f"{completed.returncode}."
        )


def forecast_archive_file(
    target_date: str,
) -> Path:
    return (
        FORECAST_ARCHIVE_DIR
        / f"{target_date}.csv"
    )


def evaluation_summary_file(
    target_date: str,
) -> Path:
    return (
        VALIDATION_DIR
        / target_date
        / "evaluation_summary.json"
    )

def inspect_data_availability(
    reference_time: datetime,
) -> dict[str, Any]:
    """
    Inspect the latest observation available in the rolling history.

    Times are evaluated in fixed NEM time, UTC+10.
    """
    result: dict[str, Any] = {
        "history_file": str(HISTORY_FILE),
        "history_available": False,
        "latest_observation_nem": None,
        "data_lag_minutes": None,
        "previous_day_complete": False,
        "history_rows": 0,
    }

    if not HISTORY_FILE.exists():
        result["warning"] = "History file does not exist."
        return result

    history = pd.read_csv(HISTORY_FILE)

    result["history_rows"] = int(len(history))

    if history.empty:
        result["warning"] = "History file is empty."
        return result

    if "local_time" not in history.columns:
        result["warning"] = (
            "History file does not contain a local_time column."
        )
        return result

    local_times = pd.to_datetime(
        history["local_time"],
        errors="coerce",
        utc=True,
    )

    local_times = local_times.dropna()

    if local_times.empty:
        result["warning"] = (
            "No valid timestamps were found in local_time."
        )
        return result

    latest_utc = local_times.max()
    latest_nem = latest_utc.tz_convert("Etc/GMT-10")

    reference_utc = pd.Timestamp(reference_time).tz_convert("UTC")

    lag_minutes = (
        reference_utc - latest_utc
    ).total_seconds() / 60.0

    today_nem = reference_time.date()

    previous_day_complete = (
        latest_nem.date() >= today_nem
    )

    result.update(
        {
            "history_available": True,
            "latest_observation_nem": (
                latest_nem.isoformat()
            ),
            "data_lag_minutes": round(
                max(lag_minutes, 0.0),
                1,
            ),
            "previous_day_complete": bool(
                previous_day_complete
            ),
        }
    )

    return result

# ---------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------

def run_pipeline(
    dry_run: bool = False,
    skip_update: bool = False,
    skip_evaluation: bool = False,
    skip_publish: bool = False,
    skip_scorecard: bool = False,
    allow_late_publication: bool = False,
    latest_publication_hour: int = (
        DEFAULT_LATEST_PUBLICATION_HOUR
    ),
) -> None:
    started_at = datetime.now(
        NEM_TZ
    )

    today = started_at.date()


    yesterday = (
    today
    - timedelta(days=1)
    )

    today_text = today.isoformat()
    yesterday_text = yesterday.isoformat()

    status: dict[str, Any] = {
        "status": "running",
        "time_standard": NEM_TZ_NAME,
        "started_at_local": started_at.isoformat(),
        "today": today_text,
        "yesterday": yesterday_text,
        "steps": {},
    }

    save_json(
        status,
        PIPELINE_STATUS_FILE,
    )

    try:
        print()
        print("==============================================")
        print(" CleanCharge Live daily pipeline")
        print("==============================================")
        print()
        print(
            f">> NEM date: {today_text}"
        )
        print(
            f">> Yesterday: {yesterday_text}"
        )
        print(
            f">> Started: {started_at.isoformat()}"
        )

        # -------------------------------------------------------------
        # 1. Update rolling observation history
        # -------------------------------------------------------------

        if skip_update:
            print(
                "\n>> Observation update skipped."
            )
            status["steps"]["history_update"] = (
                "skipped"
            )
        else:
            run_command(
                command=[
                    sys.executable,
                    "daily_update.py",
                    "--skip-forecast",
                ],
                label=(
                    "Step 1: Update rolling observation history"
                ),
                dry_run=dry_run,
            )

            status["steps"]["history_update"] = (
                "success"
                if not dry_run
                else "dry_run"
            )

            if not dry_run:
                availability = inspect_data_availability(
                    datetime.now(NEM_TZ)
                )

                status["data_availability"] = availability

                print()
                print(">> Data availability check")

                if availability["history_available"]:
                    print(
                        ">> Latest observation: "
                        f"{availability['latest_observation_nem']}"
                    )
                    print(
                        ">> Data lag: "
                        f"{availability['data_lag_minutes']} minutes"
                    )
                    print(
                        ">> Previous calendar day complete: "
                        f"{availability['previous_day_complete']}"
                    )
                else:
                    print(
                        ">> Data availability could not be determined."
                    )

        # -------------------------------------------------------------
        # 2. Evaluate yesterday's forecast
        # -------------------------------------------------------------

        yesterday_archive = (
            forecast_archive_file(
                yesterday_text
            )
        )

        yesterday_summary = (
            evaluation_summary_file(
                yesterday_text
            )
        )

        if skip_evaluation:
            print(
                "\n>> Yesterday's evaluation skipped."
            )
            status["steps"][
                "yesterday_evaluation"
            ] = "skipped"

        elif yesterday_summary.exists():
            print(
                "\n>> Yesterday's evaluation already "
                "exists and was preserved:"
            )
            print(
                f"   {yesterday_summary}"
            )

            status["steps"][
                "yesterday_evaluation"
            ] = "already_complete"

        elif not yesterday_archive.exists():
            print(
                "\n>> No archived forecast exists for "
                f"{yesterday_text}."
            )
            print(
                ">> Yesterday cannot be evaluated."
            )

            status["steps"][
                "yesterday_evaluation"
            ] = "no_archived_forecast"

        else:
            run_command(
                command=[
                    sys.executable,
                    "-m",
                    "src.live.evaluate_scientific",
                    "--target-date",
                    yesterday_text,
                ],
                label=(
                    "Step 2A: Scientific evaluation "
                    f"for {yesterday_text}"
                ),
                dry_run=dry_run,
            )

            run_command(
                command=[
                    sys.executable,
                    "-m",
                    "src.live.evaluate_decision",
                    "--target-date",
                    yesterday_text,
                ],
                label=(
                    "Step 2B: Decision evaluation "
                    f"for {yesterday_text}"
                ),
                dry_run=dry_run,
            )

            status["steps"][
                "yesterday_evaluation"
            ] = (
                "success"
                if not dry_run
                else "dry_run"
            )

        # -------------------------------------------------------------
        # 3. Publish today's official forecast
        # -------------------------------------------------------------

        today_archive = (
            forecast_archive_file(
                today_text
            )
        )

        publication_is_late = (
            started_at.hour
            > latest_publication_hour
        )

        if skip_publish:
            print(
                "\n>> Today's forecast publication skipped."
            )
            status["steps"][
                "today_publication"
            ] = "skipped"

        elif today_archive.exists():
            print(
                "\n>> Today's official forecast already "
                "exists and was preserved:"
            )
            print(
                f"   {today_archive}"
            )

            status["steps"][
                "today_publication"
            ] = "already_published"

        elif (
            publication_is_late
            and not allow_late_publication
        ):
            print()
            print(
                ">> Today's forecast was not published "
                "because the normal publication window "
                "has passed."
            )
            print(
                f">> Current NEM hour: "
                f"{started_at.hour}"
            )
            print(
                f">> Latest permitted hour: "
                f"{latest_publication_hour}"
            )
            print(
                ">> Use --allow-late-publication only "
                "for development or recovery."
            )

            status["steps"][
                "today_publication"
            ] = "late_publication_blocked"

        else:
            run_command(
                command=[
                    sys.executable,
                    "-m",
                    "src.live.publish_daily",
                    "--target-date",
                    today_text,
                ],
                label=(
                    "Step 3: Publish today's official forecast"
                ),
                dry_run=dry_run,
            )

            status["steps"][
                "today_publication"
            ] = (
                "late_success"
                if publication_is_late
                else "success"
            )

        # -------------------------------------------------------------
        # 4. Rebuild scorecard
        # -------------------------------------------------------------

        if skip_scorecard:
            print(
                "\n>> Scorecard rebuild skipped."
            )
            status["steps"][
                "scorecard"
            ] = "skipped"
        else:
            run_command(
                command=[
                    sys.executable,
                    "-m",
                    "src.live.build_scorecard",
                ],
                label=(
                    "Step 4: Rebuild rolling scorecard"
                ),
                dry_run=dry_run,
            )

            status["steps"][
                "scorecard"
            ] = (
                "success"
                if not dry_run
                else "dry_run"
            )

        completed_at = datetime.now(
            NEM_TZ
        )

        status["status"] = (
            "success"
            if not dry_run
            else "dry_run"
        )

        status["completed_at_local"] = (
            completed_at.isoformat()
        )

        status["duration_seconds"] = round(
            (
                completed_at
                - started_at
            ).total_seconds(),
            2,
        )

        save_json(
            status,
            PIPELINE_STATUS_FILE,
        )

        print()
        print("==============================================")
        print(" Daily pipeline completed successfully")
        print("==============================================")
        print()
        print(
            f"Pipeline status:\n  "
            f"{PIPELINE_STATUS_FILE}"
        )
        print()

    except Exception as exc:
        failed_at = datetime.now(
            NEM_TZ
        )

        status["status"] = "failed"
        status["failed_at_local"] = (
            failed_at.isoformat()
        )
        status["error"] = str(exc)

        save_json(
            status,
            PIPELINE_STATUS_FILE,
        )

        print()
        print("==============================================")
        print(" Daily pipeline failed")
        print("==============================================")
        print()
        print(f"Error: {exc}")
        print(
            f"\nPipeline status:\n  "
            f"{PIPELINE_STATUS_FILE}"
        )

        raise


# ---------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete CleanCharge Live "
            "daily operational pipeline."
        )
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Show the commands without executing them."
        ),
    )

    parser.add_argument(
        "--skip-update",
        action="store_true",
    )

    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
    )

    parser.add_argument(
        "--skip-publish",
        action="store_true",
    )

    parser.add_argument(
        "--skip-scorecard",
        action="store_true",
    )

    parser.add_argument(
        "--allow-late-publication",
        action="store_true",
        help=(
            "Allow today's forecast to be published "
            "after the normal publication window. "
            "Use only for development or recovery."
        ),
    )

    parser.add_argument(
        "--latest-publication-hour",
        type=int,
        default=DEFAULT_LATEST_PUBLICATION_HOUR,
        help=(
            "Latest Melbourne hour at which an official "
            "daily forecast may normally be published. "
            "Default: 3."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    run_pipeline(
        dry_run=args.dry_run,
        skip_update=args.skip_update,
        skip_evaluation=args.skip_evaluation,
        skip_publish=args.skip_publish,
        skip_scorecard=args.skip_scorecard,
        allow_late_publication=(
            args.allow_late_publication
        ),
        latest_publication_hour=(
            args.latest_publication_hour
        ),
    )


if __name__ == "__main__":
    main()