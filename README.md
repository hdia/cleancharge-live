# CleanCharge Live

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-operational-brightgreen.svg)](#system-status)

**Daily carbon-aware electric vehicle charging guidance for Victoria, Australia.**

CleanCharge Live is an operational research and decision-support platform. It retrieves recent Victorian electricity-system data, maintains a rolling observation history, forecasts regional grid carbon intensity, publishes one lower-carbon charging period for a representative electric vehicle charging scenario, and evaluates its own performance as actual observations become available.

It is the live operational companion to the published [CleanCharge](https://github.com/hdia/cleancharge) research toolkit.

---

## Live dashboard

Try the CleanCharge Live dashboard:

**https://cleancharge-live.streamlit.app/**

---

## Overview

CleanCharge Live:

- retrieves hourly Victorian electricity and emissions data from OpenElectricity
- maintains a rolling 90-day observation history with coverage checks
- forecasts regional electricity carbon intensity for the next operational day
- publishes one model-selected lower-carbon charging period for a representative charging scenario
- archives each formally published daily forecast without overwriting it
- evaluates forecast accuracy when actual observations become available
- evaluates the realised quality of the published charging decision
- maintains rolling performance scorecards
- records operational health and data-freshness metadata
- supports an interactive Streamlit dashboard

The current public implementation uses a representative **20 kWh charging session with a 7 kW charger**, requiring approximately three hourly forecast intervals.

CleanCharge Live v1 displays one primary charging period for clarity. Adjacent or alternative start times may sometimes have similar forecast emissions, but they are not currently presented as separate recommendations.

The operational pipeline uses the **Australia/Melbourne** timezone for forecast dates, publication, evaluation and dashboard display. The UTC offset therefore changes automatically between standard time and daylight saving time.

> **Research and decision-support system.** Forecasts and charging guidance are model based, subject to uncertainty and not guaranteed operational advice.

---

## Relationship to the published CleanCharge study

The original CleanCharge repository contains the reproducible research workflow, archived study datasets, infrastructure and equity analysis, charging scenarios, figures, and CleanCharge Explorer dashboard.

This repository contains the operational forecasting and evaluation system developed as a live extension of that work.

The scientific basis is described in:

> **Dia, H. (2026). _CleanCharge: Emissions-aware electric vehicle charging and infrastructure equity with open data in Melbourne._ International Journal of Sustainable Transportation, 1–27.**  
> https://doi.org/10.1080/15568318.2026.2693676

---

## System workflow

```text
OpenElectricity API
        |
        v
Rolling Victorian observation history
        |
        v
Daily carbon-intensity forecast
        |
        v
Published low-carbon charging period
        |
        +------------------------+
        |                        |
        v                        v
Scientific forecast        Decision-quality
evaluation                 evaluation
        |                        |
        +-----------+------------+
                    |
                    v
          Rolling scorecard and
          operational health status
                    |
                    v
            Streamlit dashboard
```

The daily orchestrator is `run_daily_pipeline.py`. It:

1. updates the rolling observation history
2. evaluates the previous day when an archived forecast is available
3. publishes the current day's forecast and charging period
4. rebuilds the rolling scorecard and system status

Each published forecast is assigned a permanent identifier and retained for transparent later evaluation.

---

## What the charging recommendation means

The displayed charging period is the contiguous period with the lowest forecast mean carbon intensity for the representative charging scenario.

It should be interpreted as:

- forecast-based guidance for regional Victorian grid conditions
- a comparison between feasible charging periods within the published forecast
- one primary model-selected period rather than a guarantee that nearby periods will be materially worse
- a research demonstration rather than personalised charging advice

The recommendation does not account for a user's tariff, rooftop solar, household battery, departure time, vehicle state of charge, charger availability or local network conditions.

---

## Repository structure

```text
cleancharge-live/
|-- daily_update.py
|-- live_app.py
|-- run_daily_pipeline.py
|-- data/
|   `-- live/
|       |-- vic_intensity_history.csv
|       |-- vic_latest_observations.csv
|       |-- intensity_forecast_next24.csv
|       |-- intensity_backtest_last7d.csv
|       |-- forecast_archive/
|       |-- forecasts/archive/
|       |-- status/
|       |-- validation/
|       `-- scorecard/
|-- src/
|   |-- fetch/fetch_openelectricity.py
|   `-- live/
|       |-- forecast_live.py
|       |-- publish_daily.py
|       |-- evaluate_scientific.py
|       |-- evaluate_decision.py
|       `-- build_scorecard.py
|-- requirements.txt
|-- runtime.txt
|-- .gitignore
|-- LICENSE
`-- README.md
```

Two forecast archives are retained deliberately:

- `data/live/forecast_archive/` stores immutable raw model outputs
- `data/live/forecasts/archive/` stores formally published daily forecasts used by the evaluation pipeline

---

## Quick start

### 1. Clone the repository

```bash
git clone https://github.com/hdia/cleancharge-live.git
cd cleancharge-live
```

### 2. Create and activate a Python environment

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure the OpenElectricity API key

Create a local `.env` file in the repository root:

```text
OPENELECTRICITY_API_KEY=your_api_key_here
```

The `.env` file is excluded from Git and must never be committed.

### 4. Test without publishing

```powershell
python run_daily_pipeline.py --skip-publish
```

### 5. Run the complete daily pipeline

```powershell
python run_daily_pipeline.py
```

### 6. Launch the dashboard

```powershell
streamlit run live_app.py
```

---

## Operational outputs

| Output | Purpose |
|---|---|
| `data/live/vic_intensity_history.csv` | Rolling 90-day Victorian observation history |
| `data/live/vic_latest_observations.csv` | Most recent retrieved observations |
| `data/live/intensity_forecast_next24.csv` | Current model forecast |
| `data/live/forecasts/today.csv` | Canonical published daily forecast |
| `data/live/status/today_recommendation.json` | Current model-selected charging period |
| `data/live/validation/YYYY-MM-DD/` | Scientific and decision evaluation |
| `data/live/scorecard/rolling_scorecard.json` | Longitudinal performance summary |
| `data/live/scorecard/pipeline_status.json` | Latest pipeline run and data-freshness metadata |
| `data/live/scorecard/system_status.json` | Machine-readable operational health |

---

## Evaluation framework

CleanCharge Live evaluates both **prediction quality** and **decision quality**.

### Scientific forecast accuracy

Scientific evaluation measures how closely forecast hourly carbon-intensity values match actual observations. Reported measures include mean absolute error, root mean squared error, symmetric mean absolute percentage error, coefficient of determination and bias.

It answers:

> How accurately did the model predict hourly regional grid carbon intensity?

### Recommendation quality

Decision evaluation measures how useful the published charging period was after actual observations became available. Reported measures include timing error, overlap with the actual lowest-carbon period and Carbon Savings Capture.

It answers:

> How useful was the published charging decision in practice?

These evaluations are reported separately because a forecast can be imperfect in absolute numerical terms while still identifying a highly effective lower-carbon charging period.

---

## Carbon Savings Capture

Carbon Savings Capture measures how much of the maximum possible emissions reduction was achieved by following the published charging period.

It compares the realised saving from the recommendation with the maximum saving that could have been achieved using perfect hindsight after the day had finished.

- **100%** means the recommendation achieved the maximum possible reduction.
- **90%** means it captured 90% of the maximum available reduction.
- Lower values mean that part of the available opportunity was missed.

This metric reflects the practical value of the decision, while the scientific forecast metrics describe the accuracy of the underlying forecast.

---

## Automation with GitHub Actions

The intended production workflow will:

1. check out the repository
2. install Python and dependencies
3. load the OpenElectricity API key from a GitHub Actions secret
4. run `run_daily_pipeline.py`
5. commit updated live data, forecasts, evaluations and scorecards
6. push the changes to the repository
7. allow the Streamlit deployment to refresh

Store the API key as a repository secret named:

```text
OPENELECTRICITY_API_KEY
```

Never place the API key in source code, workflow files, committed data or documentation.

---

## Data timing and availability

Forecast dates, charging periods and evaluation results are expressed in
**Melbourne local time** using the `Australia/Melbourne` timezone.

The UTC offset changes automatically between Australian Eastern Standard
Time and Australian Eastern Daylight Time. This keeps the published forecast
day aligned with the local Melbourne calendar throughout the year.

Data availability can vary, so each pipeline run records the latest
observation timestamp and observed data lag rather than assuming a fixed
delay.

---

## Dependencies

Core dependencies are listed in `requirements.txt`:

- NumPy
- pandas
- Plotly
- python-dotenv
- Requests
- scikit-learn
- Streamlit
- tzdata

Python 3.11 or later is recommended.

---

## System status

The scorecard builder classifies current health using recent forecast publication, evaluation completion, data availability and pipeline status.

Typical states include:

- **Operational**
- **Warning**
- **Degraded**
- **Unavailable**

The current machine-readable status is stored in:

```text
data/live/scorecard/system_status.json
```

---

## Limitations

CleanCharge Live currently:

- focuses on Victoria, Australia
- uses hourly regional electricity-system observations
- publishes one primary charging period for a representative 20 kWh, 7 kW charging scenario
- forecasts regional grid carbon intensity rather than charger-specific or household-specific electricity supply
- does not control a vehicle or charger directly
- does not account for every tariff, rooftop-solar profile, battery system, network constraint, battery-management rule or user preference
- depends on the continued availability and structure of the OpenElectricity API
- may be affected by forecast error, delayed data, missing observations, software faults or unusual electricity-system conditions
- should be interpreted as research-grade decision support

---

## Contributing

Contributions are welcome, particularly for bug reports, documentation, forecast modelling, operational resilience, evaluation methods and dashboard accessibility.

Please open an issue before proposing substantial architectural changes.

---

## License

CleanCharge Live is released under the [MIT License](LICENSE).

---

## Citation

When referring to the scientific basis of CleanCharge Live, please cite:

Dia, H. (2026). *CleanCharge: Emissions-aware electric vehicle charging and infrastructure equity with open data in Melbourne.* **International Journal of Sustainable Transportation**, 1–27.  
https://doi.org/10.1080/15568318.2026.2693676

Processed research datasets:

Dia, H. (2025). *CleanCharge processed electricity datasets (30-day and 90-day).* Zenodo.  
https://doi.org/10.5281/zenodo.17232110

Original research software:

Dia, H. (2025). *CleanCharge analysis and forecasting toolkit.* Zenodo.  
https://doi.org/10.5281/zenodo.17232338

A separate citation can be added after the first formal CleanCharge Live software release.

---

## Disclaimer

CleanCharge Live is an open-source research project for research, education, reproducibility and exploration of emissions-aware electric vehicle charging.

Forecasts and charging guidance may be affected by delayed or missing data, model error, unusual electricity-system conditions, software faults or upstream service changes. Users remain responsible for charging safety, electricity costs, vehicle and charger compatibility, and compliance with applicable requirements.

---

## Acknowledgements

CleanCharge Live builds on openly available electricity-system data provided through **OpenElectricity / OpenNEM**.

The author gratefully acknowledges the developers and maintainers of open-data infrastructure that enables transparent, reproducible and operational research.
