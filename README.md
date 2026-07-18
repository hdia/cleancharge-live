# CleanCharge Live

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-operational-brightgreen.svg)](#system-status)

**Real-time carbon-aware electric vehicle charging forecasts for Victoria, Australia.**

CleanCharge Live is an operational research platform that retrieves recent Victorian electricity-system data, maintains a rolling observation history, forecasts grid carbon intensity, identifies lower-emissions electric vehicle charging windows, and evaluates its own performance over time.

It is the live operational companion to the published [CleanCharge](https://github.com/hdia/cleancharge) research toolkit.

---

## Live dashboard

[Open the CleanCharge Live dashboard](https://cleancharge-live.streamlit.app/)

---

## Overview

CleanCharge Live:

- retrieves hourly Victorian electricity and emissions data from OpenElectricity
- maintains a rolling 90-day history with coverage checks
- forecasts electricity carbon intensity for the next 24 hours
- identifies a lower-emissions charging window
- archives each published daily forecast
- evaluates forecast accuracy when actual observations become available
- evaluates the quality of the recommended charging decision
- maintains rolling performance scorecards
- records operational health and data-freshness metadata
- supports an interactive Streamlit dashboard

The system uses **NEM time, fixed UTC+10**, for operational scheduling and data-availability checks.

> **Research and decision-support system.** Forecasts and recommendations are model based and should not be treated as guaranteed operational advice.

---

## Relationship to the published CleanCharge study

The original CleanCharge repository contains the reproducible workflow, archived study datasets, infrastructure and equity analysis, charging scenarios, figures, and CleanCharge Explorer dashboard.

This repository contains the operational forecasting system.

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
24-hour carbon-intensity forecast
        |
        v
Daily charging-window publication
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
3. publishes the current day's forecast
4. rebuilds the rolling scorecard and system status

---

## Repository structure

```text
cleancharge-live/
|-- daily_update.py
|-- live_app.py
|-- run_daily_pipeline.py
|-- sync_github.py
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
| `data/live/status/today_recommendation.json` | Current charging recommendation |
| `data/live/validation/YYYY-MM-DD/` | Scientific and decision evaluation |
| `data/live/scorecard/rolling_scorecard.json` | Longitudinal performance summary |
| `data/live/scorecard/pipeline_status.json` | Latest pipeline run and data-freshness metadata |
| `data/live/scorecard/system_status.json` | Machine-readable operational health |

---

## Evaluation framework

CleanCharge Live evaluates both **prediction quality** and **decision quality**.

Scientific evaluation compares forecast and observed carbon intensity using measures such as mean absolute error, root mean squared error and coefficient of determination.

Decision evaluation examines whether the recommended charging window remained useful once actual observations became available, including carbon-savings capture and start-time error.

This distinction matters because a forecast can be imperfect numerically while still supporting a good charging decision.

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

## Manual Git synchronisation

For normal local development:

```powershell
python sync_github.py
```

Optional custom commit message:

```powershell
python sync_github.py "Refine dashboard status panel"
```

The script stages, commits and pushes changes. It does not resolve merge conflicts automatically.

---

## Data timing and availability

Operational timestamps are interpreted in **NEM time, UTC+10**. This avoids daylight-saving ambiguity in market-day processing and forecast publication.

Data availability can vary, so each pipeline run records the latest observation timestamp and observed data lag rather than assuming a fixed delay.

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
- forecasts regional grid carbon intensity rather than charger-specific electricity supply
- does not control a vehicle or charger directly
- does not account for every tariff, network constraint, battery-management rule or user preference
- depends on the continued availability and structure of the OpenElectricity API
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

A separate citation can be added after the first formal CleanCharge Live release.

---

## Disclaimer

CleanCharge Live is an open-source research project for research, education, reproducibility and exploration of emissions-aware electric vehicle charging.

Forecasts and recommendations may be affected by delayed or missing data, model error, unusual electricity-system conditions, software faults or upstream service changes. Users remain responsible for charging safety, electricity costs, vehicle and charger compatibility, and compliance with applicable requirements.

---

## Acknowledgements

CleanCharge Live builds on openly available electricity-system data provided through **OpenElectricity / OpenNEM**.

The author gratefully acknowledges the developers and maintainers of open-data infrastructure that enables transparent, reproducible and operational research.
