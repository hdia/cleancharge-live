import os
import pathlib
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

print(">> Starting fetch_openelectricity.py")

load_dotenv()
API_KEY = os.getenv("OPENELECTRICITY_API_KEY", "")
BASE = "https://api.openelectricity.org.au/v4/data/network"

print(">> API key prefix/suffix:", (API_KEY[:4] if API_KEY else "NONE"), "...", (API_KEY[-4:] if API_KEY else "NONE"))

def _time_window(hours_back: int) -> tuple[str, str]:
    """
    Return timezone-naive timestamps expressed in NEM network time.

    OpenElectricity requires NEM request dates to be naive AEST
    timestamps. NEM network time is fixed at UTC+10.
    """
    nem_timezone = timezone(timedelta(hours=10))

    end_nem = datetime.now(nem_timezone)
    start_nem = end_nem - timedelta(hours=hours_back)

    return (
        start_nem.strftime("%Y-%m-%dT%H:%M:%S"),
        end_nem.strftime("%Y-%m-%dT%H:%M:%S"),
    )

def _request(url: str, params: list[tuple[str, str]]):
    headers = {"Authorization": f"Bearer {API_KEY}"}
    r = requests.get(url, params=params, headers=headers, timeout=120)
    if r.status_code == 200:
        return r.json()
    print("!! URL:", r.url)
    print("!! Status:", r.status_code, r.text[:500])
    r.raise_for_status()

def _df_from_columns_data(block: dict) -> pd.DataFrame:
    cols = block.get("columns", [])
    rows = block.get("data", [])
    if (not cols) and isinstance(rows, list) and rows and isinstance(rows[0], list) and len(rows[0]) == 2:
        return pd.DataFrame(rows, columns=["ts", "value"])
    col_names = []
    for c in cols:
        if isinstance(c, dict) and "name" in c:
            col_names.append(c["name"])
        elif isinstance(c, str):
            col_names.append(c)
        else:
            col_names.append(str(c))
    return pd.DataFrame(rows, columns=col_names) if col_names else pd.DataFrame(rows)

def _unpack(payload) -> pd.DataFrame:
    data_obj = payload.get("data", payload)
    rows = data_obj if isinstance(data_obj, list) else [data_obj]
    if not rows:
        return pd.DataFrame()
    top = rows[0]
    if "results" in top:
        res = top["results"]
        if isinstance(res, dict) and "data" in res:
            return _df_from_columns_data(res)
        elif isinstance(res, list):
            return pd.DataFrame(res)
    if "data" in top:
        return _df_from_columns_data(top)
    return pd.DataFrame(rows)

def _pick_ts_col(df: pd.DataFrame) -> str:
    for c in ["ts","trading_interval","time","timestamp","datetime","interval_start","period_start","date_start"]:
        if c in df.columns:
            return c
    for c in df.columns:
        v = str(df[c].iloc[0]) if len(df) else ""
        if "-" in v and "T" in v:
            return c
    raise ValueError(f"No timestamp column. Columns: {list(df.columns)}")

def _pick_val_col(df: pd.DataFrame, metric: str) -> str:
    if "value" in df.columns:
        return "value"
    for c in [metric, metric.lower(), metric.upper(), "amount", "result", "y", "v"]:
        if c in df.columns:
            return c
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric) == 1:
        return numeric[0]
    print("!! Sample row for value detection:", df.head(1).to_dict(orient="records"))
    raise ValueError(f"No value column for '{metric}'. Columns: {list(df.columns)}")

def _fetch_series(network_code: str, metric: str, hours_back: int = 24, interval: str = "1h") -> pd.DataFrame:
    date_start, date_end = _time_window(hours_back)
#    params = [
#        ("metrics", metric),
#        ("interval", interval),
#        ("date_start", date_start),
#        ("date_end",   date_end),
#    ]

    params = [
    ("metrics", metric),
    ("interval", interval),
    ("date_start", date_start),
    ("date_end", date_end),
    ("network_region", "VIC1"),
    ("primary_grouping", "network_region"),
    ]
    
    url = f"{BASE}/{network_code}"
    print(f">> Request metric='{metric}', interval='{interval}', window={date_start}..{date_end}")
    payload = _request(url, params)
    pts = _unpack(payload)
    if set(pts.columns) >= {"name","date_start","date_end","columns","data"} and len(pts) == 1:
        container = pts.iloc[0].to_dict()
        dat = container.get("data", [])
        if isinstance(dat, list) and dat and isinstance(dat[0], list) and len(dat[0]) == 2:
            pts = pd.DataFrame(dat, columns=["ts","value"])
            print(">> Unboxed container 'data' into ts/value with", len(pts), "rows")
    if pts.empty:
        raise ValueError(f"Empty points for metric '{metric}'")
    ts_col = _pick_ts_col(pts)
    val_col = _pick_val_col(pts, metric)
    out = pts[[ts_col, val_col]].rename(columns={ts_col: "ts", val_col: metric}).copy()
    out["ts"] = pd.to_datetime(out["ts"], utc=True, errors="coerce")
    out[metric] = pd.to_numeric(out[metric], errors="coerce")
    out = out.dropna(subset=["ts", metric]).sort_values("ts").reset_index(drop=True)
    print(f">> Got {len(out)} rows for '{metric}'")
    return out

def _fetch_energy(network_code: str, hours_back: int, interval: str) -> pd.DataFrame:
    try:
        return _fetch_series(network_code, "energy", hours_back, interval)
    except Exception as e:
        print(".. 'energy' failed; trying 'demand_energy' →", e)
        return _fetch_series(network_code, "demand_energy", hours_back, interval)

def _fetch_emissions(network_code: str, hours_back: int, interval: str) -> pd.DataFrame:
    try:
        return _fetch_series(network_code, "emissions", hours_back, interval)
    except Exception as e:
        print(".. 'emissions' failed; trying 'pollution' →", e)
        return _fetch_series(network_code, "pollution", hours_back, interval)

def normalise_and_intensity(network_code="NEM", hours_back=24, interval="1h") -> pd.DataFrame:
    df_energy    = _fetch_energy(network_code, hours_back, interval)
    df_emissions = _fetch_emissions(network_code, hours_back, interval)
    tol = pd.Timedelta(interval) if any(ch.isdigit() for ch in interval) else pd.Timedelta("1h")
    df = pd.merge_asof(df_emissions.sort_values("ts"),
                       df_energy.sort_values("ts"),
                       on="ts", direction="nearest", tolerance=tol)
    df = df.dropna(subset=["emissions", "energy"], how="any").copy()
    df["emissions"] = pd.to_numeric(df["emissions"], errors="coerce")
    df["energy"]    = pd.to_numeric(df["energy"], errors="coerce")
    df["intensity"] = (df["emissions"] / df["energy"]) * 1000.0
    df["local_time"] = df["ts"].dt.tz_convert("Australia/Melbourne")  # << NEW
    out = df[["ts", "local_time", "intensity", "emissions", "energy"]].dropna().sort_values("ts").reset_index(drop=True)
    out = out.rename(columns={"energy": "energy_mwh"})
    print(">> Final joined rows:", len(out))
    return out

def main():
    if not API_KEY:
        raise SystemExit("No OPENELECTRICITY_API_KEY found in environment/.env")
    df = normalise_and_intensity(network_code="NEM", hours_back=24, interval="1h")    
    out_csv = pathlib.Path("data/processed/openelectricity_emissions.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f">> Saved {len(df)} rows to {out_csv}")
    print(df.head())

if __name__ == "__main__":
    main()
    print(">> Done.")
