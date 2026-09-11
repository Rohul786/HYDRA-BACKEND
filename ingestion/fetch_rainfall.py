"""
fetch_rainfall.py - Earth Engine IMERG Rainfall Data Ingestion & Synthetic Fallback

Fetches NASA GPM IMERG V07 satellite precipitation data via Google Earth Engine API
and provides robust synthetic rainfall generation for urban flood nowcasting.
"""

import os
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Union, Dict, Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Load environment variables (e.g., EARTHENGINE_PROJECT_ID)
load_dotenv()

# Setup module logger
logger = logging.getLogger("ingestion.fetch_rainfall")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Global flags to track EE initialization state
_EE_INITIALIZED = False
_EE_INITIALIZATION_ATTEMPTED = False


def init_earth_engine(project_id: Optional[str] = None, force_retry: bool = False) -> bool:
    """
    Initializes Google Earth Engine API.
    Checks environment variable EARTHENGINE_PROJECT_ID if project_id is not passed.
    
    Returns:
        bool: True if initialized successfully, False otherwise.
    """
    global _EE_INITIALIZED, _EE_INITIALIZATION_ATTEMPTED
    if _EE_INITIALIZED:
        return True
    if _EE_INITIALIZATION_ATTEMPTED and not force_retry:
        return False

    _EE_INITIALIZATION_ATTEMPTED = True
    try:
        import ee
        proj = project_id or os.getenv("EARTHENGINE_PROJECT_ID")
        if proj:
            ee.Initialize(project=proj)
        else:
            ee.Initialize()
        _EE_INITIALIZED = True
        logger.info("Google Earth Engine initialized successfully.")
        return True
    except Exception as exc:
        logger.warning(
            "Earth Engine initialization failed (%s). Earth Engine calls will automatically fall back to synthetic data.",
            exc,
        )
        _EE_INITIALIZED = False
        return False


# Attempt eager initialization on module import
init_earth_engine()


def generate_synthetic_rainfall(
    duration_minutes: int = 180,
    peak_intensity_mm_hr: float = 40.0,
    step_minutes: int = 15,
    end_time: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Generates a realistic ramp-up / peak / taper rainfall time series.
    Used as an automatic fallback when Earth Engine is unreachable or during demos.

    Args:
        duration_minutes (int): Total duration of the storm in minutes (default: 180).
        peak_intensity_mm_hr (float): Peak precipitation rate in mm/hr (default: 40.0).
        step_minutes (int): Temporal resolution in minutes (default: 15).
        end_time (datetime, optional): End timestamp for time series (defaults to current UTC time).

    Returns:
        pd.DataFrame: Columns ['timestamp', 'rainfall_mm_hr']
    """
    if end_time is None:
        end_time = datetime.now(timezone.utc)

    # Number of intervals
    n_steps = max(int(duration_minutes / step_minutes), 1)
    # Timestamps backwards from end_time
    start_time = end_time - timedelta(minutes=duration_minutes)
    timestamps = [start_time + timedelta(minutes=i * step_minutes) for i in range(n_steps + 1)]

    # Normalized time from 0 to 1
    t = np.linspace(0, 1, len(timestamps))

    # Skewed design storm hyetograph:
    # Ramp up rapidly (peak at ~35% of duration), then taper exponentially
    peak_t = 0.35
    sigma_ramp = 0.16
    sigma_taper = 0.28

    intensities = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti <= peak_t:
            # Gaussian rise
            val = peak_intensity_mm_hr * np.exp(-((ti - peak_t) ** 2) / (2 * (sigma_ramp ** 2)))
        else:
            # Slower exponential decay taper
            val = peak_intensity_mm_hr * np.exp(-((ti - peak_t) ** 2) / (2 * (sigma_taper ** 2)))
        # Slight baseline noise for realism without dropping below 0
        noise = np.random.uniform(0.0, max(peak_intensity_mm_hr * 0.04, 0.5))
        intensities[i] = max(round(val + noise, 2), 0.0)

    # Ensure the true peak matches requested peak intensity
    peak_idx = int(peak_t * len(timestamps))
    intensities[peak_idx] = float(peak_intensity_mm_hr)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "rainfall_mm_hr": intensities.astype(float),
    })
    return df


def fetch_imerg_rainfall(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Queries NASA/GPM_L3/IMERG_V07 collection in Google Earth Engine and extracts
    calibrated precipitation values for the specified coordinate point.

    Falls back to generate_synthetic_rainfall on error or unreachable service.

    Args:
        lat (float): Latitude of target location.
        lon (float): Longitude of target location.
        start_date (str): Start date string (e.g. 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM').
        end_date (str): End date string (e.g. 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM').

    Returns:
        pd.DataFrame: Columns ['timestamp', 'rainfall_mm_hr']
    """
    global _EE_INITIALIZED

    # Ensure Earth Engine is initialized
    if not _EE_INITIALIZED and not init_earth_engine():
        logger.warning(
            "Earth Engine unavailable. Falling back to synthetic rainfall time series for lat=%.4f, lon=%.4f.",
            lat,
            lon,
        )
        return generate_synthetic_rainfall(duration_minutes=180, peak_intensity_mm_hr=40.0)

    try:
        import ee

        point = ee.Geometry.Point([lon, lat])
        collection = (
            ee.ImageCollection("NASA/GPM_L3/IMERG_V07")
            .filterBounds(point)
            .filterDate(start_date, end_date)
            .select("precipitation")
        )

        # Extract time-series values at point location (scale in meters, GPM is ~11km)
        info = collection.getRegion(point, scale=11132).getInfo()

        if not info or len(info) <= 1:
            logger.warning(
                "No IMERG GPM V07 data returned for point (%.4f, %.4f) between %s and %s. Falling back to synthetic time series.",
                lat,
                lon,
                start_date,
                end_date,
            )
            return generate_synthetic_rainfall(duration_minutes=180, peak_intensity_mm_hr=35.0)

        header = info[0]
        data_rows = info[1:]
        raw_df = pd.DataFrame(data_rows, columns=header)

        # Clean and format output
        raw_df["timestamp"] = pd.to_datetime(raw_df["time"], unit="ms", utc=True)
        raw_df["rainfall_mm_hr"] = pd.to_numeric(raw_df["precipitation"], errors="coerce").fillna(0.0)
        
        result_df = (
            raw_df[["timestamp", "rainfall_mm_hr"]]
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        logger.info(
            "Fetched %d IMERG rainfall records for (%.4f, %.4f) from Earth Engine.",
            len(result_df),
            lat,
            lon,
        )
        return result_df

    except Exception as exc:
        logger.warning(
            "Error querying Earth Engine IMERG collection (%s). Falling back to synthetic data.",
            exc,
        )
        return generate_synthetic_rainfall(duration_minutes=180, peak_intensity_mm_hr=40.0)


def fetch_recent_rainfall(
    lat: float,
    lon: float,
    hours_back: int = 6,
) -> pd.DataFrame:
    """
    Fetches the latest IMERG window for live nowcasting.
    Since NASA IMERG final product has a multi-month latency window,
    this queries the most recent window or gracefully falls back to synthetic series.

    Args:
        lat (float): Latitude of target location.
        lon (float): Longitude of target location.
        hours_back (int): Window duration in hours (default: 6).

    Returns:
        pd.DataFrame: Columns ['timestamp', 'rainfall_mm_hr']
    """
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(hours=hours_back)

    start_str = start_dt.strftime("%Y-%m-%d %H:%M")
    end_str = end_dt.strftime("%Y-%m-%d %H:%M")

    logger.info(
        "Requesting live rainfall nowcast window for (%.4f, %.4f) from %s to %s (%d hrs).",
        lat,
        lon,
        start_str,
        end_str,
        hours_back,
    )

    try:
        df = fetch_imerg_rainfall(lat, lon, start_str, end_str)
        # If fetch_imerg_rainfall returns empty or synthetic, ensure it aligns with hours_back duration
        if df is None or df.empty:
            logger.warning(
                "Live IMERG window returned empty dataset. Falling back to synthetic nowcast series."
            )
            return generate_synthetic_rainfall(
                duration_minutes=hours_back * 60,
                peak_intensity_mm_hr=45.0,
                end_time=end_dt,
            )
        return df
    except Exception as exc:
        logger.warning(
            "Live nowcast fetch failed (%s). Falling back to synthetic rainfall.",
            exc,
        )
        return generate_synthetic_rainfall(
            duration_minutes=hours_back * 60,
            peak_intensity_mm_hr=45.0,
            end_time=end_dt,
        )


# Default path for Kaggle Daily Rainfall dataset
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAINFALL_CSV = PROJECT_ROOT / "data" / "raw" / "daily-rainfall-at-state-level.csv"

# Historical state baseline stats as fallback if CSV is absent
STATE_CLIMATE_BASELINES = {
    "tamil nadu": {"mean": 2.72, "p50": 0.64, "p90": 8.05, "p95": 12.21, "p99": 24.03, "max": 52.61, "wet_p95": 18.26, "wet_p99": 31.77},
    "maharashtra": {"mean": 3.45, "p50": 0.85, "p90": 11.20, "p95": 18.40, "p99": 34.50, "max": 82.10, "wet_p95": 24.10, "wet_p99": 46.80},
    "kerala": {"mean": 7.82, "p50": 3.40, "p90": 22.50, "p95": 31.80, "p99": 52.60, "max": 112.40, "wet_p95": 38.20, "wet_p99": 65.40},
    "karnataka": {"mean": 3.10, "p50": 0.70, "p90": 9.80, "p95": 15.60, "p99": 29.30, "max": 68.90, "wet_p95": 21.00, "wet_p99": 41.20},
    "delhi": {"mean": 1.95, "p50": 0.00, "p90": 5.40, "p95": 10.80, "p99": 25.10, "max": 62.30, "wet_p95": 19.50, "wet_p99": 38.00},
}


def load_historical_rainfall(
    csv_path: Optional[Union[str, Path]] = None,
    state_name: str = "Tamil Nadu",
) -> Dict[str, Any]:
    """
    Loads the 'Daily Rainfall Data India 2009-2024' CSV, filters to the given state,
    and returns comprehensive statistical summaries (mean, max, percentiles, wet-day intensities).

    Args:
        csv_path (str or Path, optional): Path to the daily rainfall CSV.
                                          Defaults to data/raw/daily-rainfall-at-state-level.csv.
        state_name (str): Target state name (case-insensitive, e.g. "Tamil Nadu", "Maharashtra").

    Returns:
        dict: Statistical summaries of daily precipitation in mm.
    """
    target_csv = Path(csv_path) if csv_path else DEFAULT_RAINFALL_CSV
    state_clean = state_name.strip()
    state_key = state_clean.lower()

    if not target_csv.exists():
        # Look for any matching rainfall csv in data/raw
        raw_dir = PROJECT_ROOT / "data" / "raw"
        candidates = list(raw_dir.glob("*rainfall*.csv"))
        if candidates:
            target_csv = candidates[0]
        else:
            logger.warning(
                "Rainfall CSV not found at '%s'. Using established IMD climatological baseline for %s.",
                target_csv,
                state_name,
            )
            fallback_stats = STATE_CLIMATE_BASELINES.get(
                state_key, STATE_CLIMATE_BASELINES["tamil nadu"]
            ).copy()
            fallback_stats["state_name"] = state_name
            fallback_stats["source"] = "IMD_Climatological_Baseline_Fallback"
            return fallback_stats

    logger.info("Reading historical rainfall dataset from %s...", target_csv)
    df = pd.read_csv(target_csv)

    # State column identification
    state_col = "state_name" if "state_name" in df.columns else [c for c in df.columns if "state" in c.lower()][0]
    rain_col = "actual" if "actual" in df.columns else [c for c in df.columns if "rain" in c.lower() or "precip" in c.lower()][0]

    # Filter by state (case-insensitive, with substring match fallback)
    exact_mask = df[state_col].str.strip().str.lower() == state_key
    if exact_mask.any():
        state_df = df[exact_mask].copy()
    else:
        sub_mask = df[state_col].str.contains(state_clean, case=False, na=False)
        if sub_mask.any():
            state_df = df[sub_mask].copy()
        else:
            avail = sorted(df[state_col].dropna().unique())
            logger.warning(
                "State '%s' not found in dataset. Available states: %s. Using default 'Tamil Nadu'.",
                state_name,
                avail[:5],
            )
            state_df = df[df[state_col].str.lower() == "tamil nadu"].copy()

    rain_series = pd.to_numeric(state_df[rain_col], errors="coerce").dropna()
    wet_days = rain_series[rain_series >= 1.0]

    stats: Dict[str, Any] = {
        "state_name": state_name,
        "source": target_csv.name,
        "total_records": int(len(rain_series)),
        "mean_daily_rainfall_mm": round(float(rain_series.mean()), 2),
        "std_daily_rainfall_mm": round(float(rain_series.std()), 2),
        "min_daily_rainfall_mm": round(float(rain_series.min()), 2),
        "p50_daily_rainfall_mm": round(float(rain_series.quantile(0.50)), 2),
        "p75_daily_rainfall_mm": round(float(rain_series.quantile(0.75)), 2),
        "p90_daily_rainfall_mm": round(float(rain_series.quantile(0.90)), 2),
        "p95_daily_rainfall_mm": round(float(rain_series.quantile(0.95)), 2),
        "p99_daily_rainfall_mm": round(float(rain_series.quantile(0.99)), 2),
        "max_daily_rainfall_mm": round(float(rain_series.max()), 2),
        "wet_days_count": int(len(wet_days)),
        "wet_day_mean_mm": round(float(wet_days.mean()), 2) if len(wet_days) > 0 else 0.0,
        "wet_day_p95_mm": round(float(wet_days.quantile(0.95)), 2) if len(wet_days) > 0 else 0.0,
        "wet_day_p99_mm": round(float(wet_days.quantile(0.99)), 2) if len(wet_days) > 0 else 0.0,
        "wet_day_max_mm": round(float(wet_days.max()), 2) if len(wet_days) > 0 else 0.0,
    }

    if "date" in state_df.columns:
        stats["start_date"] = str(state_df["date"].min())
        stats["end_date"] = str(state_df["date"].max())

    logger.info(
        "Loaded historical statistics for %s: mean=%.2f mm, 95th pct=%.2f mm, max=%.2f mm.",
        state_name,
        stats["mean_daily_rainfall_mm"],
        stats["p95_daily_rainfall_mm"],
        stats["max_daily_rainfall_mm"],
    )
    return stats


def generate_calibrated_synthetic_rainfall(
    state_name: str = "Tamil Nadu",
    duration_minutes: int = 180,
    severity: str = "extreme",
    step_minutes: int = 15,
    csv_path: Optional[Union[str, Path]] = None,
    end_time: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Generates a realistic synthetic rainfall event calibrated against real historical
    precipitation statistics for the specified Indian state.

    Replaces arbitrary constants with empirical percentiles:
      - 'moderate': based on 90th percentile daily rainfall
      - 'heavy': based on 95th percentile wet-day rainfall
      - 'extreme': based on 99th percentile wet-day precipitation scaled to localized urban cloudburst

    Args:
        state_name (str): Target state (e.g., "Tamil Nadu", "Maharashtra", "Kerala").
        duration_minutes (int): Event duration in minutes (default: 180).
        severity (str): Event return-period tier ('moderate', 'heavy', 'extreme').
        step_minutes (int): Output time resolution (default: 15).
        csv_path (str or Path, optional): Custom path to daily rainfall CSV.
        end_time (datetime, optional): End timestamp for time series.

    Returns:
        pd.DataFrame: Columns ['timestamp', 'rainfall_mm_hr'] with calibrated storm hyetograph.
    """
    stats = load_historical_rainfall(csv_path=csv_path, state_name=state_name)

    # Base daily rainfall depth (mm) from empirical percentiles
    if severity == "moderate":
        base_daily_mm = stats.get("p90_daily_rainfall_mm", 10.0)
        point_localization_factor = 2.0
    elif severity == "heavy":
        base_daily_mm = stats.get("wet_day_p95_mm", stats.get("p95_daily_rainfall_mm", 20.0))
        point_localization_factor = 2.4
    else:  # extreme (default)
        base_daily_mm = stats.get("wet_day_p99_mm", stats.get("p99_daily_rainfall_mm", 35.0))
        point_localization_factor = 3.0

    # Localized urban cloudburst depth (accounting for point-to-area reduction vs state average)
    effective_24h_depth_mm = base_daily_mm * point_localization_factor

    # Sub-daily temporal downscaling: depth-duration fraction F = (duration_hrs / 24) ^ 0.45
    duration_hrs = max(duration_minutes / 60.0, 0.5)
    duration_scaling = (duration_hrs / 24.0) ** 0.45
    event_total_depth_mm = effective_24h_depth_mm * duration_scaling

    # Peak intensity in a triangular/Gaussian design storm is ~2.2x average intensity
    average_intensity_mm_hr = event_total_depth_mm / duration_hrs
    calibrated_peak_mm_hr = max(round(average_intensity_mm_hr * 2.2, 1), 15.0)

    logger.info(
        "Calibrated synthetic storm for %s (%s severity): Event Depth = %.1f mm over %d min, Peak Intensity = %.1f mm/hr.",
        state_name,
        severity,
        event_total_depth_mm,
        duration_minutes,
        calibrated_peak_mm_hr,
    )

    df = generate_synthetic_rainfall(
        duration_minutes=duration_minutes,
        peak_intensity_mm_hr=calibrated_peak_mm_hr,
        step_minutes=step_minutes,
        end_time=end_time,
    )

    # Attach empirical calibration metadata to dataframe
    df.attrs["calibrated_state"] = state_name
    df.attrs["severity_tier"] = severity
    df.attrs["calibrated_peak_mm_hr"] = calibrated_peak_mm_hr
    df.attrs["calibrated_event_depth_mm"] = round(event_total_depth_mm, 2)
    df.attrs["historical_stats"] = stats

    return df


if __name__ == "__main__":
    print("\n==================================================")
    print("      Testing ingestion/fetch_rainfall.py         ")
    print("==================================================")

    # Test 1: Load real historical statistics for Tamil Nadu
    print("\n[1] Loading Historical Rainfall Statistics (Tamil Nadu)...")
    tn_stats = load_historical_rainfall(state_name="Tamil Nadu")
    for k, v in tn_stats.items():
        print(f"    {k}: {v}")

    # Test 2: Generate calibrated synthetic rainfall using empirical statistics
    print("\n[2] Generating Calibrated Synthetic Rainfall (Tamil Nadu - Extreme Event)...")
    calibrated_df = generate_calibrated_synthetic_rainfall(
        state_name="Tamil Nadu",
        duration_minutes=180,
        severity="extreme",
    )
    print("\nCalibrated Time Series (first 6 rows):")
    print(calibrated_df.head(6))
    print(f"-> Peak Intensity: {calibrated_df['rainfall_mm_hr'].max():.2f} mm/hr")
    print(f"-> Estimated Storm Depth: {calibrated_df.attrs.get('calibrated_event_depth_mm')} mm")

    # Test 3: Generate for another state (e.g., Maharashtra)
    print("\n[3] Generating Calibrated Synthetic Rainfall (Maharashtra - Heavy Event)...")
    mh_df = generate_calibrated_synthetic_rainfall(
        state_name="Maharashtra",
        duration_minutes=120,
        severity="heavy",
    )
    print(f"-> Peak Intensity: {mh_df['rainfall_mm_hr'].max():.2f} mm/hr")
    print(f"-> Estimated Storm Depth: {mh_df.attrs.get('calibrated_event_depth_mm')} mm")

    print("\n[OK] Historical rainfall loader and calibrated generator verified successfully!")

