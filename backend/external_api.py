"""
external_api.py - Resilient External HTTP Integrations & Graceful Degradation Engine

Handles outgoing HTTP queries for:
1. Open-Meteo API (live hourly precipitation, rainfall nowcasting, weather variables).
2. Radar feeds (RainViewer API / radar reflectivity / nowcast frames).
3. Weather providers (OpenWeather, Tomorrow.io).
4. Map tile providers (Mapbox access token verification, OpenStreetMap fallback).

Reliability Guarantees:
- Explicit try...except (httpx.RequestError, httpx.HTTPStatusError, TimeoutError) blocks.
- Sensible timeout limits (5-10 seconds total, 5s connect).
- 3-tier graceful degradation on missing keys, 401 Unauthorized, 429 Rate Limit, or timeouts:
    Tier 1: Existing cached responses in cache/
    Tier 2: Static historical rainfall from data/raw/daily-rainfall-at-state-level.csv
    Tier 3: Deterministic synthetic baseline values
- Missing or empty .env environment variables will NEVER crash startup or request lifecycles.
"""

import os
import json
import time
import hashlib
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, Tuple, List, Union

import httpx
import pandas as pd
from dotenv import load_dotenv

# Ensure environment variables are loaded safely without failing if .env is missing
try:
    load_dotenv()
except Exception:
    pass

logger = logging.getLogger("backend.external_api")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Root directory and data paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_HISTORICAL_CSV = RAW_DATA_DIR / "daily-rainfall-at-state-level.csv"

# Timeout configuration: 5.0s connect, 8.0s read/write/pool (sensible 8s limit)
DEFAULT_TIMEOUT = httpx.Timeout(8.0, connect=5.0)

# Environment variables safely loaded with defaults
def get_env_var(key: str, default: str = "") -> str:
    """Safely retrieves an environment variable without crashing if missing or empty."""
    val = os.getenv(key, default)
    return val.strip() if val is not None else default


OPENWEATHER_API_KEY = get_env_var("OPENWEATHER_API_KEY", "")
TOMORROW_IO_API_KEY = get_env_var("TOMORROW_IO_API_KEY", "")
MAPBOX_ACCESS_TOKEN = get_env_var("MAPBOX_ACCESS_TOKEN", "")


# -----------------------------------------------------------------------------
# Tier 1: Disk Cache Helpers (cache/)
# -----------------------------------------------------------------------------

from backend.cache_manager import safe_load_json, atomic_write_json


def _generate_cache_key(namespace: str, params: Dict[str, Any]) -> str:
    """Generates a deterministic hash-based cache filename."""
    serialized = json.dumps(params, sort_keys=True, default=str)
    hashed = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
    safe_ns = "".join(c if c.isalnum() else "_" for c in namespace)
    return f"{safe_ns}_{hashed}.json"


def read_from_cache(namespace: str, params: Dict[str, Any], max_age_seconds: float = 3600.0) -> Optional[Dict[str, Any]]:
    """
    Tier 1 Fallback: Attempts to read an existing cached JSON payload from cache/.
    Uses safe_load_json to catch json.JSONDecodeError, invalidating and removing
    any corrupt or half-written cache files.
    """
    cache_file = CACHE_DIR / _generate_cache_key(namespace, params)
    if not cache_file.exists():
        # Also check for any existing file in cache/ matching namespace
        candidates = list(CACHE_DIR.glob(f"{namespace}_*.json"))
        if candidates:
            cache_file = candidates[0]
        else:
            return None

    data = safe_load_json(cache_file, invalidate_on_corrupt=True)
    if data is None or not isinstance(data, dict):
        return None

    cached_time = data.get("_cached_at", 0)
    age = time.time() - cached_time
    if max_age_seconds > 0 and age > max_age_seconds:
        logger.info("Cache entry %s is stale (age: %.1fs > %.0fs). Kept as disaster fallback.", cache_file.name, age, max_age_seconds)
    data["_from_cache"] = True
    data["_cache_age_seconds"] = round(age, 1)
    return data


def write_to_cache(namespace: str, params: Dict[str, Any], data: Dict[str, Any]) -> None:
    """
    Persists a response payload to cache/ using atomic_write_json
    (tempfile.NamedTemporaryFile + os.replace) to ensure zero locking conflicts
    or truncated files during concurrent requests.
    """
    try:
        cache_file = CACHE_DIR / _generate_cache_key(namespace, params)
        payload = dict(data)
        payload["_cached_at"] = time.time()
        atomic_write_json(cache_file, payload, indent=2)
    except Exception as exc:
        logger.warning("Failed to write response to cache %s: %s", namespace, exc)


# -----------------------------------------------------------------------------
# Tier 2: Static Historical CSV Loader (data/raw/daily-rainfall-at-state-level.csv)
# -----------------------------------------------------------------------------

def load_static_csv_rainfall_summary(state_name: str = "Tamil Nadu") -> Dict[str, Any]:
    """
    Tier 2 Fallback: Loads empirical daily rainfall percentiles and mean from
    data/raw/daily-rainfall-at-state-level.csv without crashing if missing.
    """
    target_csv = DEFAULT_HISTORICAL_CSV
    if not target_csv.exists():
        candidates = list(RAW_DATA_DIR.glob("*rainfall*.csv"))
        if candidates:
            target_csv = candidates[0]
        else:
            return {
                "source": "missing_csv_stub",
                "mean_daily_rainfall_mm": 3.5,
                "p90_daily_rainfall_mm": 12.0,
                "p95_daily_rainfall_mm": 25.0,
                "p99_daily_rainfall_mm": 45.0,
                "max_daily_rainfall_mm": 75.0,
            }

    try:
        df = pd.read_csv(target_csv)
        state_col = "state_name" if "state_name" in df.columns else [c for c in df.columns if "state" in c.lower()][0]
        rain_col = "actual" if "actual" in df.columns else [c for c in df.columns if "rain" in c.lower() or "precip" in c.lower()][0]

        mask = df[state_col].astype(str).str.strip().str.lower() == state_name.strip().lower()
        if not mask.any():
            mask = df[state_col].astype(str).str.contains(state_name.strip(), case=False, na=False)
        if not mask.any():
            mask = df[state_col].astype(str).str.lower().str.contains("tamil nadu", na=False)

        state_df = df[mask] if mask.any() else df
        series = pd.to_numeric(state_df[rain_col], errors="coerce").dropna()

        return {
            "source": target_csv.name,
            "state_name": state_name,
            "records_count": int(len(series)),
            "mean_daily_rainfall_mm": round(float(series.mean()), 2) if len(series) else 3.5,
            "p90_daily_rainfall_mm": round(float(series.quantile(0.90)), 2) if len(series) else 12.0,
            "p95_daily_rainfall_mm": round(float(series.quantile(0.95)), 2) if len(series) else 25.0,
            "p99_daily_rainfall_mm": round(float(series.quantile(0.99)), 2) if len(series) else 45.0,
            "max_daily_rainfall_mm": round(float(series.max()), 2) if len(series) else 75.0,
        }
    except Exception as exc:
        logger.warning("Error reading static historical rainfall CSV: %s", exc)
        return {
            "source": "csv_read_exception_stub",
            "mean_daily_rainfall_mm": 3.5,
            "p90_daily_rainfall_mm": 12.0,
            "p95_daily_rainfall_mm": 25.0,
            "p99_daily_rainfall_mm": 45.0,
            "max_daily_rainfall_mm": 75.0,
        }


# -----------------------------------------------------------------------------
# Tier 3: Deterministic Synthetic Baseline Values
# -----------------------------------------------------------------------------

def generate_deterministic_synthetic_weather(
    lat: float,
    lon: float,
    hours_back: int = 6,
    severity: str = "moderate",
) -> Dict[str, Any]:
    """
    Tier 3 Fallback: Generates deterministic baseline weather and precipitation
    values derived mathematically from coordinate hashes and severity tiers.
    Guarantees consistent, reproducible output without any external dependencies.
    """
    # Deterministic seed from coordinate rounding
    coord_key = f"{round(lat, 2)}:{round(lon, 2)}"
    seed = int(hashlib.md5(coord_key.encode("utf-8")).hexdigest()[:8], 16)

    severity_multipliers = {
        "low": 0.3,
        "moderate": 0.8,
        "heavy": 1.6,
        "extreme": 2.5,
    }
    multiplier = severity_multipliers.get(severity.lower(), 1.0)

    # Base values
    base_temp = 28.0 + (seed % 7) - 3.5
    base_humidity = min(98.0, max(50.0, 75.0 + (seed % 20) - 10.0))
    base_rain_rate = max(0.0, round((((seed % 15) + 5.0) * multiplier), 2))

    now = datetime.now(timezone.utc)
    hourly_series = []
    for h in range(hours_back, -1, -1):
        dt = now - timedelta(hours=h)
        # S-curve / bell envelope around the midpoint
        hour_fraction = 1.0 - abs((h - (hours_back / 2.0)) / (hours_back / 2.0 + 0.1))
        rain_val = max(0.0, round(base_rain_rate * hour_fraction * 1.2, 2))
        hourly_series.append({
            "time": dt.isoformat(),
            "precipitation_mm": rain_val,
        })

    return {
        "status": "deterministic_synthetic_baseline",
        "latitude": lat,
        "longitude": lon,
        "temperature_c": round(base_temp, 1),
        "relative_humidity_pct": round(base_humidity, 1),
        "current_precipitation_mm_hr": base_rain_rate,
        "weather_description": "Cloudy with precipitation (Synthetic Baseline)",
        "weather_code": 61,  # WMO code: Rain, slight
        "hourly_precipitation": hourly_series,
        "generated_at": now.isoformat(),
    }


# -----------------------------------------------------------------------------
# 1. Open-Meteo External API Integration (with 3-Tier Fallback)
# -----------------------------------------------------------------------------

async def fetch_open_meteo_weather(
    lat: float,
    lon: float,
    hours_back: int = 6,
    forecast_hours: int = 12,
    state_name: str = "Tamil Nadu",
) -> Dict[str, Any]:
    """
    Fetches live weather & precipitation from Open-Meteo API.
    Endpoint: https://api.open-meteo.com/v1/forecast

    Catches (httpx.RequestError, httpx.HTTPStatusError, TimeoutError) with 5-8s timeout,
    and falls back through:
    1. cache/
    2. static CSV daily-rainfall-at-state-level.csv
    3. deterministic synthetic baseline
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": round(lat, 4),
        "longitude": round(lon, 4),
        "current": "temperature_2m,relative_humidity_2m,precipitation,rain,weather_code",
        "hourly": "precipitation,rain,weather_code",
        "past_hours": hours_back,
        "forecast_hours": forecast_hours,
        "timezone": "auto",
    }
    cache_params = {"lat": round(lat, 3), "lon": round(lon, 3), "endpoint": "open_meteo"}

    try:
        logger.info("Querying Open-Meteo weather API for (%.4f, %.4f)...", lat, lon)
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

            current = data.get("current", {})
            hourly = data.get("hourly", {})
            times = hourly.get("time", [])
            precips = hourly.get("precipitation", [])

            hourly_series = [
                {"time": t, "precipitation_mm": float(p or 0.0)}
                for t, p in zip(times, precips)
            ]

            result = {
                "status": "live",
                "provider": "open-meteo",
                "latitude": lat,
                "longitude": lon,
                "temperature_c": current.get("temperature_2m", 28.0),
                "relative_humidity_pct": current.get("relative_humidity_2m", 80.0),
                "current_precipitation_mm_hr": float(current.get("precipitation") or current.get("rain") or 0.0),
                "weather_code": current.get("weather_code", 0),
                "hourly_precipitation": hourly_series,
                "is_fallback": False,
                "fallback_tier": None,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }

            # Save to disk cache for future fallback
            write_to_cache("weather_open_meteo", cache_params, result)
            return result

    except (httpx.RequestError, httpx.HTTPStatusError, TimeoutError) as exc:
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        logger.warning(
            "Open-Meteo HTTP query failed (%s, status=%s). Engaging multi-tier graceful degradation...",
            exc,
            status_code,
        )

        # Tier 1: Try reading disk cache
        cached = read_from_cache("weather_open_meteo", cache_params, max_age_seconds=14400.0)
        if cached:
            cached["is_fallback"] = True
            cached["fallback_tier"] = "tier1_cache"
            cached["fallback_reason"] = f"{type(exc).__name__}: {exc}"
            logger.info("Delivering Tier 1 cached Open-Meteo weather payload.")
            return cached

        # Tier 2: Try static historical rainfall CSV
        csv_stats = load_static_csv_rainfall_summary(state_name=state_name)
        if csv_stats.get("records_count", 0) > 0:
            now = datetime.now(timezone.utc)
            # Create synthetic series scaled to empirical 95th percentile
            p95_depth = csv_stats.get("p95_daily_rainfall_mm", 20.0)
            hourly_rain = round(p95_depth / 6.0, 2)
            hourly_series = [
                {
                    "time": (now - timedelta(hours=h)).isoformat(),
                    "precipitation_mm": hourly_rain if h <= 3 else round(hourly_rain * 0.4, 2),
                }
                for h in range(hours_back, -1, -1)
            ]
            logger.info("Delivering Tier 2 static CSV historical rainfall baseline (%s).", state_name)
            return {
                "status": "degraded_historical_csv",
                "provider": "open-meteo_fallback_csv",
                "latitude": lat,
                "longitude": lon,
                "temperature_c": 28.5,
                "relative_humidity_pct": 82.0,
                "current_precipitation_mm_hr": hourly_rain,
                "weather_code": 61,
                "hourly_precipitation": hourly_series,
                "historical_csv_stats": csv_stats,
                "is_fallback": True,
                "fallback_tier": "tier2_historical_csv",
                "fallback_reason": f"{type(exc).__name__}: {exc}",
                "fetched_at": now.isoformat(),
            }

        # Tier 3: Deterministic synthetic baseline
        logger.info("Delivering Tier 3 deterministic synthetic weather baseline.")
        synth = generate_deterministic_synthetic_weather(lat=lat, lon=lon, hours_back=hours_back)
        synth["is_fallback"] = True
        synth["fallback_tier"] = "tier3_synthetic_baseline"
        synth["fallback_reason"] = f"{type(exc).__name__}: {exc}"
        return synth

    except Exception as exc:
        logger.error("Unexpected error in fetch_open_meteo_weather: %s", exc, exc_info=True)
        synth = generate_deterministic_synthetic_weather(lat=lat, lon=lon, hours_back=hours_back)
        synth["is_fallback"] = True
        synth["fallback_tier"] = "tier3_synthetic_baseline"
        synth["fallback_reason"] = str(exc)
        return synth


def fetch_open_meteo_rainfall_df(
    lat: float,
    lon: float,
    hours_back: int = 6,
    state_name: str = "Tamil Nadu",
) -> pd.DataFrame:
    """
    Synchronous wrapper to fetch Open-Meteo precipitation and return a DataFrame
    with ['timestamp', 'rainfall_mm_hr'] ready for pipeline hydraulic ingestion.
    Gracefully degrades across all 3 tiers if network or API fails.
    """
    import asyncio

    try:
        # Check if an event loop is running
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                res = pool.submit(
                    asyncio.run,
                    fetch_open_meteo_weather(lat, lon, hours_back=hours_back, state_name=state_name),
                ).result()
        else:
            res = loop.run_until_complete(
                fetch_open_meteo_weather(lat, lon, hours_back=hours_back, state_name=state_name)
            )
    except Exception:
        res = asyncio.run(
            fetch_open_meteo_weather(lat, lon, hours_back=hours_back, state_name=state_name)
        )

    hourly = res.get("hourly_precipitation", [])
    if hourly:
        rows = [
            {
                "timestamp": pd.to_datetime(item["time"], utc=True),
                "rainfall_mm_hr": float(item.get("precipitation_mm", 0.0)),
            }
            for item in hourly
        ]
        df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
        df.attrs["provider"] = res.get("provider", "open-meteo")
        df.attrs["is_fallback"] = res.get("is_fallback", False)
        df.attrs["fallback_tier"] = res.get("fallback_tier", None)
        return df

    # Extreme safety fallback
    now = datetime.now(timezone.utc)
    return pd.DataFrame([
        {"timestamp": now - timedelta(hours=i), "rainfall_mm_hr": 25.0}
        for i in range(hours_back, -1, -1)
    ])


# -----------------------------------------------------------------------------
# 2. Radar & Precipitation Telemetry Integration (with 3-Tier Fallback)
# -----------------------------------------------------------------------------

async def fetch_radar_nowcast(
    lat: float,
    lon: float,
    state_name: str = "Tamil Nadu",
) -> Dict[str, Any]:
    """
    Fetches live weather radar nowcasting frames and tile layers.
    Queries the public RainViewer radar API (https://api.rainviewer.com/public/weather-maps.json).
    
    Gracefully degrades:
    - On 401, 429, timeout, or network failure ->
      1. Tier 1: cache/
      2. Tier 2: Static CSV baseline
      3. Tier 3: Deterministic synthetic radar frame collection
    """
    url = "https://api.rainviewer.com/public/weather-maps.json"
    cache_params = {"endpoint": "rainviewer_radar"}

    try:
        logger.info("Querying live radar telemetry from RainViewer API...")
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            radar_data = resp.json()

            host = radar_data.get("host", "https://tilecache.rainviewer.com")
            past = radar_data.get("radar", {}).get("past", [])
            nowcast = radar_data.get("radar", {}).get("nowcast", [])

            frames = []
            for f in past[-4:] + nowcast[:4]:
                frames.append({
                    "time": f.get("time"),
                    "path": f.get("path"),
                    "tile_url": f"{host}{f.get('path')}/256/{{z}}/{{x}}/{{y}}/2/1_1.png",
                })

            result = {
                "status": "live",
                "provider": "rainviewer",
                "latitude": lat,
                "longitude": lon,
                "radar_host": host,
                "frames_count": len(frames),
                "frames": frames,
                "is_fallback": False,
                "fallback_tier": None,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            write_to_cache("radar_rainviewer", cache_params, result)
            return result

    except (httpx.RequestError, httpx.HTTPStatusError, TimeoutError) as exc:
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        logger.warning(
            "Radar HTTP query failed (%s, status=%s). Engaging multi-tier graceful degradation...",
            exc,
            status_code,
        )

        # Tier 1: Cache
        cached = read_from_cache("radar_rainviewer", cache_params, max_age_seconds=14400.0)
        if cached:
            cached["is_fallback"] = True
            cached["fallback_tier"] = "tier1_cache"
            cached["fallback_reason"] = f"{type(exc).__name__}: {exc}"
            return cached

        # Tier 2 & 3: Synthetic radar frames baseline
        now_ts = int(time.time())
        synthetic_frames = [
            {
                "time": now_ts - (i * 600),
                "path": f"/v2/radar/synthetic_{now_ts - (i * 600)}",
                "tile_url": None,
                "simulated_reflectivity_dbz": round(max(0.0, 35.0 - (i * 2.5)), 1),
            }
            for i in range(6, -1, -1)
        ]
        return {
            "status": "degraded_synthetic_radar",
            "provider": "synthetic_radar_baseline",
            "latitude": lat,
            "longitude": lon,
            "radar_host": "local_synthetic_engine",
            "frames_count": len(synthetic_frames),
            "frames": synthetic_frames,
            "is_fallback": True,
            "fallback_tier": "tier3_synthetic_baseline",
            "fallback_reason": f"{type(exc).__name__}: {exc}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as exc:
        logger.error("Unexpected error fetching radar data: %s", exc, exc_info=True)
        return {
            "status": "error_synthetic_radar",
            "provider": "synthetic_radar_baseline",
            "latitude": lat,
            "longitude": lon,
            "frames": [],
            "is_fallback": True,
            "fallback_tier": "tier3_synthetic_baseline",
            "fallback_reason": str(exc),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


# -----------------------------------------------------------------------------
# 2b. Open-Meteo 7-Day Hourly + Daily Forecast (with 3-Tier Fallback)
# -----------------------------------------------------------------------------

# WMO Weather Interpretation Codes → human-readable descriptions
WMO_CODE_DESCRIPTIONS: Dict[int, str] = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Foggy", 48: "Depositing rime fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    66: "Light freezing rain", 67: "Heavy freezing rain",
    71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
    77: "Snow grains",
    80: "Slight showers", 81: "Moderate showers", 82: "Violent showers",
    85: "Slight snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail",
}


def _wmo_description(code: int) -> str:
    """Return a human-readable description for a WMO weather code."""
    return WMO_CODE_DESCRIPTIONS.get(int(code), f"Weather code {code}")


async def fetch_open_meteo_forecast(
    lat: float,
    lon: float,
    forecast_days: int = 7,
    state_name: str = "Tamil Nadu",
) -> Dict[str, Any]:
    """
    Fetches a 7-day hourly + daily weather forecast from Open-Meteo API.
    Endpoint: https://api.open-meteo.com/v1/forecast

    Returns structured hourly and daily forecast arrays, plus current conditions.
    Gracefully degrades through 3 tiers if the network or API is unavailable.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": round(lat, 4),
        "longitude": round(lon, 4),
        "current": (
            "temperature_2m,relative_humidity_2m,apparent_temperature,"
            "precipitation,rain,weather_code,wind_speed_10m,wind_direction_10m,"
            "uv_index,surface_pressure"
        ),
        "hourly": (
            "temperature_2m,relative_humidity_2m,precipitation_probability,"
            "precipitation,weather_code,wind_speed_10m,uv_index"
        ),
        "daily": (
            "weather_code,temperature_2m_max,temperature_2m_min,"
            "precipitation_sum,precipitation_probability_max,"
            "wind_speed_10m_max,uv_index_max,sunrise,sunset"
        ),
        "forecast_days": min(max(1, forecast_days), 16),
        "timezone": "auto",
        "wind_speed_unit": "kmh",
    }
    cache_params = {"lat": round(lat, 3), "lon": round(lon, 3), "endpoint": "open_meteo_forecast"}

    try:
        logger.info("Querying Open-Meteo 7-day forecast for (%.4f, %.4f)...", lat, lon)
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

        current = data.get("current", {})
        hourly = data.get("hourly", {})
        daily = data.get("daily", {})

        # Build hourly array (next 48 hours)
        h_times = hourly.get("time", [])
        hourly_series = []
        for i, t in enumerate(h_times[:48]):
            hourly_series.append({
                "time": t,
                "temperature_c": hourly.get("temperature_2m", [None] * (i + 1))[i],
                "humidity_pct": hourly.get("relative_humidity_2m", [None] * (i + 1))[i],
                "precipitation_mm": float(hourly.get("precipitation", [0] * (i + 1))[i] or 0),
                "precipitation_probability_pct": hourly.get("precipitation_probability", [None] * (i + 1))[i],
                "weather_code": hourly.get("weather_code", [0] * (i + 1))[i],
                "weather_description": _wmo_description(int(hourly.get("weather_code", [0] * (i + 1))[i] or 0)),
                "wind_speed_kmh": hourly.get("wind_speed_10m", [None] * (i + 1))[i],
                "uv_index": hourly.get("uv_index", [None] * (i + 1))[i],
            })

        # Build daily array
        d_times = daily.get("time", [])
        daily_series = []
        for i, t in enumerate(d_times):
            daily_series.append({
                "date": t,
                "temp_max_c": daily.get("temperature_2m_max", [None] * (i + 1))[i],
                "temp_min_c": daily.get("temperature_2m_min", [None] * (i + 1))[i],
                "precipitation_sum_mm": float(daily.get("precipitation_sum", [0] * (i + 1))[i] or 0),
                "precipitation_probability_pct": daily.get("precipitation_probability_max", [None] * (i + 1))[i],
                "weather_code": daily.get("weather_code", [0] * (i + 1))[i],
                "weather_description": _wmo_description(int(daily.get("weather_code", [0] * (i + 1))[i] or 0)),
                "wind_speed_max_kmh": daily.get("wind_speed_10m_max", [None] * (i + 1))[i],
                "uv_index_max": daily.get("uv_index_max", [None] * (i + 1))[i],
                "sunrise": daily.get("sunrise", [None] * (i + 1))[i],
                "sunset": daily.get("sunset", [None] * (i + 1))[i],
            })

        current_code = int(current.get("weather_code", 0) or 0)
        result = {
            "status": "live",
            "provider": "open-meteo",
            "latitude": lat,
            "longitude": lon,
            "state_name": state_name,
            "timezone": data.get("timezone", "UTC"),
            "current": {
                "temperature_c": current.get("temperature_2m"),
                "feels_like_c": current.get("apparent_temperature"),
                "humidity_pct": current.get("relative_humidity_2m"),
                "precipitation_mm_hr": float(current.get("precipitation") or current.get("rain") or 0),
                "weather_code": current_code,
                "weather_description": _wmo_description(current_code),
                "wind_speed_kmh": current.get("wind_speed_10m"),
                "wind_direction_deg": current.get("wind_direction_10m"),
                "uv_index": current.get("uv_index"),
                "pressure_hpa": current.get("surface_pressure"),
            },
            "hourly": hourly_series,
            "daily": daily_series,
            "is_fallback": False,
            "fallback_tier": None,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }

        write_to_cache("weather_forecast", cache_params, result)
        return result

    except (httpx.RequestError, httpx.HTTPStatusError, TimeoutError) as exc:
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        logger.warning(
            "Open-Meteo forecast query failed (%s, status=%s). Engaging fallback...",
            exc, status_code,
        )

        # Tier 1: disk cache
        cached = read_from_cache("weather_forecast", cache_params, max_age_seconds=21600.0)
        if cached:
            cached["is_fallback"] = True
            cached["fallback_tier"] = "tier1_cache"
            cached["fallback_reason"] = f"{type(exc).__name__}: {exc}"
            return cached

        # Tier 2 / 3: synthesise a 7-day forecast from CSV + deterministic baseline
        now = datetime.now(timezone.utc)
        csv_stats = load_static_csv_rainfall_summary(state_name=state_name)
        mean_rain = csv_stats.get("mean_daily_rainfall_mm", 3.5)

        synthetic_hourly = []
        for h in range(48):
            dt = now + timedelta(hours=h)
            rain = round(max(0.0, mean_rain / 24 + (h % 6) * 0.1), 2)
            synthetic_hourly.append({
                "time": dt.isoformat(),
                "temperature_c": round(28.0 + (h % 8) * 0.3 - 1.5, 1),
                "humidity_pct": 82,
                "precipitation_mm": rain,
                "precipitation_probability_pct": min(90, 40 + h % 30),
                "weather_code": 63,
                "weather_description": "Moderate rain (Synthetic)",
                "wind_speed_kmh": 18.0,
                "uv_index": max(0, 6 - abs(h % 24 - 12)),
            })

        synthetic_daily = []
        for d in range(7):
            dt = now + timedelta(days=d)
            synthetic_daily.append({
                "date": dt.strftime("%Y-%m-%d"),
                "temp_max_c": round(30.0 - d * 0.2, 1),
                "temp_min_c": round(24.0 - d * 0.1, 1),
                "precipitation_sum_mm": round(mean_rain * (1.0 + d * 0.05), 2),
                "precipitation_probability_pct": min(90, 55 + d * 3),
                "weather_code": 63,
                "weather_description": "Moderate rain (Synthetic)",
                "wind_speed_max_kmh": 22.0,
                "uv_index_max": 6,
                "sunrise": (dt.replace(hour=6, minute=10)).isoformat(),
                "sunset": (dt.replace(hour=18, minute=30)).isoformat(),
            })

        return {
            "status": "degraded_synthetic",
            "provider": "synthetic_forecast_baseline",
            "latitude": lat,
            "longitude": lon,
            "state_name": state_name,
            "current": {
                "temperature_c": 28.5, "feels_like_c": 31.0, "humidity_pct": 82,
                "precipitation_mm_hr": round(mean_rain / 24, 2),
                "weather_code": 63, "weather_description": "Moderate rain (Synthetic)",
                "wind_speed_kmh": 18.0, "wind_direction_deg": 180,
                "uv_index": 6, "pressure_hpa": 1008.0,
            },
            "hourly": synthetic_hourly,
            "daily": synthetic_daily,
            "is_fallback": True,
            "fallback_tier": "tier2_synthetic",
            "fallback_reason": f"{type(exc).__name__}: {exc}",
            "fetched_at": now.isoformat(),
        }

    except Exception as exc:
        logger.error("Unexpected error in fetch_open_meteo_forecast: %s", exc, exc_info=True)
        now = datetime.now(timezone.utc)
        return {
            "status": "error",
            "provider": "synthetic_forecast_baseline",
            "latitude": lat,
            "longitude": lon,
            "current": {
                "temperature_c": 28.5, "feels_like_c": 31.0, "humidity_pct": 82,
                "precipitation_mm_hr": 3.5, "weather_code": 63,
                "weather_description": "Moderate rain (Error Fallback)",
                "wind_speed_kmh": 18.0, "wind_direction_deg": 180,
                "uv_index": 5, "pressure_hpa": 1008.0,
            },
            "hourly": [], "daily": [],
            "is_fallback": True, "fallback_tier": "tier3_synthetic_baseline",
            "fallback_reason": str(exc),
            "fetched_at": now.isoformat(),
        }


# -----------------------------------------------------------------------------
# 3. Weather Provider Integration (OpenWeather / Tomorrow.io)
# -----------------------------------------------------------------------------

async def fetch_weather_provider(
    lat: float,
    lon: float,
    provider: str = "auto",
    state_name: str = "Tamil Nadu",
) -> Dict[str, Any]:
    """
    Fetches live weather conditions from OpenWeather or Tomorrow.io if API keys exist.
    If keys are missing, 401 Unauthorized, 429 Rate Limited, or time out,
    it seamlessly degrades to Open-Meteo or local 3-tier baselines.
    """
    openweather_key = get_env_var("OPENWEATHER_API_KEY", "")
    tomorrow_key = get_env_var("TOMORROW_IO_API_KEY", "")

    # If OpenWeather requested or auto with valid key
    if (provider == "openweather" or provider == "auto") and openweather_key and openweather_key != "your_openweather_api_key":
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {"lat": lat, "lon": lon, "appid": openweather_key, "units": "metric"}
        try:
            async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()
                rain_1h = data.get("rain", {}).get("1h", 0.0)
                return {
                    "status": "live",
                    "provider": "openweathermap",
                    "latitude": lat,
                    "longitude": lon,
                    "temperature_c": data.get("main", {}).get("temp", 28.0),
                    "relative_humidity_pct": data.get("main", {}).get("humidity", 80.0),
                    "current_precipitation_mm_hr": float(rain_1h),
                    "weather_description": data.get("weather", [{}])[0].get("description", "Clear"),
                    "is_fallback": False,
                    "fallback_tier": None,
                }
        except (httpx.RequestError, httpx.HTTPStatusError, TimeoutError) as exc:
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            logger.warning(
                "OpenWeather request failed (%s, status=%s). Gracefully falling back...",
                exc,
                status_code,
            )

    # If Tomorrow.io requested or available
    if (provider == "tomorrow" or provider == "auto") and tomorrow_key and tomorrow_key != "your_tomorrow_io_api_key":
        url = "https://api.tomorrow.io/v4/weather/realtime"
        params = {"location": f"{lat},{lon}", "apikey": tomorrow_key}
        try:
            async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()
                values = data.get("data", {}).get("values", {})
                return {
                    "status": "live",
                    "provider": "tomorrow.io",
                    "latitude": lat,
                    "longitude": lon,
                    "temperature_c": values.get("temperature", 28.0),
                    "relative_humidity_pct": values.get("humidity", 80.0),
                    "current_precipitation_mm_hr": float(values.get("precipitationIntensity", 0.0)),
                    "weather_code": values.get("weatherCode", 1000),
                    "is_fallback": False,
                    "fallback_tier": None,
                }
        except (httpx.RequestError, httpx.HTTPStatusError, TimeoutError) as exc:
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            logger.warning(
                "Tomorrow.io request failed (%s, status=%s). Gracefully falling back...",
                exc,
                status_code,
            )

    # Seamless fallback: Open-Meteo (free, keyless) + 3-tier degradation
    logger.info("Engaging Open-Meteo fallback for weather provider query...")
    return await fetch_open_meteo_weather(lat=lat, lon=lon, state_name=state_name)


# -----------------------------------------------------------------------------
# 4. Map Tile Integration & Status Check
# -----------------------------------------------------------------------------

async def verify_tile_provider() -> Dict[str, Any]:
    """
    Validates map tile provider configuration (Mapbox access token or OpenStreetMap).
    If Mapbox token is missing, unauthorized (401), rate-limited (429), or times out,
    seamlessly falls back to OpenStreetMap and local tile schemas.
    """
    mapbox_token = get_env_var("MAPBOX_ACCESS_TOKEN", "")
    has_token = bool(mapbox_token and mapbox_token != "your_mapbox_token")

    osm_url_template = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
    mapbox_url_template = f"https://api.mapbox.com/styles/v1/mapbox/dark-v11/tiles/{{z}}/{{x}}/{{y}}?access_token={mapbox_token}"

    if not has_token:
        return {
            "active_provider": "openstreetmap",
            "tile_url": osm_url_template,
            "attribution": "&copy; OpenStreetMap contributors",
            "token_status": "missing_or_default",
            "is_fallback": True,
            "fallback_reason": "MAPBOX_ACCESS_TOKEN not set or default placeholder",
        }

    # Verify token with lightweight HTTP query (5-second timeout)
    verify_url = f"https://api.mapbox.com/tokens/v2?access_token={mapbox_token}"
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(5.0, connect=3.0)) as client:
            resp = await client.get(verify_url)
            if resp.status_code == 200:
                return {
                    "active_provider": "mapbox",
                    "tile_url": mapbox_url_template,
                    "attribution": "&copy; Mapbox &copy; OpenStreetMap",
                    "token_status": "valid",
                    "is_fallback": False,
                    "fallback_reason": None,
                }
            else:
                logger.warning("Mapbox token check returned status %d. Falling back to OSM tiles.", resp.status_code)
                return {
                    "active_provider": "openstreetmap",
                    "tile_url": osm_url_template,
                    "attribution": "&copy; OpenStreetMap contributors",
                    "token_status": f"invalid_status_{resp.status_code}",
                    "is_fallback": True,
                    "fallback_reason": f"Mapbox returned HTTP {resp.status_code}",
                }
    except (httpx.RequestError, httpx.HTTPStatusError, TimeoutError) as exc:
        logger.warning("Mapbox token verification network error (%s). Falling back to OSM tiles.", exc)
        return {
            "active_provider": "openstreetmap",
            "tile_url": osm_url_template,
            "attribution": "&copy; OpenStreetMap contributors",
            "token_status": "network_timeout_or_error",
            "is_fallback": True,
            "fallback_reason": f"{type(exc).__name__}: {exc}",
        }
