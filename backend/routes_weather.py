"""
routes_weather.py - Weather, Radar, and Map Tile API Router

Exposes RESTful endpoints for:
1. Live weather conditions & precipitation nowcasting (/weather/live)
2. Weather radar telemetry & tile frames (/weather/radar)
3. Meteorological rainfall time-series (/weather/rainfall)
4. Map tile service configuration and health (/tiles/status)

Guarantees HTTP 200 responses with graceful degradation across all 3 tiers:
- Tier 1: cache/
- Tier 2: data/raw/daily-rainfall-at-state-level.csv
- Tier 3: Deterministic synthetic baseline values
"""

import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import APIRouter, Query, status
from fastapi.responses import JSONResponse

# Ensure project root is present in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.external_api import (
    fetch_open_meteo_weather,
    fetch_open_meteo_forecast,
    fetch_radar_nowcast,
    fetch_weather_provider,
    verify_tile_provider,
    load_static_csv_rainfall_summary,
    generate_deterministic_synthetic_weather,
)

logger = logging.getLogger("backend.routes_weather")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

router = APIRouter(prefix="", tags=["Weather & Geospatial Feeds"])


# Preset coordinates for popular regions
LOCATION_COORDS = {
    "adyar": (13.0067, 80.2570, "Tamil Nadu"),
    "velachery": (12.9790, 80.2180, "Tamil Nadu"),
    "chennai": (13.0827, 80.2707, "Tamil Nadu"),
    "mumbai": (19.0760, 72.8777, "Maharashtra"),
    "bengaluru": (12.9716, 77.5946, "Karnataka"),
    "bangalore": (12.9716, 77.5946, "Karnataka"),
    "kochi": (9.9312, 76.2673, "Kerala"),
    "delhi": (28.6139, 77.2090, "Delhi"),
}


def _resolve_coordinates(
    lat: Optional[float],
    lon: Optional[float],
    location: Optional[str],
) -> tuple[float, float, str]:
    """Resolves latitude, longitude, and state name from inputs with Chennai default."""
    if lat is not None and lon is not None:
        # Simple state heuristic
        state = "Tamil Nadu"
        if 18.5 <= lat <= 20.5 and 72.5 <= lon <= 74.5:
            state = "Maharashtra"
        elif 8.0 <= lat <= 12.5 and 75.0 <= lon <= 77.5:
            state = "Kerala"
        elif 12.0 <= lat <= 14.0 and 76.5 <= lon <= 78.5:
            state = "Karnataka"
        return float(lat), float(lon), state

    if location:
        loc_clean = location.strip().lower()
        for key, (k_lat, k_lon, k_state) in LOCATION_COORDS.items():
            if key in loc_clean:
                return k_lat, k_lon, k_state

    # Default: Adyar, Chennai, Tamil Nadu
    return 13.0067, 80.2570, "Tamil Nadu"


@router.get(
    "/weather/live",
    summary="Get Live Weather & Precipitation Telemetry",
    description=(
        "Fetches live weather variables and precipitation nowcast using Open-Meteo, "
        "OpenWeather, or Tomorrow.io. Features bounded timeouts and 3-tier graceful "
        "degradation (Cache -> Static CSV -> Deterministic Synthetic Baseline), "
        "guaranteeing HTTP 200 response under any network or API condition."
    ),
)
async def get_live_weather(
    lat: Optional[float] = Query(None, description="Target latitude (e.g. 13.0067)"),
    lon: Optional[float] = Query(None, description="Target longitude (e.g. 80.2570)"),
    location: Optional[str] = Query(None, description="Location query string (e.g. 'Adyar, Chennai')"),
    provider: str = Query("auto", description="Weather provider: 'auto', 'open-meteo', 'openweather', 'tomorrow'"),
    hours_back: int = Query(6, ge=1, le=48, description="Lookback window for past precipitation in hours"),
):
    target_lat, target_lon, target_state = _resolve_coordinates(lat, lon, location)

    logger.info(
        "Weather request received for (%.4f, %.4f, %s) via provider=%s...",
        target_lat,
        target_lon,
        target_state,
        provider,
    )

    if provider == "open-meteo":
        data = await fetch_open_meteo_weather(
            lat=target_lat,
            lon=target_lon,
            hours_back=hours_back,
            state_name=target_state,
        )
    else:
        data = await fetch_weather_provider(
            lat=target_lat,
            lon=target_lon,
            provider=provider,
            state_name=target_state,
        )

    headers = {
        "X-Weather-Provider": str(data.get("provider", "unknown")),
        "X-Fallback-Active": "TRUE" if data.get("is_fallback") else "FALSE",
    }
    if data.get("fallback_tier"):
        headers["X-Fallback-Tier"] = str(data.get("fallback_tier"))

    return JSONResponse(content=data, status_code=status.HTTP_200_OK, headers=headers)


@router.get(
    "/weather/radar",
    summary="Get Weather Radar Nowcast & Reflectivity Frames",
    description=(
        "Retrieves weather radar telemetry and tile layers for precipitation nowcasting. "
        "Gracefully degrades to cached or synthetic radar frames on network/API timeouts."
    ),
)
async def get_radar_telemetry(
    lat: Optional[float] = Query(None, description="Target latitude"),
    lon: Optional[float] = Query(None, description="Target longitude"),
    location: Optional[str] = Query(None, description="Location name"),
):
    target_lat, target_lon, target_state = _resolve_coordinates(lat, lon, location)
    radar_data = await fetch_radar_nowcast(lat=target_lat, lon=target_lon, state_name=target_state)

    headers = {
        "X-Radar-Provider": str(radar_data.get("provider", "rainviewer")),
        "X-Fallback-Active": "TRUE" if radar_data.get("is_fallback") else "FALSE",
    }
    return JSONResponse(content=radar_data, status_code=status.HTTP_200_OK, headers=headers)


@router.get(
    "/weather/rainfall",
    summary="Get Normalized Precipitation Hyetograph",
    description=(
        "Returns a clean time series of rainfall intensities suitable for hydraulic "
        "and flood risk modeling. Bounded by sensible timeouts with 3-tier fallback."
    ),
)
async def get_rainfall_series(
    lat: Optional[float] = Query(None, description="Target latitude"),
    lon: Optional[float] = Query(None, description="Target longitude"),
    location: Optional[str] = Query(None, description="Location name"),
    hours_back: int = Query(6, ge=1, le=48),
):
    target_lat, target_lon, target_state = _resolve_coordinates(lat, lon, location)
    weather_data = await fetch_open_meteo_weather(
        lat=target_lat,
        lon=target_lon,
        hours_back=hours_back,
        state_name=target_state,
    )

    hourly = weather_data.get("hourly_precipitation", [])
    current_intensity = weather_data.get("current_precipitation_mm_hr", 0.0)

    payload = {
        "latitude": target_lat,
        "longitude": target_lon,
        "state_name": target_state,
        "current_intensity_mm_hr": current_intensity,
        "hourly_series": hourly,
        "is_fallback": weather_data.get("is_fallback", False),
        "fallback_tier": weather_data.get("fallback_tier", None),
        "data_source": weather_data.get("provider", "open-meteo"),
    }
    return JSONResponse(content=payload, status_code=status.HTTP_200_OK)


@router.get(
    "/tiles/status",
    summary="Map Tile Service Health & Configuration",
    description=(
        "Inspects active map tile configuration (Mapbox access token or OpenStreetMap fallback). "
        "Never crashes on missing MAPBOX_ACCESS_TOKEN and returns valid tile templates."
    ),
)
async def get_tile_status():
    tile_info = await verify_tile_provider()
    headers = {
        "X-Tile-Provider": str(tile_info.get("active_provider", "openstreetmap")),
        "X-Tile-Fallback": "TRUE" if tile_info.get("is_fallback") else "FALSE",
    }
    return JSONResponse(content=tile_info, status_code=status.HTTP_200_OK, headers=headers)


@router.get(
    "/weather/current",
    summary="Get Rich Current Weather Conditions",
    description=(
        "Fetches current weather conditions including temperature, feels-like temperature, "
        "humidity, precipitation rate, wind speed/direction, UV index, and pressure "
        "from Open-Meteo. Returns HTTP 200 with graceful 3-tier degradation."
    ),
)
async def get_current_weather(
    lat: Optional[float] = Query(None, description="Target latitude"),
    lon: Optional[float] = Query(None, description="Target longitude"),
    location: Optional[str] = Query(None, description="Location name"),
):
    target_lat, target_lon, target_state = _resolve_coordinates(lat, lon, location)

    logger.info(
        "Current weather request for (%.4f, %.4f, %s)...",
        target_lat, target_lon, target_state,
    )

    forecast = await fetch_open_meteo_forecast(
        lat=target_lat,
        lon=target_lon,
        forecast_days=1,
        state_name=target_state,
    )

    # Extract only current conditions + nearest hourly slice
    current = forecast.get("current", {})
    hourly_next_6 = forecast.get("hourly", [])[:6]

    payload = {
        "status": forecast.get("status", "live"),
        "provider": forecast.get("provider", "open-meteo"),
        "latitude": target_lat,
        "longitude": target_lon,
        "state_name": target_state,
        "timezone": forecast.get("timezone", "UTC"),
        "current": current,
        "hourly_next_6h": hourly_next_6,
        "is_fallback": forecast.get("is_fallback", False),
        "fallback_tier": forecast.get("fallback_tier"),
        "fetched_at": forecast.get("fetched_at"),
    }

    headers = {
        "X-Weather-Provider": str(forecast.get("provider", "open-meteo")),
        "X-Fallback-Active": "TRUE" if forecast.get("is_fallback") else "FALSE",
    }
    return JSONResponse(content=payload, status_code=status.HTTP_200_OK, headers=headers)


@router.get(
    "/weather/forecast",
    summary="Get 7-Day Weather Forecast",
    description=(
        "Returns a structured 7-day weather forecast with hourly (next 48h) and daily arrays. "
        "Each hourly entry includes temperature, humidity, precipitation probability and intensity, "
        "weather code, wind speed, and UV index. Free Open-Meteo API — no key required. "
        "3-tier graceful degradation guarantees HTTP 200 under any network condition."
    ),
)
async def get_weather_forecast(
    lat: Optional[float] = Query(None, description="Target latitude (e.g. 13.0067)"),
    lon: Optional[float] = Query(None, description="Target longitude (e.g. 80.2570)"),
    location: Optional[str] = Query(None, description="Location name (e.g. 'Chennai')"),
    forecast_days: int = Query(7, ge=1, le=16, description="Number of forecast days (1-16)"),
):
    target_lat, target_lon, target_state = _resolve_coordinates(lat, lon, location)

    logger.info(
        "7-day forecast request for (%.4f, %.4f, %s), days=%d...",
        target_lat, target_lon, target_state, forecast_days,
    )

    forecast = await fetch_open_meteo_forecast(
        lat=target_lat,
        lon=target_lon,
        forecast_days=forecast_days,
        state_name=target_state,
    )

    headers = {
        "X-Weather-Provider": str(forecast.get("provider", "open-meteo")),
        "X-Fallback-Active": "TRUE" if forecast.get("is_fallback") else "FALSE",
    }
    if forecast.get("fallback_tier"):
        headers["X-Fallback-Tier"] = str(forecast.get("fallback_tier"))

    return JSONResponse(content=forecast, status_code=status.HTTP_200_OK, headers=headers)

