"""
routes_risk.py - Flood Risk Assessment API Router

Exposes RESTful endpoints to trigger the urban flood nowcasting pipeline
and retrieve real-time GeoJSON flood risk features with in-memory caching.
Safely falls back to physical heuristic calculation on unpickling/model errors.
"""

import sys
import json
import time
import asyncio
import logging
from pathlib import Path
from threading import Lock
from typing import Dict, Any, Optional, Tuple, List

# Ensure project root is present in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import APIRouter, Query, HTTPException, Response
from fastapi.responses import JSONResponse

from engine.pipeline import run_nowcast_pipeline
from engine.risk_classifier import (
    retrain_or_export_model,
    compute_heuristic_flood_risk,
    get_impervious_grid_summary,
)

logger = logging.getLogger("backend.routes_risk")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Router definition
router = APIRouter(prefix="", tags=["Flood Risk"])

# -----------------------------------------------------------------------------
# 30-Second In-Memory TTL Cache
# -----------------------------------------------------------------------------
CACHE_TTL_SECONDS = 30.0


class FloodRiskCache:
    """
    Thread-safe in-memory cache with 30-second TTL for nowcast GeoJSON outputs.
    """

    def __init__(self, ttl_seconds: float = CACHE_TTL_SECONDS):
        self.ttl = ttl_seconds
        self._cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
        self._lock = Lock()

    def _generate_key(self, location: str, rainfall_source: str, storm_severity: str) -> str:
        loc_norm = location.strip().lower()
        rf_norm = rainfall_source.strip().lower()
        sev_norm = storm_severity.strip().lower()
        return f"{loc_norm}::{rf_norm}::{sev_norm}"

    def get(
        self,
        location: str,
        rainfall_source: str,
        storm_severity: str = "extreme",
    ) -> Optional[Dict[str, Any]]:
        key = self._generate_key(location, rainfall_source, storm_severity)
        with self._lock:
            if key in self._cache:
                cached_time, cached_data = self._cache[key]
                age = time.time() - cached_time
                if age < self.ttl:
                    logger.info("Cache HIT for key '%s' (age: %.1fs / %.0fs)", key, age, self.ttl)
                    return cached_data
                else:
                    logger.info("Cache EXPIRED for key '%s' (age: %.1fs)", key, age)
                    del self._cache[key]
        return None

    def set(
        self,
        location: str,
        rainfall_source: str,
        data: Dict[str, Any],
        storm_severity: str = "extreme",
    ) -> None:
        key = self._generate_key(location, rainfall_source, storm_severity)
        with self._lock:
            self._cache[key] = (time.time(), data)
            logger.info("Cached result for key '%s' (TTL: %.0fs)", key, self.ttl)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


# Global cache instance
risk_cache = FloodRiskCache(ttl_seconds=CACHE_TTL_SECONDS)


# -----------------------------------------------------------------------------
# Heuristic Fallback Generator
# -----------------------------------------------------------------------------

def generate_heuristic_fallback_geojson(
    location: str,
    rainfall_source: str,
    storm_severity: str = "extreme",
    error_msg: str = "",
) -> Dict[str, Any]:
    """
    Generates a valid fallback GeoJSON FeatureCollection using physical heuristic
    risk calculation based on rainfall depth, elevation slope, and impervious grid
    surface data from data/processed/impervious_grid_*.geojson when full ML pipeline
    or model unpickling fails. Guarantees HTTP 200 response instead of HTTP 500.
    """
    processed_dir = PROJECT_ROOT / "data" / "processed"
    cached_geojson = processed_dir / "nowcast_output.geojson"

    severity_rain_map = {"moderate": 35.0, "heavy": 60.0, "extreme": 90.0}
    rain_mm = severity_rain_map.get(str(storm_severity).lower(), 75.0)

    # 1. Attempt to serve disk cached nowcast with heuristic markers
    if cached_geojson.exists():
        try:
            with open(cached_geojson, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["metadata"] = data.get("metadata", {})
            data["metadata"]["heuristic_fallback"] = True
            data["metadata"]["fallback_reason"] = error_msg
            logger.info("Delivering disk nowcast output as heuristic fallback (HTTP 200).")
            return data
        except Exception as read_err:
            logger.warning("Could not load cached nowcast output: %s", read_err)

    # 2. Build feature collection from impervious grid geojson
    grid_files = list(processed_dir.glob("impervious_grid_*.geojson"))
    features: List[Dict[str, Any]] = []

    if grid_files:
        try:
            with open(grid_files[0], "r", encoding="utf-8") as f:
                grid_json = json.load(f)

            for feat in grid_json.get("features", [])[:400]:
                props = feat.get("properties", {})
                imp_ratio = float(props.get("impervious_ratio", 0.65))

                h_res = compute_heuristic_flood_risk(
                    rainfall_intensity=rain_mm,
                    slope_deg=1.0,
                    imperviousness=imp_ratio,
                )

                risk_cat = h_res["predicted_class"]
                severe_prob = h_res["probabilities"].get("Severe", 0.05)

                features.append({
                    "type": "Feature",
                    "geometry": feat.get("geometry"),
                    "properties": {
                        "cell_id": props.get("cell_id", 0),
                        "impervious_ratio": imp_ratio,
                        "rainfall_intensity_mm_hr": rain_mm,
                        "flood_risk_level": risk_cat.upper(),
                        "ml_risk_category": risk_cat,
                        "final_risk_category": risk_cat,
                        "ml_severe_probability": severe_prob,
                        "surcharge_ratio": 0.0,
                        "heuristic_fallback": True,
                    },
                })
        except Exception as grid_err:
            logger.warning("Failed parsing impervious grid: %s", grid_err)

    # If no features from grid, construct valid baseline coordinate segment
    if not features:
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [[80.2400, 13.0180], [80.2450, 13.0120]],
            },
            "properties": {
                "name": "Fallback Road Segment",
                "flood_risk_level": "MODERATE",
                "ml_risk_category": "Moderate",
                "final_risk_category": "Moderate",
                "ml_severe_probability": 0.25,
                "heuristic_fallback": True,
            },
        })

    grid_summary = get_impervious_grid_summary()

    return {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "location": location,
            "rainfall_source": rainfall_source,
            "storm_severity": storm_severity,
            "rainfall_intensity_mm_hr": rain_mm,
            "mean_imperviousness": grid_summary["mean_imperviousness"],
            "heuristic_fallback": True,
            "fallback_reason": error_msg,
            "road_segments_count": len(features),
        },
    }


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------

@router.get(
    "/flood-risk",
    summary="Compute or Retrieve Urban Flood Risk GeoJSON",
    description=(
        "Executes the end-to-end urban flood nowcasting pipeline for a specified location "
        "and rainfall source (or retrieves the result from the 30-second in-memory cache). "
        "Returns a standard GeoJSON FeatureCollection of road segments with hydraulic backflow, "
        "ML classification, and final risk tiers. Falls back to heuristic calculation if unpickling fails."
    ),
    response_description="GeoJSON FeatureCollection representing road flood risks",
)
async def get_flood_risk(
    location: str = Query(
        "Adyar, Chennai, Tamil Nadu, India",
        description="Geographic location query (e.g. 'Adyar, Chennai, Tamil Nadu, India')",
    ),
    rainfall_source: str = Query(
        "imerg",
        description="Rainfall data source: 'imerg' (satellite), 'open-meteo' (live weather API), or 'synthetic' (calibrated historical)",
    ),
    storm_severity: str = Query(
        "extreme",
        description="Intensity tier for simulation: 'moderate', 'heavy', or 'extreme'",
    ),
    hours_back: int = Query(
        6,
        description="Hours back for IMERG satellite precipitation lookup",
        ge=1,
        le=72,
    ),
    force_refresh: bool = Query(
        False,
        description="If True, bypasses the 30-second cache and re-runs the simulation pipeline",
    ),
):
    """
    GET /flood-risk?location=...&rainfall_source=imerg
    """
    start_time = time.perf_counter()

    # 1. Check in-memory cache (unless force_refresh is requested)
    if not force_refresh:
        cached_result = risk_cache.get(location, rainfall_source, storm_severity)
        if cached_result is not None:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            headers = {
                "X-Cache": "HIT",
                "X-Cache-TTL-Remaining": f"{max(0.0, CACHE_TTL_SECONDS - (time.time() - risk_cache._cache[risk_cache._generate_key(location, rainfall_source, storm_severity)][0])):.1f}s",
                "X-Execution-Time-Ms": f"{elapsed_ms:.2f}",
            }
            return JSONResponse(content=cached_result, headers=headers)

    # 2. Cache miss or forced refresh: Execute nowcast pipeline in worker thread
    logger.info(
        "Executing nowcast pipeline for location='%s', rainfall_source='%s' (Severity: %s)...",
        location,
        rainfall_source,
        storm_severity,
    )

    is_fallback = False
    try:
        geojson_result = await asyncio.to_thread(
            run_nowcast_pipeline,
            location=location,
            rainfall_source=rainfall_source,
            storm_severity=storm_severity,
            hours_back=hours_back,
        )
    except Exception as exc:
        logger.warning(
            "Nowcast pipeline encountered error (%s). Engaging heuristic risk fallback...",
            exc,
        )
        is_fallback = True
        geojson_result = generate_heuristic_fallback_geojson(
            location=location,
            rainfall_source=rainfall_source,
            storm_severity=storm_severity,
            error_msg=str(exc),
        )

    # 3. Store result in 30-second cache
    risk_cache.set(location, rainfall_source, geojson_result, storm_severity)

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    headers = {
        "X-Cache": "MISS",
        "X-Cache-TTL": f"{CACHE_TTL_SECONDS}s",
        "X-Execution-Time-Ms": f"{elapsed_ms:.2f}",
        "X-Heuristic-Fallback": "TRUE" if is_fallback else "FALSE",
    }

    return JSONResponse(content=geojson_result, headers=headers)


@router.post(
    "/flood-risk/retrain-model",
    summary="Retrain or Cleanly Re-export Flood Risk ML Model Bundle",
    description=(
        "Re-trains or cleanly re-exports data/processed/flood_risk_model.pkl in the current "
        "Python environment using active scikit-learn, XGBoost, and NumPy versions."
    ),
)
async def api_retrain_model():
    """
    POST /flood-risk/retrain-model
    """
    try:
        exported_path = await asyncio.to_thread(retrain_or_export_model)
        return JSONResponse(
            content={
                "status": "success",
                "message": "Model retrained and cleanly exported in current environment.",
                "model_path": str(exported_path),
            },
            status_code=200,
        )
    except Exception as exc:
        logger.error("Failed retraining model: %s", exc, exc_info=True)
        return JSONResponse(
            content={"status": "error", "detail": str(exc)},
            status_code=500,
        )
