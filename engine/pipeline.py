"""
pipeline.py - Integrated Urban Flood Nowcasting Orchestrator

End-to-end simulation pipeline combining:
1. Ingestion: Road network (OSM), building footprints, imperviousness grid, DEM slope, drainage graph.
2. Meteorology: NASA GPM IMERG satellite nowcasting with calibrated historical synthetic fallback.
3. Hydraulics: Rational Method runoff generation, pipe flow propagation, and manhole surcharging.
4. Machine Learning: XGBoost / empirical flood risk classification and probability refinement.
5. Delivery: GeoJSON FeatureCollection with per-segment hydraulic telemetry and final risk tiers.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import osmnx as ox
import shapely

from ingestion.fetch_osm_data import (
    fetch_road_network,
    fetch_buildings,
    fetch_surface_drainage,
    calculate_imperviousness_grid,
)
from ingestion.fetch_dem import (
    fetch_copernicus_dem,
    get_elevation_at_point,
    calculate_slope,
)
from ingestion.drainage_graph import (
    build_sample_drainage_network,
    calibrate_with_osm_drainage,
    save_network,
    load_network,
)
from ingestion.fetch_rainfall import (
    fetch_recent_rainfall,
    generate_calibrated_synthetic_rainfall,
    load_historical_rainfall,
)
from engine.hydraulics import (
    distribute_runoff_to_nodes,
    propagate_flow,
    map_surcharge_to_road_segments,
    create_dem_slope_lookup,
)
from engine.risk_classifier import (
    map_node_features_to_model_input,
    predict_risk_ml,
    train_model,
)

logger = logging.getLogger("engine.pipeline")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Ordinal risk rank mapping for hydraulic & ML fusion
RISK_RANK = {"LOW": 0, "MODERATE": 1, "HIGH": 2, "SEVERE": 3}
RANK_TO_RISK = {0: "Low", 1: "Moderate", 2: "High", 3: "Severe"}


def _infer_state_from_location(location: str) -> str:
    """Infers the corresponding Indian state from a place name."""
    loc_lower = location.lower()
    if any(k in loc_lower for k in ["chennai", "tamil nadu", "adyar", "mylapore", "coimbatore", "madurai"]):
        return "Tamil Nadu"
    elif any(k in loc_lower for k in ["mumbai", "maharashtra", "pune", "nagpur", "thane"]):
        return "Maharashtra"
    elif any(k in loc_lower for k in ["bengaluru", "bangalore", "karnataka", "mysuru"]):
        return "Karnataka"
    elif any(k in loc_lower for k in ["kochi", "kerala", "thiruvananthapuram", "ernakulam"]):
        return "Kerala"
    elif any(k in loc_lower for k in ["delhi", "new delhi", "ncr"]):
        return "Delhi"
    elif any(k in loc_lower for k in ["hyderabad", "telangana"]):
        return "Telangana"
    elif any(k in loc_lower for k in ["kolkata", "west bengal"]):
        return "West Bengal"
    return "Tamil Nadu"


def run_nowcast_pipeline(
    location: str = "Adyar, Chennai, Tamil Nadu, India",
    rainfall_source: str = "imerg",
    storm_severity: str = "extreme",
    hours_back: int = 6,
    grid_cell_size_m: float = 150.0,
    impact_radius_m: float = 180.0,
    export_geojson_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Executes the complete urban flood nowcasting pipeline for a specified location:
    1. Ingestion: Road graph, building footprints, imperviousness grid, DEM slope, and drainage network.
    2. Rainfall: GPM IMERG satellite precipitation or empirically calibrated historical synthetic event.
    3. Hydraulics: Rational Method runoff, pipe network routing, and manhole surcharging.
    4. ML Refinement: XGBoost classification combining hydraulic surcharge with socio-environmental risk.
    5. Output: GeoJSON FeatureCollection of road segments with hydraulic metrics and final risk tiers.

    Args:
        location (str): Geographic query (e.g., "Adyar, Chennai, Tamil Nadu, India").
        rainfall_source (str): 'imerg' (satellite live with fallback) or 'synthetic' (calibrated historical).
        storm_severity (str): Event intensity tier ('moderate', 'heavy', 'extreme').
        hours_back (int): Temporal window in hours for IMERG nowcasting.
        grid_cell_size_m (float): Spatial resolution for imperviousness grid in meters.
        impact_radius_m (float): Proximity buffer for surcharging manhole flood overland effect.
        export_geojson_path (str or Path, optional): Optional file path to save resulting GeoJSON.

    Returns:
        Dict[str, Any]: Standard GeoJSON FeatureCollection dictionary.
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    safe_loc_name = "".join(c if c.isalnum() else "_" for c in location[:32]).strip("_").lower()
    inferred_state = _infer_state_from_location(location)

    logger.info("=" * 70)
    logger.info("STARTING URBAN FLOOD NOWCAST PIPELINE FOR: %s", location)
    logger.info("Rainfall Source: %s | State Baseline: %s", rainfall_source, inferred_state)
    logger.info("=" * 70)

    # -------------------------------------------------------------------------
    # STEP 1: LOAD / FETCH GEOSPATIAL INFRASTRUCTURE DATA
    # -------------------------------------------------------------------------
    # 1.1 Road Network
    road_cache_path = PROCESSED_DATA_DIR / f"road_network_{safe_loc_name}.graphml"
    road_graph = fetch_road_network(place_name=location, filepath=road_cache_path)
    roads_gdf = ox.graph_to_gdfs(road_graph, nodes=False, edges=True)

    # Compute spatial center of study area
    bounds = roads_gdf.total_bounds  # minx, miny, maxx, maxy
    center_lon = float((bounds[0] + bounds[2]) / 2.0)
    center_lat = float((bounds[1] + bounds[3]) / 2.0)
    logger.info("Study area centroid: (%.4f, %.4f), Road segments: %d", center_lat, center_lon, len(roads_gdf))

    # 1.2 Buildings & Imperviousness Grid (with disk caching)
    grid_cache_path = PROCESSED_DATA_DIR / f"impervious_grid_{safe_loc_name}.geojson"
    if grid_cache_path.exists():
        logger.info("Loading cached imperviousness grid from %s...", grid_cache_path)
        impervious_grid = gpd.read_file(grid_cache_path)
    else:
        logger.info("Generating imperviousness grid from OSM buildings & roads...")
        buildings_gdf = fetch_buildings(place_name=location)
        # Sample buildings if region is massive to preserve fast nowcasting response
        sample_buildings = buildings_gdf.head(1500) if len(buildings_gdf) > 1500 else buildings_gdf
        impervious_grid = calculate_imperviousness_grid(
            buildings_gdf=sample_buildings,
            road_gdf=roads_gdf.head(1500) if len(roads_gdf) > 1500 else roads_gdf,
            grid_cell_size_m=grid_cell_size_m,
        )
        try:
            impervious_grid.to_file(grid_cache_path, driver="GeoJSON")
            logger.info("Cached imperviousness grid to %s", grid_cache_path)
        except Exception as save_err:
            logger.warning("Could not cache imperviousness grid: %s", save_err)

    # 1.3 DEM & Slope
    dem_path = RAW_DATA_DIR / "dem.tif"
    if not dem_path.exists():
        fetch_copernicus_dem(lat=center_lat, lon=center_lon, buffer_km=4.0, output_path=dem_path)
    slope_fn = create_dem_slope_lookup(dem_path=dem_path)

    # 1.4 Drainage Network
    drain_cache_path = PROCESSED_DATA_DIR / f"drainage_network_{safe_loc_name}.graphml"
    if drain_cache_path.exists():
        drainage_graph = load_network(drain_cache_path)
    else:
        logger.info("Constructing localized drainage graph centered at (%.4f, %.4f)...", center_lat, center_lon)
        base_drain = build_sample_drainage_network(
            num_nodes=25,
            center_lat=center_lat,
            center_lon=center_lon,
            spread_km=2.2,
            num_outfalls=3,
        )
        osm_waterways = fetch_surface_drainage(place_name=location)
        drainage_graph = calibrate_with_osm_drainage(base_drain, osm_waterways, max_snap_dist_m=250.0)
        save_network(drainage_graph, drain_cache_path)

    # -------------------------------------------------------------------------
    # STEP 2: METEOROLOGICAL RAINFALL RETRIEVAL / NOWCASTING
    # -------------------------------------------------------------------------
    rf_src = rainfall_source.strip().lower()
    if rf_src in ("open-meteo", "openmeteo", "weather", "radar"):
        try:
            from backend.external_api import fetch_open_meteo_rainfall_df
            rainfall_df = fetch_open_meteo_rainfall_df(lat=center_lat, lon=center_lon, hours_back=hours_back, state_name=inferred_state)
            provider = rainfall_df.attrs.get("provider", "Open-Meteo")
            is_fallback = rainfall_df.attrs.get("is_fallback", False)
            rainfall_source_used = f"Live_{provider}" if not is_fallback else f"Degraded_{provider}"
        except Exception as om_err:
            logger.warning("Failed querying Open-Meteo rainfall (%s). Falling back to calibrated synthetic...", om_err)
            rainfall_df = generate_calibrated_synthetic_rainfall(
                state_name=inferred_state,
                duration_minutes=180,
                severity=storm_severity,
            )
            rainfall_source_used = f"Calibrated_Synthetic_{inferred_state}_{storm_severity}"
    elif rf_src == "imerg":
        rainfall_df = fetch_recent_rainfall(lat=center_lat, lon=center_lon, hours_back=hours_back)
        rainfall_source_used = "NASA_GPM_IMERG_V07"
    else:
        rainfall_df = generate_calibrated_synthetic_rainfall(
            state_name=inferred_state,
            duration_minutes=180,
            severity=storm_severity,
        )
        rainfall_source_used = f"Calibrated_Synthetic_{inferred_state}_{storm_severity}"

    # Current nowcasting intensity: peak of the active storm cell (mm/hr)
    current_rainfall_intensity = float(rainfall_df["rainfall_mm_hr"].max())
    logger.info(
        "Active Nowcast Precipitation: %.1f mm/hr (Source: %s, Records: %d)",
        current_rainfall_intensity,
        rainfall_source_used,
        len(rainfall_df),
    )

    # -------------------------------------------------------------------------
    # STEP 3: HYDRAULIC RUNOFF & SURCHARGE PROPAGATION
    # -------------------------------------------------------------------------
    # 3.1 Distribute surface runoff from roads to drainage inlets
    inflow_dict = distribute_runoff_to_nodes(
        road_network=road_graph,
        drainage_graph=drainage_graph,
        rainfall_intensity_mm_hr=current_rainfall_intensity,
        imperviousness_grid=impervious_grid,
        slope_lookup=slope_fn,
    )

    # 3.2 Propagate through underground pipe network
    sim_result = propagate_flow(
        drainage_graph=drainage_graph,
        inflow_dict=inflow_dict,
        slope_lookup=slope_fn,
    )

    # 3.3 Map overland surcharge to nearest road segments
    roads_hydraulic = map_surcharge_to_road_segments(
        surcharge_result=sim_result,
        road_network=road_graph,
        impact_radius_m=impact_radius_m,
    )

    # -------------------------------------------------------------------------
    # STEP 4: MACHINE LEARNING RISK CLASSIFICATION & FUSION
    # -------------------------------------------------------------------------
    # Train/load XGBoost classifier
    model_path = PROCESSED_DATA_DIR / "flood_risk_model.pkl"
    if not model_path.exists():
        logger.info("Flood risk ML model bundle not found. Training model...")
        try:
            train_model(output_model_path=model_path)
        except Exception as train_err:
            logger.warning("Auto-training model failed (%s); heuristic fallback will be used.", train_err)

    # Compute ML risk for each drainage node based on physical state
    node_ml_cache: Dict[str, Dict[str, Any]] = {}
    mean_imperviousness = float(impervious_grid["impervious_ratio"].mean()) if not impervious_grid.empty else 0.70

    for n_id, data in drainage_graph.nodes(data=True):
        nx_val = data.get("x", center_lon)
        ny_val = data.get("y", center_lat)
        node_slope = slope_fn(ny_val, nx_val)

        # Check if node was surcharging
        is_surcharging = n_id in sim_result.get("surcharging_nodes", {})
        surcharge_info = sim_result.get("surcharging_nodes", {}).get(n_id, {})
        surcharge_ratio = float(surcharge_info.get("surcharge_ratio", 0.0 if not is_surcharging else 1.2))

        node_dict_for_ml = {
            "node_type": data.get("node_type", "inlet"),
            "elevation": data.get("elevation", 10.0),
            "capacity_liters_per_min": data.get("capacity_liters_per_min", 30000.0),
            "surcharge_ratio": surcharge_ratio,
            "osm_channel_dist_m": data.get("osm_channel_dist_m", 250.0),
        }

        features_df = map_node_features_to_model_input(
            drainage_node=node_dict_for_ml,
            rainfall_intensity=current_rainfall_intensity,
            imperviousness=mean_imperviousness,
            slope_deg=node_slope,
        )

        ml_res = predict_risk_ml(
            features_df,
            model_path=model_path,
            rainfall_intensity=current_rainfall_intensity,
            slope_deg=node_slope,
            imperviousness=mean_imperviousness,
        )
        node_ml_cache[n_id] = ml_res

    # 4.2 Fuse Hydraulic Risk with Machine Learning Risk per Road Segment
    roads_final = roads_hydraulic.copy()

    ml_categories = []
    ml_severe_probs = []
    final_risk_categories = []

    for _, row in roads_final.iterrows():
        hyd_risk = str(row.get("flood_risk_level", "LOW")).upper()
        near_node = row.get("nearest_surcharge_node")

        if near_node and near_node in node_ml_cache:
            ml_pred = node_ml_cache[near_node]
            ml_cat = ml_pred["predicted_class"]
            ml_prob = ml_pred["probabilities"].get("Severe", 0.0)
        else:
            # Baseline ML prediction for unaffected roads under current rainfall
            if current_rainfall_intensity > 70.0:
                ml_cat = "Moderate"
                ml_prob = 0.15
            else:
                ml_cat = "Low"
                ml_prob = 0.05

        ml_categories.append(ml_cat)
        ml_severe_probs.append(round(float(ml_prob), 4))

        # Risk fusion: conservative union of physical hydraulic surcharge and ML assessment
        h_rank = RISK_RANK.get(hyd_risk, 0)
        m_rank = RISK_RANK.get(ml_cat.upper(), 0)
        final_rank = max(h_rank, m_rank)
        final_risk_categories.append(RANK_TO_RISK[final_rank])

    roads_final["ml_risk_category"] = ml_categories
    roads_final["ml_severe_probability"] = ml_severe_probs
    roads_final["final_risk_category"] = final_risk_categories
    roads_final["rainfall_intensity_mm_hr"] = current_rainfall_intensity
    roads_final["rainfall_source"] = rainfall_source_used

    # -------------------------------------------------------------------------
    # STEP 5: BUILD GEOJSON FEATURE COLLECTION
    # -------------------------------------------------------------------------
    # Ensure CRS is WGS84 EPSG:4326 for standard web mapping (Folium, Streamlit, Mapbox)
    if roads_final.crs and roads_final.crs.to_epsg() != 4326:
        roads_final = roads_final.to_crs("EPSG:4326")

    # Select and format relevant properties for JSON serialization
    export_cols = [
        "name",
        "highway",
        "length",
        "flood_risk_level",
        "ml_risk_category",
        "final_risk_category",
        "surcharge_volume_lpm",
        "surcharge_ratio",
        "ml_severe_probability",
        "rainfall_intensity_mm_hr",
        "rainfall_source",
        "geometry",
    ]
    present_cols = [c for c in export_cols if c in roads_final.columns]
    geojson_gdf = roads_final[present_cols].copy()

    # Clean non-serializable types (lists in OSM tags)
    for col in geojson_gdf.columns:
        if col != "geometry":
            geojson_gdf[col] = geojson_gdf[col].apply(
                lambda x: ", ".join(x) if isinstance(x, (list, tuple)) else (None if pd.isna(x) else x)
            )

    # Convert to Python GeoJSON FeatureCollection dictionary
    geojson_dict = json.loads(geojson_gdf.to_json())

    # Attach pipeline-level simulation metadata
    geojson_dict["metadata"] = {
        "location": location,
        "centroid": {"lat": center_lat, "lon": center_lon},
        "rainfall_source": rainfall_source_used,
        "rainfall_intensity_mm_hr": current_rainfall_intensity,
        "total_surface_runoff_lpm": sim_result["total_surface_runoff_lpm"],
        "total_surcharge_lpm": sim_result["total_surcharge_lpm"],
        "surcharging_node_count": sim_result["surcharge_node_count"],
        "surcharging_pipe_count": sim_result["surcharge_pipe_count"],
        "road_segments_count": len(roads_final),
        "risk_breakdown": roads_final["final_risk_category"].value_counts().to_dict(),
    }

    if export_geojson_path:
        out_p = Path(export_geojson_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(geojson_dict, f, indent=2)
        logger.info("Exported final nowcasting GeoJSON to %s", out_p)

    logger.info("Pipeline completed successfully. Final Risk Breakdown: %s", geojson_dict["metadata"]["risk_breakdown"])
    return geojson_dict


if __name__ == "__main__":
    print("\n========================================================")
    print("      Testing engine/pipeline.py (Full Nowcast)         ")
    print("========================================================")

    # Run complete end-to-end nowcasting pipeline for Adyar, Chennai
    study_location = "Adyar, Chennai, Tamil Nadu, India"
    geojson_result = run_nowcast_pipeline(
        location=study_location,
        rainfall_source="imerg",  # Live GPM IMERG (with calibrated synthetic fallback)
        storm_severity="extreme",
        grid_cell_size_m=200.0,
        export_geojson_path=PROJECT_ROOT / "data" / "processed" / "nowcast_output.geojson",
    )

    meta = geojson_result.get("metadata", {})
    features = geojson_result.get("features", [])

    print("\n[PIPELINE EXECUTION SUMMARY]")
    print(f"  Location:                {meta.get('location')}")
    print(f"  Coordinates (Center):    Lat {meta['centroid']['lat']:.4f}, Lon {meta['centroid']['lon']:.4f}")
    print(f"  Precipitation Used:      {meta.get('rainfall_intensity_mm_hr')} mm/hr ({meta.get('rainfall_source')})")
    print(f"  Total System Inflow:     {meta.get('total_surface_runoff_lpm'):,.1f} L/min")
    print(f"  Hydraulic Surcharges:    {meta.get('surcharging_node_count')} nodes, {meta.get('surcharging_pipe_count')} pipes")
    print(f"  Total Surcharge Volume:  {meta.get('total_surcharge_lpm'):,.1f} L/min")
    print(f"  Total Road Segments:     {meta.get('road_segments_count')}")
    print(f"  Final Risk Breakdown:    {meta.get('risk_breakdown')}")

    # Inspect first 2 sample features
    print(f"\n[SAMPLE GEOJSON FEATURE PROPERTIES (First 2 Segments)]")
    for i, feat in enumerate(features[:2]):
        props = feat.get("properties", {})
        print(f"  Segment #{i+1}: Name='{props.get('name')}', HydraulicRisk={props.get('flood_risk_level')}, MLRisk={props.get('ml_risk_category')}, FinalRisk={props.get('final_risk_category')}")

    print("\n[OK] Urban flood nowcast pipeline executed successfully!")
