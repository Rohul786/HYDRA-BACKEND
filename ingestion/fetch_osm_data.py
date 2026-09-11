"""
fetch_osm_data.py - OpenStreetMap Urban Infrastructure & Imperviousness Ingestion

Provides functions to:
1. Fetch drivable road networks with OSMnx and cache/load from data/processed/road_network.graphml.
2. Fetch building footprints as GeoDataFrames.
3. Fetch surface drainage features (drains, ditches, streams, canals).
4. Overlay buildings and roads on a metric grid to compute imperviousness coefficients per cell.
"""

import logging
from pathlib import Path
from typing import Optional, Union, Tuple, List, Any

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import osmnx as ox
import shapely
from shapely.geometry import box, Polygon, MultiPolygon

logger = logging.getLogger("ingestion.fetch_osm_data")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Default paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROAD_GRAPH_PATH = PROJECT_ROOT / "data" / "processed" / "road_network.graphml"

# Ensure OSMnx HTTP cache routines use safe JSON parsing and atomic writes
try:
    from backend.cache_manager import patch_osmnx_cache
    patch_osmnx_cache()
except Exception as patch_err:
    logger.debug("Could not auto-patch OSMnx cache: %s", patch_err)


def _fetch_features_robust(place_name: str, tags: dict, dist_m: int = 1500) -> gpd.GeoDataFrame:
    """
    Helper to fetch OSM features by place name.
    If the place geocodes to a Point rather than a Polygon in Nominatim,
    automatically falls back to features_from_point with dist_m radius.
    """
    try:
        gdf = ox.features_from_place(place_name, tags=tags)
        return gdf
    except TypeError as type_err:
        # Nominatim geocoded to Point instead of (Multi)Polygon
        logger.info(
            "Place '%s' geocoded to a Point rather than a Polygon (%s). Querying buffer radius of %dm...",
            place_name,
            type_err,
            dist_m,
        )
        pt = ox.geocode(place_name)
        gdf = ox.features_from_point(pt, tags=tags, dist=dist_m)
        return gdf
    except Exception as exc:
        logger.warning("Failed querying OSM features for '%s' with tags %s: %s", place_name, tags, exc)
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")


def fetch_road_network(
    place_name: str,
    filepath: Optional[Union[str, Path]] = None,
    force_reload: bool = False,
    dist_m: int = 1500,
) -> nx.MultiDiGraph:
    """
    Fetches the drivable road graph for a given place name and caches it as GraphML.
    Reuses the existing GraphML file if already present unless force_reload=True.

    Args:
        place_name (str): Geographic place name (e.g., "Adyar, Chennai, India").
        filepath (str or Path, optional): Custom path for the .graphml file.
                                          Defaults to data/processed/road_network.graphml.
        force_reload (bool): If True, ignores cached file and re-downloads from OSM.
        dist_m (int): Buffer distance in meters used if place geocodes to a point.

    Returns:
        nx.MultiDiGraph: Drivable road network graph.
    """
    dest_path = Path(filepath) if filepath else DEFAULT_ROAD_GRAPH_PATH
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Reuse existing GraphML file if available
    if dest_path.exists() and not force_reload:
        logger.info("Loading existing road network graph from cache: %s", dest_path)
        try:
            G = ox.load_graphml(dest_path)
            logger.info("Loaded graph with %d nodes and %d edges.", len(G.nodes), len(G.edges))
            return G
        except Exception as load_err:
            logger.warning("Failed loading cached GraphML (%s). Re-fetching from OSM...", load_err)

    # 2. Download from OSM
    logger.info("Downloading drivable road network for '%s' from OpenStreetMap...", place_name)
    try:
        G = ox.graph_from_place(place_name, network_type="drive")
    except TypeError:
        logger.info("Place '%s' geocoded to a Point. Querying point buffer radius of %dm...", place_name, dist_m)
        pt = ox.geocode(place_name)
        G = ox.graph_from_point(pt, dist=dist_m, network_type="drive")

    # If empty drivable roads (e.g., inside private campus), fallback to all roads
    if len(G.edges) == 0:
        logger.warning("Drivable graph has 0 edges; falling back to network_type='all'...")
        try:
            G = ox.graph_from_place(place_name, network_type="all")
        except TypeError:
            pt = ox.geocode(place_name)
            G = ox.graph_from_point(pt, dist=dist_m, network_type="all")

    # 3. Save to disk
    try:
        ox.save_graphml(G, dest_path)
        logger.info(
            "Saved road network graph (%d nodes, %d edges) to %s",
            len(G.nodes),
            len(G.edges),
            dest_path,
        )
    except Exception as save_err:
        logger.warning("Could not cache GraphML to %s: %s", dest_path, save_err)

    return G


def fetch_buildings(place_name: str, dist_m: int = 1500) -> gpd.GeoDataFrame:
    """
    Fetches building footprints as a GeoDataFrame via OSMnx features.

    Args:
        place_name (str): Geographic query (e.g., "Adyar, Chennai, India").
        dist_m (int): Search distance in meters if query resolves to a point.

    Returns:
        gpd.GeoDataFrame: Building footprints with geometry column.
    """
    logger.info("Fetching building footprints for '%s'...", place_name)
    gdf = _fetch_features_robust(place_name, tags={"building": True}, dist_m=dist_m)
    
    if gdf.empty:
        logger.warning("No building footprints found for '%s'.", place_name)
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    # Filter to only valid polygons/multipolygons
    poly_mask = gdf.geometry.type.isin(["Polygon", "MultiPolygon"])
    gdf = gdf[poly_mask & gdf.geometry.is_valid & ~gdf.geometry.is_empty].copy()
    logger.info("Found %d valid building footprint polygons for '%s'.", len(gdf), place_name)
    return gdf


def fetch_surface_drainage(place_name: str, dist_m: int = 1500) -> gpd.GeoDataFrame:
    """
    Fetches surface drainage waterway features tagged as drain, ditch, stream, canal.

    Args:
        place_name (str): Geographic query (e.g., "Adyar, Chennai, India").
        dist_m (int): Search distance in meters if query resolves to a point.

    Returns:
        gpd.GeoDataFrame: Drainage line and polygon features.
    """
    tags = {"waterway": ["drain", "ditch", "stream", "canal"]}
    logger.info("Fetching surface drainage features for '%s' (tags: %s)...", place_name, tags)
    gdf = _fetch_features_robust(place_name, tags=tags, dist_m=dist_m)

    if gdf.empty:
        logger.warning("No surface drainage features found for '%s'.", place_name)
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    gdf = gdf[gdf.geometry.is_valid & ~gdf.geometry.is_empty].copy()
    logger.info("Found %d surface drainage features for '%s'.", len(gdf), place_name)
    return gdf


def calculate_imperviousness_grid(
    buildings_gdf: gpd.GeoDataFrame,
    road_gdf: Union[gpd.GeoDataFrame, nx.MultiDiGraph],
    grid_cell_size_m: float = 100.0,
    road_buffer_m: float = 4.0,
) -> gpd.GeoDataFrame:
    """
    Overlays building footprints and road geometries onto a metric grid of size grid_cell_size_m,
    calculating the imperviousness coefficient (impervious_ratio: 0.0 to 1.0) per cell.

    Args:
        buildings_gdf (gpd.GeoDataFrame): Building polygons.
        road_gdf (gpd.GeoDataFrame or nx.MultiDiGraph): Road network geometries or Graph.
        grid_cell_size_m (float): Dimensions of each square grid cell in meters (default: 100m).
        road_buffer_m (float): Buffer radius for line roads in meters to simulate road width (default: 4m).

    Returns:
        gpd.GeoDataFrame: Grid cells with columns:
            - 'cell_id': int index
            - 'geometry': shapely Polygon of cell (in EPSG:4326)
            - 'impervious_ratio': float [0.0, 1.0]
            - 'impervious_area_m2': float
            - 'total_area_m2': float
            - 'centroid_lat': float
            - 'centroid_lon': float
    """
    # 1. Normalize road input: convert nx graph to edges GeoDataFrame if necessary
    if isinstance(road_gdf, (nx.MultiDiGraph, nx.DiGraph, nx.Graph)):
        roads = ox.graph_to_gdfs(road_gdf, nodes=False, edges=True)
    else:
        roads = road_gdf.copy()

    buildings = buildings_gdf.copy()

    # If both inputs are empty, return empty grid
    if (buildings.empty or len(buildings) == 0) and (roads.empty or len(roads) == 0):
        logger.warning("Both buildings and roads inputs are empty. Returning empty grid.")
        return gpd.GeoDataFrame(
            columns=["cell_id", "geometry", "impervious_ratio", "impervious_area_m2", "total_area_m2", "centroid_lat", "centroid_lon"],
            crs="EPSG:4326",
        )

    # 2. Determine metric coordinate system (UTM)
    if not buildings.empty and buildings.crs:
        metric_crs = buildings.estimate_utm_crs()
    elif not roads.empty and roads.crs:
        metric_crs = roads.estimate_utm_crs()
    else:
        metric_crs = "EPSG:3857"

    logger.info("Projecting geometries to metric CRS (%s) for accurate area computation...", metric_crs)
    b_proj = buildings.to_crs(metric_crs) if not buildings.empty else gpd.GeoDataFrame(geometry=[], crs=metric_crs)
    r_proj = roads.to_crs(metric_crs) if not roads.empty else gpd.GeoDataFrame(geometry=[], crs=metric_crs)

    # 3. Buffer linear road geometries to polygons
    impervious_geoms: List[Any] = []
    if not b_proj.empty:
        valid_b = b_proj.geometry[b_proj.geometry.is_valid & ~b_proj.geometry.is_empty]
        impervious_geoms.extend(valid_b)

    if not r_proj.empty:
        valid_r = r_proj.geometry[r_proj.geometry.is_valid & ~r_proj.geometry.is_empty]
        # Buffer linestrings to polygons representing road footprint
        buffered_roads = valid_r.buffer(road_buffer_m)
        impervious_geoms.extend(buffered_roads)

    if not impervious_geoms:
        logger.warning("No valid impervious geometries found.")
        return gpd.GeoDataFrame(
            columns=["cell_id", "geometry", "impervious_ratio", "impervious_area_m2", "total_area_m2", "centroid_lat", "centroid_lon"],
            crs="EPSG:4326",
        )

    # Combine into a single unified polygon/multipolygon for intersection
    logger.info("Dissolving %d impervious surface elements...", len(impervious_geoms))
    impervious_union = shapely.unary_union(impervious_geoms)

    # 4. Generate regular bounding grid
    minx, miny, maxx, maxy = impervious_union.bounds
    x_steps = np.arange(minx, maxx + grid_cell_size_m, grid_cell_size_m)
    y_steps = np.arange(miny, maxy + grid_cell_size_m, grid_cell_size_m)

    grid_boxes = []
    for i in range(len(x_steps) - 1):
        for j in range(len(y_steps) - 1):
            grid_boxes.append(box(x_steps[i], y_steps[j], x_steps[i + 1], y_steps[j + 1]))

    logger.info(
        "Constructed %d grid cells (cell size: %0.1fm x %0.1fm) across study region.",
        len(grid_boxes),
        grid_cell_size_m,
        grid_cell_size_m,
    )

    # 5. Compute intersection and impervious ratio
    cell_area_nominal = grid_cell_size_m * grid_cell_size_m
    grid_geoms_array = np.array(grid_boxes)

    # C-vectorized shapely intersections
    intersections = shapely.intersection(grid_geoms_array, impervious_union)
    impervious_areas = shapely.area(intersections)
    ratios = np.clip(impervious_areas / cell_area_nominal, 0.0, 1.0)

    # 6. Create GeoDataFrame and reproject to standard EPSG:4326
    grid_metric = gpd.GeoDataFrame(
        {
            "cell_id": np.arange(len(grid_boxes)),
            "geometry": grid_geoms_array,
            "impervious_ratio": np.round(ratios, 4),
            "impervious_area_m2": np.round(impervious_areas, 2),
            "total_area_m2": cell_area_nominal,
        },
        crs=metric_crs,
    )

    # Compute centroids in EPSG:4326 for straightforward map rendering & API consumption
    centroids_wgs84 = grid_metric.geometry.centroid.to_crs("EPSG:4326")
    grid_wgs84 = grid_metric.to_crs("EPSG:4326")
    grid_wgs84["centroid_lat"] = np.round(centroids_wgs84.y, 6)
    grid_wgs84["centroid_lon"] = np.round(centroids_wgs84.x, 6)

    logger.info(
        "Imperviousness calculation complete: Mean ratio = %.2f, Max ratio = %.2f.",
        grid_wgs84["impervious_ratio"].mean(),
        grid_wgs84["impervious_ratio"].max(),
    )
    return grid_wgs84


if __name__ == "__main__":
    print("\n==================================================")
    print("       Testing ingestion/fetch_osm_data.py        ")
    print("==================================================")

    # Test on a compact urban location (Adyar, Chennai, India)
    test_place = "Adyar, Chennai, Tamil Nadu, India"
    print(f"\n[1] Querying Study Area: {test_place}")

    # 1. Fetch road network
    road_graph = fetch_road_network(test_place)
    road_edges_count = len(road_graph.edges)
    print(f"-> Roads Found: {road_edges_count} edges ({len(road_graph.nodes)} intersections)")

    # 2. Fetch buildings
    buildings = fetch_buildings(test_place)
    print(f"-> Buildings Found: {len(buildings)} structures")

    # 3. Fetch surface drainage
    drainage = fetch_surface_drainage(test_place)
    print(f"-> Surface Drainage Features Found: {len(drainage)} channels")

    # 4. Calculate imperviousness grid
    print(f"\n[2] Computing Imperviousness Grid (cell_size=200m for fast test)...")
    # Take a representative spatial sample if building dataset is large
    sample_buildings = buildings.head(600) if len(buildings) > 600 else buildings
    road_gdf = ox.graph_to_gdfs(road_graph, nodes=False, edges=True)
    sample_roads = road_gdf.head(600) if len(road_gdf) > 600 else road_gdf

    grid_df = calculate_imperviousness_grid(
        buildings_gdf=sample_buildings,
        road_gdf=sample_roads,
        grid_cell_size_m=200.0,
    )

    print(f"-> Generated {len(grid_df)} grid cells")
    print(grid_df[["cell_id", "impervious_ratio", "impervious_area_m2", "centroid_lat", "centroid_lon"]].head())
    print("\n[OK] OSM ingestion module verified successfully!")
