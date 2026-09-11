"""
routes_routing.py - Flood-Aware Safe Emergency Navigation Router

Exposes RESTful endpoints to calculate shortest safe routes across the urban
road network, automatically penalizing and detouring around High and Severe
flood-risk road segments identified by hydraulic nowcasts and drainage network surcharge.
"""

import sys
import ast
import json
import time
import math
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Set, Union

# Ensure project root is present in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import networkx as nx
import osmnx as ox
from shapely.geometry import LineString, Point
from fastapi import APIRouter, Query, HTTPException, Request
from fastapi.responses import JSONResponse

from backend.routes_risk import risk_cache
from engine.pipeline import run_nowcast_pipeline
from ingestion.drainage_graph import DEFAULT_DRAINAGE_GRAPH_PATH, load_network

logger = logging.getLogger("backend.routes_routing")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

router = APIRouter(prefix="", tags=["Routing"])

DEFAULT_ROAD_GRAPH_PATH = PROJECT_ROOT / "data" / "processed" / "road_network.graphml"
DEFAULT_GEOJSON_PATH = PROJECT_ROOT / "data" / "processed" / "nowcast_output.geojson"


# -----------------------------------------------------------------------------
# Coordinate Formatting & Conversion Helpers
# -----------------------------------------------------------------------------

def to_geojson_coords(lat: float, lon: float) -> List[float]:
    """
    Ensures coordinates emitted in GeoJSON payloads strictly use [longitude, latitude],
    conforming with RFC 7946 specifications.
    """
    return [round(float(lon), 6), round(float(lat), 6)]


def to_leaflet_coords(lon: float, lat: float) -> Tuple[float, float]:
    """
    Reverses GeoJSON [longitude, latitude] into (latitude, longitude)
    when feeding Leaflet, OSMnx, or classical GIS routines expecting (lat, lon).
    """
    return (round(float(lat), 6), round(float(lon), 6))


def ensure_geojson_coord(pt: Any) -> List[float]:
    """
    Ensures a coordinate pair is formatted as strict GeoJSON [longitude, latitude].
    """
    if isinstance(pt, (list, tuple)) and len(pt) >= 2:
        return [round(float(pt[0]), 6), round(float(pt[1]), 6)]
    return [0.0, 0.0]


# -----------------------------------------------------------------------------
# Safe Attribute Parsing & Zero-Division-Safe Cost Calculation
# -----------------------------------------------------------------------------

def _safe_float(val: Any, default: float) -> float:
    """
    Safely converts arbitrary values (including OSM strings with units or lists)
    to a valid non-NaN, non-infinite float, returning `default` on error.
    """
    if val is None:
        return default
    if isinstance(val, (int, float)):
        f_val = float(val)
        return f_val if not (math.isnan(f_val) or math.isinf(f_val)) else default
    if isinstance(val, list) and val:
        return _safe_float(val[0], default)
    if isinstance(val, str):
        cleaned = val.strip().split()[0]
        try:
            f_val = float(cleaned)
            return f_val if not (math.isnan(f_val) or math.isinf(f_val)) else default
        except (ValueError, TypeError):
            return default
    return default


def calculate_edge_traversal_cost(
    edge: Dict[str, Any],
    is_flooded: bool = False,
    penalty_multiplier: float = 100.0,
    flood_depth: Optional[float] = None,
) -> float:
    """
    Calculates edge traversal cost with strict ZeroDivisionError prevention:
    - Safe dictionary access using sensible defaults:
        edge.get("length", 10.0) -> meters
        edge.get("maxspeed", 30.0) -> km/h
        edge.get("risk_score", 1.0) -> dimensionless risk multiplier
    - Dynamic flood depth and speed degradation safeguards against zero division.
    - Guaranteed strictly positive traversal cost.
    """
    raw_len = _safe_float(edge.get("length", 10.0), 10.0)
    raw_speed = _safe_float(edge.get("maxspeed", 30.0), 30.0)
    raw_risk = _safe_float(edge.get("risk_score", 1.0), 1.0)

    # Prevent zero or negative length
    safe_length = max(raw_len, 0.1)

    # Convert km/h to m/s, safeguarding against zero/negative speed
    safe_speed_kmh = max(raw_speed, 1.0)
    speed_ms = safe_speed_kmh / 3.6

    # Dynamic flood depth consideration
    depth_m = max(
        flood_depth if flood_depth is not None else _safe_float(edge.get("flood_depth", 0.0), 0.0),
        0.0,
    )

    # Speed reduction factor due to water depth (depth >= 0.5m immobilizes vehicle)
    # Floor speed factor at 0.05 so denominator never drops to zero
    speed_factor = max(0.05, 1.0 - (depth_m / 0.5)) if depth_m > 0.0 else 1.0
    effective_speed_ms = max(speed_ms * speed_factor, 0.2)

    # Base traversal travel time in seconds: safe_length / effective_speed_ms
    base_time_sec = safe_length / effective_speed_ms

    # Safe risk multiplier and penalty multiplier
    safe_risk = max(raw_risk, 0.01)
    safe_penalty = max(penalty_multiplier, 1.0)

    if is_flooded or depth_m >= 0.15:
        # High/Severe flood surcharge penalty
        cost = base_time_sec * safe_risk * safe_penalty * (1.0 + depth_m * 10.0)
    else:
        cost = base_time_sec * safe_risk

    return max(cost, 0.001)


# -----------------------------------------------------------------------------
# Graph Loaders & Traversal Utilities (Road & Drainage Networks)
# -----------------------------------------------------------------------------

def get_road_network(request: Optional[Request] = None) -> nx.MultiDiGraph:
    """
    Locates and acquires the road network graph from memory (app.state)
    or loads it from road_network.graphml on disk.
    """
    if request is not None:
        road_graph = getattr(request.app.state, "road_network", None)
        if road_graph is not None:
            return road_graph

    logger.warning("Road graph not in app.state. Searching disk for road_network.graphml...")
    road_path = DEFAULT_ROAD_GRAPH_PATH
    if not road_path.exists():
        candidates = list((PROJECT_ROOT / "data" / "processed").glob("road_network*.graphml"))
        if candidates:
            road_path = candidates[0]

    if not road_path.exists():
        raise HTTPException(
            status_code=500,
            detail="Road network graph is not available. Please run data ingestion first.",
        )

    road_graph = ox.load_graphml(road_path)
    if request is not None:
        request.app.state.road_network = road_graph
    return road_graph


def get_drainage_network(request: Optional[Request] = None) -> Optional[nx.DiGraph]:
    """
    Locates and acquires the drainage network graph from memory (app.state)
    or loads it from drainage_network.graphml on disk.
    """
    if request is not None:
        drainage_graph = getattr(request.app.state, "drainage_network", None)
        if drainage_graph is not None:
            return drainage_graph

    drainage_path = DEFAULT_DRAINAGE_GRAPH_PATH
    if not drainage_path.exists():
        candidates = list((PROJECT_ROOT / "data" / "processed").glob("drainage_network*.graphml"))
        if candidates:
            drainage_path = candidates[0]

    if drainage_path.exists():
        try:
            drainage_graph = load_network(drainage_path)
            if request is not None:
                request.app.state.drainage_network = drainage_graph
            return drainage_graph
        except Exception as exc:
            logger.warning("Could not load drainage graph: %s", exc)

    return None


def get_drainage_surcharged_locations(drainage_g: Optional[nx.DiGraph]) -> List[Tuple[float, float, float]]:
    """
    Traverses the drainage network to identify surcharged or backflowing inlets/outfalls.
    Returns list of (lat, lon, surcharge_depth).
    """
    if drainage_g is None:
        return []

    surcharged = []
    for _, data in drainage_g.nodes(data=True):
        surcharge_ratio = _safe_float(data.get("surcharge_ratio", 0.0), 0.0)
        flood_depth = _safe_float(data.get("flood_depth", 0.0), 0.0)
        is_surcharged = bool(data.get("is_surcharged", False))

        if surcharge_ratio > 1.0 or flood_depth > 0.05 or is_surcharged:
            lat = _safe_float(data.get("y"), 0.0)
            lon = _safe_float(data.get("x"), 0.0)
            if lat != 0.0 and lon != 0.0:
                depth = max(flood_depth, (surcharge_ratio - 1.0) * 0.3)
                surcharged.append((lat, lon, depth))

    return surcharged


# -----------------------------------------------------------------------------
# Dijkstra & A* Pathfinding Routines
# -----------------------------------------------------------------------------

def compute_shortest_path(
    G: nx.MultiDiGraph,
    orig_node: int,
    dest_node: int,
    weight: Any,
    algorithm: str = "dijkstra",
) -> List[int]:
    """
    Executes Dijkstra or A* pathfinding between orig_node and dest_node on G.
    - 'dijkstra': Classical shortest path exploration via priority queue.
    - 'astar': Heuristic-directed exploration using Euclidean distance between node coordinates.
    """
    algo = str(algorithm).strip().lower()

    if algo == "astar":
        dest_data = G.nodes.get(dest_node, {})
        dest_x = _safe_float(dest_data.get("x", 0.0), 0.0)
        dest_y = _safe_float(dest_data.get("y", 0.0), 0.0)

        def astar_heuristic(u: int, v: int) -> float:
            """Admissible metric heuristic (straight-line distance / max expected speed)."""
            u_data = G.nodes.get(u, {})
            ux = _safe_float(u_data.get("x", 0.0), 0.0)
            uy = _safe_float(u_data.get("y", 0.0), 0.0)

            # Flat-earth metric distance approximation
            dx_m = (dest_x - ux) * 111320.0 * math.cos(math.radians((uy + dest_y) / 2.0))
            dy_m = (dest_y - uy) * 111320.0
            dist_m = math.sqrt(dx_m * dx_m + dy_m * dy_m)

            # Divide by maximum expected speed (~80 km/h = 22.2 m/s) to keep heuristic admissible
            return dist_m / 22.2

        return nx.astar_path(G, orig_node, dest_node, heuristic=astar_heuristic, weight=weight)
    else:
        return nx.dijkstra_path(G, orig_node, dest_node, weight=weight)


def _parse_edge_identifier(raw_id: Any) -> Optional[Tuple[int, int, int]]:
    """Parses raw feature id into an (u, v, key) integer tuple."""
    if isinstance(raw_id, (tuple, list)) and len(raw_id) >= 2:
        try:
            return (
                int(raw_id[0]),
                int(raw_id[1]),
                int(raw_id[2]) if len(raw_id) > 2 else 0,
            )
        except (ValueError, TypeError):
            return None

    if isinstance(raw_id, str):
        try:
            val = ast.literal_eval(raw_id)
            if isinstance(val, (tuple, list)) and len(raw_id) >= 2:
                return (
                    int(val[0]),
                    int(val[1]),
                    int(val[2]) if len(val) > 2 else 0,
                )
        except Exception:
            pass

    return None


def get_latest_flood_risk_edges() -> Tuple[Set[Tuple], str]:
    """
    Retrieves the set of currently surcharged / high-risk road edges from:
    1. Active in-memory risk_cache.
    2. Cached nowcast_output.geojson on disk.
    3. Auto-fallback to live pipeline nowcast.

    Returns:
        Tuple[Set[Tuple], str]: Set of flooded edge tuples and source description.
    """
    flooded_edges: Set[Tuple] = set()
    source_name = "none"
    geojson_data: Optional[Dict[str, Any]] = None

    # 1. Check in-memory risk_cache
    with risk_cache._lock:
        if risk_cache._cache:
            latest_key = max(risk_cache._cache.keys(), key=lambda k: risk_cache._cache[k][0])
            cached_time, geojson_data = risk_cache._cache[latest_key]
            source_name = f"in_memory_cache:{latest_key}"

    # 2. Check disk GeoJSON if cache was empty
    if not geojson_data and DEFAULT_GEOJSON_PATH.exists():
        try:
            with open(DEFAULT_GEOJSON_PATH, "r", encoding="utf-8") as f:
                geojson_data = json.load(f)
            source_name = f"disk_cache:{DEFAULT_GEOJSON_PATH.name}"
        except Exception as e:
            logger.warning("Could not read disk nowcast file: %s", e)

    # 3. Fallback to computing nowcast if still empty
    if not geojson_data:
        logger.info("No prior flood-risk result found. Triggering baseline nowcast...")
        try:
            geojson_data = run_nowcast_pipeline()
            source_name = "live_pipeline_fallback"
        except Exception as e:
            logger.error("Failed to generate fallback flood nowcast: %s", e)
            return set(), "unavailable"

    # Extract High and Severe risk edges
    features = geojson_data.get("features", [])
    for feat in features:
        props = feat.get("properties", {})
        final_risk = str(props.get("final_risk_category", "")).upper()
        hyd_risk = str(props.get("flood_risk_level", "")).upper()

        if final_risk in ("HIGH", "SEVERE") or hyd_risk in ("HIGH", "SEVERE"):
            raw_id = feat.get("id")
            edge_tuple = _parse_edge_identifier(raw_id)
            if edge_tuple:
                u, v, k = edge_tuple
                flooded_edges.add((u, v, k))
                flooded_edges.add((u, v))
                # Add bidirectional risk for undirected street navigation safety
                flooded_edges.add((v, u, k))
                flooded_edges.add((v, u))

    logger.info(
        "Extracted %d flooded edge references from %s",
        len(flooded_edges),
        source_name,
    )
    return flooded_edges, source_name


def _build_route_linestring(
    G: nx.MultiDiGraph,
    path_nodes: List[int],
    flooded_edges: Set[Tuple],
) -> Tuple[List[List[float]], float, int]:
    """
    Constructs a detailed GeoJSON coordinate list strictly formatted as [longitude, latitude]
    and metric distance from a sequence of graph nodes, preserving high-resolution curve geometry.
    Safely retrieves edge attributes with fallback defaults.
    """
    coordinates: List[List[float]] = []
    total_distance_m = 0.0
    traversed_flooded_count = 0

    for i in range(len(path_nodes) - 1):
        u = path_nodes[i]
        v = path_nodes[i + 1]

        edge_data_dict = G.get_edge_data(u, v)
        if not edge_data_dict:
            # Fallback direct node connection
            node_u = G.nodes.get(u, {})
            node_v = G.nodes.get(v, {})
            ux, uy = _safe_float(node_u.get("x"), 0.0), _safe_float(node_u.get("y"), 0.0)
            vx, vy = _safe_float(node_v.get("x"), 0.0), _safe_float(node_v.get("y"), 0.0)
            coords = [[ux, uy], [vx, vy]]
            dist = 10.0
        else:
            # Select edge key with minimum traversal cost
            best_key = min(
                edge_data_dict.keys(),
                key=lambda k: calculate_edge_traversal_cost(
                    edge_data_dict[k],
                    is_flooded=((u, v, k) in flooded_edges or (u, v) in flooded_edges),
                    penalty_multiplier=1.0,
                ),
            )
            best_attrs = edge_data_dict[best_key]
            # Safe attribute access with sensible default
            dist = _safe_float(best_attrs.get("length", 10.0), 10.0)

            if (u, v, best_key) in flooded_edges or (u, v) in flooded_edges:
                traversed_flooded_count += 1

            if "geometry" in best_attrs and isinstance(best_attrs["geometry"], LineString):
                geom = best_attrs["geometry"]
                # Shapely LineString coords are (x, y) = (longitude, latitude)
                coords = [[float(c[0]), float(c[1])] for c in geom.coords]
            else:
                node_u = G.nodes.get(u, {})
                node_v = G.nodes.get(v, {})
                ux, uy = _safe_float(node_u.get("x"), 0.0), _safe_float(node_u.get("y"), 0.0)
                vx, vy = _safe_float(node_v.get("x"), 0.0), _safe_float(node_v.get("y"), 0.0)
                coords = [[ux, uy], [vx, vy]]

        total_distance_m += dist

        # Stitch coordinates strictly ensuring [longitude, latitude] GeoJSON format
        for pt in coords:
            geojson_pt = ensure_geojson_coord(pt)
            if not coordinates or coordinates[-1] != geojson_pt:
                coordinates.append(geojson_pt)

    return coordinates, total_distance_m, traversed_flooded_count


# -----------------------------------------------------------------------------
# RESTful Safe Route API Endpoint
# -----------------------------------------------------------------------------

@router.get(
    "/safe-route",
    summary="Compute Shortest Safe Route Avoiding Flood Zones",
    description=(
        "Calculates an emergency safe route between start and end coordinates. "
        "Road segments classified as High or Severe risk by the latest hydraulic nowcast "
        "are removed or heavily penalized, returning the shortest detour as a GeoJSON LineString."
    ),
    response_description="GeoJSON Feature with LineString geometry and routing metrics",
)
@router.get(
    "/route/safe",
    summary="Alias for /safe-route",
    include_in_schema=False,
)
async def get_safe_route(
    request: Request,
    start_lat: float = Query(..., description="Starting latitude (e.g. 13.0180)", ge=-90.0, le=90.0),
    start_lon: float = Query(..., description="Starting longitude (e.g. 80.2400)", ge=-180.0, le=180.0),
    end_lat: float = Query(..., description="Destination latitude (e.g. 13.0030)", ge=-90.0, le=90.0),
    end_lon: float = Query(..., description="Destination longitude (e.g. 80.2400)", ge=-180.0, le=180.0),
    avoid_floods: bool = Query(
        True,
        description="If True, removes/penalizes High and Severe flood risk segments.",
    ),
    penalty_multiplier: float = Query(
        100.0,
        description="Cost multiplier for flooded edges during fallback routing if path is disconnected.",
        ge=1.0,
    ),
    algorithm: str = Query(
        "dijkstra",
        description="Pathfinding algorithm to execute: 'dijkstra' or 'astar'.",
    ),
):
    """
    GET /safe-route?start_lat=...&start_lon=...&end_lat=...&end_lon=...
    """
    start_time = time.perf_counter()

    # 1. Acquire Road and Drainage Graphs
    road_graph = get_road_network(request)
    drainage_graph = get_drainage_network(request)

    # 2. Snap coordinates to nearest road network nodes (X=lon, Y=lat)
    try:
        orig_node = ox.nearest_nodes(road_graph, X=start_lon, Y=start_lat)
        dest_node = ox.nearest_nodes(road_graph, X=end_lon, Y=end_lat)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Failed snapping coordinates to road network: {str(exc)}",
        )

    if orig_node == dest_node:
        # Return trivial route with strict GeoJSON [longitude, latitude] coordinates
        same_node_coords = [
            to_geojson_coords(start_lat, start_lon),
            to_geojson_coords(end_lat, end_lon),
        ]
        return JSONResponse(
            content={
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": same_node_coords,
                },
                "properties": {
                    "distance_m": 0.0,
                    "distance_km": 0.0,
                    "safe": True,
                    "nodes_count": 1,
                    "routing_strategy": "identical_node",
                    "pathfinding_algorithm": algorithm,
                    "message": "Start and end locations snap to the exact same road node.",
                    "start": {
                        "lat": start_lat,
                        "lon": start_lon,
                        "geojson_coords": to_geojson_coords(start_lat, start_lon),
                        "leaflet_coords": to_leaflet_coords(start_lon, start_lat),
                        "snapped_node": orig_node,
                    },
                    "end": {
                        "lat": end_lat,
                        "lon": end_lon,
                        "geojson_coords": to_geojson_coords(end_lat, end_lon),
                        "leaflet_coords": to_leaflet_coords(end_lon, end_lat),
                        "snapped_node": dest_node,
                    },
                },
                "distance_m": 0.0,
                "distance_km": 0.0,
            }
        )

    # 3. Retrieve currently flooded road edges from latest nowcast & drainage surcharge
    flooded_edges: Set[Tuple] = set()
    flood_source = "disabled"
    if avoid_floods:
        flooded_edges, flood_source = get_latest_flood_risk_edges()

    # 4. Compute unconstrained baseline route using Dijkstra or A*
    base_path: Optional[List[int]] = None
    base_dist = 0.0
    base_flooded_count = 0

    def baseline_weight_fn(u, v, edge_dict):
        best = None
        for k, attrs in edge_dict.items():
            cost = calculate_edge_traversal_cost(attrs, is_flooded=False, penalty_multiplier=1.0)
            if best is None or cost < best:
                best = cost
        return best

    try:
        base_path = compute_shortest_path(
            road_graph, orig_node, dest_node, weight=baseline_weight_fn, algorithm=algorithm
        )
        _, base_dist, base_flooded_count = _build_route_linestring(
            road_graph, base_path, flooded_edges
        )
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        logger.warning("No baseline path found between nodes %s -> %s", orig_node, dest_node)

    # 5. Compute Safe Route Avoiding Flood Zones
    safe_path: Optional[List[int]] = None
    strategy = "strictly_avoid_floods"

    if avoid_floods and flooded_edges:
        # Phase 1: Try strict removal of flooded edges
        def strict_weight_fn(u, v, edge_dict):
            best = None
            for k, attrs in edge_dict.items():
                if (u, v, k) in flooded_edges or (u, v) in flooded_edges:
                    continue  # edge removed
                cost = calculate_edge_traversal_cost(attrs, is_flooded=False, penalty_multiplier=1.0)
                if best is None or cost < best:
                    best = cost
            return best

        try:
            safe_path = compute_shortest_path(
                road_graph, orig_node, dest_node, weight=strict_weight_fn, algorithm=algorithm
            )
            strategy = "strictly_avoid_floods"
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            # Phase 2: If strict removal disconnects graph, fallback to high-penalty routing
            logger.info(
                "Strict safe path disconnected between %s -> %s. Falling back to penalized routing (x%.1f)...",
                orig_node,
                dest_node,
                penalty_multiplier,
            )

            def penalized_weight_fn(u, v, edge_dict):
                best = None
                for k, attrs in edge_dict.items():
                    is_fl = (u, v, k) in flooded_edges or (u, v) in flooded_edges
                    cost = calculate_edge_traversal_cost(
                        attrs,
                        is_flooded=is_fl,
                        penalty_multiplier=penalty_multiplier,
                    )
                    if best is None or cost < best:
                        best = cost
                return best

            try:
                safe_path = compute_shortest_path(
                    road_graph, orig_node, dest_node, weight=penalized_weight_fn, algorithm=algorithm
                )
                strategy = "penalized_fallback_flood_minimized"
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                safe_path = None
    else:
        safe_path = base_path
        strategy = "baseline_unconstrained"

    if safe_path is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No viable path found between start ({start_lat}, {start_lon}) and "
                f"end ({end_lat}, {end_lon}). Road network is disconnected."
            ),
        )

    # 6. Reconstruct GeoJSON LineString coordinates strictly as [longitude, latitude]
    coords, safe_dist_m, traversed_flooded = _build_route_linestring(
        road_graph, safe_path, flooded_edges
    )

    is_safe = traversed_flooded == 0
    avoided_flooded_count = max(0, base_flooded_count - traversed_flooded)
    detour_additional_m = max(0.0, safe_dist_m - base_dist) if base_path else 0.0
    process_time_ms = (time.perf_counter() - start_time) * 1000.0

    response_payload = {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": coords,
        },
        "properties": {
            "distance_m": round(safe_dist_m, 2),
            "distance_km": round(safe_dist_m / 1000.0, 3),
            "baseline_distance_m": round(base_dist, 2),
            "detour_additional_m": round(detour_additional_m, 2),
            "safe": is_safe,
            "routing_strategy": strategy,
            "pathfinding_algorithm": algorithm,
            "flooded_segments_avoided": avoided_flooded_count,
            "flooded_segments_traversed": traversed_flooded,
            "total_flooded_edges_in_network": len(flooded_edges),
            "flood_risk_source": flood_source,
            "drainage_network_loaded": drainage_graph is not None,
            "nodes_count": len(safe_path),
            "start": {
                "lat": start_lat,
                "lon": start_lon,
                "geojson_coords": to_geojson_coords(start_lat, start_lon),
                "leaflet_coords": to_leaflet_coords(start_lon, start_lat),
                "snapped_node": orig_node,
            },
            "end": {
                "lat": end_lat,
                "lon": end_lon,
                "geojson_coords": to_geojson_coords(end_lat, end_lon),
                "leaflet_coords": to_leaflet_coords(end_lon, end_lat),
                "snapped_node": dest_node,
            },
            "calculation_time_ms": round(process_time_ms, 2),
        },
        "distance_m": round(safe_dist_m, 2),
        "distance_km": round(safe_dist_m / 1000.0, 3),
    }

    return JSONResponse(
        content=response_payload,
        headers={
            "X-Routing-Strategy": strategy,
            "X-Pathfinding-Algorithm": algorithm,
            "X-Calculation-Time-Ms": f"{process_time_ms:.2f}",
        },
    )
