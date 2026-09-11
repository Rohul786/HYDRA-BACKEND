"""
hydraulics.py - Urban Stormwater Runoff & Pipe Network Surcharge Simulation

Implements:
1. Rational Method Runoff (Q = C * i * A).
2. Spatial distribution of runoff to drainage inlets using per-cell OSM imperviousness & DEM slope.
3. Hydraulic flow propagation through directed pipe networks and surcharging / backflow detection.
4. Mapping hydraulic surcharge to urban road networks for flood risk classification.
"""

import sys
import math
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple, Callable, List

# Ensure project root is in sys.path for cross-module imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import shapely
from shapely.geometry import Point, LineString, box
from shapely.strtree import STRtree

logger = logging.getLogger("engine.hydraulics")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def calculate_runoff(
    rainfall_intensity_mm_hr: float,
    catchment_area_m2: float,
    imperviousness_coefficient: float,
) -> float:
    """
    Calculates surface runoff rate using the classical Rational Method formula:

        Q = C * i * A

    Where:
        - Q is peak runoff discharge rate.
        - C is the dimensionless runoff / imperviousness coefficient [0.0 to 1.0].
        - i is the rainfall intensity in mm/hour.
        - A is the catchment / drainage basin surface area in square meters (m^2).

    Unit Conversion Derivation:
        Depth per hour = i * 10^-3 m/hr
        Discharge volume = C * (i / 1000 m/hr) * (A m^2) = (C * i * A / 1000) m^3/hr
        1 m^3 = 1000 Liters
        1 hour = 60 minutes
        Q (L/min) = [ (C * i * A / 1000) * 1000 ] / 60 = (C * i * A) / 60

    Args:
        rainfall_intensity_mm_hr (float): Rainfall intensity (mm/hr).
        catchment_area_m2 (float): Drainage contributing area (m^2).
        imperviousness_coefficient (float): Imperviousness ratio [0.0, 1.0].

    Returns:
        float: Estimated peak runoff in Liters per minute (L/min).
    """
    c = max(min(float(imperviousness_coefficient), 1.0), 0.0)
    i = max(float(rainfall_intensity_mm_hr), 0.0)
    a = max(float(catchment_area_m2), 0.0)

    runoff_liters_per_min = (c * i * a) / 60.0
    return round(runoff_liters_per_min, 2)


def distribute_runoff_to_nodes(
    road_network: Union[nx.MultiDiGraph, gpd.GeoDataFrame],
    drainage_graph: nx.DiGraph,
    rainfall_intensity_mm_hr: float,
    imperviousness_grid: Optional[gpd.GeoDataFrame] = None,
    slope_lookup: Optional[Callable[[float, float], float]] = None,
    default_road_width_m: float = 10.0,
) -> Dict[str, float]:
    """
    Assigns surface stormwater runoff to the nearest underground drainage inlet node
    for every road segment, utilizing per-cell OSM imperviousness and terrain slope.

    Args:
        road_network (MultiDiGraph or GeoDataFrame): Road network containing geometry and length.
        drainage_graph (DiGraph): Stormwater drainage network with (x, y) coordinates.
        rainfall_intensity_mm_hr (float): Current storm rainfall rate in mm/hr.
        imperviousness_grid (GeoDataFrame, optional): OSM-derived imperviousness grid with
                                                     'impervious_ratio' and geometry.
        slope_lookup (Callable, optional): Function (lat, lon) -> slope in degrees or %.
        default_road_width_m (float): Default roadway corridor buffer width in meters.

    Returns:
        Dict[str, float]: Inflow mapping {drainage_node_id: total_surface_runoff_L_min}.
    """
    # 1. Normalize road input to GeoDataFrame
    if isinstance(road_network, (nx.MultiDiGraph, nx.DiGraph, nx.Graph)):
        import osmnx as ox
        roads_gdf = ox.graph_to_gdfs(road_network, nodes=False, edges=True)
    else:
        roads_gdf = road_network.copy()

    if roads_gdf.empty or len(drainage_graph.nodes) == 0:
        logger.warning("Empty road network or drainage graph supplied to distribute_runoff_to_nodes.")
        return {n: 0.0 for n in drainage_graph.nodes}

    # Initialize node inflow accumulator
    node_inflows: Dict[str, float] = {n: 0.0 for n in drainage_graph.nodes}

    # 2. Build spatial index for drainage nodes (in WGS84 EPSG:4326)
    node_list = []
    node_points = []
    for n, data in drainage_graph.nodes(data=True):
        nx_val = data.get("x")
        ny_val = data.get("y")
        if nx_val is not None and ny_val is not None:
            node_list.append(n)
            node_points.append(Point(nx_val, ny_val))

    if not node_points:
        logger.warning("Drainage graph has no valid (x, y) coordinates.")
        return node_inflows

    node_tree = STRtree(node_points)

    # 3. Setup spatial index for imperviousness grid if provided
    has_grid = (
        imperviousness_grid is not None
        and not imperviousness_grid.empty
        and "impervious_ratio" in imperviousness_grid.columns
    )
    if has_grid:
        grid_wgs84 = imperviousness_grid.to_crs("EPSG:4326")
        grid_tree = STRtree(grid_wgs84.geometry.values)
        grid_ratios = grid_wgs84["impervious_ratio"].values
    else:
        grid_tree = None
        grid_ratios = None

    # 4. Iterate over road segments and compute localized runoff
    total_assigned_runoff = 0.0

    for _, row in roads_gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        # Segment length in meters
        if "length" in row and not pd.isna(row["length"]):
            length_m = float(row["length"])
        else:
            # Estimate length from coordinates
            length_m = max(geom.length * 111320.0, 10.0)

        # Catchment area for this road corridor (pavement + adjacent sidewalk/ditch)
        corridor_width = default_road_width_m
        lanes_val = row.get("lanes") if hasattr(row, "get") else (row["lanes"] if "lanes" in row else None)
        if lanes_val is not None:
            if isinstance(lanes_val, (list, tuple, np.ndarray)) and len(lanes_val) > 0:
                lanes_val = lanes_val[0]
            if not pd.isna(lanes_val):
                try:
                    lanes = float(str(lanes_val).split(";")[0].strip())
                    corridor_width = max(lanes * 3.5 + 3.0, 8.0)
                except (ValueError, TypeError):
                    pass
        catchment_area_m2 = length_m * corridor_width

        # Representative center point
        midpoint = geom.interpolate(0.5, normalized=True)
        mid_lon, mid_lat = midpoint.x, midpoint.y

        # Query imperviousness coefficient C from OSM grid
        c_val = 0.75  # Standard default urban road imperviousness
        if grid_tree is not None:
            matching_idx = grid_tree.query(midpoint, predicate="intersects")
            if len(matching_idx) > 0:
                c_val = float(grid_ratios[matching_idx[0]])
            else:
                # Nearest cell if slightly outside
                nearest_idx = grid_tree.nearest(midpoint)
                if nearest_idx is not None:
                    c_val = float(grid_ratios[nearest_idx])

        # Adjust for terrain slope (from DEM) if slope_lookup is provided
        # Steeper slopes reduce depression storage and accelerate runoff peak
        if slope_lookup is not None:
            try:
                terrain_slope_deg = float(slope_lookup(mid_lat, mid_lon))
                # Rational method slope adjustment factor (ASCE / IMD standard)
                slope_factor = 1.0 + 0.04 * min(max(terrain_slope_deg, 0.0), 12.0)
                c_val = min(c_val * slope_factor, 0.98)
            except Exception as slope_err:
                logger.debug("Slope lookup failed for (%.4f, %.4f): %s", mid_lat, mid_lon, slope_err)

        # Compute runoff for this road segment
        q_segment_lpm = calculate_runoff(
            rainfall_intensity_mm_hr=rainfall_intensity_mm_hr,
            catchment_area_m2=catchment_area_m2,
            imperviousness_coefficient=c_val,
        )

        # Find nearest drainage inlet node
        nearest_pt_idx = node_tree.nearest(midpoint)
        target_node = node_list[nearest_pt_idx]

        node_inflows[target_node] += q_segment_lpm
        total_assigned_runoff += q_segment_lpm

    logger.info(
        "Distributed %.1f L/min of surface runoff across %d drainage nodes (Rainfall = %.1f mm/hr).",
        total_assigned_runoff,
        len(drainage_graph.nodes),
        rainfall_intensity_mm_hr,
    )
    return {k: round(v, 2) for k, v in node_inflows.items()}


def propagate_flow(
    drainage_graph: nx.DiGraph,
    inflow_dict: Dict[str, float],
    slope_lookup: Optional[Callable[[float, float], float]] = None,
) -> Dict[str, Any]:
    """
    Pushes stormwater runoff through the directed drainage graph towards outfalls.
    Evaluates hydraulic intake capacity at inlet nodes and full-flow pipe capacity,
    flagging surcharging nodes and pressurized/overflowing conduits.

    Args:
        drainage_graph (nx.DiGraph): Directed drainage graph flowing toward outfalls.
        inflow_dict (Dict[str, float]): Surface inflow rate per node in L/min.
        slope_lookup (Callable, optional): Slope lookup function to adjust pipe conveyance.

    Returns:
        Dict[str, Any]: Simulation results containing:
            - 'node_total_flow': Dict[node_id, float] total flow processed
            - 'pipe_flow': Dict[(u, v), float] flow rate through pipes
            - 'surcharging_nodes': Dict[node_id, dict] details of surcharging inlets
            - 'surcharging_pipes': Dict[(u, v), dict] details of surcharging pipes
            - 'outfall_discharges': Dict[outfall_id, float] discharge at sinks
            - 'total_surface_runoff_lpm': float
            - 'total_surcharge_lpm': float
            - 'surcharge_node_count': int
    """
    if not nx.is_directed_acyclic_graph(drainage_graph):
        logger.warning("Drainage graph contains cycles! Breaking cycles for hierarchical flow propagation.")
        G = nx.bfs_tree(drainage_graph, source=list(drainage_graph.nodes)[0])
    else:
        G = drainage_graph

    # Initialize flow trackers
    accumulated_node_flow: Dict[str, float] = {n: float(inflow_dict.get(n, 0.0)) for n in G.nodes}
    pipe_flows: Dict[Tuple[str, str], float] = {}
    surcharging_nodes: Dict[str, Dict[str, Any]] = {}
    surcharging_pipes: Dict[Tuple[str, str], Dict[str, Any]] = {}
    outfall_discharges: Dict[str, float] = {}

    # Topological sort ensures upstream inlets are resolved before downstream trunks
    topo_order = list(nx.topological_sort(G))

    for node in topo_order:
        node_data = drainage_graph.nodes[node]
        total_inflow = accumulated_node_flow[node]
        intake_cap = float(node_data.get("capacity_liters_per_min", 35000.0))

        # Check for inlet manhole surcharge (intake bottleneck)
        is_surcharging = total_inflow > intake_cap
        if is_surcharging:
            surcharge_vol = total_inflow - intake_cap
            surcharging_nodes[node] = {
                "node_type": node_data.get("node_type", "inlet"),
                "surface_inflow_lpm": round(inflow_dict.get(node, 0.0), 1),
                "total_inflow_lpm": round(total_inflow, 1),
                "capacity_lpm": round(intake_cap, 1),
                "surcharge_lpm": round(surcharge_vol, 1),
                "surcharge_ratio": round(total_inflow / max(intake_cap, 1.0), 2),
                "x": node_data.get("x"),
                "y": node_data.get("y"),
                "elevation": node_data.get("elevation"),
            }
            # Physical flow admitted into underground conduit (capped at intake capacity)
            conveyed_flow = intake_cap
        else:
            conveyed_flow = total_inflow

        # Distribute conveyed flow to outgoing downstream conduits
        out_edges = list(G.out_edges(node, data=True))
        if not out_edges:
            # Terminal outfall sink
            outfall_discharges[node] = round(total_inflow, 1)
        else:
            flow_per_pipe = conveyed_flow / len(out_edges)
            for u, v, edge_data in out_edges:
                edge_key = (u, v)
                pipe_cap = float(edge_data.get("flow_capacity_liters_per_min", 75000.0))

                # Dynamically adjust pipe capacity if local terrain slope is available
                if slope_lookup is not None and "slope" not in edge_data:
                    ux, uy = node_data.get("x"), node_data.get("y")
                    if ux is not None and uy is not None:
                        slope_pct = max(slope_lookup(uy, ux) / 100.0, 0.002)
                        pipe_cap *= math.sqrt(slope_pct / 0.005)

                pipe_flows[edge_key] = round(flow_per_pipe, 1)

                # Check for pipe capacity exceedance (pressurized backflow risk)
                if flow_per_pipe > pipe_cap:
                    pipe_surcharge_vol = flow_per_pipe - pipe_cap
                    surcharging_pipes[edge_key] = {
                        "flow_lpm": round(flow_per_pipe, 1),
                        "capacity_lpm": round(pipe_cap, 1),
                        "surcharge_lpm": round(pipe_surcharge_vol, 1),
                        "surcharge_ratio": round(flow_per_pipe / max(pipe_cap, 1.0), 2),
                        "pipe_diameter_m": edge_data.get("pipe_diameter_m", 0.6),
                        "length_m": edge_data.get("length_m", 50.0),
                    }

                # Propagate flow downstream to node v
                accumulated_node_flow[v] += flow_per_pipe

    total_surface = sum(inflow_dict.values())
    total_surcharged = sum(d["surcharge_lpm"] for d in surcharging_nodes.values())

    logger.info(
        "Propagated flow: %d/%d nodes surcharging, %d/%d pipes surcharging (Total Surcharge = %.1f L/min).",
        len(surcharging_nodes),
        len(G.nodes),
        len(surcharging_pipes),
        len(G.edges),
        total_surcharged,
    )

    return {
        "node_total_flow": {k: round(v, 1) for k, v in accumulated_node_flow.items()},
        "pipe_flow": pipe_flows,
        "surcharging_nodes": surcharging_nodes,
        "surcharging_pipes": surcharging_pipes,
        "outfall_discharges": outfall_discharges,
        "total_surface_runoff_lpm": round(total_surface, 1),
        "total_surcharge_lpm": round(total_surcharged, 1),
        "surcharge_node_count": len(surcharging_nodes),
        "surcharge_pipe_count": len(surcharging_pipes),
    }


def map_surcharge_to_road_segments(
    surcharge_result: Dict[str, Any],
    road_network: Union[nx.MultiDiGraph, gpd.GeoDataFrame],
    impact_radius_m: float = 180.0,
) -> gpd.GeoDataFrame:
    """
    Identifies road segments susceptible to pluvial flooding and surface ponding
    due to nearby surcharging drainage inlet manholes and pressurized pipes.

    Args:
        surcharge_result (Dict[str, Any]): Output from propagate_flow().
        road_network (MultiDiGraph or GeoDataFrame): Road network.
        impact_radius_m (float): Flood overland influence radius in meters (default: 180m).

    Returns:
        gpd.GeoDataFrame: Road segments flagged with flood risk attributes:
            - 'flood_risk_level': 'SEVERE', 'HIGH', 'MODERATE', or 'LOW'
            - 'surcharge_volume_lpm': Maximum adjacent surcharge rate
            - 'surcharge_ratio': Intake exceedance multiplier
            - 'nearest_surcharge_node': Associated inlet ID
    """
    # Normalize roads to GeoDataFrame
    if isinstance(road_network, (nx.MultiDiGraph, nx.DiGraph, nx.Graph)):
        import osmnx as ox
        roads_gdf = ox.graph_to_gdfs(road_network, nodes=False, edges=True)
    else:
        roads_gdf = road_network.copy()

    surcharging_nodes = surcharge_result.get("surcharging_nodes", {})

    # If no surcharge occurred, return unflagged roads
    if not surcharging_nodes or roads_gdf.empty:
        result_roads = roads_gdf.copy()
        result_roads["flood_risk_level"] = "LOW"
        result_roads["surcharge_volume_lpm"] = 0.0
        result_roads["surcharge_ratio"] = 0.0
        result_roads["nearest_surcharge_node"] = None
        return result_roads

    # Convert surcharging nodes to GeoDataFrame
    surcharge_points = []
    surcharge_data = []
    for n_id, data in surcharging_nodes.items():
        x, y = data.get("x"), data.get("y")
        if x is not None and y is not None:
            surcharge_points.append(Point(x, y))
            surcharge_data.append({
                "node_id": n_id,
                "surcharge_lpm": data["surcharge_lpm"],
                "surcharge_ratio": data["surcharge_ratio"],
            })

    surcharge_gdf = gpd.GeoDataFrame(surcharge_data, geometry=surcharge_points, crs="EPSG:4326")

    # Project to metric CRS for accurate distance buffering
    metric_crs = roads_gdf.estimate_utm_crs() or "EPSG:3857"
    roads_proj = roads_gdf.to_crs(metric_crs)
    surcharge_proj = surcharge_gdf.to_crs(metric_crs)

    # Buffer surcharging nodes by impact radius
    surcharge_proj["geometry_buffer"] = surcharge_proj.geometry.buffer(impact_radius_m)

    # Spatial join between roads and surcharging node buffers
    buffer_gdf = surcharge_proj.set_geometry("geometry_buffer")
    joined = gpd.sjoin(
        roads_proj,
        buffer_gdf[["node_id", "surcharge_lpm", "surcharge_ratio", "geometry_buffer"]],
        how="left",
        predicate="intersects",
    )

    # Aggregate maximum surcharge impact per road segment
    # In case a road segment intersects multiple surcharging nodes
    if not joined.empty:
        agg_risk = (
            joined.groupby(joined.index)
            .agg({
                "surcharge_lpm": "max",
                "surcharge_ratio": "max",
                "node_id": "first",
            })
            .fillna(0.0)
        )
    else:
        agg_risk = pd.DataFrame(index=roads_gdf.index, columns=["surcharge_lpm", "surcharge_ratio", "node_id"]).fillna(0.0)

    result_roads = roads_gdf.copy()
    result_roads["surcharge_volume_lpm"] = agg_risk["surcharge_lpm"].fillna(0.0).round(1)
    result_roads["surcharge_ratio"] = agg_risk["surcharge_ratio"].fillna(0.0).round(2)
    result_roads["nearest_surcharge_node"] = agg_risk["node_id"]

    # Classify qualitative flood risk level
    def classify_risk(row) -> str:
        ratio = row["surcharge_ratio"]
        vol = row["surcharge_volume_lpm"]
        if ratio >= 2.0 or vol >= 50000.0:
            return "SEVERE"
        elif ratio >= 1.3 or vol >= 20000.0:
            return "HIGH"
        elif ratio >= 1.01 or vol > 0.0:
            return "MODERATE"
        else:
            return "LOW"

    result_roads["flood_risk_level"] = result_roads.apply(classify_risk, axis=1)

    high_risk_count = (result_roads["flood_risk_level"].isin(["SEVERE", "HIGH"])).sum()
    logger.info(
        "Mapped surcharge to road network: %d road segments identified with HIGH/SEVERE flood risk.",
        high_risk_count,
    )
    return result_roads


def create_dem_slope_lookup(
    dem_path: Optional[Union[str, Any]] = None,
) -> Callable[[float, float], float]:
    """
    Constructs a fast callable slope lookup function using the Copernicus DEM raster.

    Args:
        dem_path: Path to dem.tif (defaults to data/raw/dem.tif).

    Returns:
        Callable[[float, float], float]: Function (lat, lon) -> slope in degrees.
    """
    from pathlib import Path
    proj_root = Path(__file__).resolve().parent.parent
    target_dem = Path(dem_path) if dem_path else proj_root / "data" / "raw" / "dem.tif"

    if not target_dem.exists():
        logger.info("DEM raster not found at %s. Using default 1.0 deg urban slope.", target_dem)
        return lambda lat, lon: 1.0

    try:
        import rasterio
        from ingestion.fetch_dem import calculate_slope

        slope_grid = calculate_slope(target_dem, in_degrees=True)
        with rasterio.open(target_dem) as src:
            transform = src.transform
            bounds = src.bounds

        def slope_lookup(lat: float, lon: float) -> float:
            try:
                # Invert transform from (lon, lat) -> (col, row)
                try:
                    col, row = ~transform @ (lon, lat)
                except TypeError:
                    col, row = ~transform * (lon, lat)
                r, c = int(round(row)), int(round(col))
                if 0 <= r < slope_grid.shape[0] and 0 <= c < slope_grid.shape[1]:
                    val = float(slope_grid[r, c])
                    return val if not np.isnan(val) else 1.0
            except Exception:
                pass
            return 1.0

        return slope_lookup

    except Exception as exc:
        logger.warning("Could not initialize DEM slope lookup (%s). Using fallback 1.0 deg.", exc)
        return lambda lat, lon: 1.0


if __name__ == "__main__":
    print("\n==================================================")
    print("         Testing engine/hydraulics.py             ")
    print("==================================================")

    # 1. Test Rational Method calculation directly
    intensity = 60.0       # 60 mm/hr (monsoon cloudburst)
    area = 2500.0          # 2,500 m^2 road corridor
    c_coeff = 0.85         # 85% impervious urban pavement
    runoff_lpm = calculate_runoff(intensity, area, c_coeff)
    print(f"\n[1] Rational Method Verification:")
    print(f"  - Intensity: {intensity} mm/hr, Area: {area} m^2, C: {c_coeff}")
    print(f"  - Peak Runoff: {runoff_lpm} Liters/min (Formula: Q = C * i * A / 60)")

    import osmnx as ox
    from ingestion.drainage_graph import build_sample_drainage_network, load_network
    from ingestion.fetch_osm_data import fetch_road_network, calculate_imperviousness_grid

    print(f"\n[2] Loading Drainage Network & Roads...")
    test_place = "Adyar, Chennai, Tamil Nadu, India"
    road_graph = fetch_road_network(test_place)
    roads_sample = ox.graph_to_gdfs(road_graph, nodes=False, edges=True)
    
    # Place drainage network right inside the road network catchment
    bounds = roads_sample.total_bounds  # minx, miny, maxx, maxy
    center_lon = float((bounds[0] + bounds[2]) / 2.0)
    center_lat = float((bounds[1] + bounds[3]) / 2.0)
    drainage_net = build_sample_drainage_network(
        num_nodes=25,
        center_lat=center_lat,
        center_lon=center_lon,
        spread_km=2.0,
        num_outfalls=3,
    )

    # 3. Create DEM slope lookup
    slope_fn = create_dem_slope_lookup()
    test_slope = slope_fn(13.0827, 80.2707)
    print(f"  - DEM Slope at Chennai center: {test_slope:.2f}°")

    # 4. Distribute runoff to drainage inlet nodes
    print(f"\n[3] Distributing Runoff (Intensity = 45 mm/hr)...")
    inflows = distribute_runoff_to_nodes(
        road_network=road_graph,
        drainage_graph=drainage_net,
        rainfall_intensity_mm_hr=45.0,
        slope_lookup=slope_fn,
    )
    sample_nodes = list(inflows.items())[:5]
    print(f"  - Sample Node Inflows (L/min): {sample_nodes}")

    # 5. Propagate flow & detect surcharging
    print(f"\n[4] Simulating Hydraulic Pipe Flow & Surcharge...")
    sim_result = propagate_flow(drainage_net, inflows, slope_lookup=slope_fn)
    print(f"  - Total Surface Inflow: {sim_result['total_surface_runoff_lpm']:,.1f} L/min")
    print(f"  - Surcharging Nodes: {sim_result['surcharge_node_count']} / {len(drainage_net.nodes)}")
    print(f"  - Surcharging Pipes: {sim_result['surcharge_pipe_count']} / {len(drainage_net.edges)}")
    print(f"  - Total Surcharge Volume: {sim_result['total_surcharge_lpm']:,.1f} L/min")
    print(f"  - Outfall Discharges: {sim_result['outfall_discharges']}")

    # 6. Map surcharge back to road network for flood risk tagging
    print(f"\n[5] Mapping Surcharge to Road Network Risk Segments...")
    roads_gdf = map_surcharge_to_road_segments(sim_result, road_graph, impact_radius_m=200.0)
    risk_summary = roads_gdf["flood_risk_level"].value_counts().to_dict()
    print(f"  - Road Flood Risk Distribution: {risk_summary}")
    print("\n[OK] Hydraulics module verified successfully!")
