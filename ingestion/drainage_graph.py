"""
drainage_graph.py - Municipal Stormwater Drainage Network Modeling

========================================================================================
METHODOLOGICAL NOTE & DOMAIN CONTEXT:
In Indian metropolitan areas (Chennai, Bengaluru, Mumbai, Delhi), comprehensive GIS
datasets of underground piped stormwater drainage networks (exact pipe inverts, conduit
dimensions, manhole layouts, and sub-surface gradients) are generally non-public and
siloed across municipal corporations (e.g., GCC, BWSSB, MCGM) due to security, administrative,
and infrastructural constraints.

This module implements a calibrated synthetic stormwater drainage network:
1. Hydrologically sound directed tree topology (draining from elevated inlet manholes
   through progressive trunk collectors towards natural waterbody/canal outfalls).
2. Physical hydraulic attributes (pipe diameter, length, slope, flow capacity).
3. Calibration against real surface drainage features (OSM waterways/canals/ditches)
   when available, snapping outfalls and sizing capacities to match observed channels.
========================================================================================
"""

import math
import logging
from pathlib import Path
from typing import Optional, Union, Tuple, Dict, Any, List

import numpy as np
import networkx as nx
import geopandas as gpd
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points

logger = logging.getLogger("ingestion.drainage_graph")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Default export path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DRAINAGE_GRAPH_PATH = PROJECT_ROOT / "data" / "processed" / "drainage_network.graphml"


def _estimate_pipe_capacity_manning(
    diameter_m: float,
    slope: float = 0.005,
    manning_n: float = 0.013,
) -> float:
    """
    Estimates pipe full-flow capacity using Manning's Equation for circular conduits.
    
    Q = (1 / n) * A * R^(2/3) * S^(1/2)
    For circular pipe full flow:
      A = pi * D^2 / 4
      R = D / 4
      
    Returns:
        float: Capacity in Liters per minute (L/min).
    """
    diameter = max(diameter_m, 0.2)
    s = max(slope, 0.001)
    
    area = math.pi * (diameter ** 2) / 4.0
    hydraulic_radius = diameter / 4.0
    velocity = (1.0 / manning_n) * (hydraulic_radius ** (2.0 / 3.0)) * math.sqrt(s)
    flow_m3_per_sec = area * velocity
    
    # Convert m^3/sec to Liters/min: 1 m^3 = 1000 L, 1 sec = 1/60 min
    flow_liters_per_min = flow_m3_per_sec * 1000.0 * 60.0
    return round(flow_liters_per_min, 1)


def build_sample_drainage_network(
    num_nodes: int = 25,
    center_lat: float = 13.0827,
    center_lon: float = 80.2707,
    spread_km: float = 1.8,
    num_outfalls: int = 3,
    random_seed: int = 42,
) -> nx.DiGraph:
    """
    Constructs a synthetic directed stormwater drainage graph representing manholes,
    inlets, and pipe conduits. The graph is structured as a forest of directed trees
    flowing downstream towards 2-3 outfall points.

    Node attributes:
        - elevation (float): Ground/manhole invert elevation in meters.
        - capacity_liters_per_min (float): Surface runoff inlet intake capacity.
        - node_type (str): 'inlet', 'junction', or 'outfall'.
        - x (float): Longitude.
        - y (float): Latitude.

    Edge (pipe) attributes:
        - pipe_diameter_m (float): Conduit diameter in meters.
        - flow_capacity_liters_per_min (float): Hydraulic discharge capacity.
        - length_m (float): Pipe segment distance in meters.
        - slope (float): Hydraulic gradient.

    Args:
        num_nodes (int): Total number of drainage nodes (min 10, default: 25).
        center_lat (float): Base latitude center.
        center_lon (float): Base longitude center.
        spread_km (float): Spatial catchment extent in kilometers.
        num_outfalls (int): Number of terminal drainage outfalls (2 or 3).
        random_seed (int): Seed for reproducible node layout.

    Returns:
        nx.DiGraph: Directed drainage network (edges point downstream towards outfalls).
    """
    rng = np.random.default_rng(random_seed)
    num_nodes = max(num_nodes, 10)
    num_outfalls = min(max(num_outfalls, 2), 4)

    G = nx.DiGraph(
        name="Calibrated_Synthetic_Drainage_Network",
        crs="EPSG:4326",
        description="Calibrated synthetic urban stormwater drainage network model",
    )

    # 1. Place Outfall Nodes at the downstream perimeter (lowest elevation, e.g. coastal/canal side)
    # Scale conversion: 1 deg lat ~ 111.32 km, 1 deg lon ~ 111.32 * cos(lat) km
    deg_lat_km = 1.0 / 111.32
    deg_lon_km = 1.0 / (111.32 * math.cos(math.radians(center_lat)))

    outfall_ids = [f"outfall_{i}" for i in range(num_outfalls)]
    # Place outfalls along an eastern / southeastern drainage boundary (typical for coastal cities)
    for i, out_id in enumerate(outfall_ids):
        angle = (i - (num_outfalls - 1) / 2.0) * (math.pi / 4.0)
        out_lon = center_lon + (spread_km * 0.65) * math.cos(angle) * deg_lon_km
        out_lat = center_lat + (spread_km * 0.65) * math.sin(angle) * deg_lat_km
        
        # Outfalls have the lowest elevation (e.g. 2.0m - 3.5m above sea level)
        elev = round(2.5 + float(rng.uniform(-0.5, 0.8)), 2)
        # Sinks have huge discharge capacity to natural waterbodies
        G.add_node(
            out_id,
            node_type="outfall",
            elevation=elev,
            capacity_liters_per_min=120000.0,
            x=round(out_lon, 6),
            y=round(out_lat, 6),
            calibrated_with_osm=False,
        )

    # 2. Generate Inlets & Junction Nodes upstream
    num_internal = num_nodes - num_outfalls
    internal_ids = [f"node_{i:02d}" for i in range(num_internal)]

    # Partition internal nodes among the outfall catchments
    catchments: Dict[str, List[str]] = {out_id: [] for out_id in outfall_ids}
    for i, n_id in enumerate(internal_ids):
        assigned_outfall = outfall_ids[i % num_outfalls]
        catchments[assigned_outfall].append(n_id)

    # For each catchment, build a directed tree draining towards that outfall
    for out_id, members in catchments.items():
        if not members:
            continue
        
        out_x = G.nodes[out_id]["x"]
        out_y = G.nodes[out_id]["y"]
        out_elev = G.nodes[out_id]["elevation"]

        # Sort members by distance from outfall to create hierarchical tiers
        # Tier 0 = direct collectors to outfall, higher tiers = upstream street inlets
        n_members = len(members)
        
        # Assign spatial positions and elevations (uphill from outfall)
        tree_parents: Dict[str, str] = {}
        
        for idx, m_id in enumerate(members):
            # Distance scales upstream towards northwest/west
            dist_km = (0.25 + (idx + 1) * (spread_km / (n_members + 2)))
            jitter_x = float(rng.uniform(-0.15, 0.15)) * spread_km
            jitter_y = float(rng.uniform(-0.15, 0.15)) * spread_km
            
            node_lon = out_x - dist_km * 0.8 * deg_lon_km + jitter_x * deg_lon_km
            node_lat = out_y + jitter_y * deg_lat_km
            
            # Elevation increases with upstream distance (slope ~ 0.003 - 0.008)
            node_elev = round(out_elev + (dist_km * 1000.0) * float(rng.uniform(0.003, 0.007)), 2)
            
            # Nodes near the outer edge are street inlets; intermediate are manhole junctions
            is_inlet = idx >= int(n_members * 0.4)
            node_type = "inlet" if is_inlet else "junction"
            inlet_cap = round(float(rng.uniform(6000.0, 15000.0)) if is_inlet else float(rng.uniform(20000.0, 45000.0)), 1)
            
            G.add_node(
                m_id,
                node_type=node_type,
                elevation=node_elev,
                capacity_liters_per_min=inlet_cap,
                x=round(node_lon, 6),
                y=round(node_lat, 6),
                calibrated_with_osm=False,
            )

        # Connect nodes in a branching tree topology towards outfall
        # Lowest tier connects to outfall
        trunk_size = max(1, int(n_members * 0.35))
        trunk_nodes = members[:trunk_size]
        lateral_nodes = members[trunk_size:]

        # Connect first trunk node to outfall
        tree_parents[trunk_nodes[0]] = out_id
        for k in range(1, len(trunk_nodes)):
            tree_parents[trunk_nodes[k]] = trunk_nodes[k - 1]

        # Connect lateral nodes to the nearest existing trunk/branch node
        existing_nodes = [out_id] + trunk_nodes
        for lat_id in lateral_nodes:
            # Pick a closer lower-elevation node as parent
            lat_elev = G.nodes[lat_id]["elevation"]
            candidates = [c for c in existing_nodes if G.nodes[c]["elevation"] < lat_elev]
            if not candidates:
                candidates = existing_nodes
            
            # Select candidate with minimum geographic distance
            lx, ly = G.nodes[lat_id]["x"], G.nodes[lat_id]["y"]
            parent = min(
                candidates,
                key=lambda c: (G.nodes[c]["x"] - lx) ** 2 + (G.nodes[c]["y"] - ly) ** 2,
            )
            tree_parents[lat_id] = parent
            existing_nodes.append(lat_id)

        # Add directed edges: from upstream child -> downstream parent
        for child_id, parent_id in tree_parents.items():
            cx, cy = G.nodes[child_id]["x"], G.nodes[child_id]["y"]
            px, py = G.nodes[parent_id]["x"], G.nodes[parent_id]["y"]
            
            dx_m = (px - cx) * 111320.0 * math.cos(math.radians(cy))
            dy_m = (py - cy) * 111320.0
            length_m = max(round(math.sqrt(dx_m**2 + dy_m**2), 1), 25.0)

            elev_drop = max(G.nodes[child_id]["elevation"] - G.nodes[parent_id]["elevation"], 0.1)
            slope = round(elev_drop / length_m, 4)

            # Conduit diameter increases towards outfalls:
            # Trunk lines near outfalls: 0.9m - 1.6m; Lateral collectors: 0.4m - 0.6m
            is_trunk = parent_id == out_id or parent_id in trunk_nodes
            diameter_m = 1.2 if parent_id == out_id else (0.9 if is_trunk else 0.45)
            
            flow_cap = _estimate_pipe_capacity_manning(diameter_m=diameter_m, slope=slope)

            G.add_edge(
                child_id,
                parent_id,
                pipe_diameter_m=diameter_m,
                length_m=length_m,
                slope=slope,
                flow_capacity_liters_per_min=flow_cap,
            )

    logger.info(
        "Built synthetic drainage network with %d nodes (%d outfalls) and %d pipe edges.",
        len(G.nodes),
        num_outfalls,
        len(G.edges),
    )
    return G


def calibrate_with_osm_drainage(
    graph: nx.DiGraph,
    osm_drainage_gdf: gpd.GeoDataFrame,
    max_snap_dist_m: float = 200.0,
) -> nx.DiGraph:
    """
    Calibrates node positions and capacities by aligning with observed real-world
    OpenStreetMap surface drainage features (canals, streams, ditches, drains).

    For nodes within max_snap_dist_m of an OSM drainage line:
    - Nudges outfall or collector nodes toward the nearest point on the waterway.
    - Enhances discharge capacity, reflecting direct connection to open arterial drainage.
    - Updates pipe lengths and slopes connected to nudged nodes.

    Args:
        graph (nx.DiGraph): The drainage network graph.
        osm_drainage_gdf (gpd.GeoDataFrame): GeoDataFrame of OSM drainage features.
        max_snap_dist_m (float): Maximum distance in meters to consider snapping (default: 200m).

    Returns:
        nx.DiGraph: Calibrated drainage network graph.
    """
    if osm_drainage_gdf is None or osm_drainage_gdf.empty:
        logger.info("No OSM drainage features supplied. Retaining synthetic baseline network.")
        return graph

    # Filter to valid geometries
    valid_osm = osm_drainage_gdf[osm_drainage_gdf.geometry.is_valid & ~osm_drainage_gdf.geometry.is_empty].copy()
    if valid_osm.empty:
        logger.info("OSM drainage GeoDataFrame contains no valid geometries.")
        return graph

    # Ensure projected to metric CRS for accurate distance calculation
    metric_crs = valid_osm.estimate_utm_crs() or "EPSG:3857"
    osm_proj = valid_osm.to_crs(metric_crs)
    waterway_union = osm_proj.union_all() if hasattr(osm_proj, "union_all") else osm_proj.unary_union

    calibrated_count = 0

    for node_id, data in graph.nodes(data=True):
        lon = data.get("x")
        lat = data.get("y")
        if lon is None or lat is None:
            continue

        # Convert node point to metric CRS
        node_pt_geo = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(metric_crs).iloc[0]
        dist_m = node_pt_geo.distance(waterway_union)

        # Prioritize outfalls and junctions for alignment with surface drainage channels
        is_priority_node = data.get("node_type") in ("outfall", "junction")
        threshold = max_snap_dist_m if is_priority_node else (max_snap_dist_m * 0.6)

        if dist_m <= threshold:
            # Find nearest point on the waterway geometry
            nearest_water_pt = nearest_points(waterway_union, node_pt_geo)[0]
            # Convert back to EPSG:4326
            nearest_geo = gpd.GeoSeries([nearest_water_pt], crs=metric_crs).to_crs("EPSG:4326").iloc[0]

            # Nudge node coordinates towards the observed drainage feature
            # Blend 70% toward observed waterway to maintain street alignment
            new_lon = round(lon * 0.3 + nearest_geo.x * 0.7, 6)
            new_lat = round(lat * 0.3 + nearest_geo.y * 0.7, 6)

            data["x"] = new_lon
            data["y"] = new_lat
            data["calibrated_with_osm"] = True
            data["osm_channel_dist_m"] = round(dist_m, 1)

            # Boost intake/discharge capacity (+35%) due to verified active surface drain
            orig_cap = data.get("capacity_liters_per_min", 20000.0)
            data["capacity_liters_per_min"] = round(orig_cap * 1.35, 1)
            calibrated_count += 1

    # Recalculate edge lengths and flow capacities for nudged nodes
    for u, v, edge_data in graph.edges(data=True):
        ux, uy = graph.nodes[u]["x"], graph.nodes[u]["y"]
        vx, vy = graph.nodes[v]["x"], graph.nodes[v]["y"]

        dx_m = (vx - ux) * 111320.0 * math.cos(math.radians(uy))
        dy_m = (vy - uy) * 111320.0
        new_len = max(round(math.sqrt(dx_m**2 + dy_m**2), 1), 15.0)
        edge_data["length_m"] = new_len

        # Recompute slope and capacity
        elev_u = graph.nodes[u]["elevation"]
        elev_v = graph.nodes[v]["elevation"]
        new_slope = max(round(max(elev_u - elev_v, 0.05) / new_len, 4), 0.001)
        edge_data["slope"] = new_slope
        edge_data["flow_capacity_liters_per_min"] = _estimate_pipe_capacity_manning(
            diameter_m=edge_data.get("pipe_diameter_m", 0.6),
            slope=new_slope,
        )

    logger.info(
        "Calibrated drainage network with OSM: %d / %d nodes aligned with surface drainage channels.",
        calibrated_count,
        len(graph.nodes),
    )
    return graph


def save_network(graph: nx.DiGraph, path: Optional[Union[str, Path]] = None) -> Path:
    """
    Saves the drainage graph to a GraphML file.

    Args:
        graph (nx.DiGraph): The networkx drainage graph.
        path (str or Path, optional): Output path (defaults to data/processed/drainage_network.graphml).

    Returns:
        Path: Path to saved GraphML file.
    """
    dest_path = Path(path) if path else DEFAULT_DRAINAGE_GRAPH_PATH
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure all boolean and numeric attributes are cleanly serializable
    clean_g = graph.copy()
    for _, data in clean_g.nodes(data=True):
        for k, v in list(data.items()):
            if isinstance(v, bool):
                data[k] = int(v)

    nx.write_graphml(clean_g, dest_path)
    logger.info(
        "Saved drainage network (%d nodes, %d edges) to %s",
        len(graph.nodes),
        len(graph.edges),
        dest_path,
    )
    return dest_path


def load_network(path: Optional[Union[str, Path]] = None) -> nx.DiGraph:
    """
    Loads a drainage graph from a GraphML file, casting attributes back to native types.

    Args:
        path (str or Path, optional): Input path (defaults to data/processed/drainage_network.graphml).

    Returns:
        nx.DiGraph: The loaded directed graph.
    """
    src_path = Path(path) if path else DEFAULT_DRAINAGE_GRAPH_PATH
    if not src_path.exists():
        raise FileNotFoundError(f"Drainage graph not found at {src_path.resolve()}.")

    raw_g = nx.read_graphml(src_path)
    # Convert to DiGraph if needed
    G = nx.DiGraph(raw_g)

    # Cast numeric and boolean attributes
    float_node_attrs = ("elevation", "capacity_liters_per_min", "x", "y", "osm_channel_dist_m")
    float_edge_attrs = ("pipe_diameter_m", "flow_capacity_liters_per_min", "length_m", "slope")

    for _, data in G.nodes(data=True):
        for attr in float_node_attrs:
            if attr in data and data[attr] is not None:
                try:
                    data[attr] = float(data[attr])
                except (ValueError, TypeError):
                    pass
        if "calibrated_with_osm" in data:
            data["calibrated_with_osm"] = bool(int(data["calibrated_with_osm"]))

    for _, _, edge_data in G.edges(data=True):
        for attr in float_edge_attrs:
            if attr in edge_data and edge_data[attr] is not None:
                try:
                    edge_data[attr] = float(edge_data[attr])
                except (ValueError, TypeError):
                    pass

    logger.info("Loaded drainage network (%d nodes, %d edges) from %s", len(G.nodes), len(G.edges), src_path)
    return G


if __name__ == "__main__":
    print("\n========================================================")
    print("        Testing ingestion/drainage_graph.py             ")
    print("========================================================")

    # 1. Build synthetic drainage network (25 nodes, 3 outfalls)
    center_lat, center_lon = 13.0827, 80.2707  # Chennai, India
    drainage_net = build_sample_drainage_network(
        num_nodes=25,
        center_lat=center_lat,
        center_lon=center_lon,
        num_outfalls=3,
        random_seed=42,
    )

    outfalls = [n for n, d in drainage_net.nodes(data=True) if d.get("node_type") == "outfall"]
    inlets = [n for n, d in drainage_net.nodes(data=True) if d.get("node_type") == "inlet"]
    junctions = [n for n, d in drainage_net.nodes(data=True) if d.get("node_type") == "junction"]

    print(f"\n[1] Built Drainage Network Summary:")
    print(f"  - Total Nodes: {len(drainage_net.nodes)}")
    print(f"  - Outfalls (Sinks): {len(outfalls)} -> {outfalls}")
    print(f"  - Inlets (Surface Inlets): {len(inlets)}")
    print(f"  - Junctions (Manholes): {len(junctions)}")
    print(f"  - Total Conduits (Pipes): {len(drainage_net.edges)}")

    # Sample edge inspect
    sample_edge = list(drainage_net.edges(data=True))[0]
    print(f"\n[2] Sample Pipe Conduit ({sample_edge[0]} -> {sample_edge[1]}):")
    for k, v in sample_edge[2].items():
        print(f"    {k}: {v}")

    # 2. Test OSM drainage calibration with synthetic waterway geometries
    # Create mock OSM drainage canal near outfalls
    mock_waterways = gpd.GeoDataFrame(
        {
            "waterway": ["canal", "drain"],
            "geometry": [
                LineString([(80.275, 13.080), (80.280, 13.085)]),
                LineString([(80.270, 13.078), (80.276, 13.082)]),
            ],
        },
        crs="EPSG:4326",
    )

    print(f"\n[3] Calibrating with OSM Surface Drainage...")
    calibrated_net = calibrate_with_osm_drainage(drainage_net, mock_waterways, max_snap_dist_m=300.0)
    calibrated_nodes = [n for n, d in calibrated_net.nodes(data=True) if d.get("calibrated_with_osm")]
    print(f"  - Calibrated Nodes Count: {len(calibrated_nodes)} ({calibrated_nodes})")

    # 3. Test GraphML Save and Load round-trip
    saved_path = save_network(calibrated_net)
    reloaded_net = load_network(saved_path)
    print(f"\n[4] GraphML Round-Trip Test:")
    print(f"  - Saved to: {saved_path}")
    print(f"  - Reloaded Nodes: {len(reloaded_net.nodes)}, Edges: {len(reloaded_net.edges)}")
    
    reloaded_sample = list(reloaded_net.edges(data=True))[0]
    print(f"  - Verified Numeric Type: pipe_diameter_m is {type(reloaded_sample[2]['pipe_diameter_m'])} ({reloaded_sample[2]['pipe_diameter_m']}m)")
    print("\n[OK] Drainage graph module verified successfully!")
