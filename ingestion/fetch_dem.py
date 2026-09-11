"""
fetch_dem.py - Digital Elevation Model (DEM) Ingestion & Geospatial Analysis

Provides functions to:
1. Query Copernicus GLO-30 DEM from Google Earth Engine and save to data/raw/dem.tif.
2. Provide fallback instructions for manual OpenTopography download.
3. Query elevation at specific (lat, lon) coordinates using rasterio.
4. Calculate slope array using numpy.gradient.
5. Extract full elevation grid, geotransform, and spatial bounds.
"""

import os
import io
import zipfile
import logging
from pathlib import Path
from typing import Tuple, Optional, Any, Union

import numpy as np
import requests
import rasterio
from rasterio.warp import transform as warp_transform
from rasterio.transform import Affine
from rasterio.coords import BoundingBox
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("ingestion.fetch_dem")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Default output directory and file path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DEM_PATH = PROJECT_ROOT / "data" / "raw" / "dem.tif"


def print_manual_download_instructions(
    lat: float,
    lon: float,
    buffer_km: float = 5.0,
    target_path: Optional[Path] = None,
) -> None:
    """Prints step-by-step instructions to manually download Copernicus DEM 30m from OpenTopography."""
    dest = target_path or DEFAULT_DEM_PATH
    delta_deg = buffer_km / 111.32
    min_lat = round(lat - delta_deg, 4)
    max_lat = round(lat + delta_deg, 4)
    min_lon = round(lon - delta_deg / max(np.cos(np.radians(lat)), 0.01), 4)
    max_lon = round(lon + delta_deg / max(np.cos(np.radians(lat)), 0.01), 4)

    print("\n" + "=" * 80)
    print(" [MANUAL FALLBACK] Google Earth Engine DEM Export Unavailable")
    print("=" * 80)
    print("To download the 30-meter Copernicus DEM manually via OpenTopography:")
    print("  1. Open OpenTopography Copernicus GLO-30 portal:")
    print("     https://portal.opentopography.org/raster?opentopoHash=b996c5aa42d4a51e6005c210bf974fa2")
    print("     (Or navigate: https://opentopography.org -> Data -> Global DEM -> Copernicus GLO-30)")
    print("\n  2. Select Area of Interest (Bounding Box):")
    print(f"     - South (Min Lat): {min_lat}")
    print(f"     - North (Max Lat): {max_lat}")
    print(f"     - West  (Min Lon): {min_lon}")
    print(f"     - East  (Max Lon): {max_lon}")
    print("\n  3. Data Output Format:")
    print("     - Choose 'GeoTIFF'")
    print("     - Click 'Submit' to process and download the raster.")
    print("\n  4. Move and rename the downloaded file:")
    print(f"     Place the file at: {dest.resolve()}")
    print("=" * 80 + "\n", flush=True)


def fetch_copernicus_dem(
    lat: float,
    lon: float,
    buffer_km: float = 5.0,
    output_path: Optional[Union[str, Path]] = None,
    allow_synthetic_fallback: bool = True,
) -> Optional[Path]:
    """
    Queries COPERNICUS/DEM/GLO30 from Google Earth Engine, clips to the
    bounding box around (lat, lon) with buffer_km radius, and saves as GeoTIFF.

    If Earth Engine export fails, logs a warning and prints OpenTopography instructions.

    Args:
        lat (float): Latitude of center point.
        lon (float): Longitude of center point.
        buffer_km (float): Buffer radius in kilometers (default: 5.0).
        output_path (str or Path, optional): Target file path (default: data/raw/dem.tif).
        allow_synthetic_fallback (bool): If True and EE fails, creates a synthetic DEM
                                         for demo/testing continuity if file doesn't exist.

    Returns:
        Optional[Path]: Path to the saved dem.tif if successful, or None if failed.
    """
    dest_path = Path(output_path) if output_path else DEFAULT_DEM_PATH
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Requesting Copernicus DEM (GLO-30) for center (%.4f, %.4f) with %0.1f km buffer...",
        lat,
        lon,
        buffer_km,
    )

    try:
        import ee

        # Initialize EE if not already initialized
        proj = os.getenv("EARTHENGINE_PROJECT_ID")
        try:
            if proj:
                ee.Initialize(project=proj)
            else:
                ee.Initialize()
        except Exception as init_err:
            raise RuntimeError(f"EE Initialization failed: {init_err}") from init_err

        # Construct region of interest bounding box
        roi = ee.Geometry.Point([lon, lat]).buffer(buffer_km * 1000.0).bounds()

        # Copernicus GLO-30 is an ImageCollection; select 'DEM' band and mosaic
        dem_col = ee.ImageCollection("COPERNICUS/DEM/GLO30")
        dem_image = dem_col.select("DEM").mosaic().clip(roi)

        logger.info("Generating Earth Engine download URL for DEM GeoTIFF...")
        download_url = dem_image.getDownloadURL({
            "name": "dem",
            "scale": 30,
            "crs": "EPSG:4326",
            "region": roi,
            "format": "GEO_TIFF",
        })

        logger.info("Downloading raster stream from Earth Engine...")
        resp = requests.get(download_url, stream=True, timeout=15)
        resp.raise_for_status()

        # Check if response is a zip archive or raw TIFF
        content = resp.content
        if content[:2] == b"PK":  # Zip file magic header
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                # Find the tif file inside zip
                tif_names = [name for name in zf.namelist() if name.endswith(".tif") or name.endswith(".tiff")]
                if not tif_names:
                    raise RuntimeError("No GeoTIFF found inside downloaded Earth Engine archive.")
                with zf.open(tif_names[0]) as source_file, open(dest_path, "wb") as target_file:
                    target_file.write(source_file.read())
        else:
            with open(dest_path, "wb") as f:
                f.write(content)

        logger.info("Successfully saved Copernicus DEM GeoTIFF to: %s", dest_path)
        return dest_path

    except Exception as exc:
        logger.warning("Earth Engine Copernicus DEM export failed: %s", exc)
        print_manual_download_instructions(lat=lat, lon=lon, buffer_km=buffer_km, target_path=dest_path)

        if allow_synthetic_fallback and not dest_path.exists():
            logger.info("Generating fallback synthetic DEM for development/demo continuity...")
            generate_synthetic_dem(dest_path, lat=lat, lon=lon, buffer_km=buffer_km)
            return dest_path

        return None if not dest_path.exists() else dest_path


def generate_synthetic_dem(
    output_path: Union[str, Path],
    lat: float = 13.0827,
    lon: float = 80.2707,
    buffer_km: float = 5.0,
    resolution_m: float = 30.0,
) -> Path:
    """
    Generates a realistic synthetic DEM with slopes, micro-ridges, and a drainage depression.
    Ensures seamless end-to-end testing when Earth Engine is unauthenticated.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    delta_deg = buffer_km / 111.32
    deg_per_m = 1.0 / 111320.0
    cell_deg = resolution_m * deg_per_m

    n_rows = int((2 * delta_deg) / cell_deg)
    n_cols = int((2 * delta_deg / max(np.cos(np.radians(lat)), 0.01)) / cell_deg)

    y_coords = np.linspace(lat + delta_deg, lat - delta_deg, n_rows)
    x_coords = np.linspace(
        lon - delta_deg / max(np.cos(np.radians(lat)), 0.01),
        lon + delta_deg / max(np.cos(np.radians(lat)), 0.01),
        n_cols,
    )
    xx, yy = np.meshgrid(x_coords, y_coords)

    # Base elevation with a regional slope towards southeast + undulating hills and river valley
    dist_x = (xx - lon) * 111320.0 * np.cos(np.radians(lat))
    dist_y = (yy - lat) * 111320.0
    r = np.sqrt(dist_x**2 + dist_y**2)

    # Elevation surface (coastal city style: 2m to 45m with urban hills and basin)
    base_elev = 15.0 - 0.0015 * dist_x - 0.001 * dist_y
    hills = 12.0 * np.sin(dist_x / 800.0) * np.cos(dist_y / 900.0)
    basin = -8.0 * np.exp(-(r**2) / (2 * (1500.0**2)))
    elevation = np.clip(base_elev + hills + basin + 10.0, 1.0, 120.0).astype(np.float32)

    transform = Affine.translation(x_coords[0], y_coords[0]) * Affine.scale(
        x_coords[1] - x_coords[0], y_coords[1] - y_coords[0]
    )

    with rasterio.open(
        out_file,
        "w",
        driver="GTiff",
        height=n_rows,
        width=n_cols,
        count=1,
        dtype=np.float32,
        crs="EPSG:4326",
        transform=transform,
        nodata=-9999.0,
    ) as dst:
        dst.write(elevation, 1)

    logger.info("Generated synthetic DEM (%dx%d) at %s", n_rows, n_cols, out_file)
    return out_file


def get_elevation_at_point(
    dem_path: Union[str, Path],
    lat: float,
    lon: float,
) -> float:
    """
    Performs point elevation lookup from a GeoTIFF DEM file.

    Args:
        dem_path: Path to the GeoTIFF DEM file.
        lat (float): Latitude of the point.
        lon (float): Longitude of the point.

    Returns:
        float: Elevation in meters (or np.nan if outside raster bounds / nodata).
    """
    path = Path(dem_path)
    if not path.exists():
        raise FileNotFoundError(
            f"DEM file not found at '{path.resolve()}'. "
            "Run fetch_copernicus_dem() or download manually from OpenTopography."
        )

    with rasterio.open(path) as src:
        # Reproject coordinates to raster CRS if not EPSG:4326
        if src.crs and src.crs.to_epsg() != 4326:
            xs, ys = warp_transform("EPSG:4326", src.crs, [lon], [lat])
            x_coord, y_coord = xs[0], ys[0]
        else:
            x_coord, y_coord = lon, lat

        # Check if coordinates lie inside bounding box
        bounds = src.bounds
        if not (bounds.left <= x_coord <= bounds.right and bounds.bottom <= y_coord <= bounds.top):
            logger.warning(
                "Coordinate (%.4f, %.4f) falls outside raster bounds: %s",
                lat,
                lon,
                bounds,
            )
            return float(np.nan)

        sample_gen = src.sample([(x_coord, y_coord)])
        val = next(sample_gen)[0]

        if src.nodata is not None and val == src.nodata:
            return float(np.nan)

        return float(val)


def calculate_slope(
    dem_path: Union[str, Path],
    in_degrees: bool = True,
) -> np.ndarray:
    """
    Calculates a 2D slope array using numpy.gradient on the elevation raster.

    Args:
        dem_path: Path to the GeoTIFF DEM file.
        in_degrees: If True, returns slope in degrees [0, 90].
                    If False, returns slope as rise/run percentage.

    Returns:
        np.ndarray: 2D numpy array representing terrain slope.
    """
    path = Path(dem_path)
    if not path.exists():
        raise FileNotFoundError(
            f"DEM file not found at '{path.resolve()}'. "
            "Run fetch_copernicus_dem() or download manually from OpenTopography."
        )

    with rasterio.open(path) as src:
        elevation = src.read(1).astype(np.float64)
        transform_obj = src.transform

        # Pixel resolution in coordinate units
        res_x = abs(transform_obj.a)
        res_y = abs(transform_obj.e)

        # Handle geographic (EPSG:4326) coordinate scaling to meters
        if src.crs and src.crs.is_geographic:
            bounds = src.bounds
            mid_lat = (bounds.bottom + bounds.top) / 2.0
            mid_lat_rad = np.radians(mid_lat)
            dy = res_y * 111320.0
            dx = res_x * 111320.0 * np.cos(mid_lat_rad)
        else:
            dx = res_x
            dy = res_y

        # Compute gradient (axis 0 = rows/Y, axis 1 = columns/X)
        dz_dy, dz_dx = np.gradient(elevation, dy, dx)

        # Slope magnitude = sqrt((dz/dx)^2 + (dz/dy)^2)
        slope_rise_run = np.sqrt(dz_dx**2 + dz_dy**2)

        if in_degrees:
            slope = np.degrees(np.arctan(slope_rise_run))
        else:
            slope = slope_rise_run

        # Mask nodata if present
        if src.nodata is not None:
            slope[elevation == src.nodata] = np.nan

        return slope


def get_elevation_grid(
    dem_path: Union[str, Path],
) -> Tuple[np.ndarray, Affine, BoundingBox]:
    """
    Reads the full elevation raster array, geotransform, and bounding box.

    Args:
        dem_path: Path to the GeoTIFF DEM file.

    Returns:
        Tuple[np.ndarray, Affine, BoundingBox]:
            - elevation (2D numpy.ndarray): Elevation values in meters.
            - transform (Affine): Affine geotransform matrix.
            - bounds (BoundingBox): Spatial bounding box (left, bottom, right, top).
    """
    path = Path(dem_path)
    if not path.exists():
        raise FileNotFoundError(
            f"DEM file not found at '{path.resolve()}'. "
            "Run fetch_copernicus_dem() or download manually from OpenTopography."
        )

    with rasterio.open(path) as src:
        elevation = src.read(1)
        transform = src.transform
        bounds = src.bounds
        return elevation, transform, bounds


if __name__ == "__main__":
    print("--- Testing fetch_dem.py ---")
    test_lat, test_lon = 13.0827, 80.2707  # Chennai, India
    
    # 1. Fetch DEM (downloads via EE or falls back to synthetic for demo continuity)
    dem_file = fetch_copernicus_dem(
        lat=test_lat,
        lon=test_lon,
        buffer_km=5.0,
        output_path=DEFAULT_DEM_PATH,
        allow_synthetic_fallback=True,
    )
    print(f"\nDEM file ready at: {dem_file}")

    # 2. Query elevation at point
    pt_elevation = get_elevation_at_point(dem_file, lat=test_lat, lon=test_lon)
    print(f"Elevation at ({test_lat}, {test_lon}): {pt_elevation:.2f} meters")

    # 3. Calculate slope raster
    slope_grid = calculate_slope(dem_file, in_degrees=True)
    print(f"Slope array shape: {slope_grid.shape}")
    print(f"Slope stats: Min={np.nanmin(slope_grid):.2f}°, Mean={np.nanmean(slope_grid):.2f}°, Max={np.nanmax(slope_grid):.2f}°")

    # 4. Get full elevation grid
    elev_array, affine_tf, bboxes = get_elevation_grid(dem_file)
    print(f"Elevation grid shape: {elev_array.shape}, Bounds: {bboxes}")
