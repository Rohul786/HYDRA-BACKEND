"""
main.py - Urban Flood Risk Assessment FastAPI Backend Service

Provides RESTful API endpoints for:
1. System health and graph memory pre-loading status.
2. Urban flood nowcasting pipeline orchestration.
3. GeoJSON telemetry and flood risk delivery.
"""

import sys
import time
import logging
from pathlib import Path
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

# Ensure project root is present in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import networkx as nx
import osmnx as ox
from fastapi import FastAPI, Request, Response, status, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ingestion.drainage_graph import (
    load_network,
    DEFAULT_DRAINAGE_GRAPH_PATH,
    build_sample_drainage_network,
)
from backend.routes_risk import router as risk_router
from backend.routes_routing import router as routing_router
from backend.routes_weather import router as weather_router
from backend.routes_auth import router as auth_router
from backend.routes_notifications import router as notifications_router
from backend.cache_manager import patch_osmnx_cache, audit_cache_directory

# Configure logging
logger = logging.getLogger("backend.main")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Data paths
DEFAULT_ROAD_GRAPH_PATH = PROJECT_ROOT / "data" / "processed" / "road_network.graphml"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.
    Handles startup: pre-loading road network and drainage graph into memory (app.state).
    Handles shutdown: cleaning up resources.
    """
    logger.info("Initializing Urban Flood Backend Service...")
    app.state.start_time = time.time()
    app.state.road_network = None
    app.state.drainage_network = None
    app.state.road_network_loaded = False
    app.state.drainage_network_loaded = False
    app.state.road_stats = {"nodes": 0, "edges": 0}
    app.state.drainage_stats = {"nodes": 0, "edges": 0}

    # 0. Initialize atomic & resilient cache layer
    patch_osmnx_cache()
    try:
        audit_res = audit_cache_directory()
        logger.info(
            "Cache audit: %d valid, %d purged, %d Adyar/Chennai cache entries.",
            audit_res["valid_files"],
            audit_res["corrupt_files_purged"],
            audit_res["adyar_chennai_cache_hits_count"],
        )
    except Exception as audit_err:
        logger.warning("Cache audit encountered minor issue: %s", audit_err)

    # 1. Pre-load Road Network into memory
    logger.info("Pre-loading road network into memory...")
    try:
        # Check default path or any matching processed road graphml
        road_path = DEFAULT_ROAD_GRAPH_PATH
        if not road_path.exists():
            # Search for any road_network*.graphml in data/processed
            candidates = list((PROJECT_ROOT / "data" / "processed").glob("road_network*.graphml"))
            if candidates:
                road_path = candidates[0]

        if road_path.exists():
            logger.info("Loading road graph from %s...", road_path)
            road_g = ox.load_graphml(road_path)
            app.state.road_network = road_g
            app.state.road_network_loaded = True
            app.state.road_stats = {
                "nodes": len(road_g.nodes),
                "edges": len(road_g.edges),
                "path": str(road_path),
            }
            logger.info(
                "Successfully loaded road network: %d nodes, %d edges.",
                len(road_g.nodes),
                len(road_g.edges),
            )
        else:
            logger.warning(
                "No cached road network found at %s. Road network will be loaded on demand.",
                road_path,
            )
    except Exception as exc:
        logger.error("Failed to pre-load road network: %s", exc, exc_info=True)

    # 2. Pre-load Drainage Graph into memory
    logger.info("Pre-loading drainage network into memory...")
    try:
        drainage_path = DEFAULT_DRAINAGE_GRAPH_PATH
        if not drainage_path.exists():
            candidates = list((PROJECT_ROOT / "data" / "processed").glob("drainage_network*.graphml"))
            if candidates:
                drainage_path = candidates[0]

        if drainage_path.exists():
            logger.info("Loading drainage graph from %s...", drainage_path)
            drainage_g = load_network(drainage_path)
            app.state.drainage_network = drainage_g
            app.state.drainage_network_loaded = True
            app.state.drainage_stats = {
                "nodes": len(drainage_g.nodes),
                "edges": len(drainage_g.edges),
                "path": str(drainage_path),
            }
            logger.info(
                "Successfully loaded drainage network: %d nodes, %d edges.",
                len(drainage_g.nodes),
                len(drainage_g.edges),
            )
        else:
            logger.warning(
                "No cached drainage graph found at %s. Building synthetic fallback...",
                drainage_path,
            )
            fallback_g = build_sample_drainage_network(num_nodes=25)
            app.state.drainage_network = fallback_g
            app.state.drainage_network_loaded = True
            app.state.drainage_stats = {
                "nodes": len(fallback_g.nodes),
                "edges": len(fallback_g.edges),
                "path": "synthetic_fallback",
            }
            logger.info("Initialized synthetic fallback drainage network (25 nodes).")
    except Exception as exc:
        logger.error("Failed to pre-load drainage network: %s", exc, exc_info=True)

    logger.info(
        "Startup complete. Road Graph Loaded: %s | Drainage Graph Loaded: %s",
        app.state.road_network_loaded,
        app.state.drainage_network_loaded,
    )

    yield

    # Shutdown
    logger.info("Shutting down Urban Flood Backend Service...")


# FastAPI Application instance
app = FastAPI(
    title="Urban Flood Early Warning & Hydraulic Backflow API",
    description=(
        "Production-grade FastAPI service integrating physical hydraulic modeling "
        "(Rational Method runoff, manhole surcharge backflow propagation) and XGBoost "
        "machine learning classification for urban flood risk prediction."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS configuration: enabled for all origins, methods, and headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request Logging Middleware
@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """
    Middleware that logs each incoming request method, path, client host,
    and the resulting response status code and execution duration in milliseconds.
    """
    start_time = time.perf_counter()
    client_host = request.client.host if request.client else "unknown"
    method = request.method
    path = request.url.path

    logger.info("--> %s %s [client: %s]", method, path, client_host)

    try:
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start_time) * 1000.0
        logger.info(
            "<-- %s %s - Status: %d - Time: %.2fms",
            method,
            path,
            response.status_code,
            duration_ms,
        )
        response.headers["X-Process-Time-Ms"] = f"{duration_ms:.2f}"
        return response
    except Exception as exc:
        duration_ms = (time.perf_counter() - start_time) * 1000.0
        logger.error(
            "<-- FAIL %s %s - Exception: %s - Time: %.2fms",
            method,
            path,
            exc,
            duration_ms,
            exc_info=True,
        )
        raise exc


# Exception Handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Normalized structured JSON response for HTTPExceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "status_code": exc.status_code,
            "detail": exc.detail,
            "path": request.url.path,
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Normalized structured JSON response for request validation errors."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": True,
            "status_code": 422,
            "detail": exc.errors(),
            "path": request.url.path,
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Global catch-all error handler returning structured JSON with HTTP 500."""
    logger.error("Unhandled server exception on %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": True,
            "status_code": 500,
            "message": "Internal Server Error",
            "detail": str(exc),
            "path": request.url.path,
        },
    )


@app.api_route("/", methods=["GET", "HEAD"], tags=["Root"])
@app.api_route("/api-info", methods=["GET", "HEAD"], tags=["Root"])
async def api_info():
    """API metadata and active endpoints information. Responds to both GET and HEAD for cloud health probes."""
    return {
        "title": "Urban Flood Early Warning & Backflow API",
        "status": "online",
        "docs_url": "/docs",
        "health_url": "/health",
        "endpoints": {
            "health": "/health",
            "flood_risk": "/flood-risk",
            "safe_route": "/safe-route",
            "weather_live": "/weather/live",
            "weather_radar": "/weather/radar",
            "weather_rainfall": "/weather/rainfall",
            "tiles_status": "/tiles/status",
            "documentation": "/docs",
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.api_route("/health", methods=["GET", "HEAD"], tags=["Health"])
async def health_check():
    """
    System health check endpoint.
    Reports operational status, server uptime, and whether the road network
    and drainage graph are currently pre-loaded in memory.
    """
    uptime_sec = time.time() - getattr(app.state, "start_time", time.time())
    road_loaded = getattr(app.state, "road_network_loaded", False)
    drainage_loaded = getattr(app.state, "drainage_network_loaded", False)
    road_stats = getattr(app.state, "road_stats", {})
    drainage_stats = getattr(app.state, "drainage_stats", {})

    is_healthy = road_loaded and drainage_loaded
    status_str = "healthy" if is_healthy else ("degraded" if (road_loaded or drainage_loaded) else "unhealthy")

    return {
        "status": status_str,
        "service": "urban_flood_backend",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "uptime_seconds": round(uptime_sec, 2),
        "in_memory_graphs": {
            "road_network": {
                "loaded": road_loaded,
                "nodes": road_stats.get("nodes", 0),
                "edges": road_stats.get("edges", 0),
                "source": road_stats.get("path", None),
            },
            "drainage_network": {
                "loaded": drainage_loaded,
                "nodes": drainage_stats.get("nodes", 0),
                "edges": drainage_stats.get("edges", 0),
                "source": drainage_stats.get("path", None),
            },
        },
    }


# Register Routers
app.include_router(risk_router)
app.include_router(routing_router)
app.include_router(weather_router)
app.include_router(auth_router)
app.include_router(notifications_router)

# Mount compiled React frontend from frontend/dist if available
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"
if FRONTEND_DIST.exists():
    from fastapi.staticfiles import StaticFiles
    logger.info("Serving compiled React frontend from %s", FRONTEND_DIST)
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
else:
    @app.get("/", tags=["Root"])
    async def root():
        return {
            "title": "Urban Flood Early Warning API",
            "docs": "/docs",
            "health": "/health",
        }



if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)
