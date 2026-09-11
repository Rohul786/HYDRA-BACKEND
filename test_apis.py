"""
test_apis.py - Comprehensive API Test Suite for Urban Flood Early Warning System

Role: QA & Full-Stack Engineer
Target: backend/main.py, routes_risk, routes_routing, routes_weather, and fallback mechanisms.

Covers:
1. Health & Root Checks:
   - GET /health: Operational status, uptime, road and drainage memory status.
   - GET /: Root service info.
   - GET /api-info: Active endpoints registry.
2. Flood Risk Assessment Endpoint:
   - GET /flood-risk with default parameters (Adyar, Chennai).
   - GET /flood-risk with in-memory cache hit (X-Cache header).
   - GET /flood-risk with rainfall_source=open-meteo.
   - GET /flood-risk with rainfall_source=synthetic.
3. Safe Routing Query:
   - GET /safe-route with identical coordinates (snaps to same road node).
   - GET /safe-route with live coordinates across Adyar/Chennai (Dijkstra and A*).
   - Strict RFC 7946 [longitude, latitude] GeoJSON coordinate format validation.
4. Fallback Simulations (All Assert HTTP 200):
   - Network Failure Simulation: Mocked connection/timeout error degrades gracefully.
   - Corrupted/Missing Model Simulation: Mocked unpickle failure activates heuristic fallback.
5. Weather & Geospatial Telemetry:
   - GET /weather/live, GET /weather/radar, GET /tiles/status.
"""

import os
import sys
import json
import unittest
from unittest.mock import patch
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import httpx
from fastapi.testclient import TestClient
from backend.main import app

# Initialize TestClient
client = TestClient(app)


class TestUrbanFloodAPIs(unittest.TestCase):
    """Full-stack API test suite validating routes, payloads, and fallback guarantees."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        # Coordinates in Adyar, Chennai
        cls.adyar_start_lat = 13.0067
        cls.adyar_start_lon = 80.2570
        cls.adyar_end_lat = 12.9997
        cls.adyar_end_lon = 80.2376

    # -------------------------------------------------------------------------
    # 1. Health and Base Endpoints
    # -------------------------------------------------------------------------

    def test_01_health_endpoint(self):
        """Verify GET /health returns HTTP 200 with operational metrics and in-memory graphs."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertIn("status", data)
        self.assertIn(data["status"], ["healthy", "degraded", "unhealthy"])
        self.assertEqual(data["service"], "urban_flood_backend")
        self.assertIn("uptime_seconds", data)
        self.assertIn("in_memory_graphs", data)
        self.assertIn("road_network", data["in_memory_graphs"])
        self.assertIn("drainage_network", data["in_memory_graphs"])
        print("  [OK] /health returned status:", data["status"])

    def test_02_root_endpoint(self):
        """Verify GET / returns HTTP 200."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)

    def test_03_api_info_endpoint(self):
        """Verify GET /api-info returns all registered endpoints."""
        response = self.client.get("/api-info")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        endpoints = data.get("endpoints", {})
        self.assertIn("health", endpoints)
        self.assertIn("flood_risk", endpoints)
        self.assertIn("safe_route", endpoints)
        self.assertIn("weather_live", endpoints)
        self.assertIn("weather_radar", endpoints)
        self.assertIn("tiles_status", endpoints)
        print("  [OK] /api-info registered endpoints count:", len(endpoints))

    # -------------------------------------------------------------------------
    # 2. Flood Risk Assessment Endpoint
    # -------------------------------------------------------------------------

    def test_04_flood_risk_default_query(self):
        """Verify GET /flood-risk returns GeoJSON FeatureCollection."""
        response = self.client.get(
            "/flood-risk?location=Adyar,%20Chennai,%20Tamil%20Nadu,%20India&rainfall_source=synthetic&storm_severity=heavy"
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data.get("type"), "FeatureCollection")
        self.assertIn("features", data)
        self.assertIn("metadata", data)
        self.assertGreater(len(data["features"]), 0)

        # Inspect first road segment feature
        feature = data["features"][0]
        self.assertEqual(feature.get("type"), "Feature")
        self.assertIn("geometry", feature)
        self.assertIn("properties", feature)
        props = feature["properties"]
        self.assertTrue(
            "final_risk_category" in props or "flood_risk_level" in props or "risk_tier" in props,
            f"Expected flood risk tier property in {list(props.keys())}",
        )
        print(f"  [OK] /flood-risk returned {len(data['features'])} road features (Metadata: {data['metadata'].get('rainfall_source')})")

    def test_05_flood_risk_in_memory_cache_hit(self):
        """Verify repeated GET /flood-risk triggers 30-second in-memory cache hit."""
        # Query once
        self.client.get("/flood-risk?location=Adyar,%20Chennai,%20Tamil%20Nadu,%20India&rainfall_source=synthetic&storm_severity=extreme")
        # Query immediately again
        cached_res = self.client.get("/flood-risk?location=Adyar,%20Chennai,%20Tamil%20Nadu,%20India&rainfall_source=synthetic&storm_severity=extreme")
        self.assertEqual(cached_res.status_code, 200)
        self.assertEqual(cached_res.headers.get("X-Cache"), "HIT")
        print("  [OK] /flood-risk 30s in-memory cache hit confirmed.")

    def test_06_flood_risk_open_meteo_source(self):
        """Verify GET /flood-risk supports live Open-Meteo rainfall source."""
        response = self.client.get(
            "/flood-risk?location=Adyar,%20Chennai,%20Tamil%20Nadu,%20India&rainfall_source=open-meteo&force_refresh=true"
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("type"), "FeatureCollection")
        print("  [OK] /flood-risk with open-meteo rainfall completed successfully.")

    # -------------------------------------------------------------------------
    # 3. Safe Routing Query Endpoint
    # -------------------------------------------------------------------------

    def test_07_safe_route_cached_same_node_coords(self):
        """Verify GET /safe-route returns instant 0m route when start and end snap to same road node."""
        url = (
            f"/safe-route?start_lat={self.adyar_start_lat}&start_lon={self.adyar_start_lon}"
            f"&end_lat={self.adyar_start_lat}&end_lon={self.adyar_start_lon}"
        )
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data.get("type"), "Feature")
        self.assertEqual(data["geometry"]["type"], "LineString")
        self.assertEqual(data["properties"]["distance_m"], 0.0)
        self.assertTrue(data["properties"]["safe"])

        # Check RFC 7946 coordinate ordering strictly [longitude, latitude]
        coords = data["geometry"]["coordinates"]
        self.assertAlmostEqual(coords[0][0], self.adyar_start_lon, places=4)
        self.assertAlmostEqual(coords[0][1], self.adyar_start_lat, places=4)
        print("  [OK] /safe-route identical node query verified.")

    def test_08_safe_route_live_dijkstra_pathfinding(self):
        """Verify GET /safe-route calculates safe detour path avoiding flood surcharge."""
        url = (
            f"/safe-route?start_lat={self.adyar_start_lat}&start_lon={self.adyar_start_lon}"
            f"&end_lat={self.adyar_end_lat}&end_lon={self.adyar_end_lon}"
            f"&avoid_floods=true&algorithm=dijkstra"
        )
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data.get("type"), "Feature")
        self.assertEqual(data["geometry"]["type"], "LineString")
        coords = data["geometry"]["coordinates"]
        self.assertGreater(len(coords), 1)

        # Confirm strict [longitude, latitude] coordinate formatting
        for pt in coords[:5]:
            self.assertIsInstance(pt, list)
            self.assertEqual(len(pt), 2)
            # Longitude for Chennai is ~80, Latitude is ~13
            self.assertGreater(pt[0], 70.0)  # Longitude
            self.assertLess(pt[1], 20.0)     # Latitude

        props = data["properties"]
        self.assertIn("distance_m", props)
        self.assertGreater(props["distance_m"], 0.0)
        self.assertIn("routing_strategy", props)
        print(f"  [OK] /safe-route Dijkstra path found: {props['distance_m']:.1f} meters over {props.get('nodes_count', 0)} nodes.")

    def test_09_safe_route_astar_pathfinding(self):
        """Verify GET /safe-route executes A* heuristic pathfinding."""
        url = (
            f"/safe-route?start_lat={self.adyar_start_lat}&start_lon={self.adyar_start_lon}"
            f"&end_lat={self.adyar_end_lat}&end_lon={self.adyar_end_lon}"
            f"&avoid_floods=true&algorithm=astar"
        )
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["properties"]["pathfinding_algorithm"], "astar")
        print("  [OK] /safe-route A* algorithm verified.")

    # -------------------------------------------------------------------------
    # 4. Fallback Simulations (All Assert HTTP 200)
    # -------------------------------------------------------------------------

    def test_10_network_failure_fallback_returns_200(self):
        """Simulate total external network outage (Timeout/ConnectError) and assert HTTP 200."""
        with patch("httpx.AsyncClient.get", side_effect=httpx.ConnectError("Simulated DNS failure")):
            response = self.client.get(
                f"/weather/live?lat={self.adyar_start_lat}&lon={self.adyar_start_lon}&provider=open-meteo"
            )
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertTrue(data.get("is_fallback"))
            self.assertIn(data.get("fallback_tier"), ["tier1_cache", "tier2_historical_csv", "tier3_synthetic_baseline"])
            print("  [OK] Network outage fallback asserted HTTP 200 (Tier:", data.get("fallback_tier"), ")")

    def test_11_corrupted_model_unpickle_fallback_returns_200(self):
        """Simulate pickle unpickling / ML model failure and assert HTTP 200 physical heuristic fallback."""
        with patch("joblib.load", side_effect=Exception("Simulated pickle protocol mismatch")):
            with patch("pickle.load", side_effect=Exception("Simulated pickle corruption")):
                response = self.client.get(
                    "/flood-risk?location=Adyar,%20Chennai,%20Tamil%20Nadu,%20India&rainfall_source=synthetic&force_refresh=true"
                )
                self.assertEqual(response.status_code, 200)
                data = response.json()
                self.assertEqual(data.get("type"), "FeatureCollection")
                self.assertIn("features", data)
                print("  [OK] Corrupted model unpickling fallback asserted HTTP 200.")

    # -------------------------------------------------------------------------
    # 5. Weather & Geospatial Telemetry
    # -------------------------------------------------------------------------

    def test_12_weather_radar_telemetry(self):
        """Verify GET /weather/radar returns radar telemetry frames."""
        response = self.client.get(f"/weather/radar?lat={self.adyar_start_lat}&lon={self.adyar_start_lon}")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("frames", data)
        print("  [OK] /weather/radar returned frames count:", len(data["frames"]))

    def test_13_tile_status_endpoint(self):
        """Verify GET /tiles/status returns tile configuration without crashing."""
        response = self.client.get("/tiles/status")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("active_provider", data)
        self.assertIn("tile_url", data)
        print("  [OK] /tiles/status active provider:", data["active_provider"])

    # -------------------------------------------------------------------------
    # 6. Error Handling & Validation
    # -------------------------------------------------------------------------

    def test_14_validation_error_returns_structured_422(self):
        """Verify missing required parameters return structured JSON 422."""
        # /safe-route without required coordinates
        response = self.client.get("/safe-route")
        self.assertEqual(response.status_code, 422)
        data = response.json()
        self.assertTrue(data.get("error"))
        self.assertEqual(data.get("status_code"), 422)
        print("  [OK] Structured 422 validation response verified.")


if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("       Urban Flood System - Comprehensive API Test Suite        ")
    print("=" * 65)
    unittest.main(verbosity=2)
