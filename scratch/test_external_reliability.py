"""
test_external_reliability.py - Automated Reliability & Graceful Degradation Verification Suite

Tests:
1. Missing .env safety: Backend imports and startup proceed without crash.
2. Live Open-Meteo & Radar queries: Verified with bounded timeouts.
3. TimeoutError handling: Mocked timeout triggers fallback to Tier 1 / 2 / 3.
4. HTTP 401 Unauthorized handling: Mocked 401 triggers graceful degradation.
5. HTTP 429 Rate Limit handling: Mocked 429 triggers graceful degradation.
6. Multi-tier fallback hierarchy: Cache -> CSV -> Deterministic Synthetic Baseline.
7. End-to-end FastAPI endpoint tests with TestClient (all return HTTP 200).
"""

import sys
import os
import time
import json
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import httpx
from fastapi.testclient import TestClient

from backend.main import app
from backend.external_api import (
    fetch_open_meteo_weather,
    fetch_radar_nowcast,
    fetch_weather_provider,
    verify_tile_provider,
    load_static_csv_rainfall_summary,
    generate_deterministic_synthetic_weather,
    read_from_cache,
    write_to_cache,
    DEFAULT_TIMEOUT,
)


class TestExternalApiReliability(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        self.lat = 13.0067
        self.lon = 80.2570
        self.state = "Tamil Nadu"

    def test_01_missing_env_variables_safe(self):
        """Test that missing or empty .env environment variables do not cause crashes."""
        with patch.dict(os.environ, {}, clear=True):
            # Test helper functions with empty environment
            tile_res = self.client.get("/tiles/status")
            self.assertEqual(tile_res.status_code, 200)
            data = tile_res.json()
            self.assertEqual(data.get("active_provider"), "openstreetmap")
            self.assertTrue(data.get("is_fallback"))

    def test_02_timeout_configuration(self):
        """Verify that DEFAULT_TIMEOUT is between 5 and 10 seconds."""
        self.assertLessEqual(DEFAULT_TIMEOUT.connect, 5.0)
        self.assertLessEqual(DEFAULT_TIMEOUT.read, 10.0)
        self.assertGreaterEqual(DEFAULT_TIMEOUT.read, 5.0)

    def test_03_live_open_meteo_query_or_fallback(self):
        """Verify Open-Meteo endpoint returns HTTP 200 with live or graceful fallback payload."""
        response = self.client.get(f"/weather/live?lat={self.lat}&lon={self.lon}&provider=open-meteo")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("current_precipitation_mm_hr", data)
        self.assertIn("hourly_precipitation", data)
        self.assertIn("is_fallback", data)

    def test_04_simulated_network_timeout_degradation(self):
        """Verify that a network TimeoutError triggers graceful fallback instead of HTTP 500."""
        with patch("httpx.AsyncClient.get", side_effect=httpx.TimeoutException("Connection timed out")):
            response = self.client.get(f"/weather/live?lat={self.lat}&lon={self.lon}&provider=open-meteo")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertTrue(data.get("is_fallback"))
            self.assertIn(data.get("fallback_tier"), ["tier1_cache", "tier2_historical_csv", "tier3_synthetic_baseline"])
            self.assertIn("current_precipitation_mm_hr", data)

    def test_05_simulated_401_unauthorized_degradation(self):
        """Verify that HTTP 401 Unauthorized from an external weather provider triggers fallback."""
        mock_req = httpx.Request("GET", "https://api.openweathermap.org/data/2.5/weather")
        mock_resp = httpx.Response(status_code=401, request=mock_req)
        err = httpx.HTTPStatusError("401 Unauthorized", request=mock_req, response=mock_resp)

        with patch("httpx.AsyncClient.get", side_effect=err):
            response = self.client.get(f"/weather/live?lat={self.lat}&lon={self.lon}&provider=openweather")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertTrue(data.get("is_fallback"))
            self.assertIn("current_precipitation_mm_hr", data)

    def test_06_simulated_429_rate_limit_degradation(self):
        """Verify that HTTP 429 Too Many Requests triggers graceful fallback."""
        mock_req = httpx.Request("GET", "https://api.open-meteo.com/v1/forecast")
        mock_resp = httpx.Response(status_code=429, request=mock_req)
        err = httpx.HTTPStatusError("429 Too Many Requests", request=mock_req, response=mock_resp)

        with patch("httpx.AsyncClient.get", side_effect=err):
            response = self.client.get(f"/weather/live?lat={self.lat}&lon={self.lon}&provider=open-meteo")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertTrue(data.get("is_fallback"))
            self.assertIn(data.get("fallback_tier"), ["tier1_cache", "tier2_historical_csv", "tier3_synthetic_baseline"])

    def test_07_tier1_cache_fallback(self):
        """Verify Tier 1: cached entry is delivered if available upon network failure."""
        cache_params = {"lat": round(self.lat, 3), "lon": round(self.lon, 3), "endpoint": "open_meteo"}
        mock_cached_data = {
            "status": "cached_test_payload",
            "provider": "open-meteo",
            "latitude": self.lat,
            "longitude": self.lon,
            "current_precipitation_mm_hr": 42.0,
            "hourly_precipitation": [{"time": "2026-09-10T12:00:00Z", "precipitation_mm": 42.0}],
            "is_fallback": False,
        }
        write_to_cache("weather_open_meteo", cache_params, mock_cached_data)

        # Force network error
        with patch("httpx.AsyncClient.get", side_effect=httpx.ConnectError("Network unreachable")):
            response = self.client.get(f"/weather/live?lat={self.lat}&lon={self.lon}&provider=open-meteo")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data.get("fallback_tier"), "tier1_cache")
            self.assertEqual(data.get("current_precipitation_mm_hr"), 42.0)

    def test_08_tier2_static_csv_fallback(self):
        """Verify Tier 2: loads static historical CSV summary when cache is bypassed."""
        with patch("backend.external_api.read_from_cache", return_value=None):
            with patch("httpx.AsyncClient.get", side_effect=httpx.ConnectError("No internet")):
                response = self.client.get(f"/weather/live?lat={self.lat}&lon={self.lon}&provider=open-meteo")
                self.assertEqual(response.status_code, 200)
                data = response.json()
                self.assertEqual(data.get("fallback_tier"), "tier2_historical_csv")
                self.assertIn("historical_csv_stats", data)
                self.assertGreater(data.get("current_precipitation_mm_hr"), 0.0)

    def test_09_tier3_deterministic_synthetic_baseline(self):
        """Verify Tier 3: deterministic synthetic baseline output is mathematical and reproducible."""
        synth1 = generate_deterministic_synthetic_weather(self.lat, self.lon, hours_back=6)
        synth2 = generate_deterministic_synthetic_weather(self.lat, self.lon, hours_back=6)
        self.assertEqual(synth1["temperature_c"], synth2["temperature_c"])
        self.assertEqual(synth1["current_precipitation_mm_hr"], synth2["current_precipitation_mm_hr"])
        self.assertEqual(synth1["status"], "deterministic_synthetic_baseline")

    def test_10_radar_telemetry_endpoint(self):
        """Verify /weather/radar returns HTTP 200 with frames list."""
        response = self.client.get(f"/weather/radar?lat={self.lat}&lon={self.lon}")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("frames", data)
        self.assertIn("frames_count", data)

    def test_11_rainfall_series_endpoint(self):
        """Verify /weather/rainfall returns normalized hyetograph."""
        response = self.client.get(f"/weather/rainfall?lat={self.lat}&lon={self.lon}")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("hourly_series", data)
        self.assertIn("current_intensity_mm_hr", data)

    def test_12_flood_risk_with_open_meteo_source(self):
        """Verify /flood-risk endpoint works with rainfall_source=open-meteo."""
        response = self.client.get(
            f"/flood-risk?location=Adyar,%20Chennai,%20Tamil%20Nadu,%20India&rainfall_source=open-meteo&force_refresh=true"
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("type"), "FeatureCollection")
        self.assertIn("features", data)

    def test_13_api_info_endpoint(self):
        """Verify /api-info exposes new weather and tile endpoints."""
        response = self.client.get("/api-info")
        self.assertEqual(response.status_code, 200)
        endpoints = response.json().get("endpoints", {})
        self.assertIn("weather_live", endpoints)
        self.assertIn("weather_radar", endpoints)
        self.assertIn("weather_rainfall", endpoints)
        self.assertIn("tiles_status", endpoints)


if __name__ == "__main__":
    print("\n========================================================")
    print(" Running Backend Reliability & Degradation Test Suite   ")
    print("========================================================")
    unittest.main(verbosity=2)
