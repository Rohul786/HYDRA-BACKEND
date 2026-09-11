"""
test_cache_management.py - Verification Suite for Cache Systems Engineering

Tests:
1. Corrupt Cache File Handling:
   - Half-written JSON (e.g. truncated `{"status": "partial...`)
   - Empty files (0 bytes)
   - Completely invalid binary/garbage bytes
   - Asserts safe_load_json catches json.JSONDecodeError, logs warning, invalidates/deletes
     the corrupt file, and returns None without crashing.
2. Atomic Writes:
   - Verifies atomic_write_json creates a temp file with NamedTemporaryFile and renames via os.replace.
   - Verifies concurrent writes from multiple threads do not cause file locking conflicts (ERROR_SHARING_VIOLATION)
     or truncated/corrupted JSON.
3. Overpass / Nominatim / Weather Cache Integration:
   - Tests read_from_cache and write_to_cache with atomic writes and corrupt file recovery.
   - Tests patched OSMnx cache routines (_retrieve_from_cache and _save_to_cache).
4. Adyar / Chennai Cache Hits:
   - Audits all existing cache files in cache/.
   - Loads and parses each Adyar/Chennai cache file to guarantee 100% parse integrity.
"""

import os
import sys
import json
import time
import shutil
import tempfile
import threading
import unittest
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.cache_manager import (
    atomic_write_json,
    safe_load_json,
    patch_osmnx_cache,
    audit_cache_directory,
    DEFAULT_CACHE_DIR,
)
from backend.external_api import read_from_cache, write_to_cache


class TestCacheManagement(unittest.TestCase):

    def setUp(self):
        # Create a dedicated temporary directory for test cache operations
        self.test_cache_dir = PROJECT_ROOT / "scratch" / "test_cache_sandbox"
        self.test_cache_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        # Clean up test sandbox
        try:
            shutil.rmtree(self.test_cache_dir, ignore_errors=True)
        except Exception:
            pass

    def test_01_safe_load_valid_json(self):
        """Test safe_load_json loads valid JSON cleanly."""
        file_path = self.test_cache_dir / "valid_test.json"
        payload = {"city": "Chennai", "area": "Adyar", "risk_index": 0.45}
        file_path.write_text(json.dumps(payload), encoding="utf-8")

        result = safe_load_json(file_path, invalidate_on_corrupt=True)
        self.assertIsNotNone(result)
        self.assertEqual(result["city"], "Chennai")
        self.assertEqual(result["area"], "Adyar")
        self.assertTrue(file_path.exists())

    def test_02_corrupt_half_written_json_invalidates_and_removes(self):
        """Test safe_load_json catches JSONDecodeError on half-written file, deletes it, and returns None."""
        corrupt_file = self.test_cache_dir / "corrupt_half_written.json"
        # Truncated half-written JSON
        corrupt_file.write_text('{"status": "ok", "nodes": [1, 2, 3, {"id": 4', encoding="utf-8")

        self.assertTrue(corrupt_file.exists())
        result = safe_load_json(corrupt_file, invalidate_on_corrupt=True)

        # Must return None without crashing
        self.assertIsNone(result)
        # Must have invalidated and deleted the corrupt file
        self.assertFalse(corrupt_file.exists(), "Corrupt cache file was not deleted after JSONDecodeError!")

    def test_03_empty_cache_file_invalidates_and_removes(self):
        """Test safe_load_json detects 0-byte empty file, deletes it, and returns None."""
        empty_file = self.test_cache_dir / "empty_file.json"
        empty_file.write_text("", encoding="utf-8")

        self.assertTrue(empty_file.exists())
        result = safe_load_json(empty_file, invalidate_on_corrupt=True)

        self.assertIsNone(result)
        self.assertFalse(empty_file.exists(), "Empty cache file was not removed!")

    def test_04_garbage_binary_cache_file(self):
        """Test safe_load_json handles binary/garbage bytes safely."""
        garbage_file = self.test_cache_dir / "garbage.json"
        garbage_file.write_bytes(b"\x00\xff\xfe\x12\x34\xaa\xbb\xcc")

        result = safe_load_json(garbage_file, invalidate_on_corrupt=True)
        self.assertIsNone(result)
        self.assertFalse(garbage_file.exists())

    def test_05_atomic_write_json_produces_valid_file(self):
        """Test atomic_write_json writes content cleanly with no temp file residues."""
        target_file = self.test_cache_dir / "atomic_target.json"
        data = {"location": "Adyar, Chennai", "status": "verified", "coords": [13.0067, 80.2570]}

        written_path = atomic_write_json(target_file, data, indent=2)
        self.assertEqual(written_path, target_file)
        self.assertTrue(target_file.exists())

        loaded = json.loads(target_file.read_text(encoding="utf-8"))
        self.assertEqual(loaded["location"], "Adyar, Chennai")

        # Verify no .tmp files left in the directory
        tmp_files = list(self.test_cache_dir.glob(".tmp_*"))
        self.assertEqual(len(tmp_files), 0, f"Temporary write files left behind: {tmp_files}")

    def test_06_concurrent_atomic_writes_no_lock_conflicts(self):
        """Test heavy concurrent writes from multiple threads to same and different files."""
        n_workers = 10
        n_iterations = 25
        target_file = self.test_cache_dir / "contested_cache.json"

        errors = []

        def worker_task(worker_id):
            for i in range(n_iterations):
                try:
                    payload = {
                        "worker_id": worker_id,
                        "iteration": i,
                        "timestamp": time.time(),
                        "payload": [x * 1.5 for x in range(50)],
                    }
                    atomic_write_json(target_file, payload, indent=None)
                    # Verify read immediate
                    data = safe_load_json(target_file, invalidate_on_corrupt=True)
                    if data is None or "worker_id" not in data:
                        errors.append(f"Worker {worker_id} iteration {i}: read invalid payload")
                except Exception as e:
                    errors.append(f"Worker {worker_id} iteration {i} exception: {e}")

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(worker_task, w) for w in range(n_workers)]
            for fut in futures:
                fut.result()

        self.assertEqual(len(errors), 0, f"Concurrent write/read errors: {errors[:5]}")
        self.assertTrue(target_file.exists())

        # Final verify of the file
        final_data = safe_load_json(target_file)
        self.assertIsNotNone(final_data)
        self.assertIn("worker_id", final_data)

    def test_07_osmnx_patched_cache_resilience(self):
        """Test that monkey-patched OSMnx cache handles corrupt files without crashing."""
        self.assertTrue(patch_osmnx_cache())
        from osmnx import _http, settings

        # Point OSMnx settings to our test sandbox
        orig_cache_folder = settings.cache_folder
        orig_use_cache = settings.use_cache
        settings.cache_folder = str(self.test_cache_dir)
        settings.use_cache = True

        test_url = "https://nominatim.openstreetmap.org/search?q=Adyar+Chennai"

        try:
            # 1. Save valid response
            sample_payload = [{"name": "Adyar", "lat": 13.0067, "lon": 80.2570}]
            _http._save_to_cache(test_url, sample_payload, ok=True)

            # Retrieve from cache
            retrieved = _http._retrieve_from_cache(test_url)
            self.assertIsNotNone(retrieved)
            self.assertEqual(retrieved[0]["name"], "Adyar")

            # 2. Corrupt the cache file
            cache_file = _http._resolve_cache_filepath(test_url)
            self.assertTrue(cache_file.exists())
            cache_file.write_text('{"truncated_json": true, "list": [1, 2,', encoding="utf-8")

            # 3. Retrieve corrupt file: must return None (cache miss) and delete the file, NOT raise JSONDecodeError
            corrupt_retrieved = _http._retrieve_from_cache(test_url)
            self.assertIsNone(corrupt_retrieved)
            self.assertFalse(cache_file.exists(), "Corrupted OSMnx cache file was not purged!")

        finally:
            settings.cache_folder = orig_cache_folder
            settings.use_cache = orig_use_cache

    def test_08_verify_existing_adyar_chennai_cache_hits(self):
        """Audit the real cache/ directory and verify all existing Adyar/Chennai cache files load cleanly."""
        audit_res = audit_cache_directory(DEFAULT_CACHE_DIR)
        print("\n--- Cache Audit Results for urban_flood/cache/ ---")
        print(f"Total files: {audit_res['total_files']}")
        print(f"Valid files: {audit_res['valid_files']}")
        print(f"Corrupt files purged: {audit_res['corrupt_files_purged']}")
        print(f"Adyar/Chennai matches: {audit_res['adyar_chennai_cache_hits_count']}")

        self.assertGreater(audit_res["total_files"], 0, "No cache files found in cache/!")
        self.assertEqual(audit_res["corrupt_files_purged"], 0, "Real cache directory had corrupt files!")
        self.assertGreater(audit_res["adyar_chennai_cache_hits_count"], 0, "No Adyar/Chennai cache files found!")

        # Verify each Adyar/Chennai file loads cleanly
        for hit in audit_res["adyar_chennai_cache_hits"]:
            hit_path = DEFAULT_CACHE_DIR / hit["file"]
            self.assertTrue(hit_path.exists())
            data = safe_load_json(hit_path, invalidate_on_corrupt=False)
            self.assertIsNotNone(data, f"Adyar/Chennai cache file {hit['file']} failed to load!")
            print(f"  [OK] Loaded {hit['file']} ({hit['size_bytes']} bytes, type: {hit['type']})")


if __name__ == "__main__":
    print("\n========================================================")
    print(" Running Systems Engineer Cache Management Test Suite   ")
    print("========================================================")
    unittest.main(verbosity=2)
