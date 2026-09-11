"""
cache_manager.py - Resilient & Atomic Cache Management Logic

Provides:
1. atomic_write_json: Thread-safe and process-safe atomic JSON writes using
   tempfile.NamedTemporaryFile and os.replace to prevent truncated or half-written files.
2. safe_load_json: Safe JSON loader catching json.JSONDecodeError that invalidates
   and deletes corrupt/truncated files, preventing server crashes.
3. patch_osmnx_cache: Monkey-patches OSMnx internal HTTP cache routines to use
   safe_load_json and atomic_write_json for all Overpass/Nominatim cached queries.
4. audit_cache_directory: Audits all JSON cache files in cache/, invalidating any
   corrupt files and validating Adyar/Chennai cache integrity.
"""

import os
import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Optional, Dict, Union, List

logger = logging.getLogger("backend.cache_manager")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_DIR = PROJECT_ROOT / "cache"
DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)


from threading import Lock
import time

_FILE_LOCKS: Dict[str, Lock] = {}
_LOCKS_LOCK = Lock()


def _get_file_lock(file_path: Path) -> Lock:
    """Returns a process-wide thread lock for the given canonical file path."""
    key = str(file_path.resolve()).lower()
    with _LOCKS_LOCK:
        if key not in _FILE_LOCKS:
            _FILE_LOCKS[key] = Lock()
        return _FILE_LOCKS[key]


def atomic_write_json(
    file_path: Union[str, Path],
    data: Any,
    indent: Optional[int] = 2,
    max_retries: int = 6,
) -> Path:
    """
    Atomically writes JSON data to the target file path.
    
    1. Acquires in-process lock for the target file.
    2. Creates a NamedTemporaryFile in the SAME directory as the target file
       (ensuring same filesystem for atomic rename / replace).
    3. Serializes data to JSON and flushes/fsyncs to disk.
    4. Closes the file descriptor (critical on Windows to avoid ERROR_SHARING_VIOLATION).
    5. Atomically replaces the target path using os.replace() with exponential backoff
       retries for Windows OS / file-system anti-virus transient lock contention.
    
    Args:
        file_path (str or Path): Destination path for the JSON file.
        data (Any): Python object serializable to JSON.
        indent (int, optional): JSON indentation. Default is 2.
        max_retries (int): Retries on transient OS lock contention.
        
    Returns:
        Path: Resolved path to the written file.
    """
    target = Path(file_path).resolve()
    target_dir = target.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    file_lock = _get_file_lock(target)

    with file_lock:
        temp_file = tempfile.NamedTemporaryFile(
            mode="w",
            dir=str(target_dir),
            prefix=f".tmp_{target.stem}_",
            suffix=".json",
            delete=False,
            encoding="utf-8",
        )
        temp_path = Path(temp_file.name)

        try:
            json.dump(data, temp_file, indent=indent, default=str)
            temp_file.flush()
            os.fsync(temp_file.fileno())
            temp_file.close()

            # Atomic rename / replace with backoff retries for Windows lock contention
            for attempt in range(max_retries):
                try:
                    os.replace(temp_path, target)
                    break
                except (PermissionError, OSError) as os_err:
                    if attempt < max_retries - 1:
                        time.sleep(0.01 * (2 ** attempt))
                    else:
                        raise os_err

            logger.debug("Atomically wrote cache file: %s", target.name)
            return target
        except Exception as exc:
            logger.error("Failed atomic write to %s: %s", target, exc, exc_info=True)
            temp_file.close()
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except Exception:
                pass
            raise exc


def safe_load_json(
    file_path: Union[str, Path],
    invalidate_on_corrupt: bool = True,
    max_retries: int = 4,
) -> Optional[Any]:
    """
    Safely reads and parses a JSON cache file.
    
    If the file is empty, half-written, or corrupted:
    - Catches json.JSONDecodeError, UnicodeDecodeError, and ValueError.
    - Logs a warning.
    - If invalidate_on_corrupt is True, invalidates and deletes the corrupt file
      so subsequent requests cleanly miss the cache rather than crashing with HTTP 500.
    - Returns None.
    
    Args:
        file_path (str or Path): Path to the JSON file.
        invalidate_on_corrupt (bool): Whether to delete the corrupt file. Default True.
        max_retries (int): Retries on transient Windows file contention during replace.
        
    Returns:
        Any or None: The parsed JSON object, or None on error.
    """
    target = Path(file_path).resolve()
    if not target.exists():
        return None

    file_lock = _get_file_lock(target)
    raw_bytes = None

    with file_lock:
        for attempt in range(max_retries):
            try:
                raw_bytes = target.read_bytes()
                break
            except (PermissionError, OSError):
                if attempt < max_retries - 1:
                    time.sleep(0.005 * (2 ** attempt))
                else:
                    return None

    if not raw_bytes or not raw_bytes.strip():
        logger.warning("Empty or unreadable cache file detected: %s. Invalidating...", target.name)
        if invalidate_on_corrupt:
            with file_lock:
                target.unlink(missing_ok=True)
        return None

    try:
        content = raw_bytes.decode("utf-8")
        return json.loads(content)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as decode_err:
        logger.warning(
            "Corrupt or half-written cache file detected at %s (%s).",
            target.name,
            decode_err,
        )
        if invalidate_on_corrupt:
            try:
                with file_lock:
                    target.unlink(missing_ok=True)
                logger.info("Successfully invalidated and deleted corrupt cache file: %s", target.name)
            except Exception as rm_err:
                logger.error("Could not delete corrupt cache file %s: %s", target.name, rm_err)
        return None

    except Exception as exc:
        logger.error("Unexpected error parsing cache file %s: %s", target.name, exc)
        return None


# -----------------------------------------------------------------------------
# Monkey-Patch OSMnx Cache Routines
# -----------------------------------------------------------------------------

_OSMNX_PATCHED = False


def patch_osmnx_cache() -> bool:
    """
    Patches OSMnx _http._retrieve_from_cache and _http._save_to_cache to use
    safe_load_json and atomic_write_json. This ensures OSMnx operations never
    crash on corrupt Overpass/Nominatim cache files, and concurrent queries
    never produce half-written or locked files.
    """
    global _OSMNX_PATCHED
    if _OSMNX_PATCHED:
        return True

    try:
        import osmnx as ox
        from osmnx import _http, settings, utils
        import logging as lg

        original_retrieve = _http._retrieve_from_cache
        original_save = _http._save_to_cache

        def resilient_retrieve_from_cache(url: str):
            if not settings.use_cache:
                return None
            cache_filepath = _http._check_cache(url)
            if cache_filepath is not None:
                # Use safe_load_json with corrupt file auto-invalidation
                data = safe_load_json(cache_filepath, invalidate_on_corrupt=True)
                if data is not None:
                    msg = f"Retrieved valid response from cache file {str(cache_filepath)!r}"
                    utils.log(msg, lg.INFO)
                    return data
                else:
                    msg = f"Cache hit for {url!r} was corrupt or unreadable. Invalidated file; falling back to fresh query."
                    utils.log(msg, lg.WARNING)
                    return None
            return None

        def resilient_save_to_cache(url: str, response_json: Any, ok: bool) -> None:
            if not settings.use_cache:
                return
            if not ok:
                return
            if isinstance(response_json, dict) and ("remark" in response_json):
                return

            cache_filepath = _http._resolve_cache_filepath(url)
            try:
                # Atomic write using NamedTemporaryFile and os.replace
                atomic_write_json(cache_filepath, response_json, indent=None)
                msg = f"Atomically saved response to cache file {str(cache_filepath)!r}"
                utils.log(msg, level=lg.INFO)
            except Exception as exc:
                msg = f"Failed to atomically save response to cache file {str(cache_filepath)!r}: {exc}"
                utils.log(msg, level=lg.WARNING)

        _http._retrieve_from_cache = resilient_retrieve_from_cache
        _http._save_to_cache = resilient_save_to_cache
        _OSMNX_PATCHED = True
        logger.info("Successfully patched OSMnx cache routines with atomic writes and corrupt file invalidation.")
        return True

    except Exception as exc:
        logger.warning("Could not patch OSMnx cache routines: %s", exc)
        return False


# -----------------------------------------------------------------------------
# Cache Audit & Integrity Verification
# -----------------------------------------------------------------------------

def audit_cache_directory(cache_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Audits all JSON files in the cache directory:
    1. Validates JSON integrity using safe_load_json.
    2. Automatically invalidates and purges any corrupt, truncated, or unreadable files.
    3. Identifies and validates cache hits for Adyar/Chennai infrastructure.
    
    Returns:
        Dict summary with counts of total files, valid files, corrupt files cleaned,
        and specific Adyar/Chennai cache entries.
    """
    target_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    if not target_dir.exists():
        return {"status": "not_found", "total_files": 0, "valid_files": 0, "corrupt_files": 0, "adyar_hits": []}

    json_files = list(target_dir.glob("*.json"))
    valid_count = 0
    corrupt_count = 0
    adyar_chennai_hits: List[Dict[str, Any]] = []

    for f in json_files:
        data = safe_load_json(f, invalidate_on_corrupt=True)
        if data is None:
            corrupt_count += 1
        else:
            valid_count += 1
            # Check if this cache file contains Adyar or Chennai geographic data
            content_str = json.dumps(data)
            if "adyar" in content_str.lower() or "chennai" in content_str.lower():
                adyar_chennai_hits.append({
                    "file": f.name,
                    "size_bytes": f.stat().st_size if f.exists() else 0,
                    "type": "overpass" if isinstance(data, dict) and "elements" in data else ("nominatim" if isinstance(data, list) else "weather_or_telemetry"),
                })

    return {
        "status": "completed",
        "cache_dir": str(target_dir),
        "total_files": len(json_files),
        "valid_files": valid_count,
        "corrupt_files_purged": corrupt_count,
        "adyar_chennai_cache_hits_count": len(adyar_chennai_hits),
        "adyar_chennai_cache_hits": adyar_chennai_hits,
    }


# Auto-patch OSMnx cache on module import
patch_osmnx_cache()
