#!/usr/bin/env python3
"""
setup_data.py - Data Setup & Environment Prerequisite Checker for Urban Flood Nowcasting

Tasks performed:
1. Verifies Kaggle API credentials (~/.kaggle/kaggle.json) and prints setup guide if missing.
2. Downloads and unzips:
   - "Flood Prediction Dataset" (Kaggle) -> data/raw/
   - "Daily Rainfall Data India 2009-2024" (Kaggle) -> data/raw/
3. Verifies Earth Engine API authentication (ee.Initialize()) and prints guide if unauthenticated.
4. Outputs an ASCII summary table detailing ready and missing components.
"""

import os
import sys
import shutil
import zipfile
import subprocess
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Kaggle Datasets configuration
KAGGLE_DATASETS = [
    {
        "name": "Flood Prediction Dataset",
        "slug": "aritra100/flood-prediction-dataset",
        "fallback_slugs": ["naiyakhalid/flood-prediction-dataset"],
        "expected_files": ["train.csv", "test.csv"],
        "target_subfolder": None,  # Extract directly into data/raw/
    },
    {
        "name": "Daily Rainfall Data India 2009-2024",
        "slug": "wydoinn/daily-rainfall-data-india-2009-2024",
        "fallback_slugs": ["vimalborana/india-state-level-daily-rainfall-2009-2024"],
        "expected_files": ["daily-rainfall-at-state-level.csv"],
        "target_subfolder": None,
    },
]


def print_banner(title: str, char: str = "=") -> None:
    """Print a prominent visual section header."""
    width = 80
    print("\n" + char * width)
    print(f" {title}")
    print(char * width)


def check_kaggle_credentials() -> tuple[bool, str]:
    """Check if ~/.kaggle/kaggle.json exists or Kaggle env vars are set."""
    print_banner("1. Checking Kaggle API Credentials")
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"

    has_file = kaggle_json.exists() and kaggle_json.stat().st_size > 0
    has_env = bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))

    if has_file:
        print(f"[OK] Kaggle credentials found at: {kaggle_json}")
        return True, f"Found at {kaggle_json}"
    elif has_env:
        print(f"[OK] Kaggle credentials set via environment variables (KAGGLE_USERNAME, KAGGLE_KEY)")
        return True, "Set via environment variables"
    else:
        print(f"[WARNING] Kaggle credentials not found at {kaggle_json}")
        print("\nHow to configure Kaggle API:")
        print("  1. Sign in or register at https://www.kaggle.com")
        print("  2. Open your Account Settings: https://www.kaggle.com/settings")
        print("  3. Scroll to the 'API' section and click 'Create New Token'.")
        print("  4. A file named 'kaggle.json' will download.")
        print(f"  5. Move 'kaggle.json' into: {kaggle_dir}")
        print("     On Windows: %USERPROFILE%\\.kaggle\\kaggle.json")
        print("     On Linux/macOS: ~/.kaggle/kaggle.json (chmod 600 ~/.kaggle/kaggle.json)")
        print("  6. Alternatively, set KAGGLE_USERNAME and KAGGLE_KEY in your .env or environment.")
        return False, f"Missing {kaggle_json}"


def get_kaggle_cmd() -> list[str]:
    """Determine the command prefix to execute the Kaggle CLI."""
    kaggle_bin = shutil.which("kaggle")
    if kaggle_bin:
        return [kaggle_bin]
    # Fallback to python -m kaggle
    return [sys.executable, "-m", "kaggle"]


def download_and_extract_kaggle_dataset(dataset_cfg: dict, kaggle_cmd: list[str], force: bool = False) -> tuple[bool, str]:
    """Download a dataset using the Kaggle CLI and extract zip files to data/raw/."""
    name = dataset_cfg["name"]
    primary_slug = dataset_cfg["slug"]
    slugs = [primary_slug] + dataset_cfg.get("fallback_slugs", [])
    expected = dataset_cfg["expected_files"]

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Check if already extracted
    already_present = [f for f in expected if (RAW_DATA_DIR / f).exists()]
    if already_present and not force:
        print(f"[EXISTS] {name} is already present in {RAW_DATA_DIR.relative_to(PROJECT_ROOT)}: {', '.join(already_present)}", flush=True)
        return True, f"Present ({', '.join(already_present)})"

    download_success = False
    used_slug = None

    for slug in slugs:
        print(f"\n-> Downloading '{name}' ({slug}) into {RAW_DATA_DIR.relative_to(PROJECT_ROOT)}...", flush=True)
        # We pass --unzip so the kaggle CLI extracts automatically if supported
        cmd = kaggle_cmd + ["datasets", "download", "-d", slug, "-p", str(RAW_DATA_DIR), "--unzip"]
        if force:
            cmd.append("-o")

        try:
            # Run without capturing so user can observe real-time progress
            res = subprocess.run(cmd, check=False)
            if res.returncode == 0:
                download_success = True
                used_slug = slug
                break
            else:
                print(f"   [Notice] Command returned exit code {res.returncode} for slug '{slug}'.", flush=True)
        except Exception as e:
            print(f"   [Error] Failed executing kaggle CLI: {e}", flush=True)

    # Fallback extraction for any lingering .zip files
    for zip_path in RAW_DATA_DIR.glob("*.zip"):
        print(f"-> Unzipping {zip_path.name}...", flush=True)
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(RAW_DATA_DIR)
            try:
                zip_path.unlink()
            except OSError:
                pass
        except Exception as e:
            print(f"   [Error] Failed to extract {zip_path.name}: {e}", flush=True)

    # Verify expected files exist
    found_files = [f for f in expected if (RAW_DATA_DIR / f).exists()]
    all_raw_files = [f.name for f in RAW_DATA_DIR.iterdir() if f.is_file() and not f.name.endswith(".gitkeep") and not f.name.endswith(".kaggle-partial")]

    if found_files:
        msg = f"Ready ({', '.join(found_files)})"
        print(f"[OK] {name}: {msg}", flush=True)
        return True, msg
    elif download_success and all_raw_files:
        msg = f"Extracted ({', '.join(all_raw_files[:2])})"
        print(f"[OK] {name}: {msg}", flush=True)
        return True, msg
    else:
        msg = "Download/Extraction failed or files missing"
        print(f"[FAILED] {name}: {msg}", flush=True)
        return False, msg


def check_earth_engine() -> tuple[bool, str]:
    """Check Earth Engine authentication via ee.Initialize()."""
    print_banner("3. Checking Google Earth Engine API Authentication")
    try:
        import ee
    except ImportError:
        print("[ERROR] 'earthengine-api' package is not installed. Run 'pip install earthengine-api'.")
        return False, "Package not installed"

    project = os.environ.get("EARTHENGINE_PROJECT_ID")
    if not project:
        print("[WARNING] EARTHENGINE_PROJECT_ID is not configured.")
        print("Earth Engine now requires a registered Google Cloud project (as of the 2025 non-commercial project verification requirement).")
        print("Please configure your Cloud project ID in your .env file or environment:")
        print("  EARTHENGINE_PROJECT_ID=your-cloud-project-id")
        return False, "Missing EARTHENGINE_PROJECT_ID (required)"

    sa_email = os.environ.get("EE_SERVICE_ACCOUNT_EMAIL")
    sa_key_path = os.environ.get("EE_PRIVATE_KEY_PATH")

    try:
        if sa_email and sa_key_path:
            print(f"[i] Authenticating using Service Account: {sa_email}...")
            credentials = ee.ServiceAccountCredentials(sa_email, sa_key_path)
            ee.Initialize(credentials=credentials, project=project)
            print("[OK] Google Earth Engine initialized and authenticated via Service Account successfully.")
            return True, "Authenticated (Service Account)"
        else:
            ee.Initialize(project=project)
            print("[OK] Google Earth Engine initialized and authenticated successfully.")
            return True, "Authenticated"
    except Exception as exc:
        err_msg = str(exc).strip()
        print(f"[WARNING] Earth Engine authentication failed: {err_msg}")
        print("\nHow to authenticate Google Earth Engine:")
        print("  1. Sign up for Earth Engine access: https://earthengine.google.com/signup/")
        print("  2. In your terminal, run the authentication command:")
        print("       earthengine authenticate")
        print("     or inside Python:")
        print("       python -c \"import ee; ee.Authenticate()\"")
        print("  3. Follow browser instructions to grant permissions to your Google account.")
        print("  4. Specify your Google Cloud Project ID in .env:")
        print("       EARTHENGINE_PROJECT_ID=your-cloud-project-id")
        print("  5. Or configure Service Account credentials in .env:")
        print("       EE_SERVICE_ACCOUNT_EMAIL=your-sa@project.iam.gserviceaccount.com")
        print("       EE_PRIVATE_KEY_PATH=path/to/private_key.json")
        return False, f"Auth required ({err_msg[:60]}...)" if len(err_msg) > 60 else f"Auth required ({err_msg})"


def print_summary_table(summary_items: list[dict]) -> None:
    """Print an ASCII summary table of component readiness."""
    print_banner("4. System & Dataset Readiness Summary")
    
    headers = ("Component / Resource", "Status", "Details")
    col_w1 = 36
    col_w2 = 14
    col_w3 = 42

    sep = f"+{'-' * (col_w1 + 2)}+{'-' * (col_w2 + 2)}+{'-' * (col_w3 + 2)}+"
    header_row = f"| {headers[0].ljust(col_w1)} | {headers[1].ljust(col_w2)} | {headers[2].ljust(col_w3)} |"

    print(sep)
    print(header_row)
    print(sep)

    for item in summary_items:
        comp = item["component"][:col_w1].ljust(col_w1)
        status_text = item["status"]
        if status_text in ("READY", "OK"):
            status = f"[OK]".ljust(col_w2)
        elif status_text == "EXISTS":
            status = f"[EXISTS]".ljust(col_w2)
        elif status_text == "SKIPPED":
            status = f"[SKIPPED]".ljust(col_w2)
        else:
            status = f"[ACTION REQ]".ljust(col_w2)
        details = item["details"][:col_w3].ljust(col_w3)
        print(f"| {comp} | {status} | {details} |")

    print(sep)
    print("\nNext steps:")
    print(" - Run `streamlit run frontend/app.py` or launch the backend after resolving missing items.")
    print(" - Configure .env from .env.example with your specific credentials.")


def main() -> None:
    """Main orchestration routine."""
    import argparse
    parser = argparse.ArgumentParser(description="Setup data and verify credentials for flood nowcasting.")
    parser.add_argument("--skip-download", action="store_true", help="Skip dataset downloading step.")
    parser.add_argument("--force", action="store_true", help="Force re-download even if files exist.")
    args = parser.parse_args()

    summary_items = []

    # 1. Kaggle Credentials check
    kaggle_ok, kaggle_msg = check_kaggle_credentials()
    summary_items.append({
        "component": "Kaggle API Credentials",
        "status": "OK" if kaggle_ok else "MISSING",
        "details": kaggle_msg
    })

    # 2. Download and unzip datasets
    print_banner("2. Downloading Datasets into data/raw/")
    if args.skip_download:
        print("[SKIPPED] Dataset download skipped via --skip-download.", flush=True)
        for ds_cfg in KAGGLE_DATASETS:
            found = [f for f in ds_cfg["expected_files"] if (RAW_DATA_DIR / f).exists()]
            summary_items.append({
                "component": ds_cfg["name"],
                "status": "READY" if found else "SKIPPED",
                "details": f"Present ({', '.join(found)})" if found else "Download skipped"
            })
    else:
        kaggle_cmd = get_kaggle_cmd()
        for ds_cfg in KAGGLE_DATASETS:
            ds_ok, ds_msg = download_and_extract_kaggle_dataset(ds_cfg, kaggle_cmd, force=args.force)
            summary_items.append({
                "component": ds_cfg["name"],
                "status": "READY" if ds_ok else "MISSING",
                "details": ds_msg
            })

    # 3. Earth Engine check
    ee_ok, ee_msg = check_earth_engine()
    summary_items.append({
        "component": "Earth Engine API Auth",
        "status": "OK" if ee_ok else "MISSING",
        "details": ee_msg
    })

    # 4. Final summary table
    print_summary_table(summary_items)


if __name__ == "__main__":
    main()

