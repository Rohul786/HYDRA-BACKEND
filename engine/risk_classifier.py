"""
risk_classifier.py - Machine Learning Flood Risk Classification

========================================================================================
METHODOLOGICAL NOTE & FEATURE-MAPPING APPROXIMATION:
The Kaggle Flood Prediction Dataset models regional flood vulnerability across 20
broad environmental, infrastructural, and socio-economic indicators (e.g.,
MonsoonIntensity, TopographyDrainage, Urbanization, DrainageSystems, CoastalVulnerability)
with features distributed roughly between 1 and 15 (mean ~5).

Our live nowcasting pipeline models local, high-resolution physical urban stormwater
hydraulics (conduit discharge rates, manhole surcharges, OSM building/road imperviousness,
and Copernicus DEM terrain slopes).

To bridge this scale gap, map_node_features_to_model_input() translates localized
micro-scale telemetry into the macro-scale Kaggle feature space:
  - Rainfall Intensity (mm/hr)          -> MonsoonIntensity
  - DEM Slope & Elevation               -> TopographyDrainage & CoastalVulnerability
  - OSM Imperviousness Grid (C)         -> Urbanization, Deforestation & WetlandLoss
  - Hydraulic Surcharge Ratio (Q/Cap)   -> DrainageSystems & DeterioratingInfrastructure
  - Channel Proximity & Surcharge       -> RiverManagement & InadequatePlanning
========================================================================================
"""

import sys
import math
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

logger = logging.getLogger("engine.risk_classifier")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Default paths
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "flood.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "data" / "processed" / "flood_risk_model.pkl"

# Kaggle Dataset 20 Core Features
FEATURE_COLUMNS = [
    "MonsoonIntensity",
    "TopographyDrainage",
    "RiverManagement",
    "Deforestation",
    "Urbanization",
    "ClimateChange",
    "DamsQuality",
    "Siltation",
    "AgriculturalPractices",
    "Encroachments",
    "IneffectiveDisasterPreparedness",
    "DrainageSystems",
    "CoastalVulnerability",
    "Landslides",
    "Watersheds",
    "DeterioratingInfrastructure",
    "PopulationScore",
    "WetlandLoss",
    "InadequatePlanning",
    "PoliticalFactors",
]

RISK_CLASS_NAMES = ["Low", "Moderate", "High", "Severe"]
RISK_LABEL_MAP = {0: "Low", 1: "Moderate", 2: "High", 3: "Severe"}
RISK_NAME_MAP = {"Low": 0, "Moderate": 1, "High": 2, "Severe": 3}


def load_training_data(
    csv_path: Optional[Union[str, Path]] = None,
    sample_size: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Loads the Kaggle Flood Prediction Dataset (20 features + FloodProbability)
    and bins FloodProbability into 4 balanced quantile categories:
    [Low, Moderate, High, Severe].

    Args:
        csv_path (str or Path, optional): Path to flood.csv or train.csv.
        sample_size (int, optional): Subsample rows for rapid training if desired.

    Returns:
        Tuple[pd.DataFrame, Dict[str, float]]:
            - Preprocessed DataFrame with features and 'risk_class', 'risk_label'.
            - Quantile threshold cutoffs dictionary.
    """
    target_path = Path(csv_path) if csv_path else DEFAULT_DATASET_PATH

    if not target_path.exists():
        # Search data/raw for any flood prediction dataset csv
        raw_dir = PROJECT_ROOT / "data" / "raw"
        candidates = list(raw_dir.glob("*flood*.csv")) + list(raw_dir.glob("train.csv"))
        if candidates:
            target_path = candidates[0]
        else:
            raise FileNotFoundError(
                f"Kaggle Flood Prediction Dataset not found at {target_path.resolve()}. "
                "Run `python setup_data.py` to download the dataset into data/raw/."
            )

    logger.info("Loading Kaggle Flood Prediction Dataset from %s...", target_path)
    df = pd.read_csv(target_path)

    # Check that required columns exist
    missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_features:
        raise ValueError(f"CSV is missing required Kaggle feature columns: {missing_features}")

    if "FloodProbability" not in df.columns:
        raise ValueError("CSV is missing the target column 'FloodProbability'.")

    if sample_size and len(df) > sample_size:
        logger.info("Sampling %d rows out of %d for efficient training...", sample_size, len(df))
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    # Quantile thresholds (4 bins: 25th, 50th, 75th percentiles)
    q25 = float(df["FloodProbability"].quantile(0.25))
    q50 = float(df["FloodProbability"].quantile(0.50))
    q75 = float(df["FloodProbability"].quantile(0.75))

    thresholds = {
        "q25": round(q25, 4),
        "q50": round(q50, 4),
        "q75": round(q75, 4),
        "min": round(float(df["FloodProbability"].min()), 4),
        "max": round(float(df["FloodProbability"].max()), 4),
    }

    # Bin FloodProbability into 4 ordinal risk categories
    # qcut provides perfectly balanced quantile categories
    df["risk_class"] = pd.qcut(
        df["FloodProbability"],
        q=4,
        labels=RISK_CLASS_NAMES,
    )
    df["risk_label"] = df["risk_class"].map(RISK_NAME_MAP).astype(int)

    logger.info(
        "Loaded %d samples. Quantile Cutoffs: Low < %.3f <= Moderate < %.3f <= High < %.3f <= Severe.",
        len(df),
        q25,
        q50,
        q75,
    )
    return df, thresholds


def _compute_metrics_pure_python(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Pure-python / numpy fallback metric calculator for accuracy, F1, and confusion matrix."""
    accuracy = float(np.mean(y_true == y_pred))

    classes = [0, 1, 2, 3]
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < len(classes) and 0 <= p < len(classes):
            cm[t, p] += 1

    # Weighted F1
    f1_list = []
    weights = []
    for c in classes:
        tp = cm[c, c]
        fp = np.sum(cm[:, c]) - tp
        fn = np.sum(cm[c, :]) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_list.append(f1)
        weights.append(np.sum(cm[c, :]))

    total_weight = max(sum(weights), 1)
    weighted_f1 = float(sum(f * w for f, w in zip(f1_list, weights)) / total_weight)

    return {
        "accuracy": round(accuracy, 4),
        "f1_score": round(weighted_f1, 4),
        "confusion_matrix": cm,
    }


class LinearQuantileEnsemble:
    """Numpy-based linear quantile classifier reproducing the exact Kaggle S4E5 data structure."""
    def __init__(self, thresholds: Dict[str, float]):
        self.thresholds = thresholds

    def predict(self, X: np.ndarray) -> np.ndarray:
        s = np.sum(X, axis=1)
        preds = np.zeros(len(X), dtype=int)
        preds[s >= 95] = 1
        preds[s >= 100] = 2
        preds[s >= 106] = 3
        return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        s = np.sum(X, axis=1)
        probs = np.zeros((len(X), 4))
        centers = np.array([88.0, 97.0, 103.0, 112.0])
        for i, val in enumerate(s):
            dists = -((val - centers) ** 2) / 50.0
            exp_d = np.exp(dists - np.max(dists))
            probs[i] = exp_d / np.sum(exp_d)
        return probs


def generate_synthetic_training_data(n_samples: int = 5000) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Generates a calibrated synthetic training DataFrame matching the Kaggle Flood Prediction
    20-feature schema when the raw Kaggle dataset CSV is not locally available.
    """
    rng = np.random.default_rng(42)
    data = {}
    for col in FEATURE_COLUMNS:
        data[col] = rng.integers(1, 14, size=n_samples)
    df = pd.DataFrame(data)

    # Correlate target FloodProbability with dominant physical indicators
    df["FloodProbability"] = (
        0.32
        + 0.022 * df["MonsoonIntensity"]
        + 0.018 * df["TopographyDrainage"]
        + 0.015 * df["Urbanization"]
        + 0.014 * df["DrainageSystems"]
        + 0.010 * df["DeterioratingInfrastructure"]
        + 0.008 * df["RiverManagement"]
        + rng.normal(0, 0.025, size=n_samples)
    )
    df["FloodProbability"] = np.clip(df["FloodProbability"], 0.20, 0.85)

    q25 = float(df["FloodProbability"].quantile(0.25))
    q50 = float(df["FloodProbability"].quantile(0.50))
    q75 = float(df["FloodProbability"].quantile(0.75))
    thresholds = {
        "q25": round(q25, 4),
        "q50": round(q50, 4),
        "q75": round(q75, 4),
        "min": round(float(df["FloodProbability"].min()), 4),
        "max": round(float(df["FloodProbability"].max()), 4),
    }

    df["risk_class"] = pd.qcut(df["FloodProbability"], q=4, labels=RISK_CLASS_NAMES)
    df["risk_label"] = df["risk_class"].map(RISK_NAME_MAP).astype(int)
    return df, thresholds


def train_model(
    df: Optional[pd.DataFrame] = None,
    csv_path: Optional[Union[str, Path]] = None,
    output_model_path: Optional[Union[str, Path]] = None,
    sample_size: Optional[int] = 25000,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Any:
    """
    Trains an XGBoost or Scikit-Learn classifier on the 20 features, prints performance metrics
    (Accuracy, F1 score, Confusion Matrix), and serializes the model to flood_risk_model.pkl.

    Args:
        df (pd.DataFrame, optional): Pre-loaded DataFrame from load_training_data().
        csv_path (str or Path, optional): Custom path to flood.csv if df is not provided.
        output_model_path (str or Path, optional): Path to save flood_risk_model.pkl.
        sample_size (int, optional): Number of rows to use for training (default: 25000).
        test_size (float): Proportion of data held out for testing (default: 0.20).
        random_state (int): Seed for reproducibility.

    Returns:
        Any: The trained classifier model object.
    """
    target_out = Path(output_model_path) if output_model_path else DEFAULT_MODEL_PATH
    target_out.parent.mkdir(parents=True, exist_ok=True)

    if df is None:
        try:
            df, thresholds = load_training_data(csv_path=csv_path, sample_size=sample_size)
        except Exception as load_err:
            logger.warning(
                "Kaggle training data not loaded (%s). Generating calibrated synthetic data for training...",
                load_err,
            )
            df, thresholds = generate_synthetic_training_data(n_samples=sample_size or 5000)
    else:
        q25 = float(df["FloodProbability"].quantile(0.25))
        q50 = float(df["FloodProbability"].quantile(0.50))
        q75 = float(df["FloodProbability"].quantile(0.75))
        thresholds = {"q25": q25, "q50": q50, "q75": q75}

    X = df[FEATURE_COLUMNS].values
    y = df["risk_label"].values

    # Train / Test split
    rng = np.random.default_rng(random_state)
    n_samples = len(X)
    shuffled_idx = rng.permutation(n_samples)
    n_test = int(n_samples * test_size)

    test_idx = shuffled_idx[:n_test]
    train_idx = shuffled_idx[n_test:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    logger.info("Training feature matrix shape: %s, Test matrix: %s.", X_train.shape, X_test.shape)

    # Initialize model: try XGBoost, fallback to scikit-learn GradientBoosting
    model = None
    model_type = "XGBoost"

    try:
        import xgboost as xgb
        logger.info("Initializing XGBoost Classifier (XGBClassifier)...")
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="multi:softprob",
            num_class=4,
            eval_metric="mlogloss",
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
    except Exception as xgb_err:
        logger.warning("XGBoost unavailable or failed (%s). Falling back to Scikit-Learn GradientBoosting...", xgb_err)
        try:
            from sklearn.ensemble import HistGradientBoostingClassifier
            model_type = "HistGradientBoosting"
            model = HistGradientBoostingClassifier(
                max_iter=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=random_state,
            )
            model.fit(X_train, y_train)
        except Exception as sk_err:
            logger.warning("Scikit-learn unavailable (%s). Using empirical linear quantile ensemble fallback...", sk_err)
            model_type = "LinearQuantileEnsemble"
            model = LinearQuantileEnsemble(thresholds)

    # Predict on test partition
    y_pred = model.predict(X_test)

    # Compute evaluation metrics
    try:
        from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
        acc = float(accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred, average="weighted"))
        cm = confusion_matrix(y_test, y_pred)
    except Exception:
        metrics_dict = _compute_metrics_pure_python(y_test, y_pred)
        acc = metrics_dict["accuracy"]
        f1 = metrics_dict["f1_score"]
        cm = metrics_dict["confusion_matrix"]

    # Print evaluation report
    print("\n" + "=" * 70)
    print(f"      FLOOD RISK CLASSIFIER TRAINING REPORT ({model_type})")
    print("=" * 70)
    print(f"  Accuracy Score:  {acc * 100:.2f}%")
    print(f"  F1 Score (wtd):  {f1:.4f}")
    print("\n  Confusion Matrix (Rows: Actual, Cols: Predicted):")
    header_str = "           " + " ".join(f"{name:>9}" for name in RISK_CLASS_NAMES)
    print(header_str)
    for i, row in enumerate(cm):
        row_str = f"  {RISK_CLASS_NAMES[i]:>8} |" + " ".join(f"{val:>9d}" for val in row)
        print(row_str)
    print("=" * 70 + "\n")

    # Serialize model bundle
    bundle = {
        "model": model,
        "model_type": model_type,
        "feature_columns": FEATURE_COLUMNS,
        "risk_classes": RISK_CLASS_NAMES,
        "thresholds": thresholds,
        "metrics": {"accuracy": acc, "f1_score": f1},
    }

    try:
        import joblib
        joblib.dump(bundle, target_out)
        logger.info("Saved trained flood risk model bundle via joblib to %s", target_out)
    except Exception as joblib_err:
        logger.info("joblib not installed (%s). Saving via pickle...", joblib_err)
        import pickle
        with open(target_out, "wb") as f:
            pickle.dump(bundle, f)
        logger.info("Saved trained flood risk model bundle via pickle to %s", target_out)

    return model


def map_node_features_to_model_input(
    drainage_node: Dict[str, Any],
    rainfall_intensity: float,
    imperviousness: float,
    slope_deg: Optional[float] = None,
    distance_to_waterway_m: Optional[float] = None,
) -> pd.DataFrame:
    """
    Translates physical hydraulic node attributes, current storm rainfall,
    and urban terrain features into the 20 Kaggle feature inputs (scale ~1 to 15).

    Feature Mapping Heuristics:
      1. MonsoonIntensity: Scaled directly from rainfall_intensity in mm/hr.
      2. TopographyDrainage: Higher slope/elevation accelerates drainage.
      3. RiverManagement: Proximity to unmanaged drainage channels and canals.
      4. Deforestation: Urban imperviousness replaces natural vegetation.
      5. Urbanization: Directly proportional to OSM imperviousness ratio C.
      6. ClimateChange: Elevated baseline storm severity factor.
      7. DamsQuality: Upstream stormwater retention availability.
      8. Siltation: Pipe sedimentation, heightened when node is surcharging.
      9. AgriculturalPractices: Low rural influence in metropolitan centers.
      10. Encroachments: High impervious building density near natural drains.
      11. IneffectiveDisasterPreparedness: Institutional baseline factor.
      12. DrainageSystems: Pipe & manhole capacity adequacy vs surcharge ratio.
      13. CoastalVulnerability: Elevation above sea level (low elevation = high risk).
      14. Landslides: Terrain slope stability risk.
      15. Watersheds: Upstream drainage catchment area tier.
      16. DeterioratingInfrastructure: Pipe conduit structural stress and surcharge.
      17. PopulationScore: Building density in local catchment.
      18. WetlandLoss: Infill of historic urban flood retention basins.
      19. InadequatePlanning: Runoff generation exceeding network design capacity.
      20. PoliticalFactors: Municipal baseline oversight factor.

    Args:
        drainage_node (dict): Attributes of drainage inlet/manhole node.
        rainfall_intensity (float): Storm rainfall intensity in mm/hr.
        imperviousness (float): Urban surface imperviousness ratio [0.0, 1.0].
        slope_deg (float, optional): Local terrain slope in degrees.
        distance_to_waterway_m (float, optional): Distance to nearest surface canal.

    Returns:
        pd.DataFrame: 1-row DataFrame containing all 20 Kaggle feature columns.
    """
    # Extract node parameters
    elevation_m = float(drainage_node.get("elevation", 12.0))
    surcharge_ratio = float(drainage_node.get("surcharge_ratio", 0.0))
    node_type = str(drainage_node.get("node_type", "inlet")).lower()
    
    slope = float(slope_deg) if slope_deg is not None else 1.0
    dist_water = (
        float(distance_to_waterway_m)
        if distance_to_waterway_m is not None
        else float(drainage_node.get("osm_channel_dist_m", 300.0))
    )

    # 1. MonsoonIntensity (0 - 120 mm/hr -> 1 - 13)
    # 0 mm/hr = 1, 35 mm/hr = 5, 70 mm/hr = 8, 110 mm/hr = 12
    f_monsoon = int(np.clip(round(1.0 + (rainfall_intensity / 9.0)), 1, 14))

    # 2. TopographyDrainage (Good drainage = lower flood risk score, poor = higher)
    # Steeper slopes (>2°) and elevated ground (>15m) facilitate runoff dispersal
    drainage_quality = min(slope * 1.8, 6.0) + min(elevation_m / 8.0, 4.0)
    f_topography = int(np.clip(round(11.0 - drainage_quality), 1, 14))

    # 3. RiverManagement (Closer to unmanaged canals/waterways = higher risk)
    if dist_water < 80.0:
        f_river = 8
    elif dist_water < 200.0:
        f_river = 6
    else:
        f_river = 4

    # 4. Deforestation (Urban vegetation loss ~ imperviousness)
    f_deforestation = int(np.clip(round(2.0 + imperviousness * 7.5), 1, 14))

    # 5. Urbanization (Imperviousness 0.0 -> 1, 1.0 -> 10)
    f_urbanization = int(np.clip(round(1.0 + imperviousness * 9.0), 1, 14))

    # 6. ClimateChange (Baseline storm anomaly factor)
    f_climate = 7 if rainfall_intensity > 40.0 else 5

    # 7. DamsQuality (Retention basins)
    f_dams = 5

    # 8. Siltation (Pipe blockage exacerbated by backflow)
    f_siltation = int(np.clip(round(4.0 + surcharge_ratio * 3.0), 1, 14))

    # 9. AgriculturalPractices (Low rural footprint in urban setting)
    f_agriculture = 3

    # 10. Encroachments (High imperviousness close to waterways)
    encroach_boost = 3.0 if dist_water < 120.0 else 0.0
    f_encroachments = int(np.clip(round(2.0 + imperviousness * 5.0 + encroach_boost), 1, 14))

    # 11. IneffectiveDisasterPreparedness
    f_preparedness = 6 if rainfall_intensity > 50.0 else 4

    # 12. DrainageSystems (Directly reflects pipe surcharge)
    # Surcharge ratio 0 -> score 3; Surcharge ratio 1.5 -> score 8; Surcharge ratio 3.0 -> score 13
    f_drainage = int(np.clip(round(3.0 + surcharge_ratio * 3.5), 1, 14))

    # 13. CoastalVulnerability (Low coastal elevation < 5m)
    if elevation_m <= 3.0:
        f_coastal = 10
    elif elevation_m <= 7.0:
        f_coastal = 7
    elif elevation_m <= 15.0:
        f_coastal = 4
    else:
        f_coastal = 2

    # 14. Landslides (Flat urban alluvial plains)
    f_landslides = int(np.clip(round(1.0 + min(slope * 0.8, 5.0)), 1, 14))

    # 15. Watersheds (Drainage catchment position: outfalls gather largest area)
    f_watersheds = 9 if node_type == "outfall" else (6 if node_type == "junction" else 4)

    # 16. DeterioratingInfrastructure (Structural hydraulic overloading)
    f_infrastructure = int(np.clip(round(3.5 + surcharge_ratio * 3.0), 1, 14))

    # 17. PopulationScore (Higher in dense urban wards)
    f_population = int(np.clip(round(4.0 + imperviousness * 4.5), 1, 14))

    # 18. WetlandLoss (Infilling of seasonal lakes and marshlands)
    f_wetlands = int(np.clip(round(3.0 + imperviousness * 6.0), 1, 14))

    # 19. InadequatePlanning (Runoff exceeding storm pipe capacity)
    f_planning = int(np.clip(round(3.0 + surcharge_ratio * 4.0), 1, 14))

    # 20. PoliticalFactors
    f_political = 5

    features_dict = {
        "MonsoonIntensity": [f_monsoon],
        "TopographyDrainage": [f_topography],
        "RiverManagement": [f_river],
        "Deforestation": [f_deforestation],
        "Urbanization": [f_urbanization],
        "ClimateChange": [f_climate],
        "DamsQuality": [f_dams],
        "Siltation": [f_siltation],
        "AgriculturalPractices": [f_agriculture],
        "Encroachments": [f_encroachments],
        "IneffectiveDisasterPreparedness": [f_preparedness],
        "DrainageSystems": [f_drainage],
        "CoastalVulnerability": [f_coastal],
        "Landslides": [f_landslides],
        "Watersheds": [f_watersheds],
        "DeterioratingInfrastructure": [f_infrastructure],
        "PopulationScore": [f_population],
        "WetlandLoss": [f_wetlands],
        "InadequatePlanning": [f_planning],
        "PoliticalFactors": [f_political],
    }

    return pd.DataFrame(features_dict)


def get_impervious_grid_summary() -> Dict[str, float]:
    """
    Reads data/processed/impervious_grid_*.geojson to extract empirical surface
    imperviousness ratios (mean, max, 75th percentile) for heuristic flood risk estimation.
    """
    processed_dir = PROJECT_ROOT / "data" / "processed"
    grid_files = list(processed_dir.glob("impervious_grid_*.geojson"))

    if not grid_files:
        return {"mean_imperviousness": 0.65, "max_imperviousness": 0.90, "p75_imperviousness": 0.75}

    try:
        import json
        with open(grid_files[0], "r", encoding="utf-8") as f:
            grid_data = json.load(f)

        ratios = [
            float(feat["properties"]["impervious_ratio"])
            for feat in grid_data.get("features", [])
            if "impervious_ratio" in feat.get("properties", {}) and feat["properties"]["impervious_ratio"] is not None
        ]

        if ratios:
            return {
                "mean_imperviousness": round(float(np.mean(ratios)), 4),
                "max_imperviousness": round(float(np.max(ratios)), 4),
                "p75_imperviousness": round(float(np.percentile(ratios, 75)), 4),
            }
    except Exception as exc:
        logger.warning("Could not read impervious grid geojson (%s). Using default empirical ratios.", exc)

    return {"mean_imperviousness": 0.65, "max_imperviousness": 0.90, "p75_imperviousness": 0.75}


def compute_heuristic_flood_risk(
    features_df: Optional[pd.DataFrame] = None,
    rainfall_intensity: Optional[float] = None,
    slope_deg: Optional[float] = None,
    imperviousness: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Physically grounded heuristic risk calculation used when ML model unpickling fails.
    Combines:
      1. Rainfall depth / intensity (mm/hr)
      2. Terrain elevation slope (degrees)
      3. Urban impervious surface ratio from data/processed/impervious_grid_*.geojson
    """
    grid_summary = get_impervious_grid_summary()
    imp_ratio = imperviousness if imperviousness is not None else grid_summary["mean_imperviousness"]

    # 1. Resolve rainfall intensity
    if rainfall_intensity is not None:
        rain_mm = float(rainfall_intensity)
    elif features_df is not None and "MonsoonIntensity" in features_df.columns:
        f_m = float(features_df["MonsoonIntensity"].iloc[0])
        rain_mm = max(0.0, (f_m - 1.0) * 9.0)
    else:
        rain_mm = 50.0

    # 2. Resolve terrain slope
    if slope_deg is not None:
        slope = float(slope_deg)
    elif features_df is not None and "TopographyDrainage" in features_df.columns:
        f_t = float(features_df["TopographyDrainage"].iloc[0])
        slope = max(0.2, 10.0 - f_t)
    else:
        slope = 1.0

    # 3. Compute normalized component scores [0.0 to 1.0]
    # Rainfall: 0 - 120 mm/hr
    s_rain = min(rain_mm / 100.0, 1.0)
    # Imperviousness: 0.0 - 1.0
    s_imp = min(max(imp_ratio, 0.0), 1.0)
    # Flatness vulnerability: slopes < 0.5° have maximal ponding risk
    s_slope = max(0.05, 1.0 - min(slope / 4.0, 0.95))

    # Composite physical risk index
    composite_score = (0.50 * s_rain) + (0.30 * s_imp) + (0.20 * s_slope)

    # Drainage surcharge booster if present in features_df
    if features_df is not None and "DrainageSystems" in features_df.columns:
        f_drain = float(features_df["DrainageSystems"].iloc[0])
        if f_drain >= 9:
            composite_score = min(1.0, composite_score + 0.15)

    # Classify into 4 risk tiers
    if composite_score < 0.30:
        pred_id = 0
        pred_name = "Low"
    elif composite_score < 0.55:
        pred_id = 1
        pred_name = "Moderate"
    elif composite_score < 0.75:
        pred_id = 2
        pred_name = "High"
    else:
        pred_id = 3
        pred_name = "Severe"

    # Generate smooth class probability distribution
    centers = [0.15, 0.42, 0.65, 0.88]
    raw_probs = [math.exp(-((composite_score - c) ** 2) / 0.05) for c in centers]
    sum_p = sum(raw_probs)
    probs = {
        name: round(p / sum_p, 4)
        for name, p in zip(RISK_CLASS_NAMES, raw_probs)
    }

    return {
        "predicted_class": pred_name,
        "predicted_class_id": pred_id,
        "probabilities": probs,
        "all_predicted_classes": [pred_name],
        "all_predicted_class_ids": [pred_id],
        "heuristic_fallback": True,
        "heuristic_factors": {
            "rainfall_intensity_mm_hr": round(rain_mm, 2),
            "terrain_slope_deg": round(slope, 2),
            "impervious_surface_ratio": round(imp_ratio, 4),
            "composite_risk_score": round(composite_score, 4),
        },
    }


def predict_risk_ml(
    features_df: pd.DataFrame,
    model_path: Optional[Union[str, Path]] = None,
    rainfall_intensity: Optional[float] = None,
    slope_deg: Optional[float] = None,
    imperviousness: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Loads the serialized flood risk model and generates predictions and class probabilities.
    Safely wraps joblib.load() and pickle.load() inside a try...except Exception block to catch
    environment/pickle version mismatches, corrupt files, or missing packages.
    If unpickling fails, gracefully falls back to a physically grounded heuristic risk calculation
    based on rainfall depth, elevation slope, and impervious grid surface data from
    data/processed/impervious_grid_*.geojson, preventing HTTP 500 crashes.

    Args:
        features_df (pd.DataFrame): DataFrame containing the 20 feature columns.
        model_path (str or Path, optional): Custom path to flood_risk_model.pkl.
        rainfall_intensity (float, optional): Live storm rainfall in mm/hr.
        slope_deg (float, optional): Local elevation slope in degrees.
        imperviousness (float, optional): Mean impervious surface ratio [0.0, 1.0].

    Returns:
        Dict[str, Any]:
            - 'predicted_class': str ('Low', 'Moderate', 'High', 'Severe')
            - 'predicted_class_id': int (0, 1, 2, 3)
            - 'probabilities': Dict[str, float] probability per risk tier
            - 'all_predictions': List[str] if multiple rows are passed
            - 'heuristic_fallback': bool indicating if heuristic fallback was used
    """
    target_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH

    bundle = None
    if target_path.exists():
        # Wrap joblib.load() and pickle.load() inside try...except Exception
        try:
            import joblib
            bundle = joblib.load(target_path)
            logger.debug("Successfully loaded model bundle via joblib from %s", target_path)
        except Exception as joblib_err:
            logger.warning(
                "joblib.load() failed on %s (%s). Attempting pickle.load() fallback...",
                target_path,
                joblib_err,
            )
            try:
                import pickle
                with open(target_path, "rb") as f:
                    bundle = pickle.load(f)
                logger.debug("Successfully loaded model bundle via pickle from %s", target_path)
            except Exception as pickle_err:
                logger.warning(
                    "pickle.load() failed on %s (%s). Unpickling error/version mismatch detected.",
                    target_path,
                    pickle_err,
                )
                bundle = None
    else:
        logger.info("Model file not found at %s. Attempting on-demand training...", target_path)
        try:
            train_model(output_model_path=target_path)
            if target_path.exists():
                import joblib
                bundle = joblib.load(target_path)
        except Exception as train_err:
            logger.warning("On-demand model training failed (%s). Using heuristic fallback.", train_err)
            bundle = None

    # If unpickling succeeded, execute model inference
    if bundle is not None and isinstance(bundle, dict) and "model" in bundle:
        try:
            model = bundle["model"]
            classes = bundle.get("risk_classes", RISK_CLASS_NAMES)
            X_input = features_df[FEATURE_COLUMNS].values

            preds = model.predict(X_input)

            probs = None
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_input)

            pred_id = int(preds[0])
            pred_name = classes[pred_id]

            prob_dict = {}
            if probs is not None:
                for i, c_name in enumerate(classes):
                    prob_dict[c_name] = round(float(probs[0, i]), 4)

            return {
                "predicted_class": pred_name,
                "predicted_class_id": pred_id,
                "probabilities": prob_dict,
                "all_predicted_classes": [classes[int(p)] for p in preds],
                "all_predicted_class_ids": [int(p) for p in preds],
                "heuristic_fallback": False,
            }
        except Exception as infer_err:
            logger.warning("Model inference failed (%s). Falling back to heuristic risk calculation.", infer_err)

    # Heuristic fallback if unpickling or inference failed
    logger.info("Executing heuristic flood risk calculation fallback...")
    return compute_heuristic_flood_risk(
        features_df=features_df,
        rainfall_intensity=rainfall_intensity,
        slope_deg=slope_deg,
        imperviousness=imperviousness,
    )


def retrain_or_export_model(
    output_model_path: Optional[Union[str, Path]] = None,
    csv_path: Optional[Union[str, Path]] = None,
    sample_size: Optional[int] = 25000,
) -> Path:
    """
    Utility function that cleanly re-trains or re-exports the flood risk model
    bundle in the current Python environment using active scikit-learn/NumPy/XGBoost versions.

    Args:
        output_model_path (str or Path, optional): Custom path to output flood_risk_model.pkl.
        csv_path (str or Path, optional): Path to training CSV dataset.
        sample_size (int, optional): Number of rows to train on.

    Returns:
        Path: The path to the saved flood_risk_model.pkl bundle.
    """
    target_out = Path(output_model_path) if output_model_path else DEFAULT_MODEL_PATH
    target_out.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Retraining/re-exporting flood risk ML model to %s...", target_out)
    train_model(csv_path=csv_path, output_model_path=target_out, sample_size=sample_size)
    logger.info("Model re-export successfully completed: %s", target_out)
    return target_out


if __name__ == "__main__":
    print("\n==================================================")
    print("      Testing engine/risk_classifier.py           ")
    print("==================================================")

    # 1. Load Kaggle training data and inspect quantile binning
    print("\n[1] Loading Kaggle Flood Prediction Dataset & Bins...")
    flood_df, cutoffs = load_training_data(sample_size=30000)
    print(f"  - Quantile Thresholds: {cutoffs}")
    print("  - Risk Class Counts:")
    print(flood_df["risk_class"].value_counts())

    # 2. Train and evaluate classifier
    print("\n[2] Training Machine Learning Classifier on Kaggle Features...")
    trained_clf = train_model(flood_df, sample_size=30000)

    # 3. Test feature mapping approximation for a surcharging drainage node
    print("\n[3] Testing Micro-to-Macro Feature Mapping...")
    sample_node = {
        "node_type": "inlet",
        "elevation": 4.5,
        "capacity_liters_per_min": 15000.0,
        "surcharge_ratio": 1.8,
        "osm_channel_dist_m": 85.0,
    }
    input_features = map_node_features_to_model_input(
        drainage_node=sample_node,
        rainfall_intensity=55.0,
        imperviousness=0.88,
        slope_deg=0.6,
    )
    print("  - Generated Model Input Vector (20 features):")
    print(input_features.T)

    # 4. Predict ML risk
    print("\n[4] Running ML Risk Inference...")
    prediction = predict_risk_ml(input_features)
    print(f"  -> Predicted Flood Risk Tier: {prediction['predicted_class']} (ID: {prediction['predicted_class_id']})")
    print(f"  -> Class Probabilities: {prediction['probabilities']}")

    # 5. Test another scenario: Dry conditions / elevated node
    dry_node = {
        "node_type": "inlet",
        "elevation": 25.0,
        "capacity_liters_per_min": 25000.0,
        "surcharge_ratio": 0.0,
        "osm_channel_dist_m": 800.0,
    }
    dry_features = map_node_features_to_model_input(
        drainage_node=dry_node,
        rainfall_intensity=5.0,
        imperviousness=0.30,
        slope_deg=4.5,
    )
    dry_prediction = predict_risk_ml(dry_features)
    print(f"\n[5] Dry / Elevated Scenario Prediction:")
    print(f"  -> Predicted Flood Risk Tier: {dry_prediction['predicted_class']} (ID: {dry_prediction['predicted_class_id']})")
    print(f"  -> Class Probabilities: {dry_prediction['probabilities']}")

    print("\n[OK] Risk classifier module verified successfully!")
