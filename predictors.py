from __future__ import annotations

import glob
import io
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SUMMARY_ROOT = ROOT / "vendor" / "model_summaries"
MODEL_FILE_ROOT = ROOT / "vendor" / "model_files"

MEASURED_COLUMN = "Measured DT41J  [C]"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    prediction_column: str
    benchmark_column: str
    required_columns: tuple[str, ...]


MODEL_SPECS: dict[str, ModelSpec] = {
    "E900": ModelSpec(
        key="E900",
        label="ASTM E900",
        prediction_column="E900 predicted TTS (degC)",
        benchmark_column="E900 Preds",
        required_columns=(
            "Product Form",
            "temperature_C",
            "wt_percent_Cu",
            "wt_percent_Ni",
            "wt_percent_Mn",
            "wt_percent_P",
            "flux_n_cm2_sec",
            "fluence_n_cm2",
        ),
    ),
    "EONY": ModelSpec(
        key="EONY",
        label="EONY",
        prediction_column="EONY predicted TTS (degC)",
        benchmark_column="EONY Preds",
        required_columns=(
            "Product Form",
            "temperature_C",
            "wt_percent_Cu",
            "wt_percent_Ni",
            "wt_percent_Mn",
            "wt_percent_P",
            "flux_n_cm2_sec",
            "fluence_n_cm2",
        ),
    ),
    "GBR": ModelSpec(
        key="GBR",
        label="Gradient Boosting ML",
        prediction_column="GBR predicted TTS (degC)",
        benchmark_column="GBR Preds",
        required_columns=(
            "Product Form",
            "Reactor Type",
            "temperature_C",
            "wt_percent_Cu",
            "wt_percent_Ni",
            "wt_percent_Mn",
            "wt_percent_P",
            "flux_n_cm2_sec",
            "fluence_n_cm2",
        ),
    ),
    "GKRR": ModelSpec(
        key="GKRR",
        label="Gaussian Kernel Ridge ML",
        prediction_column="GKRR predicted TTS (degC)",
        benchmark_column="GKRR Preds",
        required_columns=(
            "temperature_C",
            "fluence_n_cm2",
            "effective_fluence",
            "wt_percent_Cu",
            "wt_percent_Ni",
            "wt_percent_Mn",
            "wt_percent_P",
            "wt_percent_Si",
            "wt_percent_C",
        ),
    ),
}


COLUMN_ALIASES = {
    "model_name": "model",
    "models": "model",
    "product_form": "Product Form",
    "product form": "Product Form",
    "pf": "Product Form",
    "reactor_type": "Reactor Type",
    "reactor type": "Reactor Type",
    "reactor": "Reactor Type",
    "temperature": "temperature_C",
    "temperature_c": "temperature_C",
    "temp_c": "temperature_C",
    "t": "temperature_C",
    "cu": "wt_percent_Cu",
    "wt_cu": "wt_percent_Cu",
    "wt_percent_cu": "wt_percent_Cu",
    "ni": "wt_percent_Ni",
    "wt_ni": "wt_percent_Ni",
    "wt_percent_ni": "wt_percent_Ni",
    "mn": "wt_percent_Mn",
    "wt_mn": "wt_percent_Mn",
    "wt_percent_mn": "wt_percent_Mn",
    "p": "wt_percent_P",
    "wt_p": "wt_percent_P",
    "wt_percent_p": "wt_percent_P",
    "si": "wt_percent_Si",
    "wt_si": "wt_percent_Si",
    "wt_percent_si": "wt_percent_Si",
    "c": "wt_percent_C",
    "wt_c": "wt_percent_C",
    "wt_percent_c": "wt_percent_C",
    "flux": "flux_n_cm2_sec",
    "flux_n_cm2_s": "flux_n_cm2_sec",
    "flux_n_cm2_sec": "flux_n_cm2_sec",
    "fluence": "fluence_n_cm2",
    "fluence_n_cm2": "fluence_n_cm2",
    "effective_fluence": "effective_fluence",
    "effective fluence": "effective_fluence",
    "log_effective_fluence": "log(effective_fluence)",
    "at_cu": "at_percent_Cu",
    "at_percent_cu": "at_percent_Cu",
    "at_ni": "at_percent_Ni",
    "at_percent_ni": "at_percent_Ni",
    "at_mn": "at_percent_Mn",
    "at_percent_mn": "at_percent_Mn",
    "at_p": "at_percent_P",
    "at_percent_p": "at_percent_P",
    "at_si": "at_percent_Si",
    "at_percent_si": "at_percent_Si",
    "at_c": "at_percent_C",
    "at_percent_c": "at_percent_C",
    "alloy_id": "alloy",
    "material": "alloy",
}


NUMERIC_COLUMNS = {
    "temperature_C",
    "wt_percent_Cu",
    "wt_percent_Ni",
    "wt_percent_Mn",
    "wt_percent_P",
    "wt_percent_Si",
    "wt_percent_C",
    "effective_fluence",
    "at_percent_Cu",
    "at_percent_Ni",
    "at_percent_Mn",
    "at_percent_P",
    "at_percent_Si",
    "at_percent_C",
    "flux_n_cm2_sec",
    "fluence_n_cm2",
}

GKRR_ATOMIC_ELEMENTS = ("Cu", "Ni", "Mn", "P", "Si", "C")
ATOMIC_WEIGHTS = {
    "Fe": 55.845,
    "Cu": 63.546,
    "Ni": 58.6934,
    "Mn": 54.938044,
    "P": 30.973762,
    "Si": 28.0855,
    "C": 12.011,
}


def available_models() -> list[dict[str, Any]]:
    models = []
    for key, spec in MODEL_SPECS.items():
        stats = benchmark_stats(key)
        models.append(
            {
                "key": key,
                "label": spec.label,
                "prediction_column": spec.prediction_column,
                "required_columns": list(spec.required_columns),
                "benchmark": stats,
            }
        )
    return models


def read_input_file(filename: str, payload: bytes) -> pd.DataFrame:
    suffix = Path(filename).suffix.lower()
    stream = io.BytesIO(payload)

    if suffix in {".csv", ".txt"}:
        return pd.read_csv(stream)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(stream)
    if suffix == ".json":
        raw = json.loads(payload.decode("utf-8"))
        if isinstance(raw, list):
            return pd.DataFrame(raw)
        if isinstance(raw, dict) and isinstance(raw.get("rows"), list):
            return pd.DataFrame(raw["rows"])
        if isinstance(raw, dict):
            try:
                return pd.DataFrame(raw)
            except ValueError:
                return pd.DataFrame([raw])
        raise ValueError("JSON input must be an object, a list of objects, or an object with a rows list.")

    raise ValueError("Upload a CSV, XLSX, or JSON file.")


def normalize_input(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    normalized = df.copy()
    normalized.columns = [str(column).strip() for column in normalized.columns]

    rename_map = {}
    for column in normalized.columns:
        key = column.strip().lower()
        if key in COLUMN_ALIASES:
            rename_map[column] = COLUMN_ALIASES[key]
    normalized = normalized.rename(columns=rename_map)

    if "Product Form" in normalized:
        normalized["Product Form"] = normalized["Product Form"].astype(str).str.strip().str.upper()

    if "Reactor Type" in normalized:
        normalized["Reactor Type"] = normalized["Reactor Type"].astype(str).str.strip().str.upper()

    if "model" in normalized:
        normalized["model"] = normalized["model"].astype(str).str.strip().str.upper()

    for column in NUMERIC_COLUMNS.intersection(normalized.columns):
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    if "fluence_n_cm2" in normalized and "log(fluence_n_cm2)" not in normalized:
        normalized["log(fluence_n_cm2)"] = np.log10(normalized["fluence_n_cm2"].astype(float))
    if "flux_n_cm2_sec" in normalized and "log(flux_n_cm2_sec)" not in normalized:
        normalized["log(flux_n_cm2_sec)"] = np.log10(normalized["flux_n_cm2_sec"].astype(float))

    blank_product_form = (
        "Product Form" in normalized
        and normalized["Product Form"].replace({"": np.nan, "NAN": np.nan}).isna().any()
    )
    if blank_product_form:
        warnings.append("Blank Product Form values were treated as P, matching the original model fallback.")
        normalized["Product Form"] = normalized["Product Form"].replace({"": "P", "NAN": "P"}).fillna("P")

    return normalized, warnings


def selected_models_from_request(df: pd.DataFrame, selected_models: list[str] | None) -> list[str]:
    if selected_models:
        requested = selected_models
    elif "model" in df:
        requested = []
        for raw_value in df["model"].dropna().unique():
            requested.extend(str(raw_value).replace(";", ",").split(","))
    else:
        requested = list(MODEL_SPECS)

    models = []
    for model in requested:
        key = str(model).strip().upper()
        if key and key not in models:
            if key not in MODEL_SPECS:
                raise ValueError(f"Unsupported model '{model}'. Supported models: {', '.join(MODEL_SPECS)}.")
            models.append(key)

    if not models:
        raise ValueError("Choose at least one model.")
    return models


def validate_for_models(df: pd.DataFrame, models: list[str]) -> None:
    missing = []
    for model in models:
        for column in MODEL_SPECS[model].required_columns:
            if column not in df:
                missing.append(f"{model}: {column}")

    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    numeric_missing = []
    for column in NUMERIC_COLUMNS.intersection(df.columns):
        if df[column].isna().any():
            numeric_missing.append(column)
    if numeric_missing:
        raise ValueError("Some numeric values could not be read in: " + ", ".join(sorted(numeric_missing)))

    if "fluence_n_cm2" in df and (df["fluence_n_cm2"] <= 0).any():
        raise ValueError("fluence_n_cm2 must be greater than zero.")
    if "flux_n_cm2_sec" in df and (df["flux_n_cm2_sec"] <= 0).any():
        raise ValueError("flux_n_cm2_sec must be greater than zero.")


def predict(df: pd.DataFrame, selected_models: list[str] | None = None) -> dict[str, Any]:
    normalized, warnings = normalize_input(df)
    models = selected_models_from_request(normalized, selected_models)
    if "GBR" in models:
        normalized = ensure_gbr_defaults(normalized, warnings)
    if "GKRR" in models:
        normalized = ensure_gkrr_features(normalized, warnings)
    validate_for_models(normalized, models)

    records: list[dict[str, Any]] = []
    for input_index, row in normalized.iterrows():
        for model in models:
            if "model" in normalized and not selected_models:
                row_models = str(row.get("model", "")).upper().replace(";", ",").split(",")
                row_models = [item.strip() for item in row_models if item.strip()]
                if row_models and model not in row_models:
                    continue

            pred = predict_one(model, row)
            stats = benchmark_stats(model)
            sigma = stats.get("residual_std_degC")
            lower_95 = pred - 1.96 * sigma if sigma is not None else None
            upper_95 = pred + 1.96 * sigma if sigma is not None else None

            records.append(
                {
                    "input_index": int(input_index),
                    "alloy": _clean_optional(row.get("alloy")),
                    "model": model,
                    "model_label": MODEL_SPECS[model].label,
                    "predicted_tts_degC": _round(pred),
                    "uncertainty_1sigma_degC": _round(sigma),
                    "lower_95_degC": _round(lower_95),
                    "upper_95_degC": _round(upper_95),
                    "uncertainty_source": "five-fold benchmark residual standard deviation",
                    "temperature_C": _round(row["temperature_C"]),
                    "fluence_n_cm2": _sci(row["fluence_n_cm2"]),
                    "flux_n_cm2_sec": _sci(row["flux_n_cm2_sec"]),
                    "Product Form": row["Product Form"],
                    "Reactor Type": _clean_optional(row.get("Reactor Type")),
                    "wt_percent_Cu": _round(row["wt_percent_Cu"], 6),
                    "wt_percent_Ni": _round(row["wt_percent_Ni"], 6),
                    "wt_percent_Mn": _round(row["wt_percent_Mn"], 6),
                    "wt_percent_P": _round(row["wt_percent_P"], 6),
                    "wt_percent_Si": _round(row.get("wt_percent_Si"), 6),
                    "wt_percent_C": _round(row.get("wt_percent_C"), 6),
                }
            )

    return {
        "models": models,
        "input_rows": int(normalized.shape[0]),
        "result_rows": len(records),
        "warnings": warnings,
        "benchmark": {model: benchmark_stats(model) for model in models},
        "results": records,
    }


def predict_one(model: str, row: pd.Series) -> float:
    if model == "E900":
        return e900_tts(row)
    if model == "EONY":
        return eony_tts(row)
    if model == "GBR":
        return gbr_tts(row)
    if model == "GKRR":
        return gkrr_tts(row)
    raise ValueError(f"Unsupported model '{model}'.")


def e900_tts(row: pd.Series) -> float:
    product_form = str(row["Product Form"]).upper()
    temp = float(row["temperature_C"])
    fluence = 100.0 * 100.0 * float(row["fluence_n_cm2"])
    p = float(row["wt_percent_P"])
    ni = float(row["wt_percent_Ni"])
    mn = float(row["wt_percent_Mn"])
    cu = float(row["wt_percent_Cu"])

    if product_form in {"P", "SRM"}:
        a = 1.080
    elif product_form == "F":
        a = 1.011
    elif product_form == "W":
        a = 0.919
    else:
        a = 1.080

    tts_one = (
        a
        * (5 / 9)
        * (1.8943e-12)
        * (fluence**0.5695)
        * (((1.8 * temp + 32) / 550) ** -5.47)
        * ((0.09 + (p / 0.012)) ** 0.216)
        * ((1.66 + ((ni**8.54) / 0.63)) ** 0.39)
        * ((mn / 1.36) ** 0.3)
    )

    if product_form in {"P", "SRM"}:
        b = 0.819
    elif product_form == "F":
        b = 0.738
    elif product_form == "W":
        b = 0.968
    else:
        b = 0.819

    m = (
        b
        * max(min(113.87 * (math.log(fluence) - math.log(4.5e20)), 612.6), 0)
        * (((1.8 * temp + 32) / 550) ** -5.45)
        * ((0.1 + (p / 0.012)) ** -0.098)
        * ((0.168 + ((ni**0.58) / 0.63)) ** 0.73)
    )
    tts_two = (5 / 9) * max(min(cu, 0.28) - 0.053, 0) * m
    return float(tts_one + tts_two)


def eony_tts(row: pd.Series) -> float:
    product_form = str(row["Product Form"]).upper()
    temp_f = 32 + (9 / 5) * float(row["temperature_C"])
    flux = float(row["flux_n_cm2_sec"])
    fluence = float(row["fluence_n_cm2"])
    p = float(row["wt_percent_P"])
    ni = float(row["wt_percent_Ni"])
    mn = float(row["wt_percent_Mn"])
    cu = float(row["wt_percent_Cu"])

    if product_form == "P":
        a = 1.561e-7
    elif product_form == "F":
        a = 1.140e-7
    elif product_form == "W":
        a = 1.417e-7
    else:
        a = 1.561e-7

    eff_fluence = fluence if flux >= 4.39e10 else fluence * ((4.39e10 / flux) ** 0.259)
    tts_one = a * (1 - 0.001718 * temp_f) * (1 + 6.13 * p * (mn**2.47)) * math.sqrt(eff_fluence)

    if product_form == "PCE":
        b = 135.2
    elif product_form == "P":
        b = 102.5
    elif product_form == "SRM":
        b = 128.2
    elif product_form == "F":
        b = 102.3
    elif product_form in {"W", "W80"}:
        b = 155.0
    else:
        b = 128.2

    max_cu_e = 0.243 if product_form == "W80" else 0.301
    cu_e = 0 if cu <= 0.072 else min(cu, max_cu_e)

    if cu <= 0.072:
        func_cu_e = 0
    elif p <= 0.008:
        func_cu_e = (cu_e - 0.072) ** 0.668
    else:
        func_cu_e = (cu_e - 0.072 + 1.359 * (p - 0.008)) ** 0.668

    gfunc_cu_e = 0.5 + 0.5 * math.tanh((math.log10(eff_fluence) + 1.139 * cu_e - 0.448 * ni - 18.120) / 0.629)
    tts_two = b * (1 + 3.77 * (ni**1.191)) * func_cu_e * gfunc_cu_e
    return float((tts_one + tts_two) * (5 / 9))


def gbr_tts(row: pd.Series) -> float:
    features = pd.DataFrame(
        [
            {
                "temperature_C": float(row["temperature_C"]),
                "wt_percent_Cu": float(row["wt_percent_Cu"]),
                "wt_percent_Ni": float(row["wt_percent_Ni"]),
                "wt_percent_Mn": float(row["wt_percent_Mn"]),
                "wt_percent_P": float(row["wt_percent_P"]),
                "fluence_n_cm2": float(row["fluence_n_cm2"]),
                "flux_n_cm2_sec": float(row["flux_n_cm2_sec"]),
                "Product Form": str(row["Product Form"]).upper(),
                "Reactor Type": str(row["Reactor Type"]).upper(),
            }
        ]
    )
    encoded = encode_gbr_features(features)
    scaler = load_joblib(MODEL_FILE_ROOT / "GBR" / "fullfit" / "StandardScaler.pkl")
    model = load_joblib(MODEL_FILE_ROOT / "GBR" / "fullfit" / "GradientBoostingRegressor.pkl")
    scaled = pd.DataFrame(scaler.transform(encoded), columns=encoded.columns)
    return float(model.predict(scaled)[0])


def gkrr_tts(row: pd.Series) -> float:
    features = pd.DataFrame(
        [
            {
                "temperature_C": float(row["temperature_C"]),
                "log(fluence_n_cm2)": float(row["log(fluence_n_cm2)"]),
                "log(effective_fluence)": float(row["log(effective_fluence)"]),
                "at_percent_Cu": float(row["at_percent_Cu"]),
                "at_percent_Ni": float(row["at_percent_Ni"]),
                "at_percent_Mn": float(row["at_percent_Mn"]),
                "at_percent_P": float(row["at_percent_P"]),
                "at_percent_Si": float(row["at_percent_Si"]),
                "at_percent_C": float(row["at_percent_C"]),
            }
        ]
    )
    scaler = load_joblib(MODEL_FILE_ROOT / "GKRR" / "fullfit" / "StandardScaler.pkl")
    model = load_joblib(MODEL_FILE_ROOT / "GKRR" / "fullfit" / "KernelRidge.pkl")
    scaled = pd.DataFrame(scaler.transform(features), columns=features.columns)
    return float(model.predict(scaled)[0])


def encode_gbr_features(df: pd.DataFrame) -> pd.DataFrame:
    feature_rows: list[dict[str, float]] = []
    for _, row in df.iterrows():
        product_form = str(row["Product Form"]).upper()
        reactor_type = str(row["Reactor Type"]).upper()
        pf_values = {
            "F": (1, 0, 0, 0, 0, 0),
            "HAZ": (0, 1, 0, 0, 0, 0),
            "P": (0, 0, 1, 0, 0, 0),
            "SRM": (0, 0, 0, 1, 0, 0),
            "W": (0, 0, 0, 0, 4, 0),
            "PCE": (0, 0, 0, 0, 0, 1),
        }.get(product_form, (0, 0, 1, 0, 0, 0))
        rt_values = (1, 0) if reactor_type == "BWR" else (0, 1)
        feature_rows.append(
            {
                "temperature_C": float(row["temperature_C"]),
                "wt_percent_Cu": float(row["wt_percent_Cu"]),
                "wt_percent_Ni": float(row["wt_percent_Ni"]),
                "wt_percent_Mn": float(row["wt_percent_Mn"]),
                "wt_percent_P": float(row["wt_percent_P"]),
                "fluence_n_cm2": float(row["fluence_n_cm2"]),
                "flux_n_cm2_sec": float(row["flux_n_cm2_sec"]),
                "Product Form_0": pf_values[0],
                "Product Form_1": pf_values[1],
                "Product Form_2": pf_values[2],
                "Product Form_3": pf_values[3],
                "Product Form_4": pf_values[4],
                "Product Form_5": pf_values[5],
                "Reactor Type_0": rt_values[0],
                "Reactor Type_1": rt_values[1],
            }
        )
    return pd.DataFrame(feature_rows)


def ensure_gbr_defaults(df: pd.DataFrame, warnings: list[str]) -> pd.DataFrame:
    normalized = df.copy()
    if "Reactor Type" not in normalized:
        normalized["Reactor Type"] = "PWR"
        warnings.append("Missing Reactor Type values were treated as PWR for the GBR model.")
    else:
        blank_reactor = normalized["Reactor Type"].replace({"": np.nan, "NAN": np.nan}).isna()
        if blank_reactor.any():
            normalized.loc[blank_reactor, "Reactor Type"] = "PWR"
            warnings.append("Blank Reactor Type values were treated as PWR for the GBR model.")
    return normalized


def ensure_gkrr_features(df: pd.DataFrame, warnings: list[str]) -> pd.DataFrame:
    normalized = df.copy()
    if "effective_fluence" not in normalized:
        normalized["effective_fluence"] = normalized["fluence_n_cm2"]
        warnings.append("GKRR effective_fluence was assumed equal to fluence_n_cm2 when not supplied.")
    if "log(effective_fluence)" not in normalized:
        normalized["log(effective_fluence)"] = np.log10(normalized["effective_fluence"].astype(float))

    missing_atomic = [f"at_percent_{element}" for element in GKRR_ATOMIC_ELEMENTS if f"at_percent_{element}" not in normalized]
    if missing_atomic:
        atomic = weight_to_atomic_percent(normalized)
        for column in missing_atomic:
            normalized[column] = atomic[column]
        warnings.append(
            "GKRR at_percent_* values were computed from wt_percent_* using Fe as the balance element."
        )
    return normalized


def weight_to_atomic_percent(df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, float]] = []
    for _, row in df.iterrows():
        weights = {element: float(row[f"wt_percent_{element}"]) for element in GKRR_ATOMIC_ELEMENTS}
        fe_weight = max(100.0 - sum(weights.values()), 0.0)
        moles = {"Fe": fe_weight / ATOMIC_WEIGHTS["Fe"]}
        for element, wt_percent in weights.items():
            moles[element] = wt_percent / ATOMIC_WEIGHTS[element]
        total_moles = sum(moles.values())
        records.append(
            {
                f"at_percent_{element}": (moles[element] / total_moles) * 100.0
                for element in GKRR_ATOMIC_ELEMENTS
            }
        )
    return pd.DataFrame(records, index=df.index)


@lru_cache(maxsize=None)
def load_joblib(path: Path) -> Any:
    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError("ML models require scikit-learn/joblib. Install requirements.txt and redeploy.") from exc
    return joblib.load(path)


@lru_cache(maxsize=None)
def benchmark_stats(model: str) -> dict[str, Any]:
    spec = MODEL_SPECS[model]
    fullfit_stats = SUMMARY_ROOT / model / "fullfit" / "residual_histogram_test_statistics.csv"
    if fullfit_stats.exists():
        stats = pd.read_csv(fullfit_stats, index_col=0).iloc[:, 0]
        residual_std = float(stats.get("std", np.nan))
        return {
            "n": int(stats.get("count", 0)),
            "residual_std_degC": _round(residual_std),
            "mae_degC": None,
            "rmse_degC": None,
        }

    paths = sorted(glob.glob(str(SUMMARY_ROOT / model / "5fold" / "*.csv")))
    if not paths:
        return {
            "n": 0,
            "residual_std_degC": None,
            "mae_degC": None,
            "rmse_degC": None,
        }

    frames = [pd.read_csv(path) for path in paths]
    df = pd.concat(frames, ignore_index=True)
    residuals = pd.to_numeric(df[spec.benchmark_column], errors="coerce") - pd.to_numeric(
        df[MEASURED_COLUMN], errors="coerce"
    )
    residuals = residuals.dropna()
    return {
        "n": int(residuals.shape[0]),
        "residual_std_degC": _round(float(residuals.std(ddof=1))),
        "mae_degC": _round(float(residuals.abs().mean())),
        "rmse_degC": _round(float(np.sqrt(np.mean(np.square(residuals))))),
    }


def dataframe_to_csv(records: list[dict[str, Any]]) -> str:
    return pd.DataFrame(records).to_csv(index=False)


def _round(value: Any, ndigits: int = 3) -> float | None:
    if value is None:
        return None
    if pd.isna(value):
        return None
    return round(float(value), ndigits)


def _sci(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return f"{float(value):.6e}"


def _clean_optional(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None
