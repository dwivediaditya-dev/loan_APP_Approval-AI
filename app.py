"""LoanLens - an educational loan-approval decision-support demo (Streamlit).

Dataset: Kaggle "muhammadmusharraf444/loan-approval-dataset" (loaded with kagglehub).
If the dataset is unavailable or incompatible, seeded synthetic demonstration data is used.
"""
from __future__ import annotations

import hashlib
import logging
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import streamlit as st
import kagglehub
from kagglehub import KaggleDatasetAdapter
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

logger = logging.getLogger("loanlens")

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
APP_NAME = "LoanLens"
DATASET_HANDLE = "muhammadmusharraf444/loan-approval-dataset"
DATASET_URL = "https://www.kaggle.com/datasets/muhammadmusharraf444/loan-approval-dataset"
MODEL_PATH = Path(__file__).resolve().parent / "loan_model.joblib"
HASH_PATH = MODEL_PATH.with_suffix(".joblib.sha256")
ARTIFACT_VERSION = 1
RANDOM_STATE = 42
DECISION_THRESHOLD = 0.50
DTI_WARNING = 40.0
CURRENCY = "₹"
DISCLAIMER = ("LoanLens is an educational decision-support demonstration, "
              "not a lender or a substitute for professional underwriting.")

TARGET = "loan_status"
TARGET_MAP = {"approved": 1, "rejected": 0}
LABEL_OF = {1: "Approved", 0: "Rejected"}

NUMERIC_FEATURES = ["no_of_dependents", "income_annum", "loan_amount", "loan_term",
                    "cibil_score", "residential_assets_value", "commercial_assets_value",
                    "luxury_assets_value", "bank_asset_value"]
CATEGORICAL_FEATURES = ["education", "self_employed"]
FEATURE_ORDER = NUMERIC_FEATURES + CATEGORICAL_FEATURES

FEATURE_LABELS = {
    "no_of_dependents": "Dependents", "income_annum": "Annual income",
    "loan_amount": "Loan amount", "loan_term": "Loan term (years)",
    "cibil_score": "Credit score (CIBIL)", "residential_assets_value": "Residential assets",
    "commercial_assets_value": "Commercial assets", "luxury_assets_value": "Luxury assets",
    "bank_asset_value": "Bank assets", "education": "Education",
    "self_employed": "Self-employed",
}
# (min, max) hard validity ranges used for data cleaning and upload validation
RANGES = {
    "no_of_dependents": (0, 15), "income_annum": (1, 1e9), "loan_amount": (1, 5e9),
    "loan_term": (1, 40), "cibil_score": (300, 900),
    "residential_assets_value": (0, 5e9), "commercial_assets_value": (0, 5e9),
    "luxury_assets_value": (0, 5e9), "bank_asset_value": (0, 5e9),
}
UNITS = {
    "no_of_dependents": "count (whole number)", "income_annum": f"{CURRENCY} per YEAR (gross)",
    "loan_amount": f"{CURRENCY} total principal", "loan_term": "YEARS (not months)",
    "cibil_score": "score 300-900", "residential_assets_value": CURRENCY,
    "commercial_assets_value": CURRENCY, "luxury_assets_value": CURRENCY,
    "bank_asset_value": CURRENCY, "education": "text category", "self_employed": "text category",
}

EMERALD, ROSE, AMBER = "#10B981", "#F43F5E", "#F59E0B"
NAVY, SLATE, BORDER, TEXT, MUTED = "#0B1220", "#111C33", "#24324D", "#E6EDF7", "#94A3B8"

MAX_UPLOAD_ROWS = 10_000
MAX_UPLOAD_BYTES = 5 * 1024 * 1024


def normalize_col(name: object) -> str:
    return re.sub(r"[^0-9a-z]+", "_", str(name).strip().lower()).strip("_")


# --------------------------------------------------------------------------- #
# Data: cleaning, Kaggle loading, synthetic data
# --------------------------------------------------------------------------- #
def clean_dataset(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Normalise, validate and clean a raw loan table. Raises ValueError on schema problems."""
    df = raw.copy()
    df.columns = [normalize_col(c) for c in df.columns]
    if df.columns.duplicated().any():
        raise ValueError("duplicate column names after normalisation")
    needed = FEATURE_ORDER + [TARGET]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    report = {"rows_raw": int(len(df)),
              "dropped_columns": [c for c in df.columns if c not in needed]}  # IDs etc. never used
    df = df[needed].copy()

    for col in CATEGORICAL_FEATURES + [TARGET]:
        s = df[col].astype("string").str.strip()
        df[col] = s.mask(s.str.lower().isin(["", "nan", "none", "null"]))
    tgt = df[TARGET].str.lower()
    unknown = sorted(set(tgt.dropna()) - set(TARGET_MAP))
    if unknown:
        raise ValueError(f"unrecognised target values {unknown}; expected Approved/Rejected")
    df["target"] = tgt.map(TARGET_MAP)  # Approved -> 1, Rejected -> 0 (explicit)
    report["dropped_missing_target"] = int(df["target"].isna().sum())
    df = df.dropna(subset=["target"]).drop(columns=[TARGET])
    df["target"] = df["target"].astype(int)

    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
        df.loc[~np.isfinite(df[col]), col] = np.nan
    asset_cols = [c for c in NUMERIC_FEATURES if c.endswith("_value")]
    report["clipped_negative_assets"] = int((df[asset_cols] < 0).sum().sum())
    df[asset_cols] = df[asset_cols].clip(lower=0)
    bad = pd.Series(False, index=df.index)
    for col, (lo, hi) in RANGES.items():
        bad |= df[col].notna() & ((df[col] < lo) | (df[col] > hi))
    report["dropped_out_of_range"] = int(bad.sum())
    df = df.loc[~bad].reset_index(drop=True)
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(object).where(df[col].notna(), np.nan)

    counts = df["target"].value_counts()
    if len(df) < 200 or counts.get(0, 0) < 20 or counts.get(1, 0) < 20:
        raise ValueError("too few rows or an under-represented target class after cleaning")
    report["rows_clean"] = int(len(df))
    return df, report


def data_fingerprint(df: pd.DataFrame) -> str:
    cols = FEATURE_ORDER + ["target"]
    h = pd.util.hash_pandas_object(df[cols].astype(str), index=False).values
    return hashlib.sha256(h.tobytes()).hexdigest()[:16]


def _configure_kaggle_credentials() -> None:
    """Optional credentials from Streamlit secrets; env vars are used as-is. Never printed."""
    for key in ("KAGGLE_USERNAME", "KAGGLE_KEY"):
        if os.environ.get(key):
            continue
        try:
            value = st.secrets.get(key)
        except Exception:  # no secrets file configured
            value = None
        if value:
            os.environ[key] = str(value)


@st.cache_data(ttl=3600, show_spinner="Looking for the Kaggle dataset…")
def load_kaggle_dataset() -> dict:
    """Discover a compatible CSV, load it with kagglehub, clean it. Never raises."""
    try:
        _configure_kaggle_credentials()
        root = Path(kagglehub.dataset_download(DATASET_HANDLE))
        csvs = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".csv")
        if not csvs:
            raise FileNotFoundError("the dataset contains no CSV files")
        required = set(FEATURE_ORDER + [TARGET])
        chosen = None
        for path in csvs:  # deterministic: sorted order, first schema match wins
            try:
                cols = {normalize_col(c) for c in pd.read_csv(path, nrows=0).columns}
            except Exception:
                continue
            if required.issubset(cols):
                chosen = path
                break
        if chosen is None:
            raise ValueError("no CSV contains the required target and predictor columns")
        rel = chosen.relative_to(root).as_posix()
        raw = kagglehub.dataset_load(KaggleDatasetAdapter.PANDAS, DATASET_HANDLE, rel)
        df, report = clean_dataset(raw)
        return {"ok": True, "df": df, "report": report, "file": rel, "error": None}
    except Exception as exc:  # download, auth, selection, parsing, schema
        logger.warning("Kaggle data unavailable (%s)", type(exc).__name__)
        return {"ok": False, "df": None, "report": None, "file": None,
                "error": f"{type(exc).__name__}: {str(exc)[:140]}"}


def _sample_applicants(n: int, rng: np.random.Generator, categories: dict) -> pd.DataFrame:
    income = np.clip(rng.lognormal(math.log(5.0e6), 0.55, n), 3e5, 2.5e7)
    ratio = rng.uniform(0.5, 5.0, n)
    df = pd.DataFrame({
        "no_of_dependents": rng.integers(0, 6, n).astype(float),
        "income_annum": income,
        "loan_amount": income * ratio,
        "loan_term": rng.integers(2, 21, n).astype(float),
        "cibil_score": np.clip(rng.normal(650, 100, n), 300, 900),
        "residential_assets_value": income * rng.uniform(0, 3.0, n),
        "commercial_assets_value": income * rng.uniform(0, 2.5, n) * rng.binomial(1, 0.6, n),
        "luxury_assets_value": income * rng.uniform(0.2, 3.5, n),
        "bank_asset_value": income * rng.uniform(0, 1.2, n),
    })
    for col in NUMERIC_FEATURES:
        df[col] = df[col].round(-4) if col not in ("no_of_dependents", "loan_term", "cibil_score") else df[col].round()
    df["income_annum"] = df["income_annum"].clip(lower=3e5)
    df["loan_amount"] = df["loan_amount"].clip(lower=1e5)
    for col in CATEGORICAL_FEATURES:
        df[col] = rng.choice(categories[col], n)
    return df[FEATURE_ORDER]


DEFAULT_CATEGORIES = {"education": ["Graduate", "Not Graduate"], "self_employed": ["No", "Yes"]}


@st.cache_data(show_spinner=False)
def synthetic_training_data(n: int = 3000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Reproducible synthetic applicants with a noisy, plausible approval rule (~62% approved)."""
    rng = np.random.default_rng(seed)
    df = _sample_applicants(n, rng, DEFAULT_CATEGORIES)
    assets = df[["residential_assets_value", "commercial_assets_value",
                 "luxury_assets_value", "bank_asset_value"]].sum(axis=1)
    z = (0.028 * (df["cibil_score"] - 575) - 0.35 * df["loan_amount"] / df["income_annum"]
         + 0.15 * assets / df["income_annum"] + 0.25 * (df["education"] == "Graduate")
         - 0.03 * df["loan_term"] - 0.05 * df["no_of_dependents"] + rng.normal(0, 1.0, n))
    df["target"] = (z > np.quantile(z, 0.38)).astype(int)
    return df


def get_training_data() -> dict:
    """Prefer validated Kaggle data, otherwise synthetic demonstration data."""
    result = load_kaggle_dataset()
    if result["ok"]:
        df, source = result["df"], "kaggle"
        detail = f"Kaggle file `{result['file']}`"
    else:
        df, source = synthetic_training_data(), "synthetic"
        detail = "synthetic demonstration data"
    return {"df": df, "source": source, "detail": detail,
            "kaggle_error": result["error"], "report": result["report"],
            "fingerprint": data_fingerprint(df)}


# --------------------------------------------------------------------------- #
# Model pipeline
# --------------------------------------------------------------------------- #
def build_pipeline(resample: bool, k_neighbors: int) -> ImbPipeline:
    n_num, n_cat = len(NUMERIC_FEATURES), len(CATEGORICAL_FEATURES)
    prep = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), NUMERIC_FEATURES),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value",
                                                     unknown_value=-1, dtype=float))]),
         CATEGORICAL_FEATURES),
    ])
    cat_idx = list(range(n_num, n_num + n_cat))
    sampler = (SMOTENC(categorical_features=cat_idx, k_neighbors=k_neighbors,
                       random_state=RANDOM_STATE) if resample else "passthrough")
    encode = ColumnTransformer([
        ("num", "passthrough", list(range(n_num))),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False),
         cat_idx),
    ])
    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    return ImbPipeline([("prep", prep), ("sampler", sampler), ("encode", encode), ("clf", clf)])


def describe_encoded_features(pipe: ImbPipeline) -> tuple[list[dict], dict]:
    ordinal = pipe.named_steps["prep"].named_transformers_["cat"].named_steps["ordinal"]
    ohe = pipe.named_steps["encode"].named_transformers_["cat"]
    categories = {f: [str(c) for c in cats]
                  for f, cats in zip(CATEGORICAL_FEATURES, ordinal.categories_)}
    cols = [{"feature": f, "label": FEATURE_LABELS[f], "kind": "numeric", "reference": None}
            for f in NUMERIC_FEATURES]
    drop_idx = getattr(ohe, "drop_idx_", None)
    for j, feat in enumerate(CATEGORICAL_FEATURES):
        dropped = None if drop_idx is None else drop_idx[j]
        ref = categories[feat][int(ohe.categories_[j][dropped])] if dropped is not None else None
        for k, value in enumerate(ohe.categories_[j]):
            if dropped is not None and k == dropped:
                continue
            cols.append({"feature": feat, "kind": "categorical", "reference": ref,
                         "label": f"{FEATURE_LABELS[feat]}: {categories[feat][int(value)]}"})
    return cols, categories


def train_model(data: dict) -> dict:
    df = data["df"]
    X, y = df[FEATURE_ORDER], df["target"].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y,
                                              random_state=RANDOM_STATE)
    counts = y_tr.value_counts()
    minority, majority = int(counts.min()), int(counts.max())
    imbalance = minority / majority
    resample = imbalance < 0.8 and minority >= 3
    k = max(1, min(5, minority - 1))
    if resample:
        note = f"SMOTENC on the training partition only (k_neighbors={k})"
    elif imbalance >= 0.8:
        note = "No resampling needed (classes already balanced)"
    else:
        note = "Fallback: resampling skipped (minority class too small for SMOTENC)"
    pipe = build_pipeline(resample, k)
    pipe.fit(X_tr, y_tr)
    if list(pipe.named_steps["clf"].classes_) != [0, 1]:
        raise RuntimeError("unexpected class order")

    proba = pipe.predict_proba(X_te)[:, 1]
    pred = (proba >= DECISION_THRESHOLD).astype(int)
    metrics = {"accuracy": accuracy_score(y_te, pred),
               "precision": precision_score(y_te, pred, zero_division=0),
               "recall": recall_score(y_te, pred, zero_division=0),
               "f1": f1_score(y_te, pred, zero_division=0),
               "roc_auc": roc_auc_score(y_te, proba) if y_te.nunique() == 2 else None,
               "n_train": int(len(X_tr)), "n_test": int(len(X_te))}
    cols, categories = describe_encoded_features(pipe)
    meta = {
        "artifact_version": ARTIFACT_VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__, "feature_order": FEATURE_ORDER,
        "target_mapping": TARGET_MAP, "threshold": DECISION_THRESHOLD,
        "data_source": data["source"], "data_detail": data["detail"],
        "data_fingerprint": data["fingerprint"], "n_rows": int(len(df)),
        "metrics": metrics, "resampling": note, "feature_columns": cols,
        "categories": categories,
        "approval_rate_raw": float(y.mean()),
        "numeric_ranges": {c: (float(df[c].min()), float(df[c].max())) for c in NUMERIC_FEATURES},
    }
    return {"pipeline": pipe, "meta": meta}


def save_artifact(bundle: dict) -> bool:
    try:
        tmp = MODEL_PATH.with_suffix(".tmp")
        joblib.dump(bundle, tmp)
        digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
        tmp.replace(MODEL_PATH)
        HASH_PATH.write_text(digest)
        return True
    except Exception as exc:  # ephemeral or read-only storage is fine
        logger.warning("Could not persist model artifact (%s)", type(exc).__name__)
        return False


def load_artifact(fingerprint: str) -> dict | None:
    """Load the app's own artifact only if its checksum, schema and metadata all match."""
    try:
        if not (MODEL_PATH.exists() and HASH_PATH.exists()):
            return None
        if hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest() != HASH_PATH.read_text().strip():
            return None
        obj = joblib.load(MODEL_PATH)
        meta, pipe = obj["meta"], obj["pipeline"]
        ok = (meta["artifact_version"] == ARTIFACT_VERSION
              and meta["sklearn_version"] == sklearn.__version__
              and meta["feature_order"] == FEATURE_ORDER
              and meta["target_mapping"] == TARGET_MAP
              and meta["data_fingerprint"] == fingerprint
              and hasattr(pipe, "predict_proba")
              and list(pipe.named_steps["clf"].classes_) == [0, 1])
        return obj if ok else None
    except Exception as exc:
        logger.warning("Ignoring unusable model artifact (%s)", type(exc).__name__)
        return None


@st.cache_resource(show_spinner="Preparing the LoanLens model (first run only)…")
def get_model_bundle() -> dict:
    data = get_training_data()
    bundle = load_artifact(data["fingerprint"])
    if bundle is not None:
        origin = "loaded from the saved artifact"
    else:
        bundle = train_model(data)
        origin = ("trained this session; artifact saved" if save_artifact(bundle)
                  else "trained this session; artifact could not be saved (ephemeral storage)")
    return {"bundle": bundle, "origin": origin, "kaggle_error": data["kaggle_error"],
            "report": data["report"]}


# --------------------------------------------------------------------------- #
# Scoring, explanations, financial maths
# --------------------------------------------------------------------------- #
def decision_label(p: float) -> str:
    return "Approved" if p >= DECISION_THRESHOLD else "Declined"


def score_frame(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    return bundle["pipeline"].predict_proba(X[FEATURE_ORDER])[:, 1]


def explain_row(bundle: dict, row: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Per-feature log-odds contributions = transformed value x coefficient (aggregated)."""
    pipe, meta = bundle["pipeline"], bundle["meta"]
    xt = pipe.named_steps["encode"].transform(pipe.named_steps["prep"].transform(row[FEATURE_ORDER]))
    clf = pipe.named_steps["clf"]
    contrib = xt[0] * clf.coef_[0]
    agg: dict[str, float] = {}
    for col, c in zip(meta["feature_columns"], contrib):
        agg[col["feature"]] = agg.get(col["feature"], 0.0) + float(c)
    out = pd.DataFrame({"feature": list(agg), "label": [FEATURE_LABELS[f] for f in agg],
                        "contribution": list(agg.values())})
    return out, float(clf.intercept_[0])


def _finite(*vals) -> bool:
    return all(v is not None and isinstance(v, (int, float, np.floating, np.integer))
               and math.isfinite(v) for v in vals)


def calc_emi(principal, annual_rate_pct, term_months) -> float | None:
    if not _finite(principal, annual_rate_pct, term_months):
        return None
    if principal < 0 or annual_rate_pct < 0 or term_months <= 0:
        return None
    if annual_rate_pct == 0:
        return principal / term_months
    try:
        r = annual_rate_pct / 1200.0
        growth = (1 + r) ** term_months
        emi = principal * r * growth / (growth - 1)
    except (OverflowError, ZeroDivisionError):
        return None
    return emi if math.isfinite(emi) else None


def calc_dti(annual_income, monthly_debt, emi) -> float | None:
    if not _finite(annual_income, monthly_debt, emi) or annual_income <= 0 or monthly_debt < 0:
        return None
    return (monthly_debt + emi) / (annual_income / 12.0) * 100.0


def calc_lti(principal, annual_income) -> float | None:
    if not _finite(principal, annual_income) or annual_income <= 0 or principal < 0:
        return None
    return principal / annual_income


# --------------------------------------------------------------------------- #
# Upload validation
# --------------------------------------------------------------------------- #
ERROR_COLS = ["csv_row", "column", "issue"]


def validate_applicants(raw: pd.DataFrame, meta: dict) -> tuple[pd.DataFrame | None, pd.DataFrame, str | None]:
    """Return (valid rows, row-level error table, fatal message)."""
    df = raw.copy()
    df.columns = [normalize_col(c) for c in df.columns]
    if df.columns.duplicated().any():
        return None, pd.DataFrame(columns=ERROR_COLS), "Duplicate column names in the CSV."
    missing = [c for c in FEATURE_ORDER if c not in df.columns]
    if missing:
        return None, pd.DataFrame(columns=ERROR_COLS), f"Missing required columns: {', '.join(missing)}"
    if df.empty:
        return None, pd.DataFrame(columns=ERROR_COLS), "The CSV has no data rows."
    if len(df) > MAX_UPLOAD_ROWS:
        return None, pd.DataFrame(columns=ERROR_COLS), f"Too many rows (limit {MAX_UPLOAD_ROWS:,})."
    df = df[FEATURE_ORDER].copy()
    errors: list[dict] = []
    bad = pd.Series(False, index=df.index)

    def flag(mask: pd.Series, col: str, msg) -> None:
        nonlocal bad
        bad |= mask
        for idx in df.index[mask]:
            errors.append({"csv_row": int(idx) + 2, "column": col,
                           "issue": msg(idx) if callable(msg) else msg})

    for col in NUMERIC_FEATURES:
        s = df[col]
        blank = s.isna() | (s.astype(str).str.strip() == "")
        num = pd.to_numeric(s, errors="coerce")
        flag(blank, col, "missing value")
        flag(~blank & num.isna(), col, lambda i, c=col: f"not a number: {str(df.at[i, c])[:30]!r}")
        flag(num.notna() & ~np.isfinite(num), col, "non-finite number")
        lo, hi = RANGES[col]
        fin = num.notna() & np.isfinite(num)
        flag(fin & ((num < lo) | (num > hi)), col, f"outside valid range {lo:,.0f} to {hi:,.0f}")
        if col in ("no_of_dependents", "loan_term", "cibil_score"):
            flag(fin & (num != num.round()), col, "must be a whole number")
        df[col] = num
    for col in CATEGORICAL_FEATURES:
        allowed = meta["categories"][col]
        canon = {a.strip().lower(): a for a in allowed}
        s = df[col].astype("string").str.strip()
        blank = s.isna() | (s == "")
        mapped = s.str.lower().map(canon)
        flag(blank, col, "missing value")
        flag(~blank & mapped.isna(), col,
             lambda i, c=col, a=allowed: f"unknown category {str(df.at[i, c])[:30]!r}; allowed: {a}")
        df[col] = mapped.astype(object)
    total = len(errors)
    err_df = pd.DataFrame(errors[:2000], columns=ERROR_COLS)
    valid = df.loc[~bad].copy()
    valid.insert(0, "csv_row", valid.index + 2)
    note = f"{total:,} problems found (showing first 2,000)." if total > 2000 else None
    return valid, err_df, note


def generate_applicant_batch(meta: dict, n: int = 10) -> pd.DataFrame:
    """A fresh random batch on every call (unseeded generator)."""
    rng = np.random.default_rng()
    df = _sample_applicants(n, rng, meta["categories"])
    for col in NUMERIC_FEATURES:
        df[col] = df[col].astype(float)
    df.insert(0, "applicant", [f"SYN-{i + 1:02d}" for i in range(n)])
    return df


def sample_template() -> pd.DataFrame:
    return pd.DataFrame([
        [2, 6_000_000, 15_000_000, 10, 700, 5_000_000, 2_000_000, 8_000_000, 3_000_000, "Graduate", "No"],
        [0, 3_500_000, 12_000_000, 15, 540, 1_000_000, 0, 4_000_000, 800_000, "Not Graduate", "Yes"],
    ], columns=FEATURE_ORDER)


def column_guide(meta: dict) -> pd.DataFrame:
    rows = []
    for c in FEATURE_ORDER:
        if c in NUMERIC_FEATURES:
            lo, hi = RANGES[c]
            rows.append([c, "number", UNITS[c], f"{lo:,.0f} to {hi:,.0f}"])
        else:
            rows.append([c, "text", UNITS[c], " / ".join(meta["categories"][c])])
    return pd.DataFrame(rows, columns=["column", "type", "unit", "allowed values"])


# --------------------------------------------------------------------------- #
# Styling and charts
# --------------------------------------------------------------------------- #
def inject_css() -> None:
    st.markdown(f"""
<style>
  .block-container {{padding-top: 2rem; padding-bottom: 3rem; max-width: 1280px;}}
  .ll-brand {{font-size: 2.1rem; font-weight: 800; letter-spacing: -0.02em; margin: 0;
              background: linear-gradient(90deg, {EMERALD}, #38BDF8); -webkit-background-clip: text;
              -webkit-text-fill-color: transparent;}}
  .ll-tag {{color: {MUTED}; margin: 0 0 1rem 0;}}
  .ll-card {{border: 1px solid {BORDER}; background: {SLATE}; border-radius: 16px;
             padding: 1.4rem 1.6rem; transition: border-color .3s ease, box-shadow .3s ease;}}
  .ll-card:hover {{box-shadow: 0 0 0 1px {BORDER}, 0 8px 30px rgba(0,0,0,.35);}}
  .ll-verdict {{font-size: 2.2rem; font-weight: 800; margin: .2rem 0 .1rem 0;}}
  .ll-prob {{font-size: 1rem; color: {MUTED};}}
  .ll-bar {{position: relative; height: 14px; background: {BORDER}; border-radius: 99px;
            margin: 1rem 0 .4rem 0; overflow: hidden;}}
  .ll-fill {{height: 100%; border-radius: 99px; transition: width .6s ease;}}
  .ll-mark {{position: absolute; top: -3px; width: 2px; height: 20px; background: {TEXT};}}
  .ll-small {{font-size: .82rem; color: {MUTED};}}
  div[data-testid="stMetric"] {{background: {SLATE}; border: 1px solid {BORDER};
        border-radius: 14px; padding: .9rem 1.1rem;}}
  button[data-baseweb="tab"] {{font-weight: 600;}}
</style>""", unsafe_allow_html=True)


def style_axes(fig, ax) -> None:
    fig.patch.set_facecolor(SLATE)
    ax.set_facecolor(SLATE)
    for spine in ax.spines.values():
        spine.set_color(BORDER)
    ax.tick_params(colors=MUTED)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(TEXT)
    ax.grid(axis="x", color=BORDER, alpha=0.5, linewidth=0.6)
    ax.set_axisbelow(True)


def bar_chart(labels: list[str], values: list[float], title: str, xlabel: str):
    fig, ax = plt.subplots(figsize=(7, max(2.6, 0.42 * len(labels) + 1)))
    colors = [EMERALD if v >= 0 else ROSE for v in values]
    ax.barh(labels, values, color=colors, height=0.62)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.invert_yaxis()
    ax.set_title(title, loc="left", fontsize=11)
    ax.set_xlabel(xlabel)
    style_axes(fig, ax)
    fig.tight_layout()
    return fig


def donut_chart(approved: int, declined: int):
    parts = [(n, c, v) for n, c, v in (("Approved", EMERALD, approved), ("Declined", ROSE, declined)) if v > 0]
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.pie([p[2] for p in parts], labels=[f"{p[0]} ({p[2]})" for p in parts],
           colors=[p[1] for p in parts], startangle=90, counterclock=False,
           wedgeprops={"width": 0.38, "edgecolor": SLATE, "linewidth": 2},
           textprops={"color": TEXT})
    ax.set_aspect("equal")
    fig.patch.set_facecolor(SLATE)
    fig.tight_layout()
    return fig


def show_fig(fig) -> None:
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def money(v: float | None) -> str:
    return "N/A" if v is None else f"{CURRENCY}{v:,.0f}"


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def sidebar_inputs(meta: dict) -> dict:
    sb = st.sidebar
    sb.markdown('<p class="ll-brand" style="font-size:1.5rem">LoanLens</p>', unsafe_allow_html=True)
    sb.caption("Adjust any input - the estimate updates instantly.")
    cats = meta["categories"]
    v: dict = {}
    with sb.expander("👤 Personal Info", expanded=True):
        v["no_of_dependents"] = st.slider("Dependents", 0, 10, 2, help="Number of financial dependents.")
        v["education"] = st.selectbox("Education", cats["education"], help="Highest education category used by the model.")
        v["self_employed"] = st.selectbox("Self-employed", cats["self_employed"])
    with sb.expander("💼 Financial Profile", expanded=True):
        v["income_annum"] = st.number_input(f"Gross annual income ({CURRENCY} per year)", 0, 500_000_000,
                                            6_000_000, 100_000, help="Yearly income before tax. Not monthly.")
        if v["income_annum"] > 0:
            st.caption(f"≈ {CURRENCY}{v['income_annum'] / 12:,.0f} per month")
        v["cibil_score"] = st.slider("Credit score (CIBIL, 300-900)", 300, 900, 700,
                                     help="Credit bureau score. Higher is stronger.")
        v["residential_assets_value"] = st.number_input(f"Residential assets ({CURRENCY})", 0, 2_000_000_000,
                                                        5_000_000, 100_000)
        v["commercial_assets_value"] = st.number_input(f"Commercial assets ({CURRENCY})", 0, 2_000_000_000,
                                                       2_000_000, 100_000)
        v["luxury_assets_value"] = st.number_input(f"Luxury assets ({CURRENCY})", 0, 2_000_000_000,
                                                   8_000_000, 100_000)
        v["bank_asset_value"] = st.number_input(f"Bank assets ({CURRENCY})", 0, 2_000_000_000,
                                                3_000_000, 100_000)
        v["monthly_debt"] = st.number_input(f"Existing monthly debt payments ({CURRENCY} per month)", 0, 50_000_000,
                                            20_000, 1_000,
                                            help="Calculator assumption only - used for DTI, NOT a model input.")
    with sb.expander("🏦 Loan Details", expanded=True):
        v["loan_amount"] = st.number_input(f"Requested loan amount ({CURRENCY} total)", 0, 2_000_000_000,
                                           15_000_000, 500_000)
        v["loan_term"] = st.slider("Loan term (years)", 1, 30, 10,
                                   help="Duration in YEARS. Used by the model and, ×12, for the EMI.")
        v["interest_rate"] = st.slider("Annual interest rate (%)", 0.0, 30.0, 9.5, 0.25,
                                       help="Calculator assumption only - used for EMI and DTI, NOT a model input.")
    sb.caption("Calculator-only inputs: monthly debt and interest rate. The model never sees them.")
    return v


def applicant_row(v: dict) -> pd.DataFrame:
    row = {k: float(v[k]) for k in NUMERIC_FEATURES}
    row.update({k: v[k] for k in CATEGORICAL_FEATURES})
    return pd.DataFrame([row])[FEATURE_ORDER]


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
def tab_dashboard(bundle: dict, v: dict) -> None:
    meta = bundle["meta"]
    inputs_ok = v["income_annum"] > 0 and v["loan_amount"] > 0
    left, right = st.columns([1, 1.15], gap="large")
    with left:
        if not inputs_ok:
            st.error("Enter an annual income and loan amount greater than zero to get an estimate.")
        else:
            row = applicant_row(v)
            p = float(score_frame(bundle, row)[0])
            approved = decision_label(p) == "Approved"
            color = EMERALD if approved else ROSE
            st.markdown(f"""
<div class="ll-card">
  <div class="ll-small">MODEL ESTIMATE</div>
  <div class="ll-verdict" style="color:{color}">{'APPROVED' if approved else 'DECLINED'}</div>
  <div class="ll-prob">Estimated approval probability: <b style="color:{TEXT}">{p:.1%}</b>
       &nbsp;·&nbsp; decision threshold {DECISION_THRESHOLD:.0%}</div>
  <div class="ll-bar"><div class="ll-fill" style="width:{p * 100:.1f}%; background:{color}"></div>
       <div class="ll-mark" style="left:{DECISION_THRESHOLD * 100:.0f}%"></div></div>
  <div class="ll-small">Probability ≥ threshold → Approved. This is a statistical estimate from a demo
  model, not a guaranteed lending outcome.</div>
</div>""", unsafe_allow_html=True)
            out_of_range = [FEATURE_LABELS[c] for c, (lo, hi) in meta["numeric_ranges"].items()
                            if not lo <= float(v[c]) <= hi]
            if out_of_range:
                st.warning("Outside the model's training range (extrapolating): " + ", ".join(out_of_range))
    with right:
        if inputs_ok:
            contrib, intercept = explain_row(bundle, row)
            order = contrib.reindex(contrib["contribution"].abs().sort_values(ascending=False).index)
            top, rest = order.head(5), order.iloc[5:]
            fig = bar_chart(top["label"].tolist(), top["contribution"].tolist(),
                            "Top 5 factors for this applicant", "log-odds contribution  (→ approval / ← rejection)")
            show_fig(fig)
            total = intercept + contrib["contribution"].sum()
            st.caption(
                f"Contributions are **log-odds** (coefficient × the applicant's standardized/encoded value), "
                f"not probability points and not SHAP values. Score = baseline {intercept:+.2f} (intercept) "
                f"+ top-5 {top['contribution'].sum():+.2f} + other {len(rest)} factors "
                f"{rest['contribution'].sum():+.2f} = **{total:+.2f}** log-odds → {1 / (1 + math.exp(-total)):.1%}.")

    st.markdown("#### Financial metrics")
    months = v["loan_term"] * 12
    emi = calc_emi(v["loan_amount"], v["interest_rate"], months)
    dti = calc_dti(v["income_annum"], v["monthly_debt"], emi)
    lti = calc_lti(v["loan_amount"], v["income_annum"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Estimated EMI (per month)", money(emi),
              help=f"Equal monthly installment for principal, {v['interest_rate']:.2f}% annual rate and "
                   f"{months} months. Zero interest = principal ÷ months.")
    c2.metric("Debt-to-Income (DTI)", "N/A" if dti is None else f"{dti:.1f}%",
              help="(existing monthly debt + estimated EMI) ÷ gross monthly income × 100.")
    c3.metric("Loan-to-Income", "N/A" if lti is None else f"{lti:.2f}× annual income",
              help="Requested principal ÷ annual income, shown as a ratio (multiple of yearly income).")
    if dti is not None and dti > DTI_WARNING:
        st.markdown(f'<div class="ll-card" style="border-color:{AMBER}; color:{AMBER}">⚠ DTI is above '
                    f'{DTI_WARNING:.0f}% - an illustrative benchmark, not a universal lending rule.</div>',
                    unsafe_allow_html=True)
    if emi is None or dti is None or lti is None:
        st.info("Some metrics show N/A because income, loan amount or term is zero or invalid.")


def feature_explorer(df: pd.DataFrame) -> None:
    feat = st.selectbox("Explore a feature (original data, before any resampling)", FEATURE_ORDER,
                        format_func=lambda f: FEATURE_LABELS[f])
    plot_df = df.assign(Outcome=df["target"].map(LABEL_OF))
    palette = {"Approved": EMERALD, "Rejected": ROSE}
    fig, ax = plt.subplots(figsize=(7, 3.8))
    if feat in NUMERIC_FEATURES:
        sns.histplot(data=plot_df, x=feat, hue="Outcome", palette=palette, stat="density",
                     common_norm=False, element="step", alpha=0.35, bins=30, ax=ax)
        ax.set_ylabel("Density")
    else:
        props = (plot_df.groupby(feat)["Outcome"].value_counts(normalize=True)
                 .rename("share").reset_index())
        sns.barplot(data=props, x=feat, y="share", hue="Outcome", palette=palette, ax=ax)
        ax.set_ylabel("Share within category")
    ax.set_xlabel(f"{FEATURE_LABELS[feat]} ({UNITS[feat]})")
    ax.set_title(f"{FEATURE_LABELS[feat]} by outcome", loc="left", fontsize=11)
    style_axes(fig, ax)
    leg = ax.get_legend()
    if leg:
        leg.get_frame().set_facecolor(SLATE)
        leg.get_frame().set_edgecolor(BORDER)
        for t in leg.get_texts():
            t.set_color(TEXT)
        if leg.get_title():
            leg.get_title().set_color(MUTED)
    fig.tight_layout()
    show_fig(fig)


def tab_insights(bundle: dict) -> None:
    meta = bundle["meta"]
    clf = bundle["pipeline"].named_steps["clf"]
    cols = meta["feature_columns"]
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("#### Global coefficients")
        coefs = clf.coef_[0]
        order = np.argsort(-np.abs(coefs))
        labels = [cols[i]["label"] + ("  (contrast)" if cols[i]["kind"] == "categorical" else "") for i in order]
        show_fig(bar_chart(labels, [float(coefs[i]) for i in order], "Fitted logistic-regression coefficients",
                           "coefficient (log-odds)"))
        st.caption("Green raises the estimated approval odds, red lowers them. Numeric coefficients are per "
                   "**one standard deviation** of the feature. Categorical bars are **contrasts** against the "
                   "reference category (e.g. 'Self-employed: Yes' vs 'No'). Coefficients describe patterns in "
                   "this dataset, **not causal effects**.")
    with right:
        st.markdown("#### Feature exploration")
        feature_explorer(get_training_data()["df"])

    st.markdown("#### Held-out evaluation")
    m = meta["metrics"]
    src = "Kaggle dataset" if meta["data_source"] == "kaggle" else "SYNTHETIC demonstration data"
    cols5 = st.columns(5)
    for c, (name, key) in zip(cols5, [("Accuracy", "accuracy"), ("Precision", "precision"), ("Recall", "recall"),
                                      ("F1", "f1"), ("ROC-AUC", "roc_auc")]):
        c.metric(name, "n/a" if m[key] is None else f"{m[key]:.3f}")
    note = ("These metrics come from a single 80/20 stratified split of "
            f"{meta['n_rows']:,} rows ({m['n_test']:,} held out) of **{src}**. ")
    note += ("Synthetic-data scores only show the pipeline works; they say nothing about real-world performance."
             if meta["data_source"] == "synthetic" else
             "They are not evidence of production performance, fairness, or regulatory compliance.")
    st.caption(note)
    with st.expander("How the pipeline works", expanded=False):
        st.markdown(f"""
1. **Data validation** - column names are normalised, whitespace trimmed, `Approved→1` / `Rejected→0` mapped
   explicitly, numeric ranges checked, identifiers dropped. Gender is not used.
2. **Train/test split** - 80/20, stratified, seed {RANDOM_STATE}. The test set is never resampled.
3. **Preprocessing (fit on training rows only)** - median/most-frequent imputation, standardised numerics,
   ordinal codes for categories (unseen categories are handled safely).
4. **Resampling** - {meta['resampling']}. SMOTENC is used because predictors are mixed numeric/categorical.
5. **Model** - `LogisticRegression` on standardized numerics plus one-hot categories.
6. **Prediction** - the same fitted pipeline scores single, synthetic and uploaded applicants; the approval
   threshold is {DECISION_THRESHOLD:.0%} everywhere.
""")


def render_results(bundle: dict, valid: pd.DataFrame, label: str) -> None:
    if valid is None or valid.empty:
        st.info("No valid applicants to score yet.")
        return
    probs = score_frame(bundle, valid)
    res = valid.copy()
    res["approval_probability"] = probs
    res["decision"] = [decision_label(p) for p in probs]
    n = len(res)
    appr = int((res["decision"] == "Approved").sum())
    rate = appr / n
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Valid applicants", n)
    m2.metric("Approved", appr)
    m3.metric("Declined", n - appr)
    m4.metric("Portfolio Approval Rate", f"{rate:.1%}", help="Approved ÷ successfully scored applicants.")
    st.progress(rate)
    c1, c2 = st.columns([1, 2], gap="large")
    with c1:
        show_fig(donut_chart(appr, n - appr))
        if appr in (0, n):
            st.caption("All applicants received the same decision.")
    with c2:
        st.dataframe(res, width="stretch", hide_index=True, column_config={
            "approval_probability": st.column_config.ProgressColumn(
                "approval_probability", min_value=0.0, max_value=1.0, format="%.3f")})
    st.download_button("⬇ Download results (CSV)", res.to_csv(index=False).encode("utf-8"),
                       f"loanlens_{label}_results.csv", "text/csv")


def tab_portfolio(bundle: dict) -> None:
    meta = bundle["meta"]
    st.markdown("#### 1 · Synthetic applicants")
    if st.button("Generate 10 Synthetic Applicants", type="primary"):
        st.session_state["synthetic_batch"] = generate_applicant_batch(meta, 10)
    batch = st.session_state.get("synthetic_batch")
    if batch is not None:
        st.dataframe(batch, width="stretch", hide_index=True)
    else:
        st.caption("Click the button to create a random batch matching the model's input schema.")

    st.markdown("#### 2 · Upload your own CSV")
    st.caption("Uploaded records are processed in memory for this session only - never saved or cached.")
    with st.expander("Expected columns and units"):
        st.dataframe(column_guide(meta), width="stretch", hide_index=True)
    st.download_button("⬇ Download sample template (CSV)",
                       sample_template().to_csv(index=False).encode("utf-8"), "loanlens_template.csv", "text/csv")
    up = st.file_uploader("Upload applicants CSV", type=["csv"])
    valid_upload = None
    if up is not None:
        if up.size > MAX_UPLOAD_BYTES:
            st.error("File is larger than 5 MB.")
        else:
            try:
                raw = pd.read_csv(up, dtype=str, keep_default_na=False, na_values=[""])
            except Exception:
                st.error("Could not parse this file as CSV. Check the delimiter and encoding (UTF-8).")
                raw = None
            if raw is not None:
                valid_upload, errs, msg = validate_applicants(raw, meta)
                if valid_upload is None:
                    st.error(msg)
                else:
                    if len(errs):
                        st.warning(f"{len(errs):,} validation problem(s). Affected rows are excluded from "
                                   "scoring - they are not counted as declines.")
                        st.dataframe(errs, width="stretch", hide_index=True)
                        if msg:
                            st.caption(msg)
                    else:
                        st.success(f"All {len(valid_upload):,} rows passed validation.")

    st.markdown("#### 3 · Portfolio results")
    options = []
    if batch is not None:
        options.append("Synthetic batch")
    if valid_upload is not None:
        options.append("Uploaded CSV")
    if not options:
        st.info("Generate a synthetic batch or upload a CSV to see portfolio results.")
        return
    choice = st.radio("Portfolio to score", options, horizontal=True)
    if choice == "Synthetic batch":
        render_results(bundle, batch, "synthetic")
    else:
        render_results(bundle, valid_upload, "uploaded")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(page_title="LoanLens - Loan Approval Decision Support", page_icon="🔍",
                       layout="wide", initial_sidebar_state="expanded")
    inject_css()
    st.markdown('<p class="ll-brand">LoanLens</p>'
                '<p class="ll-tag">Transparent, interactive loan-approval estimates with explainable logistic regression.</p>',
                unsafe_allow_html=True)
    st.info(DISCLAIMER, icon="ℹ️")
    try:
        state = get_model_bundle()
    except Exception as exc:
        st.error(f"The model could not be prepared ({type(exc).__name__}). Please reload the app.")
        st.stop()
    bundle, meta = state["bundle"], state["bundle"]["meta"]
    if meta["data_source"] == "kaggle":
        st.success(f"Active model trained on **Kaggle-sourced data** ({meta['data_detail']}; {meta['n_rows']:,} "
                   f"cleaned rows) · {state['origin']}. Dataset provenance is not verified as real lending outcomes.")
    else:
        why = f" Reason: {state['kaggle_error']}" if state["kaggle_error"] else ""
        st.warning(f"Kaggle data unavailable or incompatible - the active model is trained on **synthetic "
                   f"demonstration data**. Results illustrate the workflow only.{why} · {state['origin']}.")
    v = sidebar_inputs(meta)
    t1, t2, t3 = st.tabs(["Interactive Underwriting Dashboard", "Model Insights & XAI", "Portfolio Simulation"])
    with t1:
        tab_dashboard(bundle, v)
    with t2:
        tab_insights(bundle)
    with t3:
        tab_portfolio(bundle)
    st.divider()
    st.caption(f"{DISCLAIMER} Data source: [{DATASET_HANDLE}]({DATASET_URL}). Gender and other protected "
               "characteristics are not model inputs; no fairness or compliance evaluation has been performed.")


if __name__ == "__main__":
    main()
