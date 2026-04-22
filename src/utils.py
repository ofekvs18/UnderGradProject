"""
Shared utilities for the RA biomarker pipeline.

Import from here instead of duplicating data loading, metric computation,
or path constants across method scripts.
"""

import json
import re
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from omegaconf import OmegaConf

# ── Disease config (legacy constants — used by existing method scripts) ────────
DISEASE      = "ra"
DISEASE_FULL = "Rheumatoid Arthritis"
ICD9_PATTERN = "714%"

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR    = Path("data")
RESULTS_DIR = Path("results")
DATA_PATH   = DATA_DIR / "ra_modeling_data.csv"


# ── Multi-disease helpers (used by run_pipeline.py and future method scripts) ──

def load_ml_config():
    """
    Load ML parameters from conf/ml/defaults.yaml using OmegaConf.

    Returns a DictConfig with keys: seed, baselines, method2, method3, method4.
    """
    config_path = Path("conf") / "ml" / "defaults.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"ML config not found: {config_path}")
    return OmegaConf.load(config_path)


def load_disease_config(disease_slug):
    """
    Load disease config from conf/disease/{slug}.yaml using OmegaConf.

    Returns a DictConfig with keys: name, full_name, icd_patterns, icd_version.
    """
    config_path = Path("conf") / "disease" / f"{disease_slug}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Disease config not found: {config_path}")
    return OmegaConf.load(config_path)


def make_disease_config(name, icd_patterns, icd_version, full_name=""):
    """
    Build a disease config dict for the parameterized pipeline.

    Args:
        name:         Slug used for BQ table names and output CSV (e.g. 'ra').
        icd_patterns: List of ICD LIKE patterns (e.g. ['714%'] or ['250.01%', '250.03%']).
        icd_version:  9 or 10.
        full_name:    Human-readable name for logs (defaults to name.upper()).

    Returns:
        dict with keys: disease, disease_full, icd_patterns, icd_version.
    """
    if isinstance(icd_patterns, str):
        icd_patterns = [icd_patterns]
    return {
        "disease":      name,
        "disease_full": full_name or name.upper(),
        "icd_patterns": icd_patterns,
        "icd_version":  icd_version,
    }


def data_path(disease):
    """Return the modeling CSV path for a given disease slug."""
    return DATA_DIR / f"{disease}_modeling_data.csv"


def load_data_for(disease):
    """Load the modeling CSV for a given disease slug. Returns (df, feature_names)."""
    return load_data(data_path(disease))

# ── Constants ─────────────────────────────────────────────────────────────────
META_COLS    = {"subject_id", "is_case", "split"}
BASELINE_AUC = 0.658   # all-features logistic regression (AUC-ROC)

# ── Data loading ──────────────────────────────────────────────────────────────
def load_data(path=DATA_PATH):
    """Load the modeling CSV and return (df, feature_names)."""
    df       = pd.read_csv(path)
    features = [c for c in df.columns if c not in META_COLS]
    return df, features


def get_splits(df):
    """Return (train_df, test_df) using the pre-computed split column."""
    return df[df["split"] == "train"].copy(), df[df["split"] == "test"].copy()


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_binary_metrics(y_true, y_pred):
    """Compute precision, recall, F1, F2 from binary predictions."""
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    f2 = (5 * precision * recall / (4 * precision + recall)
          if (4 * precision + recall) > 0 else 0.0)
    return {
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "f2":        f2,
        "tp":        tp,
        "fp":        fp,
        "fn":        fn,
    }


def find_youden_threshold(y_true, scores):
    """
    Find optimal threshold via Youden's index on the ROC curve.
    Returns (threshold, fpr_array, tpr_array).
    """
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    # Youden's J = sensitivity + specificity - 1 = TPR - FPR
    best_idx = int(np.argmax(tpr - fpr))
    return float(thresholds[best_idx]), fpr, tpr


# ── PR metrics ────────────────────────────────────────────────────────────────
def precision_at_recall_levels(y_train_scores, y_train, y_test_scores, y_test,
                                levels=(0.25, 0.50, 0.75)):
    """
    For each target recall level, find the score threshold on TRAIN that achieves
    recall >= level, then apply it to TEST and return {level: (precision, recall)}.
    """
    from sklearn.metrics import precision_recall_curve
    prec_tr, rec_tr, thresh_tr = precision_recall_curve(y_train, y_train_scores)
    rec_search = rec_tr[:-1]
    thresholds = thresh_tr
    results = {}
    for level in levels:
        mask = rec_search >= level
        if not mask.any():
            results[level] = (0.0, 0.0)
            continue
        best_thresh = thresholds[mask].max()
        preds = (y_test_scores >= best_thresh).astype(int)
        m = compute_binary_metrics(y_test, preds)
        results[level] = (round(m["precision"], 4), round(m["recall"], 4))
    return results


# ── Formula evaluation ────────────────────────────────────────────────────────
def eval_formula_scores(formula, df, features, bad_frac=0.10):
    """
    Safely evaluate a formula string against a DataFrame.
    Returns a numpy array of scores, or None if > bad_frac rows produce NaN/inf.
    """
    local = {f: df[f].values.astype(float) for f in features}
    local["sqrt"] = np.sqrt
    local["log"]  = np.log1p
    local["abs"]  = np.abs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            scores = eval(formula, {"__builtins__": {}}, local)  # noqa: S307
        except Exception:
            return None
    scores = np.asarray(scores, dtype=float)
    bad = ~np.isfinite(scores)
    if bad.mean() > bad_frac:
        return None
    if bad.any():
        scores[bad] = np.nanmedian(scores[~bad]) if (~bad).any() else 0.0
    return scores


def evaluate_formula_full(formula, train_df, test_df, features):
    """
    Evaluate one formula end-to-end. Returns a metrics dict or None if invalid.
    Threshold is chosen via Youden's index on TRAIN, applied to TEST.
    Handles inverse predictivity by flipping scores when AUC-ROC < 0.5.
    """
    from sklearn.metrics import roc_auc_score, average_precision_score

    tr_clean = train_df[features + ["is_case"]].dropna()
    te_clean = test_df[features + ["is_case"]].dropna()

    score_tr = eval_formula_scores(formula, tr_clean, features)
    score_te = eval_formula_scores(formula, te_clean, features)
    if score_tr is None or score_te is None:
        return None

    y_tr = tr_clean["is_case"].values
    y_te = te_clean["is_case"].values

    if y_tr.sum() < 5 or y_te.sum() < 5:
        return None

    try:
        auc_roc = float(roc_auc_score(y_te, score_te))
        auc_pr  = float(average_precision_score(y_te, score_te))
    except Exception:
        return None

    if auc_roc < 0.5:
        score_tr = -score_tr
        score_te = -score_te
        auc_roc  = 1.0 - auc_roc
        auc_pr   = float(average_precision_score(y_te, score_te))

    threshold, _, _ = find_youden_threshold(y_tr, score_tr)
    preds = (score_te >= threshold).astype(int)
    m = compute_binary_metrics(y_te, preds)
    par = precision_at_recall_levels(score_tr, y_tr, score_te, y_te)

    return {
        "formula":                formula,
        "auc_roc":                round(auc_roc, 4),
        "auc_pr":                 round(auc_pr, 4),
        "precision":              round(m["precision"], 4),
        "recall":                 round(m["recall"], 4),
        "f1":                     round(m["f1"], 4),
        "f2":                     round(m["f2"], 4),
        "precision_at_recall_25": par[0.25][0],
        "precision_at_recall_50": par[0.50][0],
        "precision_at_recall_75": par[0.75][0],
    }


# ── Output helpers ────────────────────────────────────────────────────────────
def ensure_dir(path):
    """Create directory (and parents) if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


# ── LLM helpers (MedGemma on cluster) ─────────────────────────────────────────
MEDGEMMA_MODEL_ID = "google/medgemma-4b-it"
MEDGEMMA_MAX_NEW_TOKENS = 1024
CBC_FEATURE_LIST = ["rdw", "hgb", "hct", "wbc", "plt", "mcv", "mch", "mchc", "rbc"]
THRESHOLDS_CACHE_DIR = RESULTS_DIR / "literature_thresholds"

# Prompt library cache
_PROMPTS = None

def load_prompts():
    """Load prompts from prompts.json (cached after first call)."""
    global _PROMPTS
    if _PROMPTS is None:
        prompts_path = Path(__file__).parent / "prompts.json"
        with open(prompts_path, encoding="utf-8") as f:
            _PROMPTS = json.load(f)
    return _PROMPTS


def load_medgemma():
    """Load Med-Gemma 4B IT with automatic device placement. Returns (model, processor)."""
    try:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as e:
        raise ImportError(f"Missing dependency: {e}. Install: pip install transformers torch accelerate") from e

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif torch.cuda.is_available():
        dtype = torch.float16
    else:
        dtype = torch.float32

    processor = AutoProcessor.from_pretrained(MEDGEMMA_MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        MEDGEMMA_MODEL_ID, torch_dtype=dtype, device_map="auto"
    )
    model.eval()
    return model, processor


def medgemma_generate(model, processor, prompt: str, temperature: float = 0.1,
                      max_new_tokens: int = MEDGEMMA_MAX_NEW_TOKENS) -> str:
    """Run one inference call with MedGemma. Returns decoded text (new tokens only)."""
    import torch
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    input_len = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0), temperature=temperature,
        )
    return processor.decode(output_ids[0][input_len:], skip_special_tokens=True)


def build_threshold_prompt(disease_full: str) -> str:
    """Build the LLM prompt for retrieving literature thresholds."""
    prompts = load_prompts()
    template = prompts["method1_threshold"]["literature_thresholds"]["template"]
    return template.format(disease_full=disease_full)


def get_literature_thresholds(disease_full: str, force_refresh: bool = False) -> dict:
    """
    Query MedGemma to retrieve literature-based CBC thresholds for a disease.
    Returns {feature: (threshold, direction, source)}.
    direction is "above" or "below".

    Caches results to results/literature_thresholds/<disease_slug>.json.
    """
    slug = re.sub(r'[^a-z0-9]+', '_', disease_full.lower()).strip('_')
    cache_path = THRESHOLDS_CACHE_DIR / f"{slug}.json"

    if not force_refresh and cache_path.exists():
        print(f"  Loading cached thresholds from {cache_path}")
        with open(cache_path, encoding="utf-8") as f:
            raw = json.load(f)
        return {feat: (entry["threshold"], entry["direction"], entry["source"])
                for feat, entry in raw.items()}

    # Query LLM
    print(f"  Querying MedGemma for {disease_full} thresholds...")
    prompt = build_threshold_prompt(disease_full)
    model, processor = load_medgemma()
    raw = medgemma_generate(model, processor, prompt, temperature=0.1)

    # Extract JSON from response (may have surrounding text)
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if not match:
        raise ValueError(f"MedGemma did not return valid JSON. Response: {raw[:200]}")
    data = json.loads(match.group())

    thresholds = {}
    for feat in CBC_FEATURE_LIST:
        if feat in data:
            entry = data[feat]
            thresholds[feat] = (float(entry["threshold"]), entry["direction"], entry["source"])

    # Save to disk
    ensure_dir(THRESHOLDS_CACHE_DIR)
    serializable = {feat: {"threshold": t, "direction": d, "source": s}
                    for feat, (t, d, s) in thresholds.items()}
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    print(f"  Saved thresholds to {cache_path}")

    return thresholds
