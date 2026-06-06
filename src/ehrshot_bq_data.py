"""
ehrshot_bq_data.py — Extract EHRSHOT modeling data from BigQuery OMOP tables.

Queries EHRSHOTS_DATA.{condition_occurrence, measurement} in BigQuery to build
a disease cohort CSV matching the MIMIC pipeline format:
    subject_id, is_case, split, hct, hgb, mch, mchc, mcv, plt, rbc, rdw, wbc,
    neut_pct, lym_pct, mono_pct, eos_pct, baso_pct

Case patients are identified via condition_source_value LIKE patterns from
conf/disease/{slug}.yaml. Control patients are all other individuals who have
at least one CBC measurement in the dataset. Control index dates are sampled
from the case index-date distribution (calendar-time matching).

Usage:
    python src/ehrshot_bq_data.py --disease ra
    python src/ehrshot_bq_data.py --disease crhn --key-file .secrets/bq_service_account.json
    python src/ehrshot_bq_data.py --disease t1d --output data/t1d_ehrshot_data.csv
    python src/ehrshot_bq_data.py --disease ra --project biomarkers-478606

BQ settings and OMOP concept IDs are configured in conf/ehrshot_bq.yaml.
"""

import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from omegaconf import OmegaConf

sys.path.insert(0, "src")
from utils import load_disease_config, ensure_dir, DATA_DIR

EHRSHOT_BQ_CONF = Path("conf") / "ehrshot_bq.yaml"

FEATURE_ORDER = [
    "hct", "hgb", "mch", "mchc", "mcv", "plt", "rbc", "rdw", "wbc",
    "neut_pct", "lym_pct", "mono_pct", "eos_pct", "baso_pct",
]


def load_ehrshot_bq_config():
    if not EHRSHOT_BQ_CONF.exists():
        raise FileNotFoundError(f"EHRSHOT BQ config not found: {EHRSHOT_BQ_CONF}")
    return OmegaConf.load(EHRSHOT_BQ_CONF)


def make_bq_client(key_file=None, project=None):
    try:
        import google.auth
        from google.cloud import bigquery
    except ImportError:
        sys.exit(
            "ERROR: google-cloud-bigquery is not installed.\n"
            "       Run: pip install google-cloud-bigquery"
        )
    if key_file:
        creds, project_id = google.auth.load_credentials_from_file(key_file)
    else:
        creds, project_id = google.auth.default()
    project = project or project_id
    print(f"  Auth: {'service account' if key_file else 'Application Default Credentials'} "
          f"(project: {project})")
    from google.cloud import bigquery
    return bigquery.Client(credentials=creds, project=project)


def get_icd_patterns(disease_slug, ehrshot_cfg, disease_cfg):
    """
    Return the EHRSHOT-specific ICD LIKE patterns for a disease.

    EHRSHOT stores dotted ICD-9 and ICD-10 codes, so it needs its own pattern
    set (conf/ehrshot_bq.yaml: disease_icd_patterns) rather than the dotless
    ICD-9 patterns in conf/disease/<slug>.yaml. Falls back to the disease config
    with a warning if the slug is missing from the EHRSHOT map.
    """
    bq_map = ehrshot_cfg.get("disease_icd_patterns", {})
    if disease_slug in bq_map:
        return list(bq_map[disease_slug])
    print(
        f"  WARNING: '{disease_slug}' not found in disease_icd_patterns in "
        f"{EHRSHOT_BQ_CONF}. Falling back to conf/disease/{disease_slug}.yaml "
        "patterns, which assume MIMIC dotless ICD-9 and will likely MISS or "
        "MISMATCH EHRSHOT's dotted/ICD-10 codes."
    )
    return list(disease_cfg.icd_patterns)


def build_icd_filter(patterns):
    """Build a SQL WHERE clause from ICD LIKE patterns on condition_source_value."""
    clauses = [f"condition_source_value LIKE '{p}'" for p in patterns]
    if len(clauses) == 1:
        return clauses[0]
    return "(" + " OR ".join(clauses) + ")"


def query_cases(client, dataset, icd_filter):
    """Return DataFrame(person_id, index_date) — one row per case patient."""
    q = f"""
        SELECT
            person_id,
            MIN(condition_start_date) AS index_date
        FROM `{dataset}.condition_occurrence`
        WHERE {icd_filter}
        GROUP BY person_id
    """
    print("  Querying case patients from condition_occurrence...")
    df = client.query(q).to_dataframe()
    df["index_date"] = pd.to_datetime(df["index_date"])
    return df


def query_cbc_measurements(client, dataset, concept_ids, date_from, date_to):
    """
    Download all CBC measurements within a broad date window.
    Uses COALESCE(measurement_date, DATE(measurement_datetime)) for compatibility
    with OMOP instances that populate either column.
    """
    ids_str = ", ".join(str(c) for c in concept_ids)
    q = f"""
        SELECT
            person_id,
            measurement_concept_id,
            COALESCE(measurement_date,
                     DATE(measurement_datetime)) AS meas_date,
            value_as_number
        FROM `{dataset}.measurement`
        WHERE measurement_concept_id IN ({ids_str})
          AND value_as_number IS NOT NULL
          AND COALESCE(measurement_date,
                       DATE(measurement_datetime))
              BETWEEN '{date_from}' AND '{date_to}'
    """
    print(f"  Querying CBC measurements ({date_from} to {date_to})...")
    df = client.query(q).to_dataframe()
    df["meas_date"] = pd.to_datetime(df["meas_date"])
    print(f"    {len(df):,} measurement rows downloaded.")
    return df


def assign_control_index_dates(case_index_dates, control_pids, seed):
    """
    Assign each control patient a random index date sampled from the case
    index-date distribution (calendar-time matching).
    Returns Series: person_id -> index_date.
    """
    if len(case_index_dates) == 0 or len(control_pids) == 0:
        return pd.Series(dtype="datetime64[ns]", name="index_date")
    rng = np.random.default_rng(seed)
    sampled = rng.choice(case_index_dates.values, size=len(control_pids), replace=True)
    return pd.Series(sampled, index=control_pids, name="index_date")


def extract_last_cbc_before_index(meas_df, index_dates_series, lookback_days, concept_to_feat):
    """
    For each patient, take the most recent measurement of each CBC type
    within the lookback window before their index date.

    Returns DataFrame indexed by person_id with one column per CBC feature.
    """
    lookback_td = pd.Timedelta(days=lookback_days)

    df = meas_df.copy()
    df["feature"] = df["measurement_concept_id"].map(concept_to_feat)
    df = df.dropna(subset=["feature"])

    idx_df = index_dates_series.rename("index_date").reset_index()
    idx_df.columns = ["person_id", "index_date"]
    merged = df.merge(idx_df, on="person_id", how="inner")

    mask = (
        (merged["meas_date"] <= merged["index_date"])
        & (merged["meas_date"] >= merged["index_date"] - lookback_td)
    )
    filtered = merged[mask].copy()

    if filtered.empty:
        feat_names = list(concept_to_feat.values())
        return pd.DataFrame(columns=["person_id"] + feat_names)

    filtered = filtered.sort_values("meas_date")
    last_meas = (
        filtered.groupby(["person_id", "feature"])["value_as_number"]
        .last()
        .reset_index()
    )
    wide = last_meas.pivot(index="person_id", columns="feature", values="value_as_number")
    wide.columns.name = None
    return wide


def build_modeling_df(cbc_wide, case_pids, seed, test_frac):
    """Attach is_case labels and a deterministic patient-level split."""
    if cbc_wide.empty:
        return pd.DataFrame()

    df = cbc_wide.copy()
    df["is_case"] = df.index.map(lambda pid: 1 if pid in case_pids else 0)
    df = df.reset_index().rename(columns={"person_id": "subject_id"})

    rng = np.random.default_rng(seed)
    n = len(df)
    order = rng.permutation(n)
    n_test = max(1, int(n * test_frac))
    test_set = set(order[:n_test])
    df["split"] = ["test" if i in test_set else "train" for i in range(n)]

    feat_cols = [f for f in FEATURE_ORDER if f in df.columns]
    extra = [c for c in df.columns if c not in {"subject_id", "is_case", "split"} and c not in feat_cols]
    return df[["subject_id", "is_case", "split"] + feat_cols + extra]


def main():
    parser = argparse.ArgumentParser(
        description="Extract EHRSHOT OMOP cohort from BigQuery for a given disease"
    )
    parser.add_argument("--disease", default="ra",
                        help="Disease slug matching conf/disease/{slug}.yaml (default: ra)")
    parser.add_argument("--output", default="",
                        help="Output CSV path (default: data/{disease}_ehrshot_data.csv)")
    parser.add_argument("--key-file", default="",
                        help="GCP service account JSON (default: Application Default Credentials)")
    parser.add_argument("--project", default="",
                        help="GCP project ID (default: infer from credentials)")
    args = parser.parse_args()

    disease = load_disease_config(args.disease)
    cfg = load_ehrshot_bq_config()

    key_file = args.key_file or cfg.get("key_file") or None
    project  = args.project  or cfg.get("bq_project") or None
    dataset  = cfg.bq_dataset
    out_path = (Path(args.output) if args.output
                else DATA_DIR / f"{args.disease}_ehrshot_data.csv")
    ensure_dir(out_path.parent)

    icd_patterns = get_icd_patterns(args.disease, cfg, disease)

    print("=" * 70)
    print("EHRSHOT BigQuery Cohort Extraction")
    print(f"  Disease     : {disease.full_name} ({disease.name})")
    print(f"  ICD patterns: {icd_patterns}")
    print(f"  BQ dataset  : {dataset}")
    print(f"  Output      : {out_path}")
    print("=" * 70)

    client = make_bq_client(key_file, project)

    # ── Step 1: Identify case patients ────────────────────────────────────────
    icd_filter = build_icd_filter(icd_patterns)
    cases_df = query_cases(client, dataset, icd_filter)
    case_pids = set(cases_df["person_id"])
    case_index_dates = cases_df.set_index("person_id")["index_date"]
    print(f"  Case patients found: {len(case_pids):,}")

    if not case_pids:
        sys.exit(
            "ERROR: No cases found. Check that ICD patterns in "
            f"conf/disease/{args.disease}.yaml match condition_source_value "
            f"in {dataset}.condition_occurrence."
        )

    # ── Step 2: Download CBC measurements within the case date range ──────────
    date_from = (
        cases_df["index_date"].min() - pd.Timedelta(days=cfg.cbc_lookback_days)
    ).strftime("%Y-%m-%d")
    date_to = cases_df["index_date"].max().strftime("%Y-%m-%d")

    concept_ids = list(OmegaConf.to_container(cfg.cbc_concept_ids).values())
    meas_df = query_cbc_measurements(client, dataset, concept_ids, date_from, date_to)

    if meas_df.empty:
        sys.exit(
            "ERROR: No CBC measurements returned. Check that measurement_concept_id "
            f"values in conf/ehrshot_bq.yaml match the OMOP vocabulary in {dataset}."
        )

    # ── Step 3: Assign control patients ───────────────────────────────────────
    all_meas_pids = set(meas_df["person_id"].unique())
    control_pids = sorted(all_meas_pids - case_pids)
    print(f"  Control candidates (with CBC in window): {len(control_pids):,}")

    ctrl_index_dates = assign_control_index_dates(case_index_dates, control_pids, cfg.seed)
    all_index_dates = pd.concat([case_index_dates, ctrl_index_dates])
    print(f"  Total patients with index dates: {len(all_index_dates):,}")

    # ── Step 4: Extract per-patient CBC features ───────────────────────────────
    concept_to_feat = {
        v: k for k, v in OmegaConf.to_container(cfg.cbc_concept_ids).items()
    }
    print(f"\nExtracting last CBC values (lookback={cfg.cbc_lookback_days} days)...")
    cbc_wide = extract_last_cbc_before_index(
        meas_df, all_index_dates, cfg.cbc_lookback_days, concept_to_feat
    )
    n_any = len(cbc_wide)
    n_all = int(cbc_wide.notna().all(axis=1).sum()) if not cbc_wide.empty else 0
    print(f"  Patients with any CBC feature  : {n_any:,}")
    print(f"  Patients with all CBC features : {n_all:,}")

    # ── Step 5: Build and save modeling CSV ───────────────────────────────────
    modeling_df = build_modeling_df(cbc_wide, case_pids, cfg.seed, cfg.test_frac)
    if modeling_df.empty:
        sys.exit(
            "ERROR: No data produced. Likely causes:\n"
            "  1. No patients have CBC measurements within the lookback window.\n"
            "  2. ICD patterns don't match condition_source_value values.\n"
            "  3. OMOP concept IDs don't match the measurement table vocabulary."
        )

    modeling_df.to_csv(out_path, index=False)

    n_cases    = int(modeling_df["is_case"].sum())
    prevalence = modeling_df["is_case"].mean()
    n_train    = (modeling_df["split"] == "train").sum()
    n_test     = (modeling_df["split"] == "test").sum()

    print(f"\n{'='*70}")
    print(f"Saved {len(modeling_df):,} patients to {out_path}")
    print(f"  Cases      : {n_cases:,} ({prevalence:.2%})")
    print(f"  Controls   : {len(modeling_df) - n_cases:,}")
    print(f"  Train/Test : {n_train:,} / {n_test:,}")
    print("=" * 70)


if __name__ == "__main__":
    main()
