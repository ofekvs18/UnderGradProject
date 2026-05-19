"""
ehrshot_data.py — Extract modeling data from the EHRSHOT dataset.

Reads EHRSHOT patient timeline data (MEDS parquet format), extracts CBC lab
measurements, assigns disease labels from ICD codes, and outputs a modeling
CSV matching the MIMIC-IV pipeline format (subject_id, is_case, split, features).

Usage:
    python src/ehrshot_data.py \\
        --ehrshot-dir /path/to/ehrshot \\
        --disease ra

    python src/ehrshot_data.py \\
        --ehrshot-dir /path/to/ehrshot \\
        --disease t1d \\
        --output data/t1d_ehrshot_data.csv

Expected MEDS parquet schema (any of these column names are accepted):
    patient_id / person_id / subject_id  — patient identifier
    time / timestamp / start_datetime    — event timestamp
    code / concept_code / omop_code      — event code (e.g. "OMOP/3000963")
    numeric_value / value_as_number      — lab value (float, nullable)

CBC concept codes and ICD prefix formats are configured in conf/ehrshot.yaml.
Disease ICD patterns are loaded from conf/disease/{disease}.yaml.
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

sys.path.insert(0, "src")
from utils import load_disease_config, ensure_dir, DATA_DIR

EHRSHOT_CONF = Path("conf") / "ehrshot.yaml"


def load_ehrshot_config():
    if not EHRSHOT_CONF.exists():
        raise FileNotFoundError(f"EHRSHOT config not found: {EHRSHOT_CONF}")
    return OmegaConf.load(EHRSHOT_CONF)


def find_parquet_files(ehrshot_dir):
    """
    Auto-detect MEDS parquet files in the EHRSHOT directory.
    Tries common layouts in priority order.
    """
    single_candidates = [
        ehrshot_dir / "events.parquet",
        ehrshot_dir / "data" / "events.parquet",
        ehrshot_dir / "data" / "meds" / "events.parquet",
        ehrshot_dir / "data" / "meds" / "data.parquet",
    ]
    for c in single_candidates:
        if c.exists():
            return [c]

    shard_dirs = [
        ehrshot_dir / "data" / "meds",
        ehrshot_dir / "data",
        ehrshot_dir,
    ]
    for d in shard_dirs:
        if d.is_dir():
            files = sorted(d.glob("*.parquet"))
            if files:
                return files
            # Try split subdirectories (train/ held_out/)
            all_files = sorted(d.rglob("*.parquet"))
            if all_files:
                return all_files

    return []


def load_events(parquet_files):
    """Load and concatenate MEDS parquet event files into a single DataFrame."""
    print(f"Loading {len(parquet_files)} parquet file(s)...")
    frames = []
    for f in parquet_files:
        df = pd.read_parquet(f)
        frames.append(df)
        print(f"  {f.name}: {len(df):,} rows")
    events = pd.concat(frames, ignore_index=True)
    print(f"Total events loaded: {len(events):,}")

    col_map = {}
    for col in events.columns:
        lower = col.lower()
        if lower in ("patient_id", "person_id", "subject_id"):
            col_map[col] = "patient_id"
        elif lower in ("time", "timestamp", "start_datetime", "datetime", "event_time"):
            col_map[col] = "time"
        elif lower in ("code", "concept_code", "omop_code", "standard_concept_code"):
            col_map[col] = "code"
        elif lower in ("numeric_value", "value_as_number", "value", "measurement_value"):
            col_map[col] = "numeric_value"
    events = events.rename(columns=col_map)

    required = {"patient_id", "time", "code"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(
            f"MEDS events missing required columns: {missing}. "
            f"Got columns: {list(events.columns)}"
        )

    events["time"] = pd.to_datetime(events["time"], utc=True, errors="coerce")
    if "numeric_value" not in events.columns:
        events["numeric_value"] = np.nan
    else:
        events["numeric_value"] = pd.to_numeric(events["numeric_value"], errors="coerce")

    return events


def build_icd_prefixes(disease_cfg, ehrshot_cfg):
    """Convert disease ICD LIKE patterns to MEDS code prefix strings."""
    version = int(disease_cfg.icd_version)
    prefix = OmegaConf.to_container(ehrshot_cfg.icd_prefixes)[version]
    return [prefix + p.rstrip("%") for p in disease_cfg.icd_patterns]


def find_case_index_dates(events, icd_prefixes):
    """
    Find the first ICD event matching any disease prefix per patient.
    Returns Series: patient_id -> first_icd_time.
    """
    mask = events["code"].str.startswith(tuple(icd_prefixes), na=False)
    icd_events = events[mask][["patient_id", "time"]].dropna()
    if icd_events.empty:
        return pd.Series(dtype="datetime64[ns, UTC]", name="time")
    return icd_events.groupby("patient_id")["time"].min()


def assign_control_index_dates(all_patient_ids, case_index_dates, seed):
    """
    Assign control patients a random index date sampled from the case
    index-date distribution, matching the calendar-time distribution.
    Returns Series: patient_id -> sampled_time.
    """
    case_pids = set(case_index_dates.index)
    control_pids = sorted(set(all_patient_ids) - case_pids)

    if len(case_index_dates) == 0 or len(control_pids) == 0:
        return pd.Series(dtype="datetime64[ns, UTC]", name="time")

    rng = np.random.default_rng(seed)
    sampled = rng.choice(case_index_dates.values, size=len(control_pids), replace=True)
    return pd.Series(sampled, index=control_pids, dtype="datetime64[ns, UTC]")


def extract_cbc_features(events, patient_index_dates, cbc_code_map, lookback_days):
    """
    For each patient, extract the CBC measurement closest to (and before or
    equal to) the index date, within the lookback window.

    Returns DataFrame indexed by patient_id with one column per CBC feature.
    """
    all_cbc_codes = set()
    for codes in cbc_code_map.values():
        all_cbc_codes.update(codes)

    cbc_mask = events["code"].isin(all_cbc_codes) & events["numeric_value"].notna()
    cbc_events = events[cbc_mask][["patient_id", "time", "code", "numeric_value"]].copy()

    if cbc_events.empty:
        print("  WARNING: No CBC events found. Verify cbc_codes in conf/ehrshot.yaml "
              "match the codes in your EHRSHOT data.")
        return pd.DataFrame(columns=list(cbc_code_map.keys()))

    code_to_feat = {c: feat for feat, codes in cbc_code_map.items() for c in codes}
    cbc_events["feature"] = cbc_events["code"].map(code_to_feat)
    lookback_td = pd.Timedelta(days=lookback_days)

    pids = list(patient_index_dates.index)
    rows = []
    for i, pid in enumerate(pids):
        if i % 1000 == 0 and i > 0:
            print(f"  Extracting CBC... {i:,}/{len(pids):,}")
        idx_time = patient_index_dates.get(pid)
        if pd.isna(idx_time):
            continue

        pt_cbc = cbc_events[
            (cbc_events["patient_id"] == pid)
            & (cbc_events["time"] <= idx_time)
            & (cbc_events["time"] >= idx_time - lookback_td)
        ]
        if pt_cbc.empty:
            continue

        row = {"patient_id": pid}
        for feat in cbc_code_map:
            feat_vals = pt_cbc[pt_cbc["feature"] == feat].sort_values("time")
            row[feat] = float(feat_vals["numeric_value"].iloc[-1]) if len(feat_vals) > 0 else np.nan
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["patient_id"] + list(cbc_code_map.keys()))

    return pd.DataFrame(rows).set_index("patient_id")


def build_modeling_df(cbc_df, case_pids, seed, test_frac):
    """
    Combine CBC features with case/control labels and a patient-level split.
    Returns a DataFrame matching the MIMIC modeling CSV format.
    """
    if cbc_df.empty:
        return pd.DataFrame()

    df = cbc_df.copy()
    df["is_case"] = df.index.map(lambda pid: 1 if pid in case_pids else 0)
    df = df.reset_index().rename(columns={"patient_id": "subject_id"})

    rng = np.random.default_rng(seed)
    n = len(df)
    order = rng.permutation(n)
    n_test = max(1, int(n * test_frac))
    test_positions = set(order[:n_test])
    df["split"] = ["test" if i in test_positions else "train" for i in range(n)]

    feat_cols = [c for c in df.columns if c not in {"subject_id", "is_case", "split"}]
    return df[["subject_id", "is_case", "split"] + feat_cols]


def main():
    parser = argparse.ArgumentParser(description="Extract EHRSHOT modeling data for a disease")
    parser.add_argument("--ehrshot-dir", required=True,
                        help="Root directory of the local EHRSHOT dataset")
    parser.add_argument("--disease", default="ra",
                        help="Disease slug matching conf/disease/{slug}.yaml (default: ra)")
    parser.add_argument("--output", default="",
                        help="Output CSV path (default: data/{disease}_ehrshot_data.csv)")
    args = parser.parse_args()

    disease = load_disease_config(args.disease)
    ehrshot_cfg = load_ehrshot_config()

    ehrshot_dir = Path(args.ehrshot_dir)
    if not ehrshot_dir.exists():
        sys.exit(f"ERROR: EHRSHOT directory not found: {ehrshot_dir}")

    out_path = Path(args.output) if args.output else DATA_DIR / f"{args.disease}_ehrshot_data.csv"
    ensure_dir(out_path.parent)

    print("=" * 70)
    print(f"EHRSHOT Data Extraction")
    print(f"  Disease      : {disease.full_name} ({disease.name})")
    print(f"  ICD version  : {disease.icd_version} | patterns: {list(disease.icd_patterns)}")
    print(f"  EHRSHOT dir  : {ehrshot_dir}")
    print(f"  Output       : {out_path}")
    print("=" * 70)

    parquet_files = find_parquet_files(ehrshot_dir)
    if not parquet_files:
        sys.exit(
            f"ERROR: No parquet files found under {ehrshot_dir}.\n"
            "Expected MEDS layout: events.parquet, data/events.parquet, "
            "data/meds/*.parquet, or similar."
        )
    events = load_events(parquet_files)

    cbc_code_map = OmegaConf.to_container(ehrshot_cfg.cbc_codes)
    icd_prefixes = build_icd_prefixes(disease, ehrshot_cfg)
    print(f"\nSearching for ICD codes with prefixes: {icd_prefixes}")

    case_index_dates = find_case_index_dates(events, icd_prefixes)
    case_pids = set(case_index_dates.index)
    print(f"Case patients found: {len(case_pids):,}")

    all_pids = events["patient_id"].unique()
    ctrl_index_dates = assign_control_index_dates(all_pids, case_index_dates, ehrshot_cfg.seed)
    all_index_dates = pd.concat([case_index_dates, ctrl_index_dates])
    print(f"Control patients: {len(ctrl_index_dates):,}")
    print(f"Total patients with index dates: {len(all_index_dates):,}")

    print(f"\nExtracting CBC features (lookback={ehrshot_cfg.cbc_lookback_days} days)...")
    cbc_df = extract_cbc_features(
        events, all_index_dates, cbc_code_map, ehrshot_cfg.cbc_lookback_days
    )
    n_with_any_cbc = len(cbc_df)
    n_with_all_cbc = int(cbc_df.notna().all(axis=1).sum()) if not cbc_df.empty else 0
    print(f"Patients with ≥1 CBC measurement : {n_with_any_cbc:,}")
    print(f"Patients with all CBC features   : {n_with_all_cbc:,}")

    modeling_df = build_modeling_df(cbc_df, case_pids, ehrshot_cfg.seed, ehrshot_cfg.test_frac)
    if modeling_df.empty:
        sys.exit(
            "ERROR: No data produced. Likely causes:\n"
            "  1. No patients have any CBC measurement in the lookback window.\n"
            "  2. No ICD codes matched the disease patterns — check cbc_codes and "
            "icd_prefixes in conf/ehrshot.yaml."
        )

    modeling_df.to_csv(out_path, index=False)
    n_cases = int(modeling_df["is_case"].sum())
    prevalence = modeling_df["is_case"].mean()
    n_train = (modeling_df["split"] == "train").sum()
    n_test = (modeling_df["split"] == "test").sum()

    print(f"\n{'='*70}")
    print(f"Saved {len(modeling_df):,} patients to {out_path}")
    print(f"  Cases      : {n_cases:,} ({prevalence:.2%})")
    print(f"  Controls   : {len(modeling_df) - n_cases:,}")
    print(f"  Train/Test : {n_train:,} / {n_test:,}")
    print("=" * 70)


if __name__ == "__main__":
    main()
