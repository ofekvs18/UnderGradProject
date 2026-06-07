"""
nhanes_data.py — Extract modeling data from the NHANES dataset.

Reads NHANES survey component files (SAS XPT format), extracts CBC lab
measurements, assigns disease labels from questionnaire variables, and
outputs a modeling CSV matching the MIMIC-IV pipeline format:
    subject_id, is_case, split, hct, hgb, mch, mchc, mcv, plt, rbc, rdw, wbc

Supports multiple NHANES survey cycles (default: G–J, 2011–2018).

Usage:
    python src/nhanes_data.py --nhanes-dir /path/to/nhanes --disease ra

    python src/nhanes_data.py \\
        --nhanes-dir /path/to/nhanes \\
        --disease t2d \\
        --cycles G H I \\
        --output data/t2d_nhanes_data.csv

Expected file layout (both are auto-detected):
    Flat   : <nhanes_dir>/CBC_H.XPT  (all cycles in one directory)
    Subdir : <nhanes_dir>/H/CBC_H.XPT  (one subdirectory per cycle)

CBC variable names and disease case definitions are configured in
conf/nhanes.yaml.  Verify questionnaire codes against NHANES codebooks
(https://wwwn.cdc.gov/nchs/nhanes/) before first run.
"""

import argparse
import sys
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

sys.path.insert(0, "src")
from utils import load_disease_config, ensure_dir, DATA_DIR

NHANES_CONF = Path("conf") / "nhanes.yaml"
NHANES_BASE_URL = "https://wwwn.cdc.gov/Nchs/Nhanes"
_XPT_MAGIC = b"HEADER RECORD*******LIBRARY HEADER RECORD"


def load_nhanes_config():
    if not NHANES_CONF.exists():
        raise FileNotFoundError(f"NHANES config not found: {NHANES_CONF}")
    return OmegaConf.load(NHANES_CONF)


def _validate_xpt(path):
    """Return True if path starts with the SAS XPORT magic bytes."""
    try:
        with open(path, "rb") as f:
            return f.read(len(_XPT_MAGIC)) == _XPT_MAGIC
    except OSError:
        return False


def _download_xpt(year_range, filename, dest):
    """Download a NHANES XPT file from the CDC website. Returns True on success."""
    url = f"{NHANES_BASE_URL}/{year_range}/{filename}"
    try:
        print(f"    Downloading {url} ...")
        urllib.request.urlretrieve(url, str(dest))
        return True
    except Exception as exc:
        print(f"    Download failed: {exc}")
        return False


def find_component_file(nhanes_dir, cycle_letter, filename_template, year_range=None):
    """
    Locate a NHANES component file for a given cycle.
    Tries flat layout first, then cycle-subdirectory layout.
    If year_range is given and the file is missing or corrupt, attempts to
    download it from the CDC NHANES website.
    Returns Path or None if not found/downloadable.
    """
    filename = filename_template.format(cycle_letter)
    candidates = [
        nhanes_dir / filename,
        nhanes_dir / cycle_letter / filename,
        nhanes_dir / cycle_letter.lower() / filename,
    ]
    for c in candidates:
        if c.exists():
            if _validate_xpt(c):
                return c
            # File exists but header is wrong — likely a corrupted/HTML download
            print(f"  WARNING: {c.name} exists but is not a valid XPT file (corrupted download?)")
            if year_range:
                print(f"  Attempting to re-download ...")
                if _download_xpt(year_range, filename, c) and _validate_xpt(c):
                    print(f"  Re-download OK: {c.name}")
                    return c
                print(f"  Re-download failed or still invalid — skipping {c.name}")
            return None
    # File not found — attempt download to flat layout
    if year_range:
        dest = nhanes_dir / filename
        if _download_xpt(year_range, filename, dest) and _validate_xpt(dest):
            print(f"  Downloaded OK: {filename}")
            return dest
    return None


def load_xpt(path):
    """Load a SAS XPT file into a DataFrame. Returns None on failure."""
    try:
        return pd.read_sas(str(path), format="xport", encoding="latin-1")
    except Exception as exc:
        print(f"  WARNING: Could not read {path.name}: {exc}")
        return None


def load_cbc_across_cycles(nhanes_dir, cycles, cbc_file_tpl, cbc_vars):
    """
    Load and concatenate CBC component files across survey cycles.
    Renames NHANES variable names to pipeline feature names.

    Returns DataFrame with columns: SEQN, cycle, <feature_names...>
    """
    feat_to_var = OmegaConf.to_container(cbc_vars)
    var_to_feat = {v: k for k, v in feat_to_var.items()}
    needed_vars = set(feat_to_var.values())

    frames = []
    for cycle_letter, year_range in cycles.items():
        path = find_component_file(nhanes_dir, cycle_letter, cbc_file_tpl, year_range=year_range)
        if path is None:
            print(f"  [cycle {cycle_letter} / {year_range}] CBC file not found or invalid — skipped")
            continue
        df = load_xpt(path)
        if df is None:
            continue

        df.columns = [c.strip().upper() for c in df.columns]
        if "SEQN" not in df.columns:
            print(f"  [cycle {cycle_letter}] No SEQN column — skipped")
            continue

        present_vars = needed_vars & set(df.columns)
        missing_vars = needed_vars - present_vars
        if missing_vars:
            print(f"  [cycle {cycle_letter}] Missing CBC vars: {missing_vars} — will be NaN")

        keep_cols = ["SEQN"] + sorted(present_vars)
        sub = df[keep_cols].copy()
        sub = sub.rename(columns={v: k for v, k in var_to_feat.items() if v in sub.columns})
        sub["SEQN"] = sub["SEQN"].astype(int)
        sub["cycle"] = cycle_letter

        for feat in feat_to_var:
            if feat not in sub.columns:
                sub[feat] = np.nan
            else:
                sub[feat] = pd.to_numeric(sub[feat], errors="coerce")

        n_complete = int(sub[list(feat_to_var.keys())].notna().all(axis=1).sum())
        print(f"  [cycle {cycle_letter} / {year_range}] {len(sub):,} rows | "
              f"{n_complete:,} with all CBC features  ({path.name})")
        frames.append(sub)

    if not frames:
        return pd.DataFrame()

    cbc = pd.concat(frames, ignore_index=True)
    print(f"CBC total: {len(cbc):,} participant-cycle rows")
    return cbc


def load_questionnaire_conditions(nhanes_dir, cycles, q_file_tpl, variable, values):
    """
    Load a questionnaire variable across cycles and return the set of SEQNs
    where the variable equals any of the given values.
    """
    case_seqns = set()
    values_set = set(values)

    for cycle_letter in cycles:
        year_range = cycles[cycle_letter] if isinstance(cycles, dict) else None
        path = find_component_file(nhanes_dir, cycle_letter, q_file_tpl, year_range=year_range)
        if path is None:
            continue
        df = load_xpt(path)
        if df is None:
            continue
        df.columns = [c.strip().upper() for c in df.columns]
        var_upper = variable.upper()
        if "SEQN" not in df.columns or var_upper not in df.columns:
            continue
        df["SEQN"] = df["SEQN"].astype(int)
        df[var_upper] = pd.to_numeric(df[var_upper], errors="coerce")
        matched = df[df[var_upper].isin(values_set)]["SEQN"]
        case_seqns.update(matched.tolist())

    return case_seqns


def resolve_case_seqns(nhanes_dir, cycles, conditions):
    """
    Resolve which SEQNs are cases by ANDing all conditions together.
    conditions: list of {file, variable, values} dicts.

    Returns a set of SEQN integers.
    """
    if not conditions:
        return set()

    sets = []
    for cond in conditions:
        seqns = load_questionnaire_conditions(
            nhanes_dir, cycles,
            q_file_tpl=cond["file"],
            variable=cond["variable"],
            values=list(cond["values"]),
        )
        print(f"    Condition [{cond['variable']} in {list(cond['values'])}]: {len(seqns):,} matches")
        sets.append(seqns)

    result = sets[0]
    for s in sets[1:]:
        result = result & s
    return result


def build_modeling_df(cbc_df, case_seqns, seed, test_frac):
    """
    Combine CBC features with case/control labels and a patient-level split.
    One participant may appear in multiple cycles; keep one row per SEQN
    (the row with the fewest missing CBC values, breaking ties by later cycle).
    Returns a DataFrame in MIMIC modeling CSV format.
    """
    if cbc_df.empty:
        return pd.DataFrame()

    feat_cols = [c for c in cbc_df.columns if c not in {"SEQN", "cycle"}]

    # Keep most-complete CBC row per participant
    cbc_df = cbc_df.copy()
    cbc_df["_n_obs"] = cbc_df[feat_cols].notna().sum(axis=1)
    cbc_df = (
        cbc_df.sort_values(["SEQN", "_n_obs", "cycle"], ascending=[True, False, False])
        .drop_duplicates(subset="SEQN", keep="first")
        .drop(columns=["_n_obs", "cycle"])
        .reset_index(drop=True)
    )

    cbc_df["is_case"] = cbc_df["SEQN"].apply(lambda s: 1 if s in case_seqns else 0)
    cbc_df = cbc_df.rename(columns={"SEQN": "subject_id"})

    rng = np.random.default_rng(seed)
    n = len(cbc_df)
    order = rng.permutation(n)
    n_test = max(1, int(n * test_frac))
    test_positions = set(order[:n_test])
    cbc_df["split"] = ["test" if i in test_positions else "train" for i in range(n)]

    return cbc_df[["subject_id", "is_case", "split"] + feat_cols]


def main():
    parser = argparse.ArgumentParser(description="Extract NHANES modeling data for a disease")
    parser.add_argument("--nhanes-dir", required=True,
                        help="Root directory containing NHANES XPT files")
    parser.add_argument("--disease", default="ra",
                        help="Disease slug matching conf/disease/{slug}.yaml (default: ra)")
    parser.add_argument("--cycles", nargs="+", default=[],
                        help="Cycle letters to include (default: all in conf/nhanes.yaml)")
    parser.add_argument("--output", default="",
                        help="Output CSV path (default: data/{disease}_nhanes_data.csv)")
    args = parser.parse_args()

    disease = load_disease_config(args.disease)
    nhanes_cfg = load_nhanes_config()

    nhanes_dir = Path(args.nhanes_dir)
    if not nhanes_dir.exists():
        sys.exit(f"ERROR: NHANES directory not found: {nhanes_dir}")

    out_path = Path(args.output) if args.output else DATA_DIR / f"{args.disease}_nhanes_data.csv"
    ensure_dir(out_path.parent)

    all_cycles = OmegaConf.to_container(nhanes_cfg.cycles)
    cycles = {k: v for k, v in all_cycles.items() if not args.cycles or k in args.cycles}

    # Apply per-disease cycle override from conf/nhanes.yaml
    disease_cycle_overrides = OmegaConf.to_container(nhanes_cfg.get("disease_cycles", {}))
    if args.disease in disease_cycle_overrides:
        allowed = set(disease_cycle_overrides[args.disease])
        cycles = {k: v for k, v in cycles.items() if k in allowed}
        print(f"Note: cycle override for '{args.disease}' restricts to {list(cycles.keys())}")

    if not cycles:
        sys.exit(f"ERROR: No matching cycles found. Available: {list(all_cycles.keys())}")

    print("=" * 70)
    print(f"NHANES Data Extraction")
    print(f"  Disease    : {disease.full_name} ({disease.name})")
    print(f"  Cycles     : {cycles}")
    print(f"  NHANES dir : {nhanes_dir}")
    print(f"  Output     : {out_path}")
    print("=" * 70)

    # ── Load CBC data ─────────────────────────────────────────────────────────
    print("\nLoading CBC component files...")
    cbc_df = load_cbc_across_cycles(
        nhanes_dir, cycles,
        cbc_file_tpl=nhanes_cfg.cbc_file,
        cbc_vars=nhanes_cfg.cbc_vars,
    )
    if cbc_df.empty:
        sys.exit(
            "ERROR: No CBC data loaded. Check --nhanes-dir and cycle availability.\n"
            "Expected files like CBC_H.XPT (flat) or H/CBC_H.XPT (subdir layout)."
        )

    # ── Resolve disease cases ─────────────────────────────────────────────────
    disease_defs = OmegaConf.to_container(nhanes_cfg.disease_case_defs)
    if args.disease not in disease_defs:
        sys.exit(
            f"ERROR: No case definition for disease '{args.disease}' in conf/nhanes.yaml.\n"
            f"Available: {list(disease_defs.keys())}"
        )

    print(f"\nResolving case definitions for '{args.disease}'...")
    case_seqns = resolve_case_seqns(nhanes_dir, cycles, disease_defs[args.disease])
    print(f"Case participants (all conditions met): {len(case_seqns):,}")

    # Restrict to participants with CBC data
    cbc_seqns = set(cbc_df["SEQN"].unique())
    case_seqns_with_cbc = case_seqns & cbc_seqns
    print(f"Case participants with CBC data: {len(case_seqns_with_cbc):,}")

    # ── Build modeling DataFrame ──────────────────────────────────────────────
    modeling_df = build_modeling_df(cbc_df, case_seqns, nhanes_cfg.seed, nhanes_cfg.test_frac)
    if modeling_df.empty:
        sys.exit("ERROR: No data produced. Check that CBC files and questionnaire files exist.")

    modeling_df.to_csv(out_path, index=False)

    n_cases = int(modeling_df["is_case"].sum())
    prevalence = modeling_df["is_case"].mean()
    n_train = int((modeling_df["split"] == "train").sum())
    n_test = int((modeling_df["split"] == "test").sum())
    feat_cols = [c for c in modeling_df.columns if c not in {"subject_id", "is_case", "split"}]
    n_all_cbc = int(modeling_df[feat_cols].notna().all(axis=1).sum())

    print(f"\n{'='*70}")
    print(f"Saved {len(modeling_df):,} participants to {out_path}")
    print(f"  Cases            : {n_cases:,} ({prevalence:.2%})")
    print(f"  Controls         : {len(modeling_df) - n_cases:,}")
    print(f"  Train / Test     : {n_train:,} / {n_test:,}")
    print(f"  All CBC present  : {n_all_cbc:,} ({n_all_cbc/len(modeling_df):.1%})")
    print("=" * 70)


if __name__ == "__main__":
    main()
