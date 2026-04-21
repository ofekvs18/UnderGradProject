"""
run_pipeline.py — Parameterized cohort extraction pipeline.

Executes the BigQuery pipeline defined in src/queries/cohort_pipeline.sql
for an arbitrary disease, then exports the modeling CSV to data/.

Usage:
    python src/run_pipeline.py \\
        --disease ra \\
        --disease-full "Rheumatoid Arthritis" \\
        --icd-patterns "714%" \\
        --icd-version 9 \\
        --key-file .secrets/bq_service_account.json

    # Multiple ICD patterns (umbrella diseases):
    python src/run_pipeline.py \\
        --disease dm1 \\
        --disease-full "Type 1 Diabetes" \\
        --icd-patterns "250.01%" "250.03%" \\
        --icd-version 9 \\
        --key-file .secrets/bq_service_account.json

    # ICD-10 example:
    python src/run_pipeline.py \\
        --disease crohns \\
        --disease-full "Crohn's Disease" \\
        --icd-patterns "K50%" \\
        --icd-version 10 \\
        --key-file .secrets/bq_service_account.json
"""

# Standard library
import re
import sys
from pathlib import Path

# Third-party
import google.auth
from google.cloud import bigquery
import hydra
from omegaconf import DictConfig

# ── Paths ─────────────────────────────────────────────────────────────────────
SRC_DIR      = Path(__file__).parent
PROJECT_ROOT = SRC_DIR.parent
SQL_TEMPLATE = SRC_DIR / "queries" / "cohort_pipeline.sql"
DATA_DIR     = PROJECT_ROOT / "data"

DEFAULT_BQ_DATASET = "biomarkers-478606.biomarkers"


# ── Config construction ────────────────────────────────────────────────────────

def build_icd_like(patterns, alias=None):
    """
    Build a SQL LIKE expression for one or more ICD patterns.

    Single pattern:  icd_code LIKE '714%'
    Multiple:        (icd_code LIKE '714%' OR icd_code LIKE '714.0%')
    With alias 'd':  d.icd_code LIKE '714%'
    """
    prefix = f"{alias}." if alias else ""
    clauses = [f"{prefix}icd_code LIKE '{p}'" for p in patterns]
    if len(clauses) == 1:
        return clauses[0]
    return "(" + " OR ".join(clauses) + ")"


def make_pipeline_config(cfg):
    """Return a dict of all template substitution values from Hydra config."""
    patterns = cfg.disease.icd_patterns
    return {
        "disease":              cfg.disease.name,
        "disease_full":         cfg.disease.full_name,
        "icd_version":          cfg.disease.icd_version,
        "icd_filter":           build_icd_like(patterns, alias=None),
        "icd_filter_d":         build_icd_like(patterns, alias="d"),
        "icd_patterns_display": ", ".join(patterns),
        "bq_dataset":           cfg.pipeline.bq_dataset,
    }


# ── SQL loading and substitution ───────────────────────────────────────────────

def load_sql_template():
    """Load the cohort pipeline SQL template from disk."""
    if not SQL_TEMPLATE.exists():
        sys.exit(f"ERROR: SQL template not found at {SQL_TEMPLATE}")
    return SQL_TEMPLATE.read_text(encoding="utf-8")


def substitute_config(sql, config):
    """Replace all {token} placeholders in the SQL with config values."""
    try:
        return sql.format(**config)
    except KeyError as e:
        sys.exit(f"ERROR: SQL template references unknown token {e}. "
                 f"Available tokens: {list(config.keys())}")


# ── Statement parsing ──────────────────────────────────────────────────────────

def split_statements(sql):
    """
    Split SQL into individual statements.

    Splits on semicolons followed by at least one blank line, or end-of-string.
    Filters out blocks that are pure comments or whitespace.
    """
    raw_blocks = re.split(r";\s*(?=\n\s*\n|\Z)", sql, flags=re.DOTALL)
    statements = []
    for block in raw_blocks:
        block = block.strip()
        if not block:
            continue
        # Discard blocks that have no executable SQL (all comment lines)
        non_comment = [
            ln for ln in block.splitlines()
            if ln.strip() and not ln.strip().startswith("--")
        ]
        if non_comment:
            statements.append(block)
    return statements


def classify_statement(stmt):
    """
    Return one of: 'ddl', 'export', 'select', 'unknown'.

    'export' is a SELECT that carries the [EXPORT] marker comment.
    """
    if "-- [EXPORT]" in stmt:
        return "export"
    for line in stmt.splitlines():
        line = line.strip()
        if not line or line.startswith("--"):
            continue
        upper = line.upper()
        if upper.startswith("CREATE"):
            return "ddl"
        if upper.startswith("SELECT") or upper.startswith("WITH"):
            return "select"
        break
    return "unknown"


# ── BigQuery execution ─────────────────────────────────────────────────────────

def make_bq_client(key_file=None, project=None):
    """Create a BigQuery client using ADC or a service account key file."""
    creds, project_id = google.auth.load_credentials_from_file(key_file) if key_file else google.auth.default()
    project = project or project_id
    print(f"  Authenticated via {'service account key' if key_file else 'Application Default Credentials'} (project: {project})")
    
    return bigquery.Client(credentials=creds, project=project)


def run_ddl(client, stmt, label):
    """Execute a DDL statement (CREATE OR REPLACE TABLE). Prints confirmation."""
    print(f"\n[DDL] {label}")
    job = client.query(stmt)
    job.result()  # blocks until complete
    print(f"  OK — table created/replaced.")


def run_verify(client, stmt, label):
    """Execute a verification SELECT and print the result rows."""
    print(f"\n[VERIFY] {label}")
    job = client.query(stmt)
    rows = list(job.result())
    if not rows:
        print("  (no rows returned)")
        return
    # Print column headers + rows
    try:
        fields = list(rows[0].keys())
    except Exception:
        fields = [f.name for f in job.result().schema]
    col_w = {f: max(len(f), max(len(str(r[f])) for r in rows)) for f in fields}
    header = "  " + "  ".join(f.ljust(col_w[f]) for f in fields)
    divider = "  " + "  ".join("-" * col_w[f] for f in fields)
    print(header)
    print(divider)
    for row in rows:
        print("  " + "  ".join(str(row[f]).ljust(col_w[f]) for f in fields))
    print(f"  ({len(rows)} row(s))")


def run_export(client, stmt, disease, bq_dataset):
    """
    Execute the CP5 export SELECT, download as DataFrame, save to data/.
    Returns the output path.
    """
    try:
        import pandas as pd
    except ImportError:
        sys.exit("ERROR: pandas is not installed.")

    print(f"\n[EXPORT] Running CP5 export query...")
    job = client.query(stmt)
    df = job.to_dataframe()
    print(f"  Downloaded {len(df):,} rows × {len(df.columns)} columns.")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / f"{disease}_modeling_data.csv"
    df.to_csv(out_path, index=False)
    print(f"  Saved to {out_path}")
    return out_path


# ── Checkpoint label extraction ────────────────────────────────────────────────

def extract_label(stmt):
    """
    Pull a short human-readable label from the first meaningful comment or
    first SQL keyword line of a statement.
    """
    for line in stmt.splitlines():
        line = line.strip()
        if line.startswith("-- ") and not line.startswith("-- -") and len(line) > 5:
            # Skip section separators (##...##)
            if "#" not in line:
                return line[3:].strip()
    # Fallback: first keyword line, truncated
    for line in stmt.splitlines():
        line = line.strip()
        if line and not line.startswith("--"):
            return line[:80]
    return "(unknown)"


# ── Main ───────────────────────────────────────────────────────────────────────

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Execute the cohort extraction pipeline with user-specified parameters."""
    config = make_pipeline_config(cfg)

    print("=" * 70)
    print(f"Cohort Extraction Pipeline")
    print(f"  Disease  : {config['disease_full']} ({config['disease']})")
    print(f"  ICD      : version {config['icd_version']}, "
          f"pattern(s): {config['icd_patterns_display']}")
    print(f"  Dataset  : {config['bq_dataset']}")
    print(f"  Auth     : {cfg.pipeline.key_file or 'Application Default Credentials'}")
    print("=" * 70)

    sql = load_sql_template()
    sql = substitute_config(sql, config)

    if cfg.pipeline.dry_run:
        print("\n[DRY RUN] Substituted SQL:\n")
        print(sql)
        print("\n[DRY RUN] Checking BigQuery connectivity...")
        try:
            client = make_bq_client(cfg.pipeline.key_file, cfg.pipeline.project)
            rows = list(client.query("SELECT 1 AS ok").result())
            print(f"  Connection OK — got {rows[0]['ok']} from SELECT 1.")
        except Exception as exc:
            print(f"  Connection FAILED: {exc}")
        return

    client = make_bq_client(cfg.pipeline.key_file, cfg.pipeline.project)

    statements = split_statements(sql)
    print(f"\nFound {len(statements)} executable statement(s).\n")

    export_path = None
    for i, stmt in enumerate(statements, 1):
        kind = classify_statement(stmt)
        label = extract_label(stmt)

        if kind == "ddl":
            run_ddl(client, stmt, label)
        elif kind == "export":
            export_path = run_export(client, stmt, config["disease"], config["bq_dataset"])
        elif kind == "select":
            run_verify(client, stmt, label)
        else:
            print(f"\n[SKIP] Statement {i} — unrecognized type, skipping.")

    print("\n" + "=" * 70)
    print("Pipeline complete.")
    if export_path:
        print(f"Modeling CSV: {export_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
