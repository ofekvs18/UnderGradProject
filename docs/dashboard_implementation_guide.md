# Dashboard Implementation Guide

Visualization dashboard for biomarker formula results.
See `docs/dashboard_mockup.html` for the visual reference.

---

## Step 1 — Unified Data Layer

**Goal:** Single flat CSV that all dashboard code reads from. No dashboard logic should parse raw master summaries.

### 1a. Formula normalizer (`src/formula_normalizer.py`)

Each method stores formulas in different notation:
- **M1**: `rbc < 4.0` (simple threshold — already readable)
- **M2**: `((rbc**2)) / (abs(hct)+1e-6)` (infix Python with safety epsilons)
- **M3**: `div(abs(add(hgb, plt)), mul(sub(rbc, mcv), abs(mcv)))` (GP S-expression prefix)
- **M4**: `(hgt - 40) * (plt - 100) / (hgb - 100)` (infix Python, human-authored)

Implement two outputs per formula:
- `formula_display`: human-readable, clean infix (strip `+1e-6`, `abs(…)+1`, convert `**` → `²`, convert GP prefix → infix)
- `formula_raw`: original string, preserved exactly

Feature extraction: regex-match the 9 known CBC feature names (`hct hgb mch mchc mcv plt rbc rdw wbc`) from `formula_raw` to populate `features_used` (comma-separated) and `feature_count` (int).

### 1b. Aggregator script (`src/build_dashboard_data.py`)

Read all master summaries → apply normalizer → write one output file:

```
results/dashboard_data.csv
```

Schema:

| Column | Type | Source |
|---|---|---|
| `disease` | str | master summary |
| `method` | str | `m1` / `m2` / `m3` / `m4` |
| `formula_raw` | str | master summary |
| `formula_display` | str | normalizer |
| `features_used` | str | normalizer (comma-sep) |
| `feature_count` | int | normalizer |
| `auc_pr` | float | master summary |
| `auc_roc` | float | master summary |
| `f2` | float | recomputed or from summary |
| `prec_at_recall_25` | float | from summary if available |
| `prec_at_recall_50` | float | from summary if available |
| `prec_at_recall_75` | float | from summary if available |
| `baseline_auc_pr` | float | M1 best per disease (LR baseline) |
| `beats_baseline` | bool | `auc_pr > baseline_auc_pr` |
| `baseline_delta_pct` | float | `(auc_pr - baseline) / baseline * 100` |
| `timestamp` | str | from master summary |

Run: `python src/build_dashboard_data.py`

---

## Step 2 — Dashboard App

**Goal:** Local Streamlit app with filters, charts, and formula table. Reads only `dashboard_data.csv`.

### Stack choice

Use **Streamlit** + **Plotly**. Rationale:
- Streamlit is already usable in this repo's Python environment.
- Plotly figures can be exported as static PNG/SVG via `fig.write_image()` for article screenshots without a running browser.
- No JavaScript needed; all logic stays in Python.

Install: add `streamlit plotly kaleido` to `requirements.txt`.

### File: `src/dashboard.py`

**Sidebar filters:**
- Disease multiselect (default: all)
- Method multiselect (default: all)
- Feature count range slider 1–9
- Primary metric selector (AUC-PR, AUC-ROC, F2)
- "Beats baseline only" toggle

**Main panel — 4 sections:**

1. **Summary cards row** — 4 `st.metric` widgets: total formulas shown, best AUC-PR (+ delta), # beating baseline, avg improvement %

2. **Charts row** (`st.columns(2)`):
   - Left: grouped bar chart (disease on x-axis, one bar per method, colored by method)
   - Right: PR curve overlay for selected disease (one curve per method, using precomputed recall/precision arrays stored in `dashboard_data.csv` or a companion `dashboard_curves.pkl`)

3. **Formula table** — `st.dataframe` with:
   - Columns: Disease, Method, AUC-PR, AUC-ROC, F2, k, Beats Baseline, Formula (display)
   - Sortable by clicking headers
   - Row selection triggers detail panel below

4. **Detail panel** — shows on row click: all metrics with delta vs baseline, full `formula_display`, `formula_raw` in expander, feature badges

Run: `streamlit run src/dashboard.py`

---

## Step 3 — PR Curve Data

**Goal:** Store precomputed precision/recall arrays so the dashboard can render PR curves without loading the raw modeling data.

Add to `build_dashboard_data.py`: for each (disease, method, formula) row, compute the full PR curve on the test split and serialise it.

Output: `results/dashboard_curves.parquet`

Schema: `disease, method, formula_raw, recall_pts (list), precision_pts (list)`

This keeps the dashboard fast (no raw data loading at render time) and means the dashboard can run without access to `data/`.

---

## Step 4 — Static Export for Article

**Goal:** Generate publication-quality PNG figures that can be dropped into the paper without running the dashboard.

Add `src/export_figures.py`:

1. Load `dashboard_data.csv`
2. For each disease: render the grouped bar chart (AUC-PR × method) as `results/figures/{disease}_bar.png`
3. For each disease: render the PR curve overlay as `results/figures/{disease}_pr.png`
4. Render a summary heatmap (disease × method, colour = AUC-PR) as `results/figures/summary_heatmap.png`
5. All exported at 300 DPI via `fig.write_image(..., scale=3)`

Run: `python src/export_figures.py`

---

## Step 5 — External Validation Toggle (Future)

**Goal:** When EHRSHOT and NHANES evaluation results exist, surface them in the dashboard alongside MIMIC-IV.

Extend `build_dashboard_data.py` to accept an optional `--include-external` flag. When set, it reads `results/ehrshot/combined_evaluation.csv` and `results/nhanes/combined_evaluation.csv`, adds a `dataset` column (`mimic` / `ehrshot` / `nhanes`), and appends rows to `dashboard_data.csv`.

Extend `dashboard.py` sidebar: add a `Dataset` multiselect that appears only if external rows are present in the data file. This way the dashboard degrades gracefully when external data is absent — no code changes needed.

---

## Implementation order

| Step | Estimated effort | Depends on |
|---|---|---|
| 1a Formula normalizer | ~3h | — |
| 1b Aggregator | ~1h | 1a |
| 2 Dashboard app | ~4h | 1b |
| 3 PR curve data | ~2h | 1b + raw data |
| 4 Static export | ~1h | 2 |
| 5 External toggle | ~1h | 2 + TODO 2/3 done |

Start with Step 1 — once `dashboard_data.csv` exists and looks right, Steps 2 and 4 are largely independent.
