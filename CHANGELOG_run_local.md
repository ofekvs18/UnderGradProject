# run_local.ps1 Fix Changelog

## Fix 1 — PowerShell parse error due to UTF-8 special characters

**Error:**
```
At run_local.ps1:104 char:53
+ ... ost "========================================" -ForegroundColor Green
The string is missing the terminator: "
```

**Root cause:**
`run_local.ps1` was saved as UTF-8 but PowerShell 5.1 parses `.ps1` files using the system ANSI code page (Hebrew/CP-1255 on this machine). The em dash `—` (U+2014) is encoded in UTF-8 as bytes `E2 80 94`. In CP-1255, byte `0x94` maps to the RIGHT DOUBLE QUOTATION MARK `"` (U+201D), which PowerShell 5.1 treats as a string terminator. This caused the string on line 103 (`Write-Host "  ALL DONE — ..."`) to be terminated early, leaving the rest as unparseable code and making line 104 report a "missing terminator" error.

The box-drawing character `─` (U+2500, UTF-8: `E2 94 80`) had the same `0x94` byte in the middle, but since all occurrences were inside `#` comments, they did not trigger the error.

**Fix:**
Read the file as UTF-8 in PowerShell and replaced all:
- Em dashes `—` (U+2014) → `-`
- Box-drawing horizontal bars `─` (U+2500) → `-`

Then rewrote the file as plain ASCII (no BOM). Parser confirms zero errors after the fix.

**Files changed:** `run_local.ps1`

---

## Fix 2 — `FileNotFoundError` for salt-specific data files

**Error:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'data\ra_modeling_dataseed1.csv'
```

**Root cause:**
`utils.data_path(disease, split_salt)` builds the path as `{disease}_modeling_data{split_salt}.csv`, concatenating the salt with no separator. When `run_local.ps1` passes `--split-salt seed1`, it looks for `ra_modeling_dataseed1.csv`, which doesn't exist. Only the base files (`ra_modeling_data.csv`) are present locally; per-salt data files would require running `run_pipeline.py` against BigQuery for each salt, which `run_local.ps1` intentionally skips. The `split_salt` argument is purely a label recorded in master-summary CSVs — `get_splits()` always reads the pre-computed `split` column and never uses the salt.

**Fix:**
Added a fallback in `load_data_for`: if the salt-specific file doesn't exist, fall back to the base `{disease}_modeling_data.csv`. The salt is still recorded as a label in all output CSVs.

**Files changed:** `src/utils.py`

---

## Fix 3 — Script aborts on scikit-learn `ConvergenceWarning` (stderr treated as fatal error)

**Error:**
```
python.exe : ...sklearn/linear_model/_logistic.py:599: ConvergenceWarning: lbfgs failed to converge...
    + FullyQualifiedErrorId : NativeCommandError
```
Script terminates after `k=12` in `sanity_check.py`.

**Root cause:**
`$ErrorActionPreference = "Stop"` in `run_local.ps1` causes PowerShell 5.1 to treat any stderr output from a native command (like `python.exe`) as a terminating `NativeCommandError`. Scikit-learn's `ConvergenceWarning` is written to stderr — not an error — but kills the script before `Run-Step`'s `$LASTEXITCODE` check is even reached. This makes the warning indistinguishable from a real crash.

**Fix:**
Changed `$ErrorActionPreference = "Stop"` to `$ErrorActionPreference = "Continue"` (PowerShell default). The `Run-Step` function already handles real failures via `exit $LASTEXITCODE`, so `"Stop"` was redundant and harmful here.

**Files changed:** `run_local.ps1`
