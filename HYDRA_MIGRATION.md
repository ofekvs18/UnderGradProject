# Hydra Config Migration - Verification Guide

## What Changed

Migrated from hardcoded disease configs to Hydra-based YAML configuration system. All disease-specific parameters now live in `conf/disease/*.yaml` files.

## Files Modified

### New Files
- `conf/config.yaml` - Main Hydra config with defaults
- `conf/disease/ra.yaml` - Rheumatoid Arthritis config
- `conf/disease/test.yaml` - Test disease config (for verification)
- `test_hydra_config.py` - Verification test script

### Updated Files
- `requirements.txt` - Added `hydra-core>=1.3`
- `src/utils.py` - Added `load_disease_config()` function
- `src/run_pipeline.py` - Migrated to `@hydra.main` decorator
- `src/method_threshold.py` - Added `--disease` arg, uses config loader
- `src/method2_random_formula.py` - Added `--disease` arg, uses config loader

### Still TODO
- None — all method files migrated

## Verification Tests

### Test 1: Core Config Loading
```bash
python test_hydra_config.py
```
Expected: All 5 tests pass

### Test 2: Method Scripts with --disease Flag
```bash
# Default (RA)
python src/method_threshold.py --dry-run

# Explicit disease selection
python src/method_threshold.py --disease ra --dry-run
python src/method_threshold.py --disease test --dry-run

# Method 2
python src/method2_random_formula.py --help
```

### Test 3: Config File Validation
```bash
python -c "from src.utils import load_disease_config; print(load_disease_config('ra'))"
```
Expected: Prints RA disease config

### Test 4: Multi-Disease Support
```bash
# Create new disease config
cat > conf/disease/dm1.yaml << EOF
name: dm1
full_name: "Type 1 Diabetes"
icd_patterns:
  - "250.01%"
  - "250.03%"
icd_version: 9
EOF

# Test it
python src/method_threshold.py --disease dm1 --dry-run
```
Expected: Prompt references "Type 1 Diabetes"

## Known Issues

### Python 3.14 Compatibility
**Issue:** Hydra 1.3.2 has incompatibility with Python 3.14's argparse
**Affects:** `src/run_pipeline.py` (uses `@hydra.main` decorator)
**Workaround:** Use Python 3.13 or wait for Hydra 1.4
**Status:** Method scripts work fine (use argparse directly), only run_pipeline.py affected

Error when running `python src/run_pipeline.py`:
```
TypeError: argument of type 'LazyCompletionHelp' is not a container or iterable
```

## Usage Examples

### Running Methods with Different Diseases

```bash
# Method 1 (Threshold) - RA (default)
python src/method_threshold.py

# Method 1 - Diabetes
python src/method_threshold.py --disease dm1

# Method 2 (Random) - RA
python src/method2_random_formula.py

# Method 2 - Custom disease
python src/method2_random_formula.py --disease test
```

### BigQuery Pipeline (requires Python 3.13)

```bash
# Dry run with default disease (ra)
python src/run_pipeline.py pipeline.dry_run=true

# Override disease at runtime
python src/run_pipeline.py disease=dm1 pipeline.dry_run=true
```

## Adding New Diseases

1. Create `conf/disease/<slug>.yaml`:
```yaml
name: crohns
full_name: "Crohn's Disease"
icd_patterns:
  - "K50%"
icd_version: 10
```

2. Run any method with `--disease crohns`:
```bash
python src/method_threshold.py --disease crohns --dry-run
```

No code changes needed!

## Rollback Plan

If issues arise, revert to hardcoded configs:
1. `git checkout HEAD -- src/utils.py src/run_pipeline.py`
2. Restore original method files from git
3. Remove `conf/` directory
4. Uninstall hydra: `pip uninstall hydra-core omegaconf`

## Next Steps

1. **Fix remaining method files** (method3_gp.py, method4_llm.py, sanity_check.py)
2. **Test on Python 3.13** for full run_pipeline.py verification
3. **Add more diseases** (Crohn's, Type 1 Diabetes, etc.)
4. **Update documentation** in README.md with new usage patterns
5. **Consider pyenv** for managing Python 3.13 alongside 3.14
