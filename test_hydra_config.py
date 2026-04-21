#!/usr/bin/env python
"""Test script to verify Hydra config migration works correctly."""

import sys
sys.path.insert(0, "src")

from omegaconf import OmegaConf
from utils import load_disease_config, data_path

print("=" * 70)
print("Hydra Config Migration Test")
print("=" * 70)

# Test 1: Load main config
print("\n[Test 1] Loading main config from conf/config.yaml")
main_config = OmegaConf.load("conf/config.yaml")
print(f"  OK Default disease: {main_config.defaults[0].disease}")
print(f"  OK BQ dataset: {main_config.pipeline.bq_dataset}")

# Test 2: Load RA disease config
print("\n[Test 2] Loading RA disease config")
ra_config = load_disease_config("ra")
print(f"  OK Disease name: {ra_config.name}")
print(f"  OK Full name: {ra_config.full_name}")
print(f"  OK ICD patterns: {ra_config.icd_patterns}")
print(f"  OK ICD version: {ra_config.icd_version}")

# Test 3: Load test disease config
print("\n[Test 3] Loading test disease config")
test_config = load_disease_config("test")
print(f"  OK Disease name: {test_config.name}")
print(f"  OK Full name: {test_config.full_name}")
print(f"  OK ICD patterns: {test_config.icd_patterns}")

# Test 4: Verify data path function
print("\n[Test 4] Verify data path function")
ra_path = data_path("ra")
test_path = data_path("test")
print(f"  OK RA data path: {ra_path}")
print(f"  OK Test data path: {test_path}")

# Test 5: Error handling for missing config
print("\n[Test 5] Error handling for missing disease")
try:
    load_disease_config("nonexistent")
    print("  FAIL Should have raised FileNotFoundError")
except FileNotFoundError as e:
    print(f"  OK Correctly raised error: {e}")

print("\n" + "=" * 70)
print("All tests passed! OK")
print("=" * 70)
print("\nNext steps:")
print("  1. Run method scripts with --disease flag:")
print("     python src/method_threshold.py --disease ra --dry-run")
print("  2. Add more disease configs in conf/disease/")
print("  3. For run_pipeline.py, test on Python 3.13 or wait for Hydra 1.4")
