-- ============================================================================
-- BIOMARKER PIPELINE: PARAMETERIZED COHORT EXTRACTION
-- Disease: {disease_full} | {lookback_days}-day lookback | MIMIC-IV v3.1
-- ICD-{icd_version} patterns: {icd_patterns_display}
-- ============================================================================
-- This template is parameterized via run_pipeline.py
-- Dataset: {bq_dataset}
-- Source: physionet-data.mimiciv_3_1_hosp
-- ============================================================================

-- ############################################################################
-- SETUP VALIDATION (run these first to confirm access)
-- ############################################################################

-- Test 1: MIMIC-IV access (expect ~500k)
SELECT COUNT(*) AS total_admissions
FROM `physionet-data.mimiciv_3_1_hosp.admissions`;

-- Test 2: Project access (expect 9 rows after adding RDW)
SELECT * FROM `{bq_dataset}.ref_cbc_tests`;

-- Test 3: Disease patient count
SELECT COUNT(DISTINCT subject_id) AS disease_patients
FROM `physionet-data.mimiciv_3_1_hosp.diagnoses_icd`
WHERE {icd_filter}
  AND icd_version = {icd_version};

-- Test 4: Join check (expect future dates due to MIMIC anonymization)
SELECT
  d.subject_id,
  d.hadm_id,
  d.icd_code,
  a.admittime
FROM `physionet-data.mimiciv_3_1_hosp.diagnoses_icd` d
JOIN `physionet-data.mimiciv_3_1_hosp.admissions` a
  ON d.subject_id = a.subject_id AND d.hadm_id = a.hadm_id
WHERE {icd_filter_d}
  AND d.icd_version = {icd_version}
LIMIT 5;

-- Test 5: CBC records exist (expect ~33M)
SELECT COUNT(*) AS cbc_records
FROM `physionet-data.mimiciv_3_1_hosp.labevents`
WHERE itemid IN (SELECT itemid FROM `{bq_dataset}.ref_cbc_tests`);


-- ############################################################################
-- CHECKPOINT 2: LABEL TABLE
-- ############################################################################
-- One row per patient. Cases = first disease admission. Controls = random admission.

CREATE OR REPLACE TABLE `{bq_dataset}.{disease}_cohort_labels` AS

WITH
disease_cases AS (
  SELECT
    d.subject_id,
    MIN(a.admittime) AS index_admittime
  FROM `physionet-data.mimiciv_3_1_hosp.diagnoses_icd` d
  JOIN `physionet-data.mimiciv_3_1_hosp.admissions` a
    ON d.subject_id = a.subject_id
    AND d.hadm_id = a.hadm_id
  WHERE {icd_filter_d}
    AND d.icd_version = {icd_version}
  GROUP BY d.subject_id
),

non_disease_patients AS (
  SELECT DISTINCT a.subject_id
  FROM `physionet-data.mimiciv_3_1_hosp.admissions` a
  WHERE a.subject_id NOT IN (SELECT subject_id FROM disease_cases)
),

control_anchors AS (
  SELECT
    a.subject_id,
    a.admittime AS index_admittime,
    ROW_NUMBER() OVER (
      PARTITION BY a.subject_id
      ORDER BY FARM_FINGERPRINT(CAST(a.hadm_id AS STRING))
    ) AS rn
  FROM `physionet-data.mimiciv_3_1_hosp.admissions` a
  WHERE a.subject_id IN (SELECT subject_id FROM non_disease_patients)
),

controls AS (
  SELECT subject_id, index_admittime
  FROM control_anchors
  WHERE rn = 1
)

SELECT subject_id, index_admittime, 1 AS is_case FROM disease_cases
UNION ALL
SELECT subject_id, index_admittime, 0 AS is_case FROM controls;


-- ----- CHECKPOINT 2 VERIFICATION -----

-- V2.1: Counts
SELECT
  is_case,
  COUNT(*) AS n_patients
FROM `{bq_dataset}.{disease}_cohort_labels`
GROUP BY is_case;

-- V2.2: No duplicates (expect zero rows)
SELECT subject_id, COUNT(*) AS cnt
FROM `{bq_dataset}.{disease}_cohort_labels`
GROUP BY subject_id
HAVING COUNT(*) > 1;


-- ############################################################################
-- CHECKPOINT 3: CBC FEATURE EXTRACTION WITH TEMPORAL GUARD
-- ############################################################################
-- Sources: (a) prior admissions within {lookback_days}-day lookback
--          (b) first 24h of index admission (routine admission panel)
-- Aggregation: last value per test per patient

CREATE OR REPLACE TABLE `{bq_dataset}.{disease}_cbc_features` AS

WITH

index_admissions AS (
  SELECT
    l.subject_id,
    l.index_admittime,
    a.hadm_id AS index_hadm_id
  FROM `{bq_dataset}.{disease}_cohort_labels` l
  JOIN `physionet-data.mimiciv_3_1_hosp.admissions` a
    ON l.subject_id = a.subject_id
    AND a.admittime = l.index_admittime
),

prior_cbc AS (
  SELECT
    lab.subject_id,
    ref.test_abbrev,
    lab.valuenum,
    lab.charttime
  FROM `physionet-data.mimiciv_3_1_hosp.labevents` lab
  JOIN `{bq_dataset}.ref_cbc_tests` ref
    ON lab.itemid = ref.itemid
  JOIN `{bq_dataset}.{disease}_cohort_labels` l
    ON lab.subject_id = l.subject_id
  JOIN index_admissions ia
    ON lab.subject_id = ia.subject_id
  WHERE lab.charttime < l.index_admittime
    AND lab.charttime >= DATETIME_SUB(l.index_admittime, INTERVAL {lookback_days} DAY)
    AND lab.hadm_id != ia.index_hadm_id
    AND lab.valuenum IS NOT NULL
),

early_index_cbc AS (
  SELECT
    lab.subject_id,
    ref.test_abbrev,
    lab.valuenum,
    lab.charttime
  FROM `physionet-data.mimiciv_3_1_hosp.labevents` lab
  JOIN `{bq_dataset}.ref_cbc_tests` ref
    ON lab.itemid = ref.itemid
  JOIN index_admissions ia
    ON lab.subject_id = ia.subject_id
    AND lab.hadm_id = ia.index_hadm_id
  WHERE lab.charttime >= ia.index_admittime
    AND lab.charttime < DATETIME_ADD(ia.index_admittime, INTERVAL {index_window_hours} HOUR)
    AND lab.valuenum IS NOT NULL
),

control_index AS (
  SELECT
    l.subject_id,
    l.index_admittime,
    a.hadm_id AS index_hadm_id
  FROM `{bq_dataset}.{disease}_cohort_labels` l
  JOIN `physionet-data.mimiciv_3_1_hosp.admissions` a
    ON l.subject_id = a.subject_id
    AND a.admittime = l.index_admittime
  WHERE l.is_case = 0
),

control_prior_cbc AS (
  SELECT
    lab.subject_id,
    ref.test_abbrev,
    lab.valuenum,
    lab.charttime
  FROM `physionet-data.mimiciv_3_1_hosp.labevents` lab
  JOIN `{bq_dataset}.ref_cbc_tests` ref
    ON lab.itemid = ref.itemid
  JOIN `{bq_dataset}.{disease}_cohort_labels` l
    ON lab.subject_id = l.subject_id
  JOIN control_index ci
    ON lab.subject_id = ci.subject_id
  WHERE l.is_case = 0
    AND lab.charttime < l.index_admittime
    AND lab.charttime >= DATETIME_SUB(l.index_admittime, INTERVAL {lookback_days} DAY)
    AND lab.hadm_id != ci.index_hadm_id
    AND lab.valuenum IS NOT NULL
),

control_early_cbc AS (
  SELECT
    lab.subject_id,
    ref.test_abbrev,
    lab.valuenum,
    lab.charttime
  FROM `physionet-data.mimiciv_3_1_hosp.labevents` lab
  JOIN `{bq_dataset}.ref_cbc_tests` ref
    ON lab.itemid = ref.itemid
  JOIN control_index ci
    ON lab.subject_id = ci.subject_id
    AND lab.hadm_id = ci.index_hadm_id
  WHERE lab.charttime >= ci.index_admittime
    AND lab.charttime < DATETIME_ADD(ci.index_admittime, INTERVAL {index_window_hours} HOUR)
    AND lab.valuenum IS NOT NULL
),

all_cbc AS (
  SELECT * FROM prior_cbc
  UNION ALL
  SELECT * FROM early_index_cbc
  UNION ALL
  SELECT * FROM control_prior_cbc
  UNION ALL
  SELECT * FROM control_early_cbc
),

last_values AS (
  SELECT
    subject_id,
    test_abbrev,
    valuenum,
    ROW_NUMBER() OVER (
      PARTITION BY subject_id, test_abbrev
      ORDER BY charttime DESC
    ) AS rn
  FROM all_cbc
),

filtered AS (
  SELECT subject_id, test_abbrev, valuenum
  FROM last_values
  WHERE rn = 1
)

SELECT
  l.subject_id,
  l.is_case,
  MAX(IF(f.test_abbrev = 'HCT', f.valuenum, NULL)) AS hct,
  MAX(IF(f.test_abbrev = 'HGB', f.valuenum, NULL)) AS hgb,
  MAX(IF(f.test_abbrev = 'MCH', f.valuenum, NULL)) AS mch,
  MAX(IF(f.test_abbrev = 'MCHC', f.valuenum, NULL)) AS mchc,
  MAX(IF(f.test_abbrev = 'MCV', f.valuenum, NULL)) AS mcv,
  MAX(IF(f.test_abbrev = 'PLT', f.valuenum, NULL)) AS plt,
  MAX(IF(f.test_abbrev = 'RBC', f.valuenum, NULL)) AS rbc,
  MAX(IF(f.test_abbrev = 'RDW', f.valuenum, NULL)) AS rdw,
  MAX(IF(f.test_abbrev = 'WBC', f.valuenum, NULL)) AS wbc
FROM `{bq_dataset}.{disease}_cohort_labels` l
LEFT JOIN filtered f ON l.subject_id = f.subject_id
GROUP BY l.subject_id, l.is_case;


-- ----- CHECKPOINT 3 VERIFICATION -----

-- V3.1: Coverage check
SELECT
  is_case,
  COUNT(*) AS total,
  COUNTIF(hgb IS NOT NULL) AS has_cbc,
  ROUND(100.0 * COUNTIF(hgb IS NOT NULL) / COUNT(*), 1) AS pct_with_cbc
FROM `{bq_dataset}.{disease}_cbc_features`
GROUP BY is_case;

-- V3.2: Leakage check — verify all selected labs have valid timing
WITH all_cbc_sources AS (
  SELECT
    lab.subject_id, ref.test_abbrev, lab.valuenum, lab.charttime,
    'prior_admission' AS source
  FROM `physionet-data.mimiciv_3_1_hosp.labevents` lab
  JOIN `{bq_dataset}.ref_cbc_tests` ref ON lab.itemid = ref.itemid
  JOIN `{bq_dataset}.{disease}_cohort_labels` l ON lab.subject_id = l.subject_id
  JOIN `physionet-data.mimiciv_3_1_hosp.admissions` a
    ON l.subject_id = a.subject_id AND a.admittime = l.index_admittime
  WHERE l.is_case = 1
    AND lab.charttime < l.index_admittime
    AND lab.charttime >= DATETIME_SUB(l.index_admittime, INTERVAL {lookback_days} DAY)
    AND lab.hadm_id != a.hadm_id
    AND lab.valuenum IS NOT NULL

  UNION ALL

  SELECT
    lab.subject_id, ref.test_abbrev, lab.valuenum, lab.charttime,
    'index_24h' AS source
  FROM `physionet-data.mimiciv_3_1_hosp.labevents` lab
  JOIN `{bq_dataset}.ref_cbc_tests` ref ON lab.itemid = ref.itemid
  JOIN `{bq_dataset}.{disease}_cohort_labels` l ON lab.subject_id = l.subject_id
  JOIN `physionet-data.mimiciv_3_1_hosp.admissions` a
    ON l.subject_id = a.subject_id AND a.admittime = l.index_admittime
  WHERE l.is_case = 1
    AND lab.hadm_id = a.hadm_id
    AND lab.charttime >= a.admittime
    AND lab.charttime < DATETIME_ADD(a.admittime, INTERVAL {index_window_hours} HOUR)
    AND lab.valuenum IS NOT NULL
),

ranked AS (
  SELECT *, ROW_NUMBER() OVER (
    PARTITION BY subject_id, test_abbrev ORDER BY charttime DESC
  ) AS rn
  FROM all_cbc_sources
)

SELECT
  r.subject_id, l.index_admittime, r.charttime,
  DATETIME_DIFF(r.charttime, l.index_admittime, HOUR) AS hours_relative,
  r.source, r.valuenum AS value
FROM ranked r
JOIN `{bq_dataset}.{disease}_cohort_labels` l ON r.subject_id = l.subject_id
WHERE r.rn = 1 AND r.test_abbrev = 'HGB'
ORDER BY FARM_FINGERPRINT(CAST(r.subject_id AS STRING))
LIMIT 20;


-- ############################################################################
-- CHECKPOINT 4: PATIENT-LEVEL TRAIN/TEST SPLIT
-- ############################################################################
-- 80/20 deterministic split. Only patients with CBC data.
-- This table is IMMUTABLE after creation.

CREATE OR REPLACE TABLE `{bq_dataset}.{disease}_splits` AS
SELECT
  subject_id,
  is_case,
  CASE
    WHEN MOD(ABS(FARM_FINGERPRINT(CAST(subject_id AS STRING))), 5) = 0
    THEN 'test'
    ELSE 'train'
  END AS split
FROM `{bq_dataset}.{disease}_cbc_features`
WHERE hgb IS NOT NULL;


-- ----- CHECKPOINT 4 VERIFICATION -----

-- V4.1: Split sizes and class balance
SELECT
  split,
  is_case,
  COUNT(*) AS n
FROM `{bq_dataset}.{disease}_splits`
GROUP BY split, is_case
ORDER BY split, is_case;


-- ############################################################################
-- CHECKPOINT 5: EXPORT FOR MODELING
-- ############################################################################
-- [EXPORT]
-- This query will be downloaded as {disease}_modeling_data.csv

SELECT
  f.subject_id,
  f.is_case,
  s.split,
  f.hct, f.hgb, f.mch, f.mchc, f.mcv, f.plt, f.rbc, f.rdw, f.wbc
FROM `{bq_dataset}.{disease}_cbc_features` f
JOIN `{bq_dataset}.{disease}_splits` s
  ON f.subject_id = s.subject_id
WHERE f.hgb IS NOT NULL;
