# run_local.ps1 - Run all CPU methods + post-compute locally (no GPU, no BigQuery).
#
# Skips:
#   - run_pipeline.py       (modeling CSVs already in data/)
#   - medgemma_generate     (needs GPU)
#   - ehrshot_bq_data.py    (EHRSHOT CSVs already in data/)
#   - nhanes steps          (no XPT files locally)
#
# Usage:
#   .\run_local.ps1                                    # no salt
#   .\run_local.ps1 -Salts seed1                       # one salt
#   .\run_local.ps1 -Salts seed1,seed2,seed3           # multiple salts - loops over each

param(
    [string[]]$Salts = @("")
)

$ErrorActionPreference = "Continue"
$PYTHON = "..\\.venv\\Scripts\\python.exe"
$DISEASES = @("ra", "crhn", "t1d", "t2d", "psr", "lup")
$START = Get-Date

function Run-Step {
    param([string]$Label, [string[]]$Cmd)
    Write-Host ""
    Write-Host "-- [$Label] $(Get-Date -Format 'HH:mm:ss') --" -ForegroundColor Cyan
    $t = Get-Date
    & $Cmd[0] $Cmd[1..($Cmd.Length-1)]
    if ($LASTEXITCODE -ne 0) {
        Write-Host "-- [$Label] FAILED (exit $LASTEXITCODE) --" -ForegroundColor Red
        exit $LASTEXITCODE
    }
    $elapsed = [int]((Get-Date) - $t).TotalSeconds
    Write-Host "-- [$Label] done in $([int]($elapsed/60))m $($elapsed%60)s --" -ForegroundColor Green
}

# -- Loop over salts ------------------------------------------------------------
foreach ($salt in $Salts) {
    $saltLabel = if ($salt) { $salt } else { "(default)" }
    Write-Host ""
    Write-Host "########################################" -ForegroundColor Magenta
    Write-Host "  SALT: $saltLabel" -ForegroundColor Magenta
    Write-Host "########################################" -ForegroundColor Magenta

    $saltArgs = if ($salt) { @("--split-salt", $salt) } else { @() }

    # -- Per-disease: methods ---------------------------------------------------
    foreach ($disease in $DISEASES) {
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Yellow
        Write-Host "  DISEASE: $disease  SALT: $saltLabel" -ForegroundColor Yellow
        Write-Host "========================================" -ForegroundColor Yellow

        Run-Step "sanity check   [$disease/$saltLabel]" (@($PYTHON, "-u", "src/sanity_check.py",           "--disease", $disease) + $saltArgs)
        Run-Step "method1        [$disease/$saltLabel]" (@($PYTHON, "-u", "src/method_threshold.py",        "--disease", $disease) + $saltArgs)
        Run-Step "method2        [$disease/$saltLabel]" (@($PYTHON, "-u", "src/method2_random_formula.py",  "--disease", $disease) + $saltArgs)
        Run-Step "method3 GP     [$disease/$saltLabel]" (@($PYTHON, "-u", "src/method3_gp.py",              "--disease", $disease) + $saltArgs)

        # Seeded GP: one run per seed file
        $seedDir = "data/llm_seeds/$disease"
        if (Test-Path $seedDir) {
            foreach ($seedFile in (Get-ChildItem "$seedDir/*.csv")) {
                $seedName = $seedFile.BaseName
                Run-Step "method3 seeded($seedName) [$disease/$saltLabel]" (@(
                    $PYTHON, "-u", "src/method3_gp.py",
                    "--disease", $disease,
                    "--seed-file", $seedFile.FullName
                ) + $saltArgs)
            }
        }

        Run-Step "method4 eval   [$disease/$saltLabel]" (@($PYTHON, "-u", "src/method4_llm.py", "evaluate", "--disease", $disease) + $saltArgs)
        Run-Step "cross-method   [$disease/$saltLabel]" (@($PYTHON, "-u", "src/cross_method_correlation.py", "--disease", $disease))
    }

    # -- Compare all methods (once per salt) -----------------------------------
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Yellow
    Write-Host "  COMPARE METHODS  SALT: $saltLabel" -ForegroundColor Yellow
    Write-Host "========================================" -ForegroundColor Yellow
    Run-Step "compare methods [$saltLabel]" @($PYTHON, "-u", "src/compare_methods.py")
}

# -- Post-compute: once per disease (salt-independent) -------------------------
# CI and evaluation read from master summaries which now contain all salts.
foreach ($disease in $DISEASES) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Yellow
    Write-Host "  POST-COMPUTE: $disease" -ForegroundColor Yellow
    Write-Host "========================================" -ForegroundColor Yellow

    Run-Step "mimic CI       [$disease]" @($PYTHON, "-u", "src/mimic_compute_ci.py",    "--disease", $disease, "--n-bootstrap", "1000")
    Run-Step "ehrshot eval   [$disease]" @($PYTHON, "-u", "src/ehrshot_evaluate.py",    "--disease", $disease)
    Run-Step "ehrshot CI     [$disease]" @($PYTHON, "-u", "src/ehrshot_compute_ci.py",  "--disease", $disease, "--n-bootstrap", "1000")
    Run-Step "dashboard data [$disease]" @($PYTHON, "-u", "src/build_dashboard_data.py","--disease", $disease)
    Run-Step "forest plot    [$disease]" @($PYTHON, "-u", "src/plot_ci_forest.py",      "--disease", $disease)
}

# -- Done ----------------------------------------------------------------------
$total = [int]((Get-Date) - $START).TotalSeconds
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  ALL DONE - $([int]($total/3600))h $([int](($total%3600)/60))m $($total%60)s" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
