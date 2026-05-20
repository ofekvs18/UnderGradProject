"""
nhanes_download.py — Download all NHANES XPT files needed by nhanes_data.py.

Downloads CBC and questionnaire component files for cycles G–J (2011–2018)
from the CDC NHANES website into data/nhanes/<cycle>/<file>.XPT.

Usage:
    python src/nhanes_download.py
    python src/nhanes_download.py --out-dir /custom/path
    python src/nhanes_download.py --cycles G H     # subset of cycles
    python src/nhanes_download.py --force           # re-download existing files
"""

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public"

# cycle letter → (year_range_label, start_year)
CYCLES = {
    "G": ("2011-2012", 2011),
    "H": ("2013-2014", 2013),
    "I": ("2015-2016", 2015),
    "J": ("2017-2018", 2017),
}

# All component files required by nhanes_data.py across all diseases
COMPONENTS = [
    "CBC",   # Complete Blood Count (lab)
    "MCQ",   # Medical Conditions Questionnaire (ra, lup, crhn, psr)
    "DIQ",   # Diabetes Questionnaire (t2d, t1d)
]


def download_file(url: str, dest: Path, force: bool) -> str:
    """
    Download url to dest. Returns one of: 'skipped', 'ok', 'missing', 'error'.
    """
    if dest.exists() and not force:
        return "skipped"
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, dest)
        return "ok"
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return "missing"
        return f"error({exc.code})"
    except Exception as exc:
        return f"error({exc})"


def main():
    parser = argparse.ArgumentParser(description="Download NHANES XPT files")
    parser.add_argument(
        "--out-dir", default="data/nhanes",
        help="Root directory for downloaded files (default: data/nhanes)",
    )
    parser.add_argument(
        "--cycles", nargs="+", default=list(CYCLES.keys()),
        choices=list(CYCLES.keys()),
        help="Cycle letters to download (default: all G H I J)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download files that already exist",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    cycles = {k: CYCLES[k] for k in args.cycles}

    print("=" * 70)
    print("NHANES Download")
    print(f"  Output dir : {out_dir.resolve()}")
    print(f"  Cycles     : { {k: v[0] for k, v in cycles.items()} }")
    print(f"  Components : {COMPONENTS}")
    print("=" * 70)

    counts = {"ok": 0, "skipped": 0, "missing": 0, "error": 0}

    for letter, (years, start_year) in cycles.items():
        print(f"\n[Cycle {letter} / {years}]")
        for component in COMPONENTS:
            filename = f"{component}_{letter}.XPT"
            url = f"{BASE_URL}/{start_year}/DataFiles/{filename}"
            dest = out_dir / letter / filename
            status = download_file(url, dest, args.force)

            if status == "ok":
                size_kb = dest.stat().st_size // 1024
                print(f"  Downloaded  {filename}  ({size_kb:,} KB)")
                counts["ok"] += 1
            elif status == "skipped":
                print(f"  Skipped     {filename}  (already exists)")
                counts["skipped"] += 1
            elif status == "missing":
                print(f"  Not found   {filename}  (404 — may not exist for this cycle)")
                counts["missing"] += 1
            else:
                print(f"  FAILED      {filename}  {status}")
                counts["error"] += 1

    total = sum(counts.values())
    print(f"\n{'='*70}")
    print(f"Done. {total} files checked.")
    print(f"  Downloaded : {counts['ok']}")
    print(f"  Skipped    : {counts['skipped']}  (already present; use --force to re-download)")
    print(f"  Not found  : {counts['missing']}  (404 from CDC; normal for some cycle/component combos)")
    print(f"  Errors     : {counts['error']}")

    if counts["error"]:
        sys.exit(1)

    print(f"\nNext step:")
    print(f"  python src/nhanes_data.py --nhanes-dir {out_dir} --disease ra")


if __name__ == "__main__":
    main()
