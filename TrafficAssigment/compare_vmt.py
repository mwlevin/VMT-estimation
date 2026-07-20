"""
VMT Comparison Report
=====================
Side-by-side comparison of county VMT from:
    (a) Simplified-network UE assignment   (county_vmt.csv)
    (b) Approach 5 - AON on full network   (county_vmt_full_aon.csv)
    (c) Approach 1 - HPMS urban add-back   (county_vmt_with_hpms_addback.csv)

Optional column:
    (d) MnDOT HPMS observed totals (CSV path supplied via --hpms-observed)

Outputs a single CSV (county_vmt_comparison.csv) and a printed table.

Usage:
    python compare_vmt.py --year 2024
    python compare_vmt.py --year 2024 --hpms-observed path/to/hpms_dvmt_2024.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys


def read_csv_kv(path: str, key_col: str, val_col: str) -> dict:
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                out[row[key_col]] = float(row[val_col])
            except (KeyError, ValueError):
                continue
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--year", "-y", required=True, choices=["2017", "2019", "2024"])
    ap.add_argument(
        "--hpms-observed",
        default=None,
        help="Optional CSV with columns county,vmt giving observed county DVMT",
    )
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    simp_dir = os.path.join(root, f"NetworkFiles{args.year}_WithZones_Simplified")
    full_dir = os.path.join(root, f"NetworkFiles{args.year}_Connected")

    paths = {
        "simplified": os.path.join(simp_dir, "county_vmt.csv"),
        "approach5":  os.path.join(full_dir, "county_vmt_full_aon.csv"),
        "approach1":  os.path.join(simp_dir, "county_vmt_with_hpms_addback.csv"),
    }

    sim   = read_csv_kv(paths["simplified"], "county", "vmt")
    aon   = read_csv_kv(paths["approach5"],  "county", "vmt")
    hpms  = read_csv_kv(paths["approach1"],  "county", "total_vmt")
    obs = {}
    if args.hpms_observed and os.path.exists(args.hpms_observed):
        obs = read_csv_kv(args.hpms_observed, "county", "vmt")

    counties = sorted(set(sim) | set(aon) | set(hpms) | set(obs))
    if not counties:
        print("ERROR: no input county VMT files found. Looked at:")
        for k, v in paths.items():
            print(f"  {k}: {v} ({'exists' if os.path.exists(v) else 'MISSING'})")
        sys.exit(1)

    out_csv = args.output or os.path.join(simp_dir, "county_vmt_comparison.csv")

    rows = []
    for c in counties:
        rows.append({
            "county": c,
            "simplified": sim.get(c, ""),
            "approach5_aon_full": aon.get(c, ""),
            "approach1_hpms_addback": hpms.get(c, ""),
            "hpms_observed": obs.get(c, ""),
        })

    cols = ["county", "simplified", "approach5_aon_full", "approach1_hpms_addback", "hpms_observed"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out_csv}")

    # Print top 15 by simplified VMT
    rows_sorted = sorted(
        rows,
        key=lambda r: (r["simplified"] if isinstance(r["simplified"], float) else 0),
        reverse=True,
    )
    print(f"\nTop 15 counties (year {args.year}):\n")
    print(f"{'County':<22}{'Simplified':>14}{'AON Full':>14}{'+HPMS':>14}{'Observed':>14}")
    print("-" * 78)
    for r in rows_sorted[:15]:
        def fmt(v):
            return f"{v:>14,.0f}" if isinstance(v, float) else f"{'-':>14}"
        print(
            f"{r['county']:<22}{fmt(r['simplified'])}{fmt(r['approach5_aon_full'])}"
            f"{fmt(r['approach1_hpms_addback'])}{fmt(r['hpms_observed'])}"
        )

    def total(d):
        return sum(d.values())
    print("\nTotals:")
    print(f"  Simplified            : {total(sim):>15,.0f}")
    print(f"  Approach 5 (AON full) : {total(aon):>15,.0f}")
    print(f"  Approach 1 (+HPMS)    : {total(hpms):>15,.0f}")
    if obs:
        print(f"  HPMS observed         : {total(obs):>15,.0f}")


if __name__ == "__main__":
    main()
