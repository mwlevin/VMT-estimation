"""Build a side-by-side county VMT comparison CSV.

Combines:
  - OLD: county_vmt.csv  (simplified-network VMT, all 87 counties)
  - NEW: county_vmt_subarea_addback.csv  (sub-area-corrected VMT)

Output: county_vmt_old_vs_new.csv with columns
    county, old_vmt, new_vmt, delta, pct_change
"""

import argparse
import csv
import os


def load_old(path):
    out = {}
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            out[row["county"]] = float(row["vmt"])
    return out


def load_new(path):
    out = {}
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            out[row["county"]] = float(row["total_vmt"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", "-y", required=True, choices=["2017", "2019", "2024"])
    args = ap.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    simp_dir = os.path.join(repo_root, f"NetworkFiles{args.year}_WithZones_Simplified")
    old_csv = os.path.join(simp_dir, "county_vmt.csv")
    new_csv = os.path.join(simp_dir, "county_vmt_subarea_addback.csv")
    out_csv = os.path.join(simp_dir, "county_vmt_old_vs_new.csv")

    old = load_old(old_csv)
    new = load_new(new_csv)

    counties = sorted(set(old) | set(new))
    rows = []
    for c in counties:
        o = old.get(c, 0.0)
        n = new.get(c, 0.0)
        d = n - o
        pct = (d / o * 100) if o > 0 else 0.0
        rows.append((c, o, n, d, pct))
    rows.sort(key=lambda x: -x[2])

    old_total = sum(old.values())
    new_total = sum(new.values())

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["county", "old_vmt", "new_vmt", "delta", "pct_change"])
        for c, o, n, d, pct in rows:
            w.writerow([c, f"{o:.2f}", f"{n:.2f}", f"{d:.2f}", f"{pct:.2f}"])
        w.writerow(["TOTAL", f"{old_total:.2f}", f"{new_total:.2f}",
                    f"{new_total - old_total:.2f}",
                    f"{(new_total - old_total)/old_total*100:.2f}" if old_total > 0 else "0.00"])

    print(f"Wrote {out_csv}")
    print(f"  counties: {len(rows)}")
    print(f"  OLD total VMT: {old_total:,.0f}")
    print(f"  NEW total VMT: {new_total:,.0f}")
    print(f"  Delta:         {new_total - old_total:+,.0f}  "
          f"({(new_total-old_total)/old_total*100:+.2f}%)")
    print(f"\nTop 15 counties (by NEW VMT):")
    print(f"  {'County':<22}{'OLD':>14}{'NEW':>14}{'Δ':>14}{'% chg':>10}")
    print("  " + "-" * 74)
    for c, o, n, d, pct in rows[:15]:
        print(f"  {c:<22}{o:>14,.0f}{n:>14,.0f}{d:>+14,.0f}{pct:>+9.1f}%")


if __name__ == "__main__":
    main()
