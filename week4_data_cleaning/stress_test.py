"""Stress test: inject known errors into a copy of the REAL data, run the cleaning protocol, and measure recovery.
The corrupted data is simulated and used only to test the protocol; it is not real data."""
import sys, numpy as np, pandas as pd
from cleaning_pipeline import *
src = sys.argv[1]
truth = load_faostat(src)
bad = truth.copy().astype(object); bad["Year"] = truth.Year
def setv(year, col, fn): i = bad.index[bad.Year == year][0]; bad.at[i, col] = fn(truth.loc[i, col])
# injected problems (known positions)
inj = []
for y in [1975, 1989, 2003, 2011]: setv(y, "yield_kgha", lambda v: np.nan); inj.append((y, "yield_kgha", "missing"))
for y in [1995, 2016]: setv(y, "prod_t", lambda v: np.nan); inj.append((y, "prod_t", "missing"))
setv(1982, "area_ha", lambda v: np.nan); inj.append((1982, "area_ha", "missing"))
for y in [1998, 2008]: setv(y, "yield_kgha", lambda v: v * 10); inj.append((y, "yield_kgha", "unit error x10 (hg/ha)"))
setv(1993, "area_ha", lambda v: v * 10); inj.append((1993, "area_ha", "extra zero x10"))
setv(2019, "prod_t", lambda v: v / 10); inj.append((2019, "prod_t", "decimal slip /10"))
setv(2001, "yield_kgha", lambda v: f"{v:,.1f}"); inj.append((2001, "yield_kgha", "text with thousands separator"))
setv(1971, "yield_kgha", lambda v: v * 1.25); inj.append((1971, "yield_kgha", "small error +25% (not detectable by rule)"))
dup = bad[bad.Year == 2012]; bad = pd.concat([bad, dup]).sort_values("Year", kind="stable").reset_index(drop=True); inj.append((2012, "all", "duplicate row"))
bad.to_csv("corrupted_data.csv", index=False)
print("Injected problems:", len(inj))
cleaned, log = clean(bad)
print("\n".join(log))
# recovery accuracy
m = cleaned.merge(truth, on="Year", suffixes=("_c", "_t"))
for c in COLS:
    err = ((m[c + "_c"] - m[c + "_t"]).abs() / m[c + "_t"] * 100)
    print(f"{c}: rows exactly/closely recovered (<0.1% error): {(err < 0.1).sum()}/{len(m)}; max error {err.max():.1f}% in {int(m.Year[err.idxmax()])}")
print("rows:", len(cleaned), "years unique:", cleaned.Year.is_unique)
# impact on forecasting (same expanding-window linear trend baseline as Week 3; always scored against TRUE yield)
def trend_rmse(hist_df):
    h = hist_df.set_index("Year").yield_kgha.astype(float); t_true = truth.set_index("Year").yield_kgha; errs = []
    for t in range(1991, 2025):
        past = h[h.index < t].dropna(); k, b = np.polyfit(past.index, past.values, 1); errs.append(k * t + b - t_true[t])
    return float(np.sqrt(np.mean(np.square(errs))))
raw_uncleaned = bad.copy().drop_duplicates("Year")
for c in COLS: raw_uncleaned[c] = pd.to_numeric(raw_uncleaned[c], errors="coerce")  # what a careless analysis would do
res = {"Original real data": trend_rmse(truth), "Corrupted, not cleaned": trend_rmse(raw_uncleaned), "Corrupted, then cleaned": trend_rmse(cleaned)}
print({k: round(v, 1) for k, v in res.items()})
pd.Series(res).round(1).rename("RMSE_kgha").to_csv("impact_rmse.csv")
cleaned.to_csv("cleaned_from_corrupted.csv", index=False)
pd.DataFrame(inj, columns=["year", "variable", "problem"]).to_csv("injected_problems.csv", index=False)
