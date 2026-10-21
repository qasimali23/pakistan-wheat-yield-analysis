import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, pandas as pd, numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from cleaning_pipeline import *
G, LG, O, LO, B, LB, R, GR = "#1F5C3A", "#DCEBDF", "#D98324", "#FBE8D0", "#2C6FA3", "#DCE9F4", "#B03A2E", "#EDEDED"
def box(ax, x, y, w, h, t, fc=LG, ec=G, fs=8, bold=False, tc="black"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1", fc=fc, ec=ec, lw=1.3)); ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", color=tc)
def arr(ax, a, b, col="#444", ls="-"): ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=13, lw=1.3, color=col, linestyle=ls))
# ---- Fig 1: cleaning protocol flowchart ----
f, ax = plt.subplots(figsize=(10, 6.2)); ax.set_xlim(0, 10); ax.set_ylim(0, 6.2); ax.axis("off")
ax.set_title("Figure 1: Data cleaning protocol for agricultural statistics", fontsize=12, fontweight="bold", color=G)
box(ax, 0.2, 4.9, 1.7, 0.9, "Raw data\nsaved unchanged\n(data/raw)", fc=LB, ec=B)
steps = [("1. Structure\nformat, types,\nduplicates,\nyear continuity", 2.3), ("2. Units and\nscale\nconvert, compare\nwith neighbours", 4.2), ("3. Missing and\nimplausible\nvalues: flag first", 6.1), ("4. Repair\nidentity first,\nthen interpolate", 8.0)]
for t, x in steps: box(ax, x, 4.8, 1.7, 1.1, t)
arr(ax, (1.9, 5.35), (2.3, 5.35))
for a, b in zip([2.3, 4.2, 6.1], [4.2, 6.1, 8.0]): arr(ax, (a + 1.7, 5.35), (b, 5.35))
steps2 = [("5. Consistency\narea x yield\n= production?", 8.0), ("6. Outlier screen\nIQR and MAD:\nflag, never delete", 6.1), ("7. Transform\nscale, log, detrend\n(fit on training\nyears only)", 4.2), ("8. Save and log\nprocessed data +\nchange log", 2.3)]
for t, x in steps2: box(ax, x, 2.7, 1.7, 1.2, t, fc=LB if x != 2.3 else G, ec=B if x != 2.3 else G, tc="black" if x != 2.3 else "white")
arr(ax, (8.85, 4.8), (8.85, 3.9))
for a, b in zip([8.0, 6.1, 4.2], [6.1, 4.2, 2.3]): arr(ax, (a, 3.3), (b + 1.7, 3.3))
box(ax, 0.2, 0.5, 4.3, 1.3, "Decision rule for every flagged value\n1. Is it a recording or unit error? Fix and log.\n2. Is it a real extreme (consistent with other\nvariables, official flag)? Keep and mark.\n3. Cannot tell? Keep original, mark for review.", fc="#FFF6D6", ec=O, fs=8)
box(ax, 5.0, 0.5, 4.8, 1.3, "Change log records, for every edit:\nrow and variable, old value, new value, rule used,\nand the date. Raw files are never overwritten, so\nevery cleaning step can be repeated or undone.", fc=GR, ec="#666", fs=8)
arr(ax, (2.3, 3.3), (2.0, 1.8), col=O, ls="--")
f.tight_layout(); f.savefig("figures/f1_cleaning_protocol.png", dpi=170); plt.close(f)

# ---- Fig 2: stress test before / after ----
truth = load_faostat("../data/raw/FAOSTAT_data_en_10-9-2026.csv")
bad = pd.read_csv("corrupted_data.csv"); bad["yield_kgha"] = pd.to_numeric(bad.yield_kgha.astype(str).str.replace(",", ""), errors="coerce")
bad = bad.drop_duplicates("Year"); cl = pd.read_csv("cleaned_from_corrupted.csv")
f, ax = plt.subplots(1, 2, figsize=(10, 4.2), sharey=False)
ax[0].plot(bad.Year, bad.yield_kgha, "o-", color=R, ms=3, lw=1); ax[0].set(title="Corrupted data (not cleaned)", xlabel="Year", ylabel="Yield (kg/ha)")
ax[0].annotate("x10 unit errors\n(1998, 2008)", (2003, 23500), fontsize=8, ha="center", color=R)
ax[1].plot(truth.Year, truth.yield_kgha, color="black", lw=2.2, label="True (real FAOSTAT)"); ax[1].plot(cl.Year, cl.yield_kgha, color=G, lw=1.4, ls="--", label="After cleaning")
ax[1].plot([1971], [cl.set_index("Year").yield_kgha[1971]], "o", color=O, ms=8, mfc="none", mew=2, label="+25% error not detected (1971)")
ax[1].set(title="After cleaning vs true values", xlabel="Year", ylabel="Yield (kg/ha)"); ax[1].legend(fontsize=7.5, loc="upper left")
f.suptitle("Figure 2: Stress test on a corrupted copy of the real wheat data", fontweight="bold", color=G); f.tight_layout(); f.savefig("figures/f2_stress_test.png", dpi=170); plt.close(f)

# ---- Fig 3: impact on forecasting ----
im = pd.read_csv("impact_rmse.csv", index_col=0).iloc[:, 0]
f, ax = plt.subplots(figsize=(7.5, 4.2)); bars = ax.bar(["Original\nreal data", "Corrupted,\nnot cleaned", "Corrupted,\nthen cleaned"], im.values, color=[G, R, "#5E9C76"])
for b_, v in zip(bars, im.values): ax.text(b_.get_x() + b_.get_width() / 2, v + 25, f"{v:.0f}", ha="center", fontsize=10)
ax.set(ylabel="Forecast error, RMSE (kg/ha)", title="Figure 3: Effect of data quality on a simple yield forecast", ylim=(0, 2200)); ax.title.set_fontweight("bold"); ax.title.set_color(G)
f.tight_layout(); f.savefig("figures/f3_impact.png", dpi=170); plt.close(f)
