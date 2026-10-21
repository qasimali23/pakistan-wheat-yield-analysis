import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
src = sys.argv[1]; fc_path = sys.argv[2]; out = sys.argv[3]
G, LG, O, LO, B, LB, R, GR = "#1F5C3A", "#DCEBDF", "#D98324", "#FBE8D0", "#2C6FA3", "#DCE9F4", "#B03A2E", "#EDEDED"
d = pd.read_csv(src); w = d.pivot(index="Year", columns="Element", values="Value")
f = pd.read_csv(fc_path, index_col=0)
def box(ax, x, y, wd, h, t, fc=LG, ec=G, fs=8, bold=False, tc="black"):
    ax.add_patch(FancyBboxPatch((x, y), wd, h, boxstyle="round,pad=0.02,rounding_size=0.1", fc=fc, ec=ec, lw=1.3)); ax.text(x + wd / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", color=tc)
def arr(ax, a, b): ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=13, lw=1.3, color="#444"))

# Fig 1: growth index
ix = w / w.iloc[0] * 100
fg, ax = plt.subplots(figsize=(9, 4.4))
ax.plot(ix.index, ix["Production"], color=R, lw=2.4, label=f"Production ({ix['Production'].iloc[-1]:.0f})")
ax.plot(ix.index, ix["Yield"], color=G, lw=2.4, label=f"Yield per hectare ({ix['Yield'].iloc[-1]:.0f})")
ax.plot(ix.index, ix["Area harvested"], color=B, lw=2.4, label=f"Area harvested ({ix['Area harvested'].iloc[-1]:.0f})")
ax.axhline(100, color="grey", lw=0.8, ls=":")
ax.set(title="Figure 1: Pakistan wheat, growth since 1961 (1961 = 100)", xlabel="Year", ylabel="Index (1961 = 100)"); ax.title.set_fontweight("bold"); ax.title.set_color(G); ax.legend(title="Value in 2024 in brackets", loc="upper left")
fg.tight_layout(); fg.savefig(f"{out}/g1_growth_index.png", dpi=170); plt.close(fg)

# Fig 2: forecast accuracy
m = f.trend_expanding; act = f.actual; err = m - act; area = w.loc[f.index, "Area harvested"]; errmt = err * area / 1000 / 1e6
fg, ax = plt.subplots(1, 2, figsize=(10, 4.3), gridspec_kw={"width_ratios": [1.5, 1]})
ax[0].fill_between(f.index, act * 0.95, act * 1.05, color="#BFD9C7", label="Within 5% of actual")
ax[0].plot(f.index, act, color="black", lw=2, label="Actual yield"); ax[0].plot(f.index, m, color=G, lw=1.8, ls="--", label="Trend forecast (made one year ahead)")
ax[0].set(title="Forecast vs actual, 1991-2024", xlabel="Year", ylabel="Yield (kg/ha)"); ax[0].legend(fontsize=7.5, loc="upper left")
ax[1].bar(f.index, errmt, color=[R if v < 0 else B for v in errmt]); ax[1].axhline(0, color="black", lw=0.8)
ax[1].set_ylim(-3, 2.2); ax[1].set(title="Forecast error in production (million t)", xlabel="Year", ylabel="Million tonnes (forecast - actual)")
ax[1].text(0.02, 0.04, "Red = forecast too low\nBlue = forecast too high", transform=ax[1].transAxes, fontsize=7.5)
fg.suptitle("Figure 2: How accurate is the simple trend forecast?", fontweight="bold", color=G); fg.tight_layout(); fg.savefig(f"{out}/g2_forecast_accuracy.png", dpi=170); plt.close(fg)

# Fig 3: project journey
fg, ax = plt.subplots(figsize=(10, 3.7)); ax.set_xlim(0, 10); ax.set_ylim(0, 3.7); ax.axis("off")
ax.set_title("Figure 3: The five-week project at a glance", fontsize=12, fontweight="bold", color=G)
wk = [("Week 1\nStrategy", "3 public data\nsources assessed;\nplan set"), ("Week 2\nExploration", "64 years of\nwheat data;\n6 charts"), ("Week 3\nModelling plan", "Variables, method,\n4 diagrams;\nbaseline test"), ("Week 4\nCleaning", "8-step protocol;\nstress test"), ("Week 5\nEvaluation", "Metrics,\ninsights,\nrecommendations")]
for i, (t, s) in enumerate(wk):
    x = 0.15 + i * 1.97
    box(ax, x, 2.2, 1.7, 1.0, t, fc=G if i == 4 else LG, ec=G, fs=9, bold=True, tc="white" if i == 4 else "black")
    box(ax, x, 0.55, 1.7, 1.35, s, fc=LB, ec=B, fs=8)
    if i < 4: arr(ax, (x + 1.7, 2.7), (x + 1.97, 2.7))
fg.tight_layout(); fg.savefig(f"{out}/g3_project_journey.png", dpi=170); plt.close(fg)

# Fig 4: roadmap
fg, ax = plt.subplots(figsize=(10, 4.3)); ax.set_xlim(0, 10); ax.set_ylim(0, 4.3); ax.axis("off")
ax.set_title("Figure 4: Recommended roadmap for further analysis", fontsize=12, fontweight="bold", color=G)
cols = [("Next 1 to 2 months", ["Add provincial data\n(PBS / SBP tables)", "Download NASA POWER\nwheat-season weather", "Check 2024 record\nagainst PBS"], LG, G),
        ("Months 3 to 6", ["Build weather + yield\ntable; test Ridge / Lasso /\nRandom Forest", "Compare with trend\nbaseline (must beat 3.4%)", "Add fertiliser and\nprice variables"], LB, B),
        ("Months 6 to 12", ["Mid-season and\npre-harvest forecasts", "Forecast ranges and\ndrought / heat flags", "Annual refit and\nmonitoring dashboard"], LO, O)]
for i, (t, items, fc, ec) in enumerate(cols):
    x = 0.2 + i * 3.3
    box(ax, x, 3.3, 3.0, 0.7, t, fc=ec, ec=ec, fs=10, bold=True, tc="white")
    for j, it in enumerate(items): box(ax, x, 2.2 - j * 0.95 + 0.0, 3.0, 0.8, it, fc=fc, ec=ec, fs=8)
    if i < 2: arr(ax, (x + 3.0, 3.65), (x + 3.3, 3.65))
fg.tight_layout(); fg.savefig(f"{out}/g4_roadmap.png", dpi=170); plt.close(fg)
print("figures done")
