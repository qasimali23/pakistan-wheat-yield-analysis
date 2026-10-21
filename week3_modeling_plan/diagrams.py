import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, pandas as pd, numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
G, LG, O, LO, B, LB, R, GR = "#1F5C3A", "#DCEBDF", "#D98324", "#FBE8D0", "#2C6FA3", "#DCE9F4", "#B03A2E", "#EDEDED"
def box(ax, x, y, w, h, text, fc=LG, ec=G, fs=9, bold=False, tc="black"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", color=tc, wrap=True)
def arrow(ax, p1, p2, color="#444", style="-|>", rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=14, lw=1.4, color=color, connectionstyle=f"arc3,rad={rad}", linestyle=ls))
def canvas(w, h, title):
    f, ax = plt.subplots(figsize=(w, h)); ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off"); ax.set_title(title, fontsize=12, fontweight="bold", color=G, pad=8); return f, ax

# ---- Diagram 1: variable framework ----
f, ax = canvas(10, 6.4, "Figure 1: Variable framework for wheat yield prediction")
box(ax, 3.3, 2.7, 3.4, 1.0, "TARGET\nWheat yield (kg/ha)\nPakistan / Punjab, per crop year", fc=G, ec=G, fs=10, bold=True, tc="white")
groups = [
 (0.2, 4.7, "Climate (NASA POWER)", "Seasonal rainfall\nMean / min / max temperature\nHeat-stress days in grain filling\nSolar radiation", LB, B),
 (3.55, 4.9, "Soil and land", "Soil texture, organic carbon, pH\nIrrigated share of area\nland suitability class", LG, G),
 (6.9, 4.7, "Management and inputs (PBS / SBP)", "Fertiliser use per hectare\nVariety / seed type\nArea sown and cropping pattern", LO, O),
 (0.2, 0.5, "Market and policy (PBS / SBP)", "Wholesale wheat price\nSupport / procurement price\nFertiliser to wheat price ratio", LO, O),
 (3.55, 0.3, "Time and technology", "Year trend (improved seed,\npractices, mechanisation)\nPrevious-year yield (lag)", GR, "#666"),
 (6.9, 0.5, "Shocks and events", "Drought / flood / heat-wave flags\nRust or pest outbreak flags\n(where records exist)", GR, "#666"),
]
for x, y, t, items, fc, ec in groups:
    box(ax, x, y, 2.9, 1.35, "", fc=fc, ec=ec)
    ax.text(x + 1.45, y + 1.17, t, ha="center", va="center", fontsize=8.6, fontweight="bold", color=ec if ec != "#666" else "black")
    ax.text(x + 1.45, y + 0.58, items, ha="center", va="center", fontsize=7.8)
for x, y, *_ in groups:
    cx, cy = x + 1.45, y + (0 if y > 3 else 1.35)
    tx = 5.0 + (cx - 5.0) * 0.35; ty = 3.7 if y > 3 else 2.7
    arrow(ax, (cx, cy), (tx, ty))
f.tight_layout(); f.savefig("figures/d1_variable_framework.png", dpi=170); plt.close(f)

# ---- Diagram 2: data pipeline ----
f, ax = canvas(10, 5.2, "Figure 2: Data pipeline for the yield forecasting project")
src = [("FAOSTAT\nyield, area, production\n(annual, national)", 3.9), ("PBS / SBP tables\nprovince, irrigation,\nfertiliser, prices", 2.5), ("NASA POWER\ndaily weather\n(grid cells)", 1.1)]
for t, y in src: box(ax, 0.2, y, 1.9, 1.1, t, fc=LB, ec=B, fs=8)
steps = [("1. Ingest\nraw files saved\nunchanged (data/raw)", LG), ("2. Clean\nunits, missing values,\noutliers, reconcile\nsources", LG), ("3. Aggregate\ndaily weather to\nwheat-season features", LG), ("4. Join\nkey = province +\ncrop year", LG)]
xs = [2.65, 4.2, 5.75, 7.3]
for (t, fc), x in zip(steps, xs): box(ax, x, 2.45, 1.35, 1.35, t, fc=fc, ec=G, fs=7.6)
for t, y in src: arrow(ax, (2.1, y + 0.55), (2.65, 3.1))
for a, b in zip(xs[:-1], xs[1:]): arrow(ax, (a + 1.35, 3.12), (b, 3.12))
box(ax, 8.85, 2.45, 1.0, 1.35, "Analysis-ready\ntable\n(data/\nprocessed)", fc=G, ec=G, fs=7.6, bold=True, tc="white")
arrow(ax, (8.65, 3.12), (8.85, 3.12))
box(ax, 2.65, 0.45, 2.9, 1.0, "Quality checks at every step\nrange tests, area x yield = production,\nrow counts, logged corrections", fc=LO, ec=O, fs=7.8)
box(ax, 6.2, 0.45, 3.65, 1.0, "Outputs: EDA (Week 2), feature table, models (Week 3 plan),\nevaluation and executive summary (Week 5)", fc=GR, ec="#666", fs=7.8)
arrow(ax, (4.1, 1.45), (4.9, 2.45), color=O, ls="--"); arrow(ax, (9.35, 2.45), (8.2, 1.45))
f.tight_layout(); f.savefig("figures/d2_data_pipeline.png", dpi=170); plt.close(f)

# ---- Diagram 3: model development flow ----
f, ax = canvas(10, 5.6, "Figure 3: Model development workflow")
row1 = ["Define problem\ntarget = yield (kg/ha)\nhorizon = next season", "Feature engineering\nseason weather, lags,\nrates, trend term", "Feature selection\ncorrelation screen,\nregularisation,\nimportance", "Time-ordered split\nexpanding window\n(no shuffling)"]
x0 = [0.2, 2.7, 5.2, 7.7]
for t, x in zip(row1, x0): box(ax, x, 3.6, 2.1, 1.3, t, fc=LG, ec=G, fs=8)
for a, b in zip(x0[:-1], x0[1:]): arrow(ax, (a + 2.1, 4.25), (b, 4.25))
row2 = ["Baselines\nnaive, 3-year mean,\nlinear trend", "Candidate models\nRidge / Lasso,\nRandom Forest,\nGradient boosting, ARIMAX", "Tune and validate\ngrid search inside\ntraining years only", "Evaluate\nRMSE, MAE, MAPE\nvs baseline"]
x1 = [7.7, 5.2, 2.7, 0.2]
for t, x in zip(row2, x1): box(ax, x, 1.5, 2.1, 1.3, t, fc=LB if "Evaluate" not in t else LO, ec=B if "Evaluate" not in t else O, fs=8)
arrow(ax, (8.75, 3.6), (8.75, 2.8))
for a, b in zip(x1[:-1], x1[1:]): arrow(ax, (a, 2.15), (b + 2.1, 2.15))
box(ax, 3.4, 0.15, 3.2, 0.85, "Beats baseline?  Yes: interpret and report\nNo: revisit features / simplify model", fc="#FFF6D6", ec=O, fs=8, bold=True)
arrow(ax, (1.25, 1.5), (3.4, 0.6)); ax.plot([5.0,5.0,5.5],[1.0,3.3,3.3], color=R, lw=1.4, ls="--"); arrow(ax,(5.5,3.3),(5.5,3.6),color=R)
ax.text(5.08, 1.22, "no: iterate", color=R, fontsize=8, va="center")
f.tight_layout(); f.savefig("figures/d3_model_workflow.png", dpi=170); plt.close(f)

# ---- Diagram 4: validation scheme ----
f, ax = canvas(10, 3.9, "Figure 4: Expanding-window validation (time-ordered)")
ax.text(0.2, 3.35, "Each row = one forecasting round. Model is trained on all earlier years and tested on the next year only.", fontsize=8.5)
rows = [("Round 1", 5, 1), ("Round 2", 6, 1), ("Round 3", 7, 1), ("...", 8, 1), ("Last round", 9, 1)]
for i, (lab, tr, te) in enumerate(rows):
    y = 2.6 - i * 0.55
    ax.text(0.2, y + 0.18, lab, fontsize=8.5, va="center")
    ax.add_patch(FancyBboxPatch((1.7, y), tr * 0.75, 0.38, boxstyle="round,pad=0.01", fc=LG, ec=G)); ax.text(1.7 + tr * 0.375, y + 0.19, "train", ha="center", va="center", fontsize=8)
    ax.add_patch(FancyBboxPatch((1.7 + tr * 0.75 + 0.04, y), 0.7, 0.38, boxstyle="round,pad=0.01", fc=LO, ec=O)); ax.text(1.7 + tr * 0.75 + 0.39, y + 0.19, "test", ha="center", va="center", fontsize=8)
arrow(ax, (1.7, 0.1), (9.7, 0.1)); ax.text(5.7, -0.18, "time (crop years)", ha="center", fontsize=8.5)
ax.set_ylim(-0.4, 3.7)
f.tight_layout(); f.savefig("figures/d4_validation.png", dpi=170); plt.close(f)

# ---- Fig 5: baseline results from real data ----
r = pd.read_csv("baseline_forecasts.csv", index_col=0); m = pd.read_csv("baseline_metrics.csv")
f, ax = plt.subplots(1, 2, figsize=(10, 4.3), gridspec_kw={"width_ratios": [1.7, 1]})
ax[0].plot(r.index, r.actual, color="black", lw=2, label="Actual")
ax[0].plot(r.index, r.naive_last_year, color=R, lw=1.2, ls=":", label="Naive (last year)")
ax[0].plot(r.index, r.trend_expanding, color=G, lw=1.8, label="Linear trend (expanding)")
ax[0].set(title="One-step-ahead forecasts, 1991-2024", xlabel="Year", ylabel="Yield (kg/ha)"); ax[0].legend(fontsize=8)
lab = ["Naive", "3-yr mean", "Trend\n(expanding)", "Trend\n(20-yr)"]
ax[1].bar(lab, m.RMSE_kgha, color=[R, O, G, "#5E9C76"]); ax[1].set(title="RMSE (kg/ha), lower is better")
for i, v in enumerate(m.RMSE_kgha): ax[1].text(i, v + 2, f"{v:.0f}", ha="center", fontsize=9)
ax[1].tick_params(axis="x", labelsize=8)
f.suptitle("Figure 5: Baseline model results on real FAOSTAT yield data", fontweight="bold", color=G); f.tight_layout(); f.savefig("figures/d5_baseline_results.png", dpi=170); plt.close(f)
