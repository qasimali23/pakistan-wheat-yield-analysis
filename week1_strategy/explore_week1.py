"""Week 1 preliminary exploration of the FAOSTAT wheat data (Pakistan). Usage: python explore_week1.py <faostat_csv> <out_dir>"""
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
src, out = sys.argv[1], sys.argv[2]
G, LG, O, LO, B, LB, R, GR = "#1F5C3A", "#DCEBDF", "#D98324", "#FBE8D0", "#2C6FA3", "#DCE9F4", "#B03A2E", "#EDEDED"
d = pd.read_csv(src); w = d.pivot(index="Year", columns="Element", values="Value")
w = w.rename(columns={"Area harvested": "area_ha", "Production": "prod_t", "Yield": "yield_kgha"})
print(w.loc[[2023, 2024]])
# growth rates and decomposition
rows = []
for a, b in [(1961, 1980), (1980, 2000), (2000, 2024), (2010, 2024), (1961, 2024)]:
    n = b - a; ln = np.log(w.loc[b]) - np.log(w.loc[a]); tot = ln.prod_t
    rows.append((f"{a}-{b}", (w.prod_t[b] / w.prod_t[a]) ** (1 / n) * 100 - 100, (w.yield_kgha[b] / w.yield_kgha[a]) ** (1 / n) * 100 - 100, (w.area_ha[b] / w.area_ha[a]) ** (1 / n) * 100 - 100, 100 * ln.area_ha / tot, 100 * ln.yield_kgha / tot))
gr = pd.DataFrame(rows, columns=["period", "prod_cagr", "yield_cagr", "area_cagr", "area_share", "yield_share"]).round(2); print(gr); gr.to_csv("growth_decomposition.csv", index=False)
g = w.yield_kgha.pct_change() * 100
# Fig 1 yield with annotated events
f, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(w.index, w.yield_kgha, color=G, lw=2)
for y, txt, dy, dx in [(1968, "1968: +32.3%\nlargest rise", -450, 0), (1987, "1987: -17.1%\nlargest fall", -520, 0), (2024, "2024: 3,268 kg/ha\nhighest value", 0, -22)]:
    ax.annotate(txt, (y, w.yield_kgha[y]), xytext=(y + dx, w.yield_kgha[y] + dy - (150 if y == 2024 else 0)), fontsize=8, arrowprops=dict(arrowstyle="-", color="grey"))
    ax.plot([y], [w.yield_kgha[y]], "o", color=R)
ax.set(title="Figure 1: Pakistan wheat yield, 1961-2024 (FAOSTAT)", xlabel="Year", ylabel="Yield (kg/ha)"); ax.title.set_fontweight("bold"); ax.title.set_color(G)
f.tight_layout(); f.savefig(f"{out}/f1_yield_annotated.png", dpi=170); plt.close(f)
# Fig 2 decomposition
gg = gr.iloc[:4]; f, ax = plt.subplots(figsize=(8.4, 4.2)); x = np.arange(len(gg))
ax.bar(x, gg.yield_share, color=G, label="Yield"); ax.bar(x, gg.area_share, bottom=gg.yield_share, color=B, label="Area")
for i, r_ in gg.iterrows(): ax.text(i, r_.yield_share / 2, f"{r_.yield_share:.0f}%", ha="center", color="white", fontweight="bold"); ax.text(i, r_.yield_share + r_.area_share / 2, f"{r_.area_share:.0f}%", ha="center", color="white", fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels([f"{p}\n(prod. +{(np.exp((gg.area_share[i]+gg.yield_share[i])/100*0)*0):.0f}" if False else p for i, p in enumerate(gg.period)])
ax.set(ylabel="Share of production growth (%)", title="Figure 2: Where production growth came from, by period"); ax.title.set_fontweight("bold"); ax.title.set_color(G); ax.legend(loc="lower right"); ax.set_ylim(0, 105)
f.tight_layout(); f.savefig(f"{out}/f2_growth_sources.png", dpi=170); plt.close(f)
# Fig 3 methods map
def box(ax, x, y, wd, h, t, fc=LG, ec=G, fs=8, bold=False, tc="black"):
    ax.add_patch(FancyBboxPatch((x, y), wd, h, boxstyle="round,pad=0.02,rounding_size=0.1", fc=fc, ec=ec, lw=1.3)); ax.text(x + wd / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", color=tc)
def arr(ax, a, b): ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=13, lw=1.3, color="#444"))
f, ax = plt.subplots(figsize=(10, 5.4)); ax.set_xlim(0, 10); ax.set_ylim(0, 5.4); ax.axis("off")
ax.set_title("Figure 3: Analytical methods map (stage, methods, output)", fontsize=12, fontweight="bold", color=G)
st = [("1. Collect and integrate", "FAOSTAT CSV\nPBS / SBP tables\nNASA POWER API\njoin on province + year", "One analysis-ready\ntable"), ("2. Clean and prepare", "type / unit checks\nidentity check\nneighbour-ratio rule\nseason aggregation", "Cleaned table\n+ change log"), ("3. Explore (EDA)", "YoY change, CAGR\nlog decomposition\ndetrending, correlation\nline, scatter, box plots", "Patterns, anomalies,\nhypotheses"), ("4. Model", "baselines, trend\nRidge / Lasso\nRandom Forest\nARIMAX", "Yield forecasts"), ("5. Evaluate", "expanding-window test\nRMSE, MAE, MAPE\nbias, hit rate", "Accuracy vs\nbenchmark")]
for i, (t, m, o) in enumerate(st):
    x = 0.1 + i * 1.98
    box(ax, x, 3.9, 1.8, 0.9, t, fc=G, ec=G, fs=8.5, bold=True, tc="white"); box(ax, x, 1.95, 1.8, 1.7, m, fc=LB, ec=B, fs=8); box(ax, x, 0.55, 1.8, 1.1, o, fc=LO, ec=O, fs=8)
    if i < 4: arr(ax, (x + 1.8, 4.35), (x + 1.98, 4.35))
ax.text(0.1, 0.15, "Top row: stage.   Middle row: methods.   Bottom row: output of the stage.", fontsize=8, color="#555")
f.tight_layout(); f.savefig(f"{out}/f3_methods_map.png", dpi=170); plt.close(f)
