"""Week 2 EDA: Pakistan wheat (FAOSTAT). Usage: python eda_wheat.py <faostat_csv> <out_dir>"""
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns

src = sys.argv[1] if len(sys.argv) > 1 else "FAOSTAT_data_en_10-9-2026.csv"
out = sys.argv[2] if len(sys.argv) > 2 else "figs"
raw = pd.read_csv(src)

# ---- 1. reshape: one row per year ----
w = raw.pivot(index="Year", columns="Element", values="Value")
w = w.rename(columns={"Area harvested": "area_ha", "Yield": "yield_kgha", "Production": "prod_t"})
w["area_mha"] = w.area_ha / 1e6
w["prod_mt"] = w.prod_t / 1e6

# ---- 2. integrity check: production should equal area x yield ----
w["calc_prod_t"] = w.area_ha * w.yield_kgha / 1000
w["check_pct"] = (w.calc_prod_t / w.prod_t - 1) * 100
print("Rows:", len(w), "| missing:", int(w[["area_ha","yield_kgha","prod_t"]].isna().sum().sum()),
      "| max |area x yield - production| %:", round(w.check_pct.abs().max(), 4))

# ---- 3. derived metrics ----
for c in ["area_ha", "yield_kgha", "prod_t"]:
    w[c + "_g"] = w[c].pct_change() * 100
w["yield_roll10"] = w.yield_kgha.rolling(10).mean()
k, b0 = np.polyfit(w.index, w.yield_kgha, 1)
w["yield_trend"] = k * w.index + b0
w["yield_resid"] = w.yield_kgha - w.yield_trend
w["resid_z"] = (w.yield_resid - w.yield_resid.mean()) / w.yield_resid.std()
print("Linear trend slope (kg/ha per year):", round(k, 1))
print("Residual outliers |z|>2:\n", w.loc[w.resid_z.abs() > 2, ["yield_kgha", "yield_trend", "resid_z"]].round(2))
w.to_csv(f"{out}/wheat_pakistan_analysis_table.csv")

sns.set_theme(style="whitegrid", font_scale=1.0)
G, O, R, B = "#1F5C3A", "#D98324", "#B03A2E", "#2C6FA3"

# ---- Fig 1: yield trend ----
f, a = plt.subplots(figsize=(9, 4.6))
a.plot(w.index, w.yield_kgha, color=G, lw=2, label="Yield")
a.plot(w.index, w.yield_roll10, color=O, lw=2, ls="--", label="10-year rolling mean")
a.plot(w.index, w.yield_trend, color="grey", lw=1.3, ls=":", label=f"Linear trend (+{k:.0f} kg/ha per year)")
a.set(title="Figure 1: Wheat yield in Pakistan, 1961-2024", xlabel="Year", ylabel="Yield (kg/ha)")
a.legend(); f.tight_layout(); f.savefig(f"{out}/fig1_yield_trend.png", dpi=170); plt.close(f)

# ---- Fig 2: area vs production ----
f, a = plt.subplots(figsize=(9, 4.6))
a.bar(w.index, w.area_mha, color="#9CC5A8", label="Area harvested (million ha)")
a.set(xlabel="Year", ylabel="Area harvested (million ha)", title="Figure 2: Wheat area and production, 1961-2024")
a2 = a.twinx(); a2.plot(w.index, w.prod_mt, color=R, lw=2.2, label="Production (million t)")
a2.set_ylabel("Production (million t)"); a2.grid(False)
h1, l1 = a.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
a.legend(h1 + h2, l1 + l2, loc="upper left"); f.tight_layout(); f.savefig(f"{out}/fig2_area_vs_production.png", dpi=170); plt.close(f)

# ---- Fig 3: year-on-year yield change, flagged ----
g = w.yield_kgha_g.dropna()
f, a = plt.subplots(figsize=(9, 4.6))
cols = [R if v <= -10 else (B if v >= 10 else "#A9A9A9") for v in g]
a.bar(g.index, g.values, color=cols)
a.axhline(0, color="black", lw=0.8)
for y in g.index[(g.abs() >= 10)]:
    a.annotate(str(y), (y, g[y]), ha="center", va="bottom" if g[y] > 0 else "top", fontsize=8)
a.set(title="Figure 3: Year-on-year change in wheat yield (red/blue = change of 10% or more)", xlabel="Year", ylabel="Change (%)")
f.tight_layout(); f.savefig(f"{out}/fig3_yield_yoy_change.png", dpi=170); plt.close(f)

# ---- Fig 4: area vs yield scatter ----
f, a = plt.subplots(figsize=(7.4, 5))
sc = a.scatter(w.area_mha, w.yield_kgha, c=w.index, cmap="viridis", s=45)
m, c0 = np.polyfit(w.area_mha, w.yield_kgha, 1)
xs = np.linspace(w.area_mha.min(), w.area_mha.max(), 50); a.plot(xs, m * xs + c0, color="grey", ls="--")
cb = f.colorbar(sc, ax=a); cb.set_label("Year")
r = w.area_mha.corr(w.yield_kgha)
a.set(title=f"Figure 4: Area harvested vs yield (r = {r:.2f})", xlabel="Area harvested (million ha)", ylabel="Yield (kg/ha)")
f.tight_layout(); f.savefig(f"{out}/fig4_area_vs_yield.png", dpi=170); plt.close(f)

# ---- Fig 5: decade summary ----
w["decade"] = (w.index // 10 * 10).astype(str) + "s"
dec = w.groupby("decade").agg(yield_mean=("yield_kgha", "mean"), vol=("yield_kgha_g", "std"))
f, ax = plt.subplots(1, 2, figsize=(10, 4.4))
ax[0].bar(dec.index, dec.yield_mean, color=G); ax[0].set(title="Average yield by decade", ylabel="kg/ha")
ax[1].bar(dec.index, dec.vol, color=O); ax[1].set(title="Volatility of yield change (std. dev., %)", ylabel="%")
f.suptitle("Figure 5: Decade comparison"); f.tight_layout(); f.savefig(f"{out}/fig5_decade_summary.png", dpi=170); plt.close(f)
print(dec.round(2))

# ---- Fig 6: correlation of year-on-year changes ----
cm = w[["area_ha_g", "yield_kgha_g", "prod_t_g"]].dropna().rename(columns={"area_ha_g": "Area change", "yield_kgha_g": "Yield change", "prod_t_g": "Production change"}).corr()
f, a = plt.subplots(figsize=(5.8, 4.6))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Greens", vmin=0, vmax=1, ax=a, cbar=False)
a.set_title("Figure 6: Correlation of yearly changes"); f.tight_layout(); f.savefig(f"{out}/fig6_change_correlation.png", dpi=170); plt.close(f)
print("Done.")
