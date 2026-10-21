"""Week 5: evaluation metrics for the baseline yield forecasts (1991-2024), from the Week 3 forecast table.
Usage: python evaluation_metrics.py <faostat_csv> <baseline_forecasts.csv>"""
import sys, numpy as np, pandas as pd
d = pd.read_csv(sys.argv[1]); w = d.pivot(index="Year", columns="Element", values="Value")
f = pd.read_csv(sys.argv[2], index_col=0)
area = w.loc[f.index, "Area harvested"]; prod = w.loc[f.index, "Production"]
rows = []
for m in ["naive_last_year", "mean_last_3", "trend_expanding", "trend_rolling20"]:
    e = f[m] - f.actual; pe = e / f.actual * 100; mt = e * area / 1000 / 1e6
    rows.append(dict(method=m, RMSE_kgha=np.sqrt((e ** 2).mean()), MAE_kgha=e.abs().mean(), MAPE_pct=pe.abs().mean(), bias_kgha=e.mean(),
                     years_within_5pct=int((pe.abs() <= 5).sum()), years_within_10pct=int((pe.abs() <= 10).sum()), n_years=len(e),
                     max_abs_error_kgha=e.abs().max(), p90_abs_error_kgha=e.abs().quantile(.9), MAE_production_Mt=mt.abs().mean(),
                     MAE_pct_of_production=(mt.abs() * 1e6 / prod).mean() * 100))
res = pd.DataFrame(rows).round(2); print(res.T.to_string()); res.to_csv("evaluation_metrics.csv", index=False)
e = f.trend_expanding - f.actual
for a, b in [(1991, 2007), (2008, 2024)]:
    s = e.loc[a:b]; print(f"{a}-{b}: RMSE {np.sqrt((s**2).mean()):.1f}, MAPE {(s.abs()/f.actual.loc[a:b]).mean()*100:.2f}%")
