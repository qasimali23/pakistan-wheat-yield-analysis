"""Week 3: baseline one-step-ahead yield forecasts on FAOSTAT wheat data (expanding window, time-ordered)."""
import sys, numpy as np, pandas as pd
src = sys.argv[1]
d = pd.read_csv(src)
y = d[d.Element == "Yield"].set_index("Year")["Value"].sort_index()
first_test = 1991
rows = []
for t in y.index[y.index >= first_test]:
    hist = y[y.index < t]
    naive = hist.iloc[-1]
    ma3 = hist.iloc[-3:].mean()
    k, b = np.polyfit(hist.index, hist.values, 1); trend_all = k * t + b
    h20 = hist.iloc[-20:]; k2, b2 = np.polyfit(h20.index, h20.values, 1); trend20 = k2 * t + b2
    rows.append((t, y[t], naive, ma3, trend_all, trend20))
r = pd.DataFrame(rows, columns=["Year", "actual", "naive_last_year", "mean_last_3", "trend_expanding", "trend_rolling20"]).set_index("Year")
out = []
for m in r.columns[1:]:
    e = r[m] - r.actual
    out.append((m, np.sqrt((e ** 2).mean()), e.abs().mean(), (e.abs() / r.actual).mean() * 100))
res = pd.DataFrame(out, columns=["method", "RMSE_kgha", "MAE_kgha", "MAPE_pct"]).round(1)
print("Test years:", r.index.min(), "to", r.index.max(), "n =", len(r)); print(res.to_string(index=False))
r.round(1).to_csv("baseline_forecasts.csv"); res.to_csv("baseline_metrics.csv", index=False)
