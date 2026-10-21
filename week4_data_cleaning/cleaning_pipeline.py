"""Week 4: reusable cleaning protocol for annual crop statistics (Year, area_ha, yield_kgha, prod_t).
Run on real FAOSTAT data, then on a deliberately corrupted copy to test the protocol."""
import numpy as np, pandas as pd

COLS = ["area_ha", "yield_kgha", "prod_t"]

# ---------- loading ----------
def load_faostat(path):
    d = pd.read_csv(path)
    w = d.pivot(index="Year", columns="Element", values="Value")
    w = w.rename(columns={"Area harvested": "area_ha", "Yield": "yield_kgha", "Production": "prod_t"})[COLS]
    return w.reset_index()

# ---------- step 1: structure and types ----------
def fix_types(df, log):
    df = df.copy()
    for c in COLS:
        if df[c].dtype == object:  # numbers stored as text, e.g. "2,490.7"
            bad = df[c].astype(str).str.contains(",", na=False).sum()
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "", regex=False), errors="coerce")
            log.append(f"types: converted text numbers in {c} ({bad} had thousands separators)")
    return df

def drop_duplicates(df, log):
    n = df.duplicated("Year").sum()
    if n: log.append(f"duplicates: removed {n} repeated year row(s), kept first")
    return df.drop_duplicates("Year", keep="first").sort_values("Year").reset_index(drop=True)

def check_years(df, log):
    full = set(range(int(df.Year.min()), int(df.Year.max()) + 1)); miss = sorted(full - set(df.Year))
    log.append(f"years: {df.Year.min()}-{df.Year.max()}, missing years: {miss if miss else 'none'}")

# ---------- step 2: implausible values via neighbour comparison ----------
def flag_scale_errors(df, log, lo=0.5, hi=2.0, half=3):
    """Flag a value if it is more than 2x or less than 0.5x the median of its +/-3 neighbouring years.
    Catches unit slips (x10, /10) and decimal errors but not small errors. Real extreme years stay unflagged."""
    flags = pd.DataFrame(False, index=df.index, columns=COLS)
    for c in COLS:
        s = df[c]
        for i in df.index:
            nb = pd.concat([s.iloc[max(0, i - half):i], s.iloc[i + 1:i + 1 + half]]).dropna()
            if pd.notna(s.iloc[i]) and len(nb) >= 3:
                ratio = s.iloc[i] / nb.median()
                if ratio > hi or ratio < lo: flags.loc[i, c] = True
    for c in COLS:
        for i in df.index[flags[c]]:
            log.append(f"scale error flagged: {c} in {int(df.Year[i])} = {df.at[i, c]:,.1f}")
    return flags

# ---------- step 3: repair using the identity production = area x yield / 1000, else interpolation ----------
def repair(df, flags, log):
    df = df.copy()
    for c in COLS: df.loc[flags[c], c] = np.nan
    for i in df.index:
        row = df.loc[i, COLS]; miss = row[row.isna()].index.tolist()
        if len(miss) == 1:
            m = miss[0]
            if m == "prod_t": df.at[i, m] = df.at[i, "area_ha"] * df.at[i, "yield_kgha"] / 1000
            elif m == "yield_kgha": df.at[i, m] = df.at[i, "prod_t"] / df.at[i, "area_ha"] * 1000
            else: df.at[i, m] = df.at[i, "prod_t"] / df.at[i, "yield_kgha"] * 1000
            log.append(f"repaired {m} in {int(df.Year[i])} from the other two variables")
    for c in ["area_ha", "yield_kgha"]:  # anything still missing: linear interpolation in time
        n = df[c].isna().sum()
        if n:
            yrs = df.loc[df[c].isna(), "Year"].tolist()
            df[c] = df[c].interpolate(limit_area="inside"); log.append(f"interpolated {c} in {yrs}")
    still = df.prod_t.isna()
    if still.any(): df.loc[still, "prod_t"] = df.area_ha * df.yield_kgha / 1000; log.append("production rebuilt from area x yield for remaining gaps")
    return df

# ---------- step 4: consistency report ----------
def consistency(df):
    return ((df.area_ha * df.yield_kgha / 1000) / df.prod_t - 1).abs().max() * 100

# ---------- step 5: statistical outlier screen (flag only, never delete) ----------
def outlier_screen(df):
    g = df.yield_kgha.pct_change() * 100
    q1, q3 = g.quantile([.25, .75]); iqr = q3 - q1
    iqr_flag = (g < q1 - 1.5 * iqr) | (g > q3 + 1.5 * iqr)
    mad = (g - g.median()).abs().median(); mz = 0.6745 * (g - g.median()) / mad
    return pd.DataFrame({"Year": df.Year, "yield_change_pct": g.round(1), "iqr_flag": iqr_flag, "modified_z": mz.round(2), "mad_flag": mz.abs() > 3.5})

# ---------- step 6: transformations ----------
def minmax(x, ref=None): ref = x if ref is None else ref; return (x - ref.min()) / (ref.max() - ref.min())
def zscore(x, ref=None): ref = x if ref is None else ref; return (x - ref.mean()) / ref.std()
def clean(df):
    log = []
    df = fix_types(df, log); df = drop_duplicates(df, log); check_years(df, log)
    flags = flag_scale_errors(df, log); df = repair(df, flags, log)
    log.append(f"consistency: max |area x yield - production| = {consistency(df):.4f}%")
    return df, log
