"""Rainfall context at Taw Kye: the 26-27 Sep 2026 trigger vs the satellite record, and the reported 1997 episode.

IMERG V07 (0.1 deg, mean of the two cells spanning 13.7-13.9 N, 98.1-98.2 E), Myanmar-time days, 1998 - 2 Oct 2026.
CHIRPS v2 (0.05 deg, Taw Kye cell), 1981 - Aug 2026: the only daily record reaching back to 1997.
Return periods: Gumbel fitted (L-moments) to the 1998-2025 IMERG annual maxima, 2026 excluded from the fit.
"""
import json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, PROC = ROOT / "data/raw", ROOT / "data/processed"
im = pd.read_csv(RAW / "ee/rain_imerg.csv", parse_dates=["date"]).set_index("date").mm
ch = pd.read_csv(RAW / "ee/rain_chirps.csv", parse_dates=["date"]).set_index("date").mm

def gumbel_T(x, fit):
    """Return period of x under a Gumbel fitted to `fit` by L-moments."""
    s = np.sort(fit.values); n = s.size
    b0 = s.mean(); b1 = np.sum((np.arange(n) / (n - 1)) * s) / n
    l1, l2 = b0, 2 * b1 - b0
    a = l2 / np.log(2); u = l1 - 0.5772 * a
    F = np.exp(-np.exp(-(x - u) / a))
    return 1 / (1 - F), dict(u=u, alpha=a)

out = dict(source_imerg="NASA/GPM_L3/IMERG_V07", source_chirps="UCSB-CHG/CHIRPS/DAILY")
win = {}
for k in (1, 2, 3, 5):
    r = im.rolling(k).sum()
    am = r.groupby(r.index.year).max()
    am_fit = am.loc[1998:2025]
    ev = r["2026-09-20":"2026-10-02"]
    x = float(ev.max())
    T, par = gumbel_T(x, am_fit)
    prev = am_fit.sort_values(ascending=False)
    win[f"{k}d"] = dict(event_mm=round(x, 1), event_end=str(ev.idxmax().date()), prev_max_mm=round(float(prev.iloc[0]), 1),
                        prev_max_year=int(prev.index[0]), ratio=round(x / float(prev.iloc[0]), 2),
                        gumbel_T_years=round(float(T), 0), median_annual_max=round(float(am_fit.median()), 1),
                        annual_max=[[int(y), round(float(v), 1)] for y, v in am.items()])
out["windows"] = win
out["sep2026_daily"] = [[str(d.date()), round(float(v), 1)] for d, v in im["2026-09-01":"2026-10-02"].items()]
out["sep2026_total"] = round(float(im["2026-09-01":"2026-09-30"].sum()), 1)
sep = im[im.index.month == 9].groupby(im[im.index.month == 9].index.year).sum()
out["sep_total_rank"] = int(sep.rank(ascending=False)[2026]); out["sep_years"] = int(sep.size)
out["sep_prev_max"] = [int(sep.loc[:2025].idxmax()), round(float(sep.loc[:2025].max()), 1)]

# CHIRPS: 1997 in context
c1 = ch.groupby(ch.index.year).max(); c3 = ch.rolling(3).sum().groupby(ch.index.year).max()
ann = ch.groupby(ch.index.year).sum()
full = [y for y in ann.index if ch[str(y)].size >= 365]
ann = ann.loc[full]
out["chirps_1997"] = dict(max1d=round(float(c1[1997]), 1), rank1d=int(c1.rank(ascending=False)[1997]), n_years=int(c1.size),
                          max3d=round(float(c3[1997]), 1), rank3d=int(c3.rank(ascending=False)[1997]),
                          annual=round(float(ann[1997]), 0), rank_annual=int(ann.rank(ascending=False)[1997]), n_annual=int(ann.size),
                          top_days=[[str(d.date()), round(float(v), 1)] for d, v in ch["1997"].nlargest(5).items()])
out["chirps_annual_max1d"] = [[int(y), round(float(v), 1)] for y, v in c1.items()]
out["chirps_last"] = str(ch.index.max().date())
json.dump(out, open(PROC / "rain.json", "w"), indent=0)
for k, v in win.items():
    print(k, {kk: vv for kk, vv in v.items() if kk != "annual_max"})
print("Sep 2026 total", out["sep2026_total"], "rank", out["sep_total_rank"], "/", out["sep_years"], "prev max", out["sep_prev_max"])
print("CHIRPS 1997", out["chirps_1997"])
