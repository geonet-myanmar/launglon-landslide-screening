"""Historical screening of the 2026 Taw Kye landslide paths with the Landsat archive (dry seasons 1988-2026).

For each exactly-mapped 2026 landslide (drone inventory) we take its 30 m footprint (cells >= 50 % covered) and track
  s_y = median NDVI(footprint) - median NDVI(ring 90-240 m around it)        (a local, self-referencing contrast)
  a_y = s_y - median_y(s_y)                                                    (anomaly vs the path's own record)
A disturbance on the same path (fresh scar, slide, clearing) pushes a_y negative. Because haze, drought and
composite gaps also push NDVI around, each anomaly is tested against a SHAPE-TRANSLATION PLACEBO: the same footprint
shape moved to N_PL random forested hillslope positions in the window with a similar slope, run through the identical
calculation. p_y = share of placebo placements with an anomaly at least as negative.
"""
import json
from pathlib import Path
import numpy as np, rasterio, geopandas as gpd, pandas as pd, affine
from rasterio.features import rasterize
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
RAW, PROC = ROOT / "data/raw", ROOT / "data/processed"
N_PL, SEED = 600, 7
rng = np.random.default_rng(SEED)

with rasterio.open(RAW / "ee/landsat_tk.tif") as r:
    L = r.read(); T = r.transform
YRS = np.arange(1988, 2027)
nd = L[0::2].astype("float64"); nobs = L[1::2]; nd[nd < -1] = np.nan
sh = nd.shape[1:]
meta = json.load(open(PROC / "terrain_meta.json")); ter = np.load(PROC / "terrain.npz")
TG = affine.Affine(*meta["transform"])
r0, c0 = int(round((TG.f - T.f) / 30)), int(round((T.c - TG.c) / 30))
win = (slice(r0, r0 + sh[0]), slice(c0, c0 + sh[1]))
slope = ter["slope"][win]; land = ter["land"][win]
with rasterio.open(RAW / "ee/hansen.tif") as r: hz = r.read()
tc2000, lossyear = hz[0][win], hz[1][win]

ls = gpd.read_file(RAW / "tawkye_2026/tawkye_landslide_2026.gpkg", layer="landslides")
T10 = T * affine.Affine.scale(0.1)
def frac(geom):
    m = rasterize([geom], out_shape=(sh[0] * 10, sh[1] * 10), transform=T10).astype("float32")
    return m.reshape(sh[0], 10, sh[1], 10).mean((1, 3))
allscar = ndi.binary_dilation(frac(ls.union_all()) > 0, iterations=3)
base = np.nanmedian(nd, axis=0)
eligible = land & np.isfinite(base) & (base > 0.6) & ~allscar

def series(rows, cols):
    """s_y for a footprint given by absolute (rows, cols)."""
    m = np.zeros(sh, bool); m[rows, cols] = True
    ring = ndi.binary_dilation(m, iterations=8) & ~ndi.binary_dilation(m, iterations=3) & land & ~allscar
    a = np.nanmedian(nd[:, m], axis=1); b = np.nanmedian(nd[:, ring], axis=1)
    return a - b

out, detail = [], {}
for row in ls.itertuples():
    f = frac(row.geometry); m = f >= 0.5
    rr, cc = np.nonzero(m)
    s = series(rr, cc); a = s - np.nanmedian(s)
    msl = float(np.nanmean(slope[m]))
    # placebo placements
    pl = []
    tries = 0
    while len(pl) < N_PL and tries < N_PL * 60:
        tries += 1
        dr, dc = rng.integers(-rr.min() + 10, sh[0] - rr.max() - 10), rng.integers(-cc.min() + 10, sh[1] - cc.max() - 10)
        R, C = rr + dr, cc + dc
        if not eligible[R, C].all() or abs(np.nanmean(slope[R, C]) - msl) > 4: continue
        sp = series(R, C); pl.append(sp - np.nanmedian(sp))
    pl = np.array(pl)
    p = (pl <= a[None, :]).mean(0)
    loss = lossyear[m]; lossy = sorted(set(int(v) + 2000 for v in loss[loss > 0]))
    detail[row.id] = dict(anom=np.round(a, 3).tolist(), p=np.round(p, 3).tolist(),
                          pl_p05=np.round(np.nanpercentile(pl, 5, axis=0), 3).tolist(),
                          pl_p50=np.round(np.nanpercentile(pl, 50, axis=0), 3).tolist(),
                          n_cells=int(m.sum()), n_placebo=int(len(pl)), slope_mean=round(msl, 1),
                          tc2000=round(float(np.mean(tc2000[m])), 1), loss_years=lossy,
                          loss_frac=round(float((loss > 0).mean()), 3))
    flag = [int(y) for y, pv in zip(YRS, p) if pv < 0.05]
    out.append(dict(id=row.id, n_cells=int(m.sum()), n_placebo=len(pl), slope_mean=round(msl, 1),
                    flagged_years=" ".join(map(str, flag)), anom_1998=round(float(a[YRS == 1998][0]), 3),
                    p_1998=round(float(p[YRS == 1998][0]), 3), hansen_loss_years=" ".join(map(str, lossy)),
                    hansen_loss_frac=round(float((loss > 0).mean()), 3)))
    print(out[-1])

df = pd.DataFrame(out); df.to_csv(PROC / "history_landsat.csv", index=False)
# coherence: how many paths are flagged in the same year, vs. the 5 % expected by chance per path
P = np.array([detail[k]["p"] for k in detail])
nflag = (P < 0.05).sum(0)
from scipy.stats import binom
coh = [dict(year=int(y), n_flag=int(n), p_binom=round(float(binom.sf(n - 1, len(P), 0.05)), 4) if n else 1.0)
       for y, n in zip(YRS, nflag)]
# regional context: share of all forested hillslope cells with an anomaly < -0.10 (vs the cell's own median)
hill = eligible & (slope >= 12)
an = nd - base
regional = [round(float(np.nanmean(an[k][hill] < -0.10)), 3) for k in range(len(YRS))]
scenes = json.load(open(RAW / "ee/landsat_tk_scenes.json"))
json.dump(dict(years=YRS.tolist(), paths=detail, coherence=coh, regional_hill_frac_lt_m010=regional,
               scenes={int(k): v for k, v in scenes.items()},
               clear_obs_median=[float(np.nanmedian(nobs[k][hill])) for k in range(len(YRS))]),
          open(PROC / "history_landsat.json", "w"), indent=0)
print("years with >=2 paths flagged:", [(c["year"], c["n_flag"], c["p_binom"]) for c in coh if c["n_flag"] >= 2])
