"""Village-level debris-flow screening for all 113 MIMU villages of Launglon.

  1. Buildings: VIDA combined footprints (all Microsoft here) + OSM buildings > 8 m from any VIDA centroid
     (the Taw Kye work showed OSM carries houses that Microsoft misses). Each building -> nearest village <= 2 km.
  2. Runout exposure: buildings inside reach zone A (open-slope alpha) / B (channel-only alpha) from 05_terrain.
  3. Melton ratio of every stream entering the village (500 m circle): M = relief / sqrt(area), length L.
     Wilford et al. (2004): debris flow M >= 0.6 and L < 2.7 km; debris flood M 0.3-0.6 (or M >= 0.6, L >= 2.7 km);
     flood M < 0.3.
  4. Forest loss 2001-2025 (Hansen v1.13) on the steep (>= 20 deg) part of those basins.
"""
import json
from collections import deque
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd, rasterio, affine
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
RAW, PROC = ROOT / "data/raw", ROOT / "data/processed"
R_OUT, R_BLD, CH_KM2, RES = 500.0, 2000.0, 0.05, 30.0
meta = json.load(open(PROC / "terrain_meta.json")); T = affine.Affine(*meta["transform"])
ter = np.load(PROC / "terrain.npz")
z, slope, fd, acc, zone, land = (ter[k] for k in ["z", "slope", "fd", "acc_km2", "zone", "land"])
ny, nx = z.shape
with rasterio.open(RAW / "ee/hansen.tif") as r: tc2000, lossyear = r.read(1), r.read(2)

twp = gpd.read_file(RAW / "mimu/township.geojson").to_crs(32647)
vil = gpd.read_file(RAW / "mimu/villages.geojson").to_crs(32647).reset_index(drop=True)
vil_ll = vil.to_crs(4326)

# ---------------- buildings
b = pd.read_csv(RAW / "ee/buildings_vida.csv")
bg = gpd.GeoDataFrame(b, geometry=gpd.points_from_xy(b.lon, b.lat), crs=4326).to_crs(32647)
osm = json.load(open(RAW / "osm/buildings.json", encoding="utf8"))["elements"]
og = gpd.GeoDataFrame(dict(src=["osm"] * len(osm)), crs=4326,
                      geometry=gpd.points_from_xy([e["center"]["lon"] for e in osm],
                                                  [e["center"]["lat"] for e in osm])).to_crs(32647)
d, _ = cKDTree(np.c_[bg.geometry.x, bg.geometry.y]).query(np.c_[og.geometry.x, og.geometry.y])
og = og[d > 8]
bld = gpd.GeoDataFrame(pd.concat([bg[["src", "geometry"]], og[["src", "geometry"]]], ignore_index=True), crs=32647)
bld = bld[bld.within(twp.geometry.iloc[0].buffer(100))].reset_index(drop=True)
bx, by = bld.geometry.x.values, bld.geometry.y.values
col = ((bx - T.c) / RES).astype(int); row = ((T.f - by) / RES).astype(int)
bld["zone"] = zone[row, col]
dv, iv = cKDTree(np.c_[vil.geometry.x, vil.geometry.y]).query(np.c_[bx, by])
bld["vil"] = np.where(dv <= R_BLD, iv, -1)
print(f"buildings in township: {len(bld)} (VIDA {int((bld.src != 'osm').sum())}, OSM-only {int((bld.src == 'osm').sum())}); "
      f"assigned to a village {int((bld.vil >= 0).sum())}")

# ---------------- D8 donors (CSR) for upstream traversal
DR = {64: -1, 128: -1, 1: 0, 2: 1, 4: 1, 8: 1, 16: 0, 32: -1}
DC = {64: 0, 128: 1, 1: 1, 2: 1, 4: 0, 8: -1, 16: -1, 32: -1}
rec = np.full(ny * nx, -1, np.int64)
rr, cc = np.indices((ny, nx))
for k in DR:
    m = fd == k
    r2, c2 = rr[m] + DR[k], cc[m] + DC[k]
    ok = (r2 >= 0) & (r2 < ny) & (c2 >= 0) & (c2 < nx)
    rec[(rr[m] * nx + cc[m])[ok]] = r2[ok] * nx + c2[ok]
has = rec >= 0
order = np.argsort(rec[has], kind="stable"); donors_of = np.nonzero(has)[0][order]; keys = rec[has][order]
idx = np.arange(ny * nx)
start = np.searchsorted(keys, idx); stop = np.searchsorted(keys, idx, side="right")
zf, sf, lf, accf = z.ravel(), slope.ravel(), lossyear.ravel(), acc.ravel()

def basin(o):
    """Upstream cells of outlet o and the longest flow distance to the outlet (m)."""
    cells, dist, q, Lmax = [o], {o: 0.0}, deque([o]), 0.0
    while q:
        c = q.popleft()
        for dnr in donors_of[start[c]:stop[c]]:
            diag = (dnr // nx != c // nx) and (dnr % nx != c % nx)
            dist[dnr] = dist[c] + RES * (1.4142 if diag else 1.0)
            Lmax = max(Lmax, dist[dnr]); cells.append(dnr); q.append(dnr)
    return np.array(cells), Lmax

def wilford(M, L):
    if M >= 0.6 and L < 2.7: return "debris flow"
    if M >= 0.3: return "debris flood"
    return "flood"

def inside(i, vx, vy):
    r_, c_ = divmod(i, nx)
    return np.hypot((c_ + .5) * RES + T.c - vx, T.f - (r_ + .5) * RES - vy) <= R_OUT

stream = ((acc >= CH_KM2) & land).ravel()
basins_rows, vrows = [], []
for v in vil.itertuples():
    vx, vy = v.geometry.x, v.geometry.y
    c0, r0 = int((vx - T.c) / RES), int((T.f - vy) / RES)
    k = int(R_OUT / RES) + 1
    outs = []
    for r_ in range(max(r0 - k, 0), min(r0 + k + 1, ny)):
        for c_ in range(max(c0 - k, 0), min(c0 + k + 1, nx)):
            i = r_ * nx + c_
            if not stream[i] or not inside(i, vx, vy): continue
            j = rec[i]
            # outlet = last cell of a stream inside the circle: it leaves the circle or joins a bigger stream
            if j < 0 or not inside(j, vx, vy) or accf[j] > 2.0 * accf[i] + 0.05: outs.append(i)
    best, steep_total, loss_steep, n_df, df_area = None, 0, 0, 0, 0.0
    for o in outs:
        cells, L = basin(o)
        A = cells.size * RES * RES / 1e6
        relief = (np.nanmax(zf[cells]) - zf[o]) / 1000.0
        M = relief / np.sqrt(A); cls = wilford(M, L / 1000)
        st = sf[cells] >= 20
        steep_total += int(st.sum()); loss_steep += int((st & (lf[cells] > 0)).sum())
        if cls == "debris flow": n_df += 1; df_area += A
        rr_, cc_ = divmod(o, nx)
        basins_rows.append(dict(vil=v.Index, outlet_x=round((cc_ + .5) * RES + T.c, 1), outlet_y=round(T.f - (rr_ + .5) * RES, 1),
                                area_km2=round(A, 3), relief_m=round(relief * 1000, 1), melton=round(M, 3),
                                length_km=round(L / 1000, 2), cls=cls, steep_frac=round(float(st.mean()), 3),
                                loss_frac=round(float((lf[cells] > 0).mean()), 3)))
        rank = {"debris flow": 2, "debris flood": 1, "flood": 0}[cls]
        if best is None or (rank, M) > (best[0], best[1]): best = (rank, M, A, relief * 1000, L / 1000, cls)
    vb = bld[bld.vil == v.Index]
    k2 = int(R_OUT / RES)
    win = (slice(max(r0 - k2, 0), r0 + k2 + 1), slice(max(c0 - k2, 0), c0 + k2 + 1))
    g = vil_ll.geometry.iloc[v.Index]
    vrows.append(dict(
        idx=v.Index, pcode=v.VLG_PCODE, name=v.VILLAGE, name_my=v.VLG_MMR, vt=v.VT, vt_pcode=v.VT_PCODE,
        lon=round(g.x, 5), lat=round(g.y, 5),
        bld=int(len(vb)), bld_A=int((vb.zone == 2).sum()), bld_B=int((vb.zone == 1).sum()),
        n_outlets=len(outs), n_df=n_df, df_area_km2=round(df_area, 3),
        melton_max=round(best[1], 3) if best else None, best_cls=best[5] if best else "no stream",
        best_area_km2=round(best[2], 3) if best else None, best_relief_m=round(best[3], 1) if best else None,
        best_len_km=round(best[4], 2) if best else None,
        slope_p90_500m=round(float(np.nanpercentile(slope[win], 90)), 1),
        relief_500m=round(float(np.nanmax(z[win]) - np.nanmin(z[win])), 1),
        steep_loss_pct=round(100 * loss_steep / steep_total, 1) if steep_total else None,
        steep_basin_km2=round(steep_total * RES * RES / 1e6, 3)))

V = pd.DataFrame(vrows)
V["pct_A"] = (100 * V.bld_A / V.bld.clip(lower=1)).round(1)

# Tiers rest on buildings inside the calibrated reach zone A. Melton is NOT used to rank: on the 30 m DSM it
# classes Taw Kye itself as "debris flood" (M 0.55), i.e. it misses the open-slope / hollow failures of 2026.
# A debris-flow-class basin entering the village only lifts tier 3 to tier 2.
def tier(r):
    if r.bld_A >= 50: return 1
    if r.bld_A >= 20 or (r.bld_A >= 5 and r.best_cls == "debris flow"): return 2
    if r.bld_A >= 5: return 3
    return 4
V["tier"] = V.apply(tier, axis=1)
V.to_csv(PROC / "villages_screening.csv", index=False, encoding="utf-8")
pd.DataFrame(basins_rows).to_csv(PROC / "village_basins.csv", index=False)
bld[["src", "zone", "vil"]].assign(x=bx.round(1), y=by.round(1)).to_csv(PROC / "buildings_zone.csv", index=False)
print("tiers", V.tier.value_counts().sort_index().to_dict())
print("buildings: total", len(bld), "zone A", int((bld.zone == 2).sum()), "zone B", int((bld.zone == 1).sum()))
print(V.sort_values(["tier", "bld_A"], ascending=[True, False]).head(25)[
    ["name", "vt", "bld", "bld_A", "bld_B", "pct_A", "best_cls", "melton_max", "best_area_km2", "n_df",
     "steep_loss_pct", "tier"]].to_string())
print(V[V.name == "Taw Kye"].T)
