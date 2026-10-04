"""Terrain analysis on the 30 m GLO-30 DEM: hydro-conditioning, D8 flow, slope, debris-flow runout reach.

Runout reach follows the energy-line (reach-angle) rule calibrated on the 2026 Taw Kye drone inventory
(published product, outputs/summary.json): sources are hillslope cells >= SRC_SLOPE; from each source we
walk the D8 flow path and keep a cell while atan(drop / path length) >= alpha.
  zone A: alpha_open = 17.6 deg (lowest reach angle of the open-slope 2026 flows)
  zone B: alpha_channel = 13.4 deg, only once the path is inside a channel (LS-04 stream-channel flow)
"""
import json
from pathlib import Path
import numpy as np, rasterio, geopandas as gpd, pandas as pd
if not hasattr(np, "in1d"): np.in1d = np.isin      # pysheds 0.5 still calls the removed numpy alias
from numba import njit
from pysheds.grid import Grid
from pysheds.sview import Raster
from rasterio.features import rasterize
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
RAW, PROC, OUT = ROOT / "data/raw", ROOT / "data/processed", ROOT / "outputs/rasters"
OUT.mkdir(parents=True, exist_ok=True)
summ = json.load(open(RAW / "tawkye_2026/summary.json"))
A_OPEN, A_CH, SRC_SLOPE = summ["runout"]["alpha_open"], summ["runout"]["alpha_channel"], summ["runout"]["src_slope"]
SRC_RELIEF = 40.0          # m within 210 m window: keeps sources on real hillslopes, not canopy steps in the DSM
CH_KM2 = 0.05              # channel initiation area
RES = 30.0
DIRMAP = (64, 128, 1, 2, 4, 8, 16, 32)            # N NE E SE S SW W NW (pysheds default)
DR = {64: -1, 128: -1, 1: 0, 2: 1, 4: 1, 8: 1, 16: 0, 32: -1}
DC = {64: 0, 128: 1, 1: 1, 2: 1, 4: 0, 8: -1, 16: -1, 32: -1}

src = RAW / "ee/dem.tif"
with rasterio.open(src) as r:
    z = r.read(1).astype("float64"); prof = r.profile; T = r.transform
z[~np.isfinite(z)] = np.nan
land = np.isfinite(z) & (z > 0.5)
twp = gpd.read_file(RAW / "mimu/township.geojson").to_crs(32647)
in_twp = rasterize([twp.geometry.iloc[0]], out_shape=z.shape, transform=T).astype(bool)

# ---- slope (Horn) on the raw DEM
zp = np.pad(np.where(np.isfinite(z), z, np.nanmin(z)), 1, mode="edge")
a, b, c = zp[:-2, :-2], zp[:-2, 1:-1], zp[:-2, 2:]
d, f = zp[1:-1, :-2], zp[1:-1, 2:]
g, h, i = zp[2:, :-2], zp[2:, 1:-1], zp[2:, 2:]
dzdx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8 * RES)
dzdy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8 * RES)
slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
slope[~land] = np.nan
relief = ndi.maximum_filter(np.nan_to_num(z, nan=-1e4), 7) - ndi.minimum_filter(np.nan_to_num(z, nan=1e4), 7)

# ---- hydro-conditioning + D8 (pysheds)
grid = Grid.from_raster(str(src))
dem = grid.read_raster(str(src))
# depression filling by morphological reconstruction (pysheds' fill_depressions does not compile on this numba)
from skimage.morphology import reconstruction
arr = np.asarray(dem, dtype="float64"); nod = ~np.isfinite(arr) | (arr <= 0.5)
arr[nod] = 0.0                                     # sea / nodata act as the outlet level
seed = arr.copy(); seed[1:-1, 1:-1] = arr.max(); seed[nod] = 0.0
filled = reconstruction(seed, arr, method="erosion")
dem = grid.resolve_flats(Raster(filled, viewfinder=dem.viewfinder))
fdir = grid.flowdir(dem, dirmap=DIRMAP)
acc = grid.accumulation(fdir, dirmap=DIRMAP)
zc = np.asarray(dem, dtype="float64"); fd = np.asarray(fdir).astype(np.int64); acc = np.asarray(acc, dtype="float64")
zc[~np.isfinite(z)] = np.nan
acc_km2 = acc * RES * RES / 1e6
chan = (acc_km2 >= CH_KM2) & land

lut_r = np.zeros(129, np.int64); lut_c = np.zeros(129, np.int64)
for k in DIRMAP: lut_r[k], lut_c[k] = DR[k], DC[k]

@njit(cache=True)
def runout(zc, fd, chan, src_r, src_c, lut_r, lut_c, tan_open, tan_ch, res):
    ny, nx = zc.shape
    zone = np.zeros((ny, nx), np.uint8)          # 2 = A (open), 1 = B (channel only)
    nsrc = np.zeros((ny, nx), np.int32)          # number of sources reaching (zone A)
    for k in range(src_r.size):
        r, c = src_r[k], src_c[k]; z0 = zc[r, c]; L = 0.0; inch = False
        zone[r, c] = 2
        for step in range(4000):
            dcode = fd[r, c]
            if dcode <= 0 or dcode > 128: break
            rr, cc = r + lut_r[dcode], c + lut_c[dcode]
            if rr < 0 or cc < 0 or rr >= ny or cc >= nx or not np.isfinite(zc[rr, cc]): break
            L += res * (1.4142135623730951 if (lut_r[dcode] != 0 and lut_c[dcode] != 0) else 1.0)
            t = (z0 - zc[rr, cc]) / L
            inch = inch or chan[rr, cc]
            if t >= tan_open:
                zone[rr, cc] = 2; nsrc[rr, cc] += 1
            elif inch and t >= tan_ch:
                if zone[rr, cc] == 0: zone[rr, cc] = 1
            else:
                break
            r, c = rr, cc
    return zone, nsrc

is_src = land & (slope >= SRC_SLOPE) & (relief >= SRC_RELIEF)
sr, sc = np.nonzero(is_src)
zone, nsrc = runout(zc, fd, chan, sr.astype(np.int64), sc.astype(np.int64), lut_r, lut_c,
                    np.tan(np.radians(A_OPEN)), np.tan(np.radians(A_CH)), RES)
# lateral spread: one cell either side of each path (fans spread wider than a D8 line)
zA = ndi.binary_dilation(zone == 2, np.ones((3, 3), bool)) & land
zB = ndi.binary_dilation(zone >= 1, np.ones((3, 3), bool)) & land & ~zA
zone2 = np.where(zA, 2, np.where(zB, 1, 0)).astype(np.uint8)

np.savez_compressed(PROC / "terrain.npz", z=z.astype("float32"), zc=zc.astype("float32"), slope=slope.astype("float32"),
                    fd=fd.astype("int16"), acc_km2=acc_km2.astype("float32"), zone=zone2, is_src=is_src,
                    land=land, in_twp=in_twp, relief=relief.astype("float32"))
json.dump(dict(transform=list(T)[:6], shape=z.shape, crs="EPSG:32647", A_OPEN=A_OPEN, A_CH=A_CH, SRC_SLOPE=SRC_SLOPE,
               SRC_RELIEF=SRC_RELIEF, CH_KM2=CH_KM2), open(PROC / "terrain_meta.json", "w"), indent=1)

p = prof.copy(); p.update(dtype="uint8", nodata=255, compress="deflate", count=1)
with rasterio.open(OUT / "runout_zone_30m.tif", "w", **p) as w: w.write(np.where(land, zone2, 255).astype("uint8"), 1)
p.update(dtype="float32", nodata=np.nan)
with rasterio.open(OUT / "slope_deg_30m.tif", "w", **p) as w: w.write(slope.astype("float32"), 1)

# ---- calibration check against the 2026 drone polygons
ls = gpd.read_file(RAW / "tawkye_2026/tawkye_landslide_2026.gpkg", layer="landslides")
rows = []
for rr_ in ls.itertuples():
    m = rasterize([rr_.geometry], out_shape=z.shape, transform=T, all_touched=True).astype(bool)
    rows.append(dict(id=rr_.id, cells=int(m.sum()), in_A=float((zone2[m] == 2).mean()), in_AB=float((zone2[m] >= 1).mean()),
                     src_cells=int(is_src[m].sum())))
cal = pd.DataFrame(rows); cal.to_csv(PROC / "runout_calibration.csv", index=False)
print(cal.to_string(index=False))
lt = land & in_twp
print(f"township land {lt.sum()*900/1e6:.0f} km2; sources {is_src[in_twp].sum()*900/1e6:.1f} km2; "
      f"zone A {(zone2[lt]==2).sum()*900/1e6:.1f} km2; zone B {(zone2[lt]==1).sum()*900/1e6:.1f} km2")
