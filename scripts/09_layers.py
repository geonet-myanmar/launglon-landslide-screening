"""Map layers for the dashboard, warped from the 30 m UTM grid to Web Mercator at ~30 m (native-equivalent).

  layers/relief_light.webp, relief_dark.webp  hillshade + land/sea tint, one per theme (no tile server needed)
  layers/runout.png                           reach zone A / B, RGBA
  layers/vectors.json                         township, village tracts, envelopes, 2026 scars, buildings in zone A/B
"""
import json, io
from pathlib import Path
import numpy as np, rasterio, affine, geopandas as gpd, pandas as pd
from rasterio.warp import reproject, Resampling, calculate_default_transform, transform_bounds
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RAW, PROC = ROOT / "data/raw", ROOT / "data/processed"
OUT = PROC / "layers"; OUT.mkdir(exist_ok=True)
meta = json.load(open(PROC / "terrain_meta.json")); T = affine.Affine(*meta["transform"])
ter = np.load(PROC / "terrain.npz")
z, zone, land = ter["z"].astype("float64"), ter["zone"], ter["land"]
ny, nx = z.shape

# hillshade (az 315, alt 45)
zz = np.where(np.isfinite(z), z, 0)
gy, gx = np.gradient(zz * 2.0, 30.0)          # z-factor 2: the peninsula relief is low
slp = np.arctan(np.hypot(gx, gy)); asp = np.arctan2(-gx, gy)
az, alt = np.radians(315), np.radians(45)
hs = np.sin(alt) * np.cos(slp) + np.cos(alt) * np.sin(slp) * np.cos(az - asp - np.pi / 2)
hs = np.clip(hs, 0, 1)

# target grid in EPSG:3857
src_crs = "EPSG:32647"
bounds = (T.c, T.f + T.e * ny, T.c + T.a * nx, T.f)
dT, dw, dh = calculate_default_transform(src_crs, "EPSG:3857", nx, ny, *bounds, resolution=30.0)
def warp(a, rs, dtype="float32", fill=np.nan):
    dst = np.full((dh, dw), fill, dtype)
    reproject(a.astype(dtype), dst, src_transform=T, src_crs=src_crs, dst_transform=dT, dst_crs="EPSG:3857",
              resampling=rs, src_nodata=fill if dtype == "float32" else None, dst_nodata=fill if dtype == "float32" else None)
    return dst
hs_m = warp(hs, Resampling.bilinear)
land_m = warp(land.astype("float32"), Resampling.nearest)
zone_m = warp(zone.astype("float32"), Resampling.nearest)
w, s, e, n = dT.c, dT.f + dT.e * dh, dT.c + dT.a * dw, dT.f
lon0, lat0, lon1, lat1 = transform_bounds("EPSG:3857", "EPSG:4326", w, s, e, n)
print("overlay", dw, "x", dh, "px; bounds", lon0, lat0, lon1, lat1)

def tint(land_rgb, sea_rgb, shade_dark, shade_lite):
    L = np.nan_to_num(land_m) > 0.5
    h = np.nan_to_num(hs_m, nan=0.7)
    base = np.where(L[..., None], np.array(land_rgb, float), np.array(sea_rgb, float))
    # shading relative to a flat surface (hs of flat = sin(45) = 0.707)
    k = (h - 0.707)[..., None]
    out = np.where(k < 0, base + k / 0.707 * (base - np.array(shade_dark, float)), base + k / 0.293 * (np.array(shade_lite, float) - base) * 0.8)
    out = np.where(L[..., None], out, base)
    return np.clip(out, 0, 255).astype("uint8")
Image.fromarray(tint((233, 236, 228), (210, 223, 229), (96, 110, 102), (252, 253, 250))).save(OUT / "relief_light.webp", quality=82, method=6)
Image.fromarray(tint((33, 40, 37), (17, 26, 31), (4, 7, 6), (88, 102, 95))).save(OUT / "relief_dark.webp", quality=82, method=6)

rgba = np.zeros((dh, dw, 4), "uint8")
zA, zB = np.nan_to_num(zone_m) == 2, np.nan_to_num(zone_m) == 1
rgba[zA] = (122, 92, 199, 120); rgba[zB] = (122, 92, 199, 55)
Image.fromarray(rgba).save(OUT / "runout.png", optimize=True)
json.dump(dict(bounds=[[lat0, lon0], [lat1, lon1]], px=[dw, dh], res_m=30), open(OUT / "overlay_meta.json", "w"))

# vectors
def gj(g, cols, tol=0):
    g = g.to_crs(32647)
    if tol: g["geometry"] = g.geometry.simplify(tol)
    g = g.to_crs(4326)
    js = json.loads(g[cols + ["geometry"]].to_json(drop_id=True))
    for f in js["features"]:
        def rnd(c): return [rnd(x) for x in c] if isinstance(c[0], list) else [round(c[0], 5), round(c[1], 5)]
        f["geometry"]["coordinates"] = rnd(f["geometry"]["coordinates"])
    return js
twp = gpd.read_file(RAW / "mimu/township.geojson")
vt = gpd.read_file(RAW / "mimu/vtracts.geojson")
inv = gpd.read_file(PROC / "inventory.geojson")
scars = gpd.read_file(RAW / "tawkye_2026/tawkye_landslide_2026.gpkg", layer="landslides")
ls_csv = pd.read_csv(RAW / "tawkye_2026/landslides.csv")
scars = scars[["id", "geometry"]].merge(ls_csv[["id", "name", "area_m2", "reach_angle_deg", "destroyed", "damaged"]], on="id")
V = pd.read_csv(PROC / "villages_screening.csv")
vt_agg = V.groupby("vt_pcode").agg(bld=("bld", "sum"), bld_A=("bld_A", "sum")).reset_index()
vt = vt.merge(vt_agg, left_on="VT_PCODE", right_on="vt_pcode", how="left").fillna({"bld": 0, "bld_A": 0})
bz = pd.read_csv(PROC / "buildings_zone.csv"); bz = bz[bz.zone > 0]
bzg = gpd.GeoDataFrame(bz, geometry=gpd.points_from_xy(bz.x, bz.y), crs=32647).to_crs(4326)
vec = dict(
    township=gj(twp, ["TS"], 15),
    vtracts=gj(vt, ["VT", "VT_MMR", "VT_PCODE", "bld", "bld_A"], 15),
    envelopes=gj(inv, ["id", "title", "cls", "style", "interpretation"]),
    scars=gj(scars, ["id", "name", "area_m2", "reach_angle_deg", "destroyed", "damaged"], 0.5),
    bld_zone=[[round(x, 5), round(y, 5), int(zv)] for x, y, zv in zip(bzg.geometry.x, bzg.geometry.y, bz.zone)],
)
json.dump(vec, open(OUT / "vectors.json", "w"), separators=(",", ":"))
for f in sorted(OUT.iterdir()): print(f.name, f.stat().st_size // 1024, "kB")
