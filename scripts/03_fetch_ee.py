"""Earth Engine downloads (cached in data/raw/ee). Every raster is fetched at its sensor's native resolution.

  dem.tif         Copernicus GLO-30 DEM (2024_1 release), 30 m, UTM 47N, township + 2 km    (bilinear from the 1" grid)
  hansen.tif      Hansen GFC v1.13: treecover2000, lossyear (2001-2025), 30 m, same grid (nearest)
  landsat_tk.tif  Landsat 5/7/8/9 C2 L2 dry-season median NDVI 1988-2026 + clear-obs counts, 30 m, Taw Kye window
  rain_imerg.csv  GPM IMERG V07 daily (Myanmar day, 00:00 MMT) 1998-06 .. 2026-10, 2 cells covering Taw Kye
  rain_chirps.csv CHIRPS v2 daily 1981 .. latest at the Taw Kye cell (0.05 deg)
  buildings_vida.csv  VIDA combined (Google+Microsoft) footprints in the township window -> centroid + area
Set EE_PROJECT in the environment.
"""
import os, io, json, zipfile, datetime as dt
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests, ee

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/ee"; RAW.mkdir(parents=True, exist_ok=True)
ee.Initialize(project=os.environ["EE_PROJECT"])

# 30 m analysis grid (UTM 47N), township bounds + ~2 km, snapped to 30 m
X0, Y1, NX, NY = 379200, 1574250, 1335, 2614          # 379200..419250 E, 1495830..1574250 N
TK = dict(x0=404400, y1=1529400, nx=240, ny=270)        # Taw Kye window, 7.2 x 8.1 km
TK_PT = (98.1596, 13.79993)

def grid(x0, y1, nx, ny, res=30):
    return dict(crs="EPSG:32647", crs_transform=[res, 0, x0, 0, -res, y1], dimensions=f"{nx}x{ny}")

def download(img, path, g):
    if path.exists(): return
    url = img.getDownloadURL(dict(g, format="GEO_TIFF"))
    r = requests.get(url, timeout=900); r.raise_for_status()
    path.write_bytes(r.content); print("wrote", path.name, len(r.content) // 1024, "kB")

def fetch_static():
    G = grid(X0, Y1, NX, NY)
    dem = ee.ImageCollection("COPERNICUS/DEM/GLO30_2024_1").filterBounds(ee.Geometry.Rectangle([97.8, 13.4, 98.35, 14.3])) \
        .select("DEM").mosaic() \
        .setDefaultProjection("EPSG:4326", None, 1 / 3600).resample("bilinear").toFloat()
    download(dem, RAW / "dem.tif", G)
    h = ee.Image("UMD/hansen/global_forest_change_2025_v1_13")
    download(h.select(["treecover2000", "lossyear"]).toUint8(), RAW / "hansen.tif", G)

# ---------- Landsat dry-season NDVI, Taw Kye window ----------
def prep(img):
    sid = ee.String(img.get("SPACECRAFT_ID"))
    l89 = ee.List(["LANDSAT_8", "LANDSAT_9"]).contains(sid)
    nir = ee.Image(ee.Algorithms.If(l89, img.select("SR_B5"), img.select("SR_B4")))
    red = ee.Image(ee.Algorithms.If(l89, img.select("SR_B4"), img.select("SR_B3")))
    nir = nir.multiply(2.75e-5).add(-0.2); red = red.multiply(2.75e-5).add(-0.2)
    qa = img.select("QA_PIXEL")
    bad = qa.bitwiseAnd(0b11111).neq(0)            # fill, dilated cloud, cirrus, cloud, shadow
    sat = img.select("QA_RADSAT").neq(0)
    ok = bad.Or(sat).Not().And(nir.gt(0)).And(red.gt(0))
    return nir.subtract(red).divide(nir.add(red)).rename("ndvi").updateMask(ok)

def landsat(year):
    a, b = f"{year-1}-11-01", f"{year}-05-01"
    col = ee.ImageCollection([])
    for c in ["LANDSAT/LT05/C02/T1_L2", "LANDSAT/LE07/C02/T1_L2", "LANDSAT/LC08/C02/T1_L2", "LANDSAT/LC09/C02/T1_L2"]:
        col = col.merge(ee.ImageCollection(c).filterBounds(ee.Geometry.Point(TK_PT).buffer(5000)).filterDate(a, b))
    col = col.map(prep)
    return col.median().rename(f"ndvi_{year}"), col.count().rename(f"n_{year}"), col.size()

def fetch_landsat():
    out = RAW / "landsat_tk.tif"
    if out.exists(): return
    years = list(range(1988, 2027))
    imgs, meta = [], {}
    for y in years:
        m, n, k = landsat(y); imgs += [m.unmask(-9).toFloat(), n.toFloat()]; meta[y] = k
    sizes = ee.Dictionary({str(k): v for k, v in meta.items()}).getInfo()
    json.dump(sizes, open(RAW / "landsat_tk_scenes.json", "w"), indent=1)
    stack = ee.Image.cat(imgs)
    download(stack, out, grid(**TK))
    print("scenes per dry season:", sizes)

# ---------- rainfall ----------
MMT = dt.timedelta(hours=6, minutes=30)
def imerg_year(y):
    reg = ee.Geometry.Rectangle([98.1, 13.7, 98.2, 13.9])
    start = dt.datetime(y, 1, 1) - MMT
    end = min(dt.datetime(y + 1, 1, 1), dt.datetime(2026, 10, 3)) - MMT
    col = ee.ImageCollection("NASA/GPM_L3/IMERG_V07").select("precipitation").filterDate(start, end)
    ndays = (end - start).days
    def day(i):
        d0 = ee.Date(start.isoformat()).advance(i, "day")
        s = col.filterDate(d0, d0.advance(1, "day"))
        v = s.sum().multiply(0.5).reduceRegion(ee.Reducer.mean(), reg, 11132).get("precipitation")
        return ee.Feature(None, dict(date=d0.advance(6.5, "hour").format("YYYY-MM-dd"), mm=v, n=s.size()))
    fc = ee.FeatureCollection(ee.List.sequence(0, ndays - 1).map(day))
    return [f["properties"] for f in fc.getInfo()["features"]]

def fetch_rain():
    f = RAW / "rain_imerg.csv"
    if not f.exists():
        with ThreadPoolExecutor(6) as ex: rows = sum(ex.map(imerg_year, range(1998, 2027)), [])
        rows = [r for r in rows if r.get("n")]
        f.write_text("date,mm,n_halfhours\n" + "".join(f"{r['date']},{r['mm']:.2f},{r['n']}\n" for r in rows if r.get("mm") is not None))
        print("imerg days", len(rows))
    f = RAW / "rain_chirps.csv"
    if not f.exists():
        col = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").select("precipitation")
        def yr(y):
            r = col.filterDate(f"{y}-01-01", f"{y+1}-01-01").getRegion(ee.Geometry.Point(TK_PT), 5566).getInfo()
            return [(dt.datetime.utcfromtimestamp(t / 1000).strftime("%Y-%m-%d"), v) for _, _, _, t, v in r[1:]]
        with ThreadPoolExecutor(6) as ex: rows = sum(ex.map(yr, range(1981, 2027)), [])
        f.write_text("date,mm\n" + "".join(f"{d},{v:.2f}\n" for d, v in sorted(rows) if v is not None))
        print("chirps days", len(rows))

# ---------- buildings ----------
def fetch_buildings():
    f = RAW / "buildings_vida.csv"
    if f.exists(): return
    fc = ee.FeatureCollection("projects/sat-io/open-datasets/VIDA_COMBINED/MMR")
    def tile(i):
        x = 97.88 + 0.1 * (i % 4); y = 13.50 + 0.1 * (i // 4)
        r = ee.Geometry.Rectangle([x, y, x + 0.1, y + 0.1])
        def cen(g):
            c = g.geometry().centroid(1).coordinates()
            return ee.Feature(None, dict(lon=c.get(0), lat=c.get(1), area=g.get("area_in_meters"), src=g.get("bf_source")))
        sub = fc.filterBounds(r).map(cen).filter(ee.Filter.And(
            ee.Filter.gte("lon", x), ee.Filter.lt("lon", x + 0.1), ee.Filter.gte("lat", y), ee.Filter.lt("lat", y + 0.1)))
        url = sub.getDownloadURL(filetype="csv", selectors=["lon", "lat", "area", "src"])
        t = requests.get(url, timeout=900).text.splitlines()
        return t[1:]
    with ThreadPoolExecutor(8) as ex: rows = sum(ex.map(tile, range(4 * 8)), [])
    f.write_text("lon,lat,area,src\n" + "\n".join(rows) + "\n")
    print("buildings", len(rows))

if __name__ == "__main__":
    fetch_static(); fetch_buildings(); fetch_landsat(); fetch_rain()

