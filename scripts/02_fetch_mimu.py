"""Pull MIMU layers for Launglon township from the GeoNode GeoServer WFS (cached in data/raw/mimu)."""
import json, urllib.parse
from pathlib import Path
import requests, geopandas as gpd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/mimu"; RAW.mkdir(parents=True, exist_ok=True)
WFS = "https://geonode.themimu.info/geoserver/wfs"
TS = "Launglon"
BBOX = (97.85, 13.45, 98.45, 14.35)   # generous Launglon window for line/point layers

LAYERS = {
    "township":   ("mmr_polbnda_adm3_250k_mimu_1", f"TS='{TS}'"),
    "district_ts":("mmr_polbnda_adm3_250k_mimu_1", "DT='Dawei'"),
    "vtracts":    ("mmr_tni_polbnda_adm4_unhcr_mimu_250k", f"TS='{TS}'"),
    "villages":   ("mmr_tni_pplp2_250k_mimu", f"TS='{TS}'"),
    "roads":      ("mmr_rdsl_mimu_250k", None),
    "rivers":     ("myanmar_river_network", None),
    "health":     ("health_facilities_myanmar2020_v20241016", None),
    "schools":    ("formal_sector_school_location_lowermyanmar_2019", None),
}

def fetch(layer, cql):
    p = dict(service="WFS", version="1.0.0", request="GetFeature", typeName=f"geonode:{layer}",
             outputFormat="application/json", srsName="EPSG:4326")
    if cql: p["CQL_FILTER"] = cql
    else:   p["BBOX"] = ",".join(map(str, BBOX)) + ",EPSG:4326"
    r = requests.get(WFS, params=p, timeout=300); r.raise_for_status()
    return r.json()

for key, (layer, cql) in LAYERS.items():
    f = RAW / f"{key}.geojson"
    if not f.exists():
        js = fetch(layer, cql)
        f.write_text(json.dumps(js, ensure_ascii=False), encoding="utf8")
    g = gpd.read_file(f)
    print(f"{key:12s} {len(g):5d} features  crs={g.crs}  bounds={[round(x,3) for x in g.total_bounds] if len(g) else None}")

# 2026 Taw Kye drone inventory (published product) -> data/raw/tawkye_2026
TK = ROOT / "data/raw/tawkye_2026"; TK.mkdir(parents=True, exist_ok=True)
for f in ["landslides.csv", "summary.json", "buildings.csv", "tawkye_landslide_2026.gpkg"]:
    if not (TK / f).exists():
        r = requests.get(f"https://raw.githubusercontent.com/geonet-myanmar/tawkye-landslide-2026/main/outputs/{f}", timeout=300)
        r.raise_for_status(); (TK / f).write_bytes(r.content)
print("Taw Kye drone inventory:", sorted(p.name for p in TK.iterdir()))
