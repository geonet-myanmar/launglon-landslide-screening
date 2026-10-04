"""Assemble the dashboard: src/template.html (a body fragment, single source of truth) + payload + layers
  -> index.html                  standalone page (doctype, charset, viewport) for GitHub Pages / local use
  -> dist/artifact-body.html     the same fragment for a Claude Artifact (the platform supplies the head)
Every number on the page is read from data/processed at build time. Never hand-edit index.html.
"""
import base64, json, datetime as dt
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, PROC = ROOT / "data/raw", ROOT / "data/processed"
L = PROC / "layers"
SITE = "https://geonet-myanmar.github.io/launglon-landslide-screening/"
DESC = ("Historical-landslide screening for Launglon Township, Myanmar, after the 26 Sep 2026 Taw Kye debris flows: "
        "calibrated debris-flow reach, village exposure, Landsat history and rainfall context.")

def uri(p, mime):
    return f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode()

V = pd.read_csv(PROC / "villages_screening.csv")
basins = pd.read_csv(PROC / "village_basins.csv")
cal = pd.read_csv(PROC / "runout_calibration.csv")
hist = json.load(open(PROC / "history_landsat.json"))
histc = pd.read_csv(PROC / "history_landsat.csv").fillna("")
rain = json.load(open(PROC / "rain.json"))
kml = json.load(open(PROC / "kml_meta.json", encoding="utf8"))
tmeta = json.load(open(PROC / "terrain_meta.json"))
tk = json.load(open(RAW / "tawkye_2026/summary.json"))
ls = pd.read_csv(RAW / "tawkye_2026/landslides.csv")
bz = pd.read_csv(PROC / "buildings_zone.csv")
import geopandas as gpd
inv = gpd.read_file(PROC / "inventory.geojson")
env_ha = dict(zip(inv.id, (inv.to_crs(32647).area / 1e4).round(2)))
import numpy as np
ter = np.load(PROC / "terrain.npz")
lt = ter["land"] & ter["in_twp"]

inventory = []
for r in inv.itertuples():
    s = ls[ls.id == r.id].iloc[0]; c = cal[cal.id == r.id].iloc[0]; h = histc[histc.id == r.id].iloc[0]
    inventory.append(dict(id=r.id, title=r.title, cls=r.cls, style=r.style, interp=r.interpretation,
                          env_ha=env_ha[r.id], scar_ha=round(s.area_m2 / 1e4, 2), reach=float(s.reach_angle_deg),
                          H=float(s.H_m), L=float(s.L_m), destroyed=int(s.destroyed), damaged=int(s.damaged),
                          in_AB=round(float(c.in_AB), 3), in_A=round(float(c.in_A), 3),
                          flags=str(h.flagged_years), anom98=float(h.anom_1998), p98=float(h.p_1998),
                          loss=str(h.hansen_loss_years), loss_frac=float(h.hansen_loss_frac)))

top_basins = {}
for vi, g in basins.groupby("vil"):
    g = g.sort_values("melton", ascending=False).head(4)
    top_basins[int(vi)] = g[["area_km2", "relief_m", "melton", "length_km", "cls", "loss_frac"]].values.tolist()

V = V.sort_values(["tier", "bld_A"], ascending=[True, False])
cols = ["idx", "pcode", "name", "name_my", "vt", "lon", "lat", "bld", "bld_A", "bld_B", "pct_A", "best_cls", "melton_max",
        "best_area_km2", "best_relief_m", "best_len_km", "n_outlets", "n_df", "slope_p90_500m", "relief_500m",
        "steep_loss_pct", "tier"]
vil = [[None if (isinstance(x, float) and np.isnan(x)) else x for x in row] for row in V[cols].values.tolist()]
rank_tk = int((V.bld_A > int(V[V.name == "Taw Kye"].bld_A.iloc[0])).sum()) + 1

summary = dict(
    built=dt.date.today().isoformat(),
    n_env=len(inv), env_ha=round(sum(env_ha.values()), 1), scar_ha=round(tk["area_total_m2"] / 1e4, 1),
    tk_destroyed=tk["buildings_status"]["destroyed"], tk_damaged=tk["buildings_status"]["damaged"],
    tk_deaths=tk["report"]["deaths"], tk_hit_A=tk["reach"]["affected_in_A"], tk_standing_A=tk["reach"]["standing_in_A"], tk_houses_report=tk["report"]["houses_destroyed"],
    alpha_open=tmeta["A_OPEN"], alpha_ch=tmeta["A_CH"], src_slope=tmeta["SRC_SLOPE"], src_relief=tmeta["SRC_RELIEF"],
    cal_ok=int((cal.in_AB >= 0.85).sum()), cal_n=len(cal),
    land_km2=round(float(lt.sum()) * 900 / 1e6, 0), zoneA_km2=round(float((ter["zone"][lt] == 2).sum()) * 900 / 1e6, 1),
    zoneB_km2=round(float((ter["zone"][lt] == 1).sum()) * 900 / 1e6, 1),
    src_km2=round(float((ter["is_src"] & lt).sum()) * 900 / 1e6, 1),
    n_bld=len(bz), n_bld_osm=int((bz.src == "osm").sum()), bld_A=int((bz.zone == 2).sum()), bld_B=int((bz.zone == 1).sum()),
    n_vil=len(V), tiers={int(k): int(v) for k, v in V.tier.value_counts().sort_index().items()},
    tk_rank=rank_tk, melton_counts={k: int(v) for k, v in V.best_cls.value_counts().items()},
    n_vt=int(V.vt.nunique()))

payload = dict(summary=summary, villages=vil, vcols=cols, basins=top_basins, inventory=inventory,
               history=hist, rain=rain, evidence=kml["evidence"], kml_doc=kml["document"],
               overlay=json.load(open(L / "overlay_meta.json")),
               vectors=json.load(open(L / "vectors.json")))

tpl = (ROOT / "src/template.html").read_text(encoding="utf8")
css = (ROOT / "src/vendor/leaflet-1.9.4.min.css").read_text(encoding="utf8")
body = (tpl.replace("/*__LEAFLET_CSS__*/", css)
           .replace("__RELIEF_LIGHT__", uri(L / "relief_light.webp", "image/webp"))
           .replace("__RELIEF_DARK__", uri(L / "relief_dark.webp", "image/webp"))
           .replace("__RUNOUT__", uri(L / "runout.png", "image/png"))
           .replace("/*__DATA__*/null", json.dumps(payload, ensure_ascii=False, separators=(",", ":"))))
(ROOT / "dist").mkdir(exist_ok=True)
(ROOT / "dist/artifact-body.html").write_text(body, encoding="utf8")
icon = ("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Cpath d='M2 28 L12 10 L17 18 "
        "L21 12 L30 28Z' fill='%237a5cc7'/%3E%3C/svg%3E")
page = ("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
        f"<link rel=\"icon\" href=\"{icon}\">\n"
        f"<meta name=\"description\" content=\"{DESC}\">\n<link rel=\"canonical\" href=\"{SITE}\">\n"
        f"<meta property=\"og:type\" content=\"website\">\n<meta property=\"og:title\" content=\"Launglon Landslide Screening\">\n"
        f"<meta property=\"og:description\" content=\"{DESC}\">\n<meta property=\"og:url\" content=\"{SITE}\">\n"
        f"</head>\n<body>\n{body}\n</body>\n</html>\n")
(ROOT / "index.html").write_text(page, encoding="utf8")
print("index.html", len(page) // 1024, "kB; Taw Kye rank by buildings in zone A:", rank_tk)
print(json.dumps(summary, indent=0))
