"""Parse the supplied screening KML into GeoJSON layers + a JSON of evidence/QA notes."""
import json, re, xml.etree.ElementTree as ET
from pathlib import Path
import geopandas as gpd
from shapely.geometry import Polygon, Point

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/source/launglon_landslide_inventory_v1.kml"
OUT = ROOT / "data/processed"
NS = {"k": "http://www.opengis.net/kml/2.2"}

def coords(txt):
    return [tuple(map(float, c.split(",")[:2])) for c in txt.split()]

def field(desc, label):
    m = re.search(rf"<b>{label}:</b>\s*(.*?)<br/>", desc or "", re.S)
    return re.sub(r"\s+", " ", m.group(1)).strip() if m else None

tree = ET.parse(SRC)
folders = tree.getroot().findall(".//k:Folder", NS)
inv, ev, vil, notes = [], [], [], []
for f in folders:
    fname = f.find("k:name", NS).text
    for pm in f.findall("k:Placemark", NS):
        name = pm.find("k:name", NS).text.strip()
        desc = (pm.findtext("k:description", default="", namespaces=NS) or "").strip()
        style = (pm.findtext("k:styleUrl", default="", namespaces=NS) or "").lstrip("#")
        data = {d.get("name"): d.findtext("k:value", namespaces=NS) for d in pm.findall(".//k:Data", NS)}
        poly = pm.find(".//k:Polygon//k:coordinates", NS)
        pt = pm.find(".//k:Point/k:coordinates", NS)
        if poly is not None:
            lid, _, title = name.partition(" — ")
            inv.append(dict(id=lid.strip(), title=title.strip(), style=style,
                            cls=data.get("class"), confidence=data.get("confidence"),
                            source_crs=data.get("source_crs"),
                            interpretation=field(desc, "Interpretation"),
                            geometry=Polygon(coords(poly.text))))
        elif fname.startswith("Historical"):
            ev.append(dict(date=name, note=desc, lon=coords(pt.text)[0][0], lat=coords(pt.text)[0][1]))
        elif fname.startswith("Launglon village"):
            x, y = coords(pt.text)[0]
            vil.append(dict(name_my=name, geometry=Point(x, y)))
        else:
            notes.append(dict(name=name, html=desc))

g = gpd.GeoDataFrame(inv, crs=4326)
g.to_file(OUT / "inventory.geojson", driver="GeoJSON")
v = gpd.GeoDataFrame(vil, crs=4326)
v.insert(0, "kml_idx", range(len(v)))
v.to_file(OUT / "villages_kml.geojson", driver="GeoJSON")
json.dump(dict(evidence=ev, notes=notes, document=tree.getroot().findtext(".//k:Document/k:name", namespaces=NS)),
          open(OUT / "kml_meta.json", "w", encoding="utf8"), ensure_ascii=False, indent=1)
u = g.to_crs(32647)
print(len(g), "envelopes; total", round(u.area.sum() / 1e4, 2), "ha")
for r, a in zip(g.itertuples(), u.area): print(f"  {r.id} {a/1e4:6.2f} ha  {r.cls}  | {r.title}")
print(len(v), "village points; bbox", [round(x, 4) for x in v.total_bounds])
print(len(ev), "evidence dates;", len(notes), "notes")
