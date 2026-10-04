"""OSM building centroids + roads for the Launglon window via Overpass (cached in data/raw/osm)."""
import json, time
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/osm"; RAW.mkdir(parents=True, exist_ok=True)
BB = "13.50,97.88,14.25,98.27"   # S,W,N,E
URLS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"]
Q = {
    "buildings": f'[out:json][timeout:600];(way["building"]({BB});relation["building"]({BB}););out center tags;',
    "roads":     f'[out:json][timeout:600];way["highway"~"^(trunk|primary|secondary|tertiary|unclassified|residential|track)$"]({BB});out geom tags;',
}
for k, q in Q.items():
    f = RAW / f"{k}.json"
    if f.exists(): continue
    for u in URLS * 2:
        try:
            r = requests.post(u, data={"data": q}, timeout=900, headers={"User-Agent": "launglon-landslide-screening"})
            r.raise_for_status(); f.write_text(r.text, encoding="utf8"); break
        except Exception as e:
            print("retry", u, e); time.sleep(20)
    print(k, len(json.load(open(f, encoding="utf8"))["elements"]))
