# Launglon Landslide Screening

Historical-landslide screening for Launglon Township (Dawei District, Tanintharyi) after the
26–27 Sep 2026 Taw Kye debris flows. It starts from the supplied KML
`data/source/launglon_landslide_inventory_v1.kml` (9 screening envelopes, Google Earth evidence dates, 113 village points).

**Live:** https://geonet-myanmar.github.io/launglon-landslide-screening/

Open `index.html`. Every number in it is computed by the pipeline below; never hand-edit `index.html`.

| step | script | output |
|---|---|---|
| 1 | `01_parse_kml.py` | envelopes, villages, evidence → `data/processed/` |
| 2 | `02_fetch_mimu.py` | MIMU township, village tracts, villages (GeoServer WFS) |
| 3 | `03_fetch_ee.py` | GLO-30 DEM, Hansen GFC, Landsat 1988–2026 NDVI, IMERG + CHIRPS rain, VIDA buildings (needs `EE_PROJECT`) |
| 4 | `04_fetch_osm.py` | OSM buildings and roads (Overpass) |
| 5 | `05_terrain.py` | D8 flow, slope, debris-flow reach zones A/B calibrated on the 2026 drone inventory |
| 6 | `06_history.py` | Landsat history of each 2026 path with a shape-matched placebo test |
| 7 | `07_villages.py` | buildings in reach per village, Melton ratio, tiers |
| 8 | `08_rain.py` | rain trigger vs record; 1997 in CHIRPS |
| 9 | `09_layers.py` | map overlays (30 m, Web Mercator) |
| 10 | `10_build.py` | `src/template.html` → `index.html` + `dist/artifact-body.html` |

Step 2 also downloads the 2026 drone inventory from
[geonet-myanmar/tawkye-landslide-2026](https://github.com/geonet-myanmar/tawkye-landslide-2026) (`outputs/`), cached in `data/raw/tawkye_2026/`.
`data/raw/` is gitignored and must be re-fetched (steps 2–4) before rebuilding from scratch.

This is a screening product. It ranks where to look first and does not replace a site assessment.

## Deployment

GitHub Pages serves `index.html` from the root of `main` (`.nojekyll` present). The page is self-contained:
data and map overlays are embedded, and only Leaflet, Google Fonts and the optional Esri imagery layer load from the network.
To update, re-run `python scripts/10_build.py` (or the steps that changed), then commit and push.
