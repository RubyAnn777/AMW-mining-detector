# Measures tile overlaps in Madre de Dios and at a UTM zone seam (open_design_decisions.Rmd D5, D6).
# Usage: python panel_pipeline/checks/overlap/tiles.py
import warnings; warnings.filterwarnings("ignore")
import geopandas as gpd, numpy as np, shapely, time, collections
from descarteslabs.geo import DLTile
from pathlib import Path
S = str(Path(__file__).resolve().parents[3] / "data" / "boundaries") + "/"   # the repo's boundary files
# tile overlaps in Madre de Dios
shape = gpd.read_file(S+"madre_de_dios_mining_aoi.geojson").union_all()
tiles = list(DLTile.iter_from_shape(shape, tilesize=552, resolution=10, pad=12))
bb = gpd.GeoSeries([shapely.box(*t.geometry.bounds) for t in tiles], crs=4326).to_crs(6933)
nopad = gpd.GeoSeries([shapely.box(*DLTile.from_key(t.key.replace(":12:",":0:",1)).geometry.bounds) for t in tiles], crs=4326).to_crs(6933)
for name, gs in [("downloaded bbox (pad 12)", bb), ("bbox without pad", nopad)]:
    g = gs.values; tr = shapely.STRtree(g); i,j = tr.query(g, predicate="intersects"); m=i<j
    ov = shapely.area(shapely.intersection(g[i[m]], g[j[m]])); ov=ov[ov>1]
    print(f"{name}: tiles {len(g)}, overlapping pairs {len(ov)}, overlap/tile area median {np.median(ov)/gs.area.mean():.2%} max {ov.max()/gs.area.mean():.2%}; sum/union {gs.area.sum()/gs.union_all().area:.4f}")
# subregions
t0=time.time(); keys=collections.Counter(); zones=collections.Counter(); per={}
subs={n: gpd.read_file(S+f"Amazon_ACA/Amazon_ACA_{n}.geojson").union_all() for n in range(1,7)}
for n,s in subs.items():
    ks=[t.key for t in DLTile.iter_from_shape(s, tilesize=552, resolution=10, pad=12)]
    per[n]=len(ks); keys.update(ks); zones.update(k.split(":")[3] for k in ks)
print("tiles per subregion", per, "total", sum(per.values()), "unique", len(keys), "in 2+ subregions", sum(v>1 for v in keys.values()), "in 3+", sum(v>2 for v in keys.values()), f"{time.time()-t0:.0f}s")
print("UTM zones (tile count, with dups)", dict(sorted(zones.items())))
ov=[(a,b,subs[a].intersection(subs[b]).area) for a in subs for b in subs if a<b]
print("max pairwise subregion intersection area (deg2)", max(x[2] for x in ov))
