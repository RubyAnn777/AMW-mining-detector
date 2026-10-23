# Measures how tiles and chips overlap (open_design_decisions.Rmd D5, D6). Usage: python panel_pipeline/checks/overlap/overlap.py <boundary.geojson>
import sys, warnings, numpy as np, geopandas as gpd
warnings.filterwarnings("ignore")
from descarteslabs.geo import DLTile
from shapely.geometry import box
from shapely import STRtree, area, intersection
import shapely
path = sys.argv[1]
shape = gpd.read_file(path).union_all()
tiles = list(DLTile.iter_from_shape(shape, tilesize=552, resolution=10, pad=12))
rows = []
for t in tiles:
    w, s, e, n = t.geometry.bounds
    W = 576; xp = (e-w)/W; yp = (n-s)/W
    for ci, col in enumerate(range(0, 529, 24)):
        for ri, row in enumerate(range(0, 529, 24)):
            rows.append((t.zone, t.ti, t.tj, ci, ri, w+col*xp, n-(row+48)*yp, w+(col+48)*xp, n-row*yp, t.key))
import pandas as pd
df = pd.DataFrame(rows, columns="zone ti tj col row minx miny maxx maxy key".split())
g = gpd.GeoDataFrame(df, geometry=shapely.box(df.minx, df.miny, df.maxx, df.maxy), crs=4326).to_crs(6933)
print(f"tiles {len(tiles)}, zones {sorted(set(t.zone for t in tiles))}, chips {len(g)}")
A = g.geometry.area
print(f"chip area m2: mean {A.mean():.0f} min {A.min():.0f} max {A.max():.0f}  (48px*10m)^2 = 230400")
U = g.geometry.union_all().area
print(f"full set: sum area / union area = {A.sum()/U:.3f}")
# true UTM squares vs degree bbox
t = tiles[0]
print("tile bbox vs UTM square area ratio:", gpd.GeoSeries([box(*t.geometry.bounds)],crs=4326).to_crs(6933).area[0]/gpd.GeoSeries([t.geometry],crs=4326).to_crs(6933).area[0])
sub = g[((g.ti*23 + g.col) % 2 == 0) & ((g.tj*23 + g.row) % 2 == 0)].reset_index(drop=True)
# NB tj may run north->south opposite to row; parity is unaffected because 23 is odd
geoms = sub.geometry.values
tree = STRtree(geoms)
i, j = tree.query(geoms, predicate="intersects")
m = i < j
i, j = i[m], j[m]
ov = area(intersection(geoms[i], geoms[j]))
pos = ov > 1.0
i, j, ov = i[pos], j[pos], ov[pos]
sa = sub.geometry.area
print(f"every-other subset: {len(sub)} chips ({len(sub)/len(g):.3f} of all)")
print(f"  pairs with overlap >1 m2: {len(ov)}; chips involved: {len(set(i)|set(j))} ({len(set(i)|set(j))/len(sub):.1%})")
same = (sub.key.values[i] == sub.key.values[j])
print(f"  of which within the same tile: {same.sum()}")
if len(ov):
    q = np.quantile(ov/230400, [0.5, 0.9, 0.99, 1])
    print(f"  overlap as share of a chip: median {q[0]:.3%} p90 {q[1]:.3%} p99 {q[2]:.3%} max {q[3]:.3%}")
    print(f"  overlap in px (10m): median {np.median(ov)/100:.0f} max {ov.max()/100:.0f}")
Us = sub.geometry.union_all()
print(f"  sum area / union area = {sa.sum()/Us.area:.5f}")
core = g.geometry.union_all()
gap = core.difference(Us).area
# gaps only inside the region's interior (ignore outer edge strip)
inner = shape
inner_m = gpd.GeoSeries([shape], crs=4326).to_crs(6933)[0].buffer(-6000)
gap_in = inner_m.intersection(core.difference(Us)).area
print(f"  uncovered area inside region interior (6 km in from edge): {gap_in/1e6:.3f} km2 of {inner_m.area/1e6:.1f} km2 = {gap_in/inner_m.area:.3%}")
print(f"  overlap area inside interior: {(inner_m.intersection(Us).area - 0) and ((sub[sub.intersects(inner_m)].geometry.intersection(inner_m).area.sum() - inner_m.intersection(Us).area)/inner_m.intersection(Us).area):.3%} double-counted")
