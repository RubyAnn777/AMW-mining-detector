# >> FILE OVERVIEW
# >> Geometry helpers: split a region into square tiles, fix downloaded array sizes,
# >> and cut each tile into overlapping 48 px chips with a map polygon per chip.
# >> Pipeline position: step 2 (region -> DLTile tiles) and step 5 (pixels -> chips).
# >> Upstream file (Amazon Mining Watch), byte-identical to upstream.
# >> CRS note: DLTiles live in UTM metres; everything returned here is EPSG:4326.

from affine import Affine
# >> descarteslabs is used ONLY for DLTile grid maths (no Descartes Labs API calls).
from descarteslabs.geo import DLTile
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point, Polygon, box

# >> A tile centred on a chosen lat/lon instead of snapped to the global DLTile grid.
# >> Mimics DLTile's attributes (key, crs, bounds, geotrans, shape, geometry, tilesize,
# >> pad=0) so gee.get_tile_data can download it the same way.
# >> Used by tiles_for_geometry for Point inputs (utils/gee_data_pull.py).
# >> Not used by the production inference path (that uses DLTile only).
class CenteredTile:
    """
    DLTile-like tile centered on a given lat/lon, not snapped to DL global grid.

    Attributes match DLTile where possible (key, crs, bounds, geotrans, shape).
    """

    # >> Inputs: lat/lon in degrees (EPSG:4326), tilesize in px (default 48), resolution
    # >> in metres per pixel (default 10). Builds the square in UTM, then reprojects.
    def __init__(self, lat, lon, tilesize=48, resolution=10.0):
        self.lat = float(lat)
        self.lon = float(lon)
        self.tilesize = int(tilesize)
        self.resolution = float(resolution)
        # >> No padding for centred tiles.
        self.pad = 0

        # >> Borrow a real DLTile at this point only to learn which UTM zone (CRS) to use.
        snap_tile = DLTile.from_latlon(
            self.lat, self.lon,
            resolution=self.resolution,
            tilesize=self.tilesize,
            pad=self.pad
        )
        # >> CRS: the UTM zone of the DLTile grid at this location (metres).
        self.crs = snap_tile.crs

        # >> CRS CONVERSION: the centre point lon/lat (EPSG:4326) -> UTM metres.
        point = gpd.GeoSeries(
            [Point(self.lon, self.lat)], crs="EPSG:4326").to_crs(self.crs)
        x_center, y_center = float(point.x.iloc[0]), float(point.y.iloc[0])

        # >> Half the tile width in metres (48 px * 10 m / 2 = 240 m by default).
        half_m = (self.tilesize * self.resolution) / 2.0
        raw_minx = x_center - half_m
        raw_maxx = x_center + half_m
        raw_miny = y_center - half_m
        raw_maxy = y_center + half_m

        # >> Snap the upper-left corner to the 10 m pixel lattice (floor x, ceil y), so
        # >> the centre can shift by up to 1 px from the requested point.
        ulx = np.floor(raw_minx / self.resolution) * self.resolution
        uly = np.ceil(raw_maxy / self.resolution) * self.resolution

        # >> Affine: pixel (col, row) -> UTM (x, y); y scale is negative (row 0 = north).
        self.geotrans = Affine.translation(ulx, uly) * Affine.scale(
            self.resolution, -self.resolution)

        width = height = self.tilesize
        minx = ulx
        maxx = ulx + width * self.resolution
        maxy = uly
        miny = uly - height * self.resolution
        # >> Bounds in UTM metres (minx, miny, maxx, maxy), like DLTile.bounds.
        self.bounds = (minx, miny, maxx, maxy)
        
        self.shape = (self.tilesize, self.tilesize)
        # >> Key used in cache filenames, e.g. custom_-12.5_-69.1_10.0_48px.
        self.key = (f"custom_{round(self.lat, 6)}_{round(self.lon, 6)}" +
                    f"_{self.resolution:.1f}_{self.tilesize}px")

        # >> CRS CONVERSION: UTM square -> EPSG:4326 polygon (slightly tilted in lon/lat).
        # >> gee.py uses geometry.bounds of this polygon for the EE request and the
        # >> affine.
        bbox_proj = box(minx, miny, maxx, maxy)
        self.geometry = gpd.GeoSeries(
            [bbox_proj], crs=self.crs).to_crs("EPSG:4326").iloc[0]

    # >> Short printable description of the tile, for debugging.
    def __repr__(self):
        return f"<CenteredTile key={self.key} res={self.resolution} size={self.tilesize}>"


# >> Split a region into DLTile tiles (the production tiling). Returns a list.
# >> region: shapely/GeoJSON geometry in EPSG:4326; tilesize/pad in px; res in m.
# >> Production: tilesize 552, pad 12, 10 m -> 576x576 px tiles; SAM2 uses 256 + 12.
# >> Each tile key looks like 552:12:10.0:19:-15:-263 = tilesize:pad:res:UTMzone:x:y.
# >> Tiles are squares in UTM metres on a fixed global grid, so the same area always
# >> gives the same keys (this is what makes the image cache reusable across runs).
# >> Called by: inference_pipeline.main, InferenceEngine.bulk_mask_polygons,
# >> time_tiles.py, timeseries/panel.py, tiles_for_geometry below.
def create_tiles(region, tilesize, pad, resolution=10):
    """
    Create a set of tiles that cover a region.
    Inputs:
        - region: a geojson polygon
        - tilesize: the size of the tiles in pixels
        - pad: the number of pixels to pad each tile
        - resolution: spatial resolution in meters
    Outputs:
        - tiles: a list of DLTile objects
    """
    # >> DLTile.iter_from_shape: local grid maths only; yields every tile touching region.
    tiles = DLTile.iter_from_shape(
        region, tilesize=tilesize, resolution=resolution, pad=pad
    )
    return list(tiles)


# >> Pick the tiling style by geometry type: a Point gets one CenteredTile around it;
# >> polygons get the DLTile grid (create_tiles).
# >> Gotcha: `pad` is ignored for Points (CenteredTile always has pad 0).
# >> Called by: utils/gee_data_pull.py.
def tiles_for_geometry(geom, tilesize, pad=0, resolution=10):
    """
    Return tiles covering a geometry. For Point/MultiPoint, uses CenteredTile
    (one tile per point). For Polygon/MultiPolygon, uses DLTile.iter_from_shape.
    """
    if geom.geom_type == "Point":
        return [CenteredTile(geom.y, geom.x, tilesize=tilesize, resolution=resolution)]
    if geom.geom_type == "MultiPoint":
        return [
            CenteredTile(p.y, p.x, tilesize=tilesize, resolution=resolution)
            for p in geom.geoms
        ]
    return create_tiles(geom, tilesize, pad, resolution)


# >> Force an (H, W, C) array to exactly height x width: centre-crop extra rows/cols,
# >> then reflect-pad (mirror edge pixels) if too small.
# >> Why: EE sometimes returns a pixel more or less than requested; the chipper and
# >> model need an exact 576x576 grid.
# >> Called by: gee.get_tile_data (and timeseries/coverage.py).
# >> Gotcha: needs a 3-D array; reflect padding fails if a side is smaller than the
# >> padding needed (only an issue for badly broken downloads).
def ensure_tile_shape(raster, height, width=None):
    """
    Ensure tile raster data has the exact requested shape by trimming 
    (from center) if too large or padding (reflect) if too small.

    Parameters
    ----------
    raster : np.ndarray
        Input raster with shape (H, W, C).
    height : int
        Desired height.
    width : int, optional
        Desired width. If None, equals `height`.

    Returns
    -------
    np.ndarray
        Raster of shape (height, width, C).
    """
    if width is None:
        width = height

    # >> Current size; `_` = number of bands (unchanged).
    raster_height, raster_width, _ = raster.shape

    # --- Compute cropping (trim from center if too big) ---
    # >> How many extra rows/cols to remove (0 if not too big).
    dh = max(raster_height - height, 0)
    dw = max(raster_width - width, 0)

    # >> Split the trim between both sides (top gets the smaller half).
    trim_top = dh // 2
    trim_bottom = raster_height - (dh - trim_top)
    trim_left = dw // 2
    trim_right = raster_width - (dw - trim_left)

    cropped = raster[trim_top:trim_bottom, trim_left:trim_right, :]

    # --- Compute padding (reflect if too small) ---
    # >> How many rows/cols are missing (0 if not too small).
    ph = max(height - cropped.shape[0], 0)
    pw = max(width - cropped.shape[1], 0)

    pad_top = ph // 2
    pad_bottom = ph - pad_top
    pad_left = pw // 2
    pad_right = pw - pad_left

    # >> np.pad mode='reflect': mirror the edge pixels into the missing border.
    # >> Bands axis is never padded.
    padded = np.pad(
        cropped,
        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode="reflect"
    )

    return padded

# >> Cut a tile into overlapping square chips and make a lon/lat polygon for each.
# >> Inputs: pixels (H, W, B) e.g. (576, 576, 13) normalised float32; bbox =
# >> (west, south, east, north) in degrees; chip_size 48; stride 24; crs 'epsg:4326'.
# >> Outputs: chips array (N, 48, 48, B) and a GeoDataFrame of N chip squares.
# >> 576 px, 48 px chip, stride 24 -> 23 positions per axis -> 23 x 23 = 529 chips.
# >> Chip = 48 px x 10 m = ~480 m (~23 ha); each pixel is in up to 4 chips.
# >> Called by: InferenceEngine.predict_on_tile_pixels, timeseries/panel.py.
# >> Gotcha: if (W - chip_size) is not a multiple of stride, the last chips are cut
# >> short at the edge and np.array would fail on ragged shapes.
def cut_chips(pixels, bbox, chip_size, stride, crs):
    """ Create chips and chip geometries from a pixel array."""
    # >> Tile bounds in degrees; pixel row 0 = north edge, column 0 = west edge.
    west, south, east, north = bbox
    H = pixels.shape[0]  
    W = pixels.shape[1]  

    # >> Degrees per pixel in each direction (lon/lat grid, matches the EE download).
    x_per_pixel = (east - west) / W
    y_per_pixel = (north - south) / H

    chips = []
    chip_polys = []

    # >> LOOP ORDER defines chip order: OUTER over columns (west -> east), INNER over rows
    # >> (north -> south). Chip i in the output = column-major position; the fork's
    # >> timeseries/panel.py relies on this order.
    # >> range(0, 576 - 48 + 24, 24) = 0, 24, ..., 528 (23 values).
    for col in range(0, W - chip_size + stride, stride):
        for row in range(0, H - chip_size + stride, stride):
            # >> Slice (48, 48, B) out of (576, 576, B); a view, copied later by np.array.
            chip = pixels[row : row + chip_size, col : col + chip_size]
            chips.append(chip)

            # >> Pixel offsets -> lon/lat corners. North-west corner: x grows east, y
            # >> shrinks as rows go south.
            nw_x = west + col * x_per_pixel
            nw_y = north - row * y_per_pixel

            ne_x = west + (col + chip_size) * x_per_pixel
            ne_y = nw_y

            sw_x = nw_x
            sw_y = north - (row + chip_size) * y_per_pixel

            se_x = ne_x
            se_y = sw_y

            # >> Closed polygon ring NW -> SW -> SE -> NE -> NW, in lon/lat degrees.
            poly = Polygon([
                (nw_x, nw_y),
                (sw_x, sw_y),
                (se_x, se_y),
                (ne_x, ne_y),
                (nw_x, nw_y)
            ])

            chip_polys.append(poly)

    # >> GeoDataFrame of chip squares, CRS as passed in (EPSG:4326 in production).
    chip_geoms = gpd.GeoDataFrame(geometry=chip_polys, crs=crs)
    # >> np.array stacks the list: N x (48, 48, B) -> (N, 48, 48, B).
    return np.array(chips), chip_geoms
