# =============================================================================
# FILE OVERVIEW: gee/tile_utils.py
# -----------------------------------------------------------------------------
# Purpose: tiling and chipping utilities for the Amazon Mining Watch (AMW)
#   mining-detector pipeline.  Converts vector regions (GeoJSON, EPSG:4326)
#   into fixed-size raster tiles on the Descartes Labs "DLTile" UTM grid, and
#   slices downloaded tile rasters into small overlapping "chips" that are the
#   classifier's unit of prediction.
#
# Invocation: imported as a library; never run directly.  Main consumers are
#   gee/gee.py (GEE_Data_Extractor / InferenceEngine) and
#   gee/inference_pipeline.py (CLI bulk inference), plus training notebooks.
#
# On-disk inputs/outputs: none.  All functions operate on in-memory numpy
#   arrays, shapely geometries, and GeoDataFrames.
#
# Pipeline position: FIRST geometric step.  region GeoJSON -> tiles
#   (create_tiles / tiles_for_geometry; 576 px at 10 m/px = 5.76 km per side,
#   ~33.2 km^2 each) -> per-tile imagery from Google Earth Engine (gee.py) ->
#   chips (cut_chips; 48 px = 480 m, ~23 ha, stride 24 -> 23x23 = 529
#   overlapping chips/tile) -> SSL4EO embeddings -> Keras MLP ensemble ->
#   per-chip mining confidence.
#
# Why DLTile: descarteslabs.geo.DLTile defines a FIXED, deterministic global
#   grid of UTM tiles (the sole reason the descarteslabs package is a
#   dependency).  The same region always yields the same tiles, and tile.key
#   is a unique string ID used in cache filenames -- so runs over different
#   date ranges align exactly, enabling time-series comparison of the same
#   ground footprints.
# =============================================================================

from affine import Affine  # affine geotransforms (pixel <-> map coordinates)
from descarteslabs.geo import DLTile  # fixed global UTM tile grid (see overview)
import geopandas as gpd  # GeoDataFrame/GeoSeries containers + CRS reprojection
import matplotlib.pyplot as plt  # not used at runtime; kept for debugging plots
import numpy as np  # array math for rasters/chips
from shapely.geometry import Point, Polygon, box  # geometry constructors

class CenteredTile:
    """
    DLTile-like tile centered on a given lat/lon, not snapped to DL global grid.

    Attributes match DLTile where possible (key, crs, bounds, geotrans, shape).

    Purpose: a DLTile "impostor" for point-based scoring.  A real DLTile is
    snapped to the fixed global grid, so an arbitrary lat/lon usually falls
    off-center; this class instead builds a tile whose footprint is centered
    exactly on the requested point (snapped only to the 10 m pixel grid).
    Used by tiles_for_geometry() when the input geometry is Point/MultiPoint.

    Constructor inputs:
        lat (float): latitude of tile center, decimal degrees (EPSG:4326).
        lon (float): longitude of tile center, decimal degrees (EPSG:4326).
        tilesize (int): tile edge length in pixels (default 48).
        resolution (float): ground sample distance, meters/pixel (default
            10.0, matching Sentinel-2's 10 m bands).

    Attributes produced:
        crs (str): UTM zone EPSG code, inherited from a grid-snapped DLTile.
        geotrans (Affine): north-up geotransform in UTM meters.
        bounds (tuple): (minx, miny, maxx, maxy) in UTM meters.
        shape (tuple): (tilesize, tilesize) raster shape in pixels.
        geometry (shapely Polygon): footprint reprojected to EPSG:4326.
        key (str): unique ID embedding lat/lon/resolution/size; used in
            cache filenames so repeated runs share the same tile identity.

    Side effects: none (pure in-memory construction).
    """

    def __init__(self, lat, lon, tilesize=48, resolution=10.0):
        """Build the centered tile (see class docstring for parameters).

        Output: none (populates the attributes documented on the class).
        Side effects: none on disk; performs two in-memory CRS reprojections.
        """
        self.lat = float(lat)    # center latitude, degrees
        self.lon = float(lon)    # center longitude, degrees
        self.tilesize = int(tilesize)        # tile edge, pixels
        self.resolution = float(resolution)  # ground sample distance, m/px
        self.pad = 0  # no border pixels; attribute kept for DLTile API parity

        # Build a REAL grid-snapped DLTile at this location first, solely to
        # inherit the correct UTM zone CRS for this lon/lat; the snapped tile
        # itself is then discarded (we re-center everything below).
        snap_tile = DLTile.from_latlon(
            self.lat, self.lon,
            resolution=self.resolution,
            tilesize=self.tilesize,
            pad=self.pad
        )
        self.crs = snap_tile.crs  # UTM zone EPSG code, e.g. "EPSG:32720"

        # Project the requested center point from lon/lat degrees into UTM
        # meters -- all tile-extent math below happens in meters, not degrees.
        point = gpd.GeoSeries(
            [Point(self.lon, self.lat)], crs="EPSG:4326").to_crs(self.crs)
        x_center, y_center = float(point.x.iloc[0]), float(point.y.iloc[0])  # m

        # Half the tile's ground extent in meters: tilesize px * m/px / 2
        # (48 px * 10 m / 2 = 240 m for the default point-scoring tile).
        half_m = (self.tilesize * self.resolution) / 2.0
        raw_minx = x_center - half_m  # unsnapped west edge (m)
        raw_maxx = x_center + half_m  # unsnapped east edge (m)
        raw_miny = y_center - half_m  # unsnapped south edge (m)
        raw_maxy = y_center + half_m  # unsnapped north edge (m)

        # Snap the upper-left corner OUTWARD onto the 10 m pixel grid so pixel
        # edges land on whole multiples of the resolution: floor() pushes the
        # west edge west, ceil() pushes the north edge north.
        ulx = np.floor(raw_minx / self.resolution) * self.resolution
        uly = np.ceil(raw_maxy / self.resolution) * self.resolution

        # Standard north-up geotransform: translate to the UL corner, then
        # scale by the pixel size.  The y pixel size is NEGATIVE because the
        # row index grows southward while the y (northing) coordinate grows
        # northward.
        self.geotrans = Affine.translation(ulx, uly) * Affine.scale(
            self.resolution, -self.resolution)

        width = height = self.tilesize
        minx = ulx
        maxx = ulx + width * self.resolution
        maxy = uly
        miny = uly - height * self.resolution
        self.bounds = (minx, miny, maxx, maxy)
        
        self.shape = (self.tilesize, self.tilesize)
        self.key = (f"custom_{round(self.lat, 6)}_{round(self.lon, 6)}" +
                    f"_{self.resolution:.1f}_{self.tilesize}px")

        bbox_proj = box(minx, miny, maxx, maxy)
        self.geometry = gpd.GeoSeries(
            [bbox_proj], crs=self.crs).to_crs("EPSG:4326").iloc[0]

    def __repr__(self):
        return f"<CenteredTile key={self.key} res={self.resolution} size={self.tilesize}>"


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
    tiles = DLTile.iter_from_shape(
        region, tilesize=tilesize, resolution=resolution, pad=pad
    )
    return list(tiles)


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

    raster_height, raster_width, _ = raster.shape

    # --- Compute cropping (trim from center if too big) ---
    dh = max(raster_height - height, 0)
    dw = max(raster_width - width, 0)

    trim_top = dh // 2
    trim_bottom = raster_height - (dh - trim_top)
    trim_left = dw // 2
    trim_right = raster_width - (dw - trim_left)

    cropped = raster[trim_top:trim_bottom, trim_left:trim_right, :]

    # --- Compute padding (reflect if too small) ---
    ph = max(height - cropped.shape[0], 0)
    pw = max(width - cropped.shape[1], 0)

    pad_top = ph // 2
    pad_bottom = ph - pad_top
    pad_left = pw // 2
    pad_right = pw - pad_left

    padded = np.pad(
        cropped,
        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode="reflect"
    )

    return padded

def cut_chips(pixels, bbox, chip_size, stride, crs):
    """ Create chips and chip geometries from a pixel array."""
    west, south, east, north = bbox
    H = pixels.shape[0]  
    W = pixels.shape[1]  

    x_per_pixel = (east - west) / W
    y_per_pixel = (north - south) / H

    chips = []
    chip_polys = []

    for col in range(0, W - chip_size + stride, stride):
        for row in range(0, H - chip_size + stride, stride):
            chip = pixels[row : row + chip_size, col : col + chip_size]
            chips.append(chip)

            nw_x = west + col * x_per_pixel
            nw_y = north - row * y_per_pixel

            ne_x = west + (col + chip_size) * x_per_pixel
            ne_y = nw_y

            sw_x = nw_x
            sw_y = north - (row + chip_size) * y_per_pixel

            se_x = ne_x
            se_y = sw_y

            poly = Polygon([
                (nw_x, nw_y),
                (sw_x, sw_y),
                (se_x, se_y),
                (ne_x, ne_y),
                (nw_x, nw_y)
            ])

            chip_polys.append(poly)

    chip_geoms = gpd.GeoDataFrame(geometry=chip_polys, crs=crs)
    return np.array(chips), chip_geoms
