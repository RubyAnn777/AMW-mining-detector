
# =============================================================================
# FILE OVERVIEW — scripts/concatenate.py
#
# "Dumb" merge of GeoJSON FeatureCollections: reads each input file with the
# lightweight `geojson` package, pools all features into one list, and writes
# a single combined FeatureCollection. No geometry processing, no
# deduplication, no reprojection — purely list concatenation.
#
# Pipeline role: utility for combining per-region inference outputs
# (chip-square mine detections with a "confidence" property, EPSG:4326) into
# one file, e.g. before dissolving/thresholding with postprocess.py or
# dissolve.py. (postprocess.py can also concatenate inputs itself.)
#
# Invocation:
#   python scripts/concatenate.py a.geojson b.geojson c.geojson \
#       --outpath combined_detections.geojson
#
# Inputs on disk : two or more GeoJSON FeatureCollections. Each MUST contain
#   an identical top-level 'crs' member. NOTE: the modern GeoJSON spec
#   (RFC 7946) dropped the 'crs' member entirely; older GDAL versions wrote
#   it. Files from RFC 7946-compliant writers lack 'crs' -> KeyError here.
# Outputs on disk: one merged GeoJSON at --outpath
#   (default ./combined_detections.geojson).
# =============================================================================

import argparse  # CLI parsing

import geojson  # lightweight GeoJSON (de)serialization; no geometry ops

def concatenate(paths, outpath):
    """Concatenate FeatureCollections.

    Pools the 'features' arrays of several GeoJSON files into a single
    FeatureCollection, insisting that every file declares the same 'crs'.

    Inputs:
        paths (list[str]): paths of GeoJSON FeatureCollections to merge;
            features are typically chip-square detections in EPSG:4326.
        outpath (str): path for the merged output GeoJSON.
    Returns: None.
    Side effects: reads each input file; writes the merged
        FeatureCollection to ``outpath``.
    Raises:
        ValueError: if any input's 'crs' differs from the first file's.
        KeyError: if an input lacks a 'crs' member (RFC 7946 files do).
    Role in pipeline: pre-step to postprocessing when inference was run in
        several regional batches.
    """
    features = []  # accumulator for every feature dict across all inputs
    crss = []      # one 'crs' member per input file, for the consistency check
    
    for path in paths:  # read the inputs one at a time
        with open(path) as f:
            data = geojson.load(f)  # parse file into a FeatureCollection dict

        # Record this file's 'crs' member for the consistency check below.
        # KeyError here if the file has no 'crs' (see RFC 7946 note above).
        crss.append(data['crs'])
        features += data['features']  # append this file's features to the pool

    # All files must agree on CRS; compare each one against the first.
    common_crs = crss[0]
    for crs in crss:
        if crs != common_crs:
            raise ValueError(f'Inconsisent CRS: {crs} / {common_crs}')

    # Build one FeatureCollection, carrying the shared 'crs' member forward.
    fc = geojson.FeatureCollection(features=features, crs=common_crs)
    with open(outpath, 'w') as f:
        geojson.dump(fc, f)  # serialize the merged collection to disk
        
if __name__ == '__main__':

    # CLI: positional input paths plus an optional output path.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "paths", nargs="*", 
        help="Names of files with model detections to concatenate.",
        default=[])  # zero or more input GeoJSONs (positional args)
    parser.add_argument(
        "--outpath", type=str, 
        help="Path to write concatenated detections",
        default='./combined_detections.geojson')  # default output filename

    args = parser.parse_args()   # parse CLI into a Namespace
    concatenate(**vars(args))    # forward args as keyword arguments
