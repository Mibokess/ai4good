import osmnx as ox
import geopy
import geopy.distance
import ee

from itertools import accumulate
import math

import warnings
warnings.filterwarnings("ignore")

ee.Initialize()
ox.config(use_cache=True, log_console=False)


def generate_points_from_name(name, distance=6):
    area = ox.geocode_to_gdf(name)

    w, s, e, n = area.bounds.values[0]

    nw_point = geopy.Point((n, w))
    ne_point = geopy.Point((n, e))
    sw_point = geopy.Point((s, w))
    se_point = geopy.Point((s, e))

    ns_distance = math.ceil((geopy.distance.geodesic(nw_point, sw_point).kilometers / distance))
    we_distance = math.ceil((geopy.distance.geodesic(nw_point, ne_point).kilometers / distance))

    distance_kilometer = geopy.distance.distance(kilometers=distance)

    points_horizontal = list(accumulate(list(range(we_distance - 1)), lambda x, y: distance_kilometer.destination(x, bearing=90), initial=nw_point))
    points = [list(accumulate(range(ns_distance - 1), lambda x, y: distance_kilometer.destination(x, bearing=180), initial=point)) for point in points_horizontal]
    points = [[point[:2] for point in points_x] for points_x in points]

    return points


def get_streets_from_point(point, name, index, folder, distance=3000):
    fig, ax = ox.plot_figure_ground(
        point=point,
        dist=distance,
        network_type="drive_service",
        default_width=0.8,
        street_widths= {
            'motorway':          1.4,
            'motorway-link':     0.8,
            'trunk':             1.4,
            'trunk-link':        0.8,
            'primary':           1.2,
            'primary-link':      0.8,
            'secondary':         1.1,
            'secondary-link':    0.8,
            'tertiary':          0.7,
            'tertiary-link':     0.5,
            'residential':       0.5,
            'living-street':     0.8,
            'bridleway':         0.8,
            'footway':           0.8,
            'cycleway':          0.8,
            'track':             0.8,
            'track-grade1':      0.8,
            'track-grade2':      0.8,
        },
        filepath=f'{folder}/{name.replace(" ", "")}_{index}_streets.png'.replace(" ", ""),
        dpi=900,
        save=True,
        show=False,
        close=True,
    )


def get_buildings_from_point(point, name, index, folder, distance=3000):
    tags = { 'building': True }

    gdf = ox.geometries_from_point(point, tags, dist=distance)
    bbox = ox.utils_geo.bbox_from_point(point=point, dist=distance)

    fig, ax = ox.plot_footprints(
        gdf,
        bbox=bbox,
        color="w",
        dpi=900,
        filepath=f'{folder}/{name}_{index}_buildings.png'.replace(" ", ""),
        save=True,
        show=False,
        close=True,
    )


def get_satellite_from_point(point, name, index, distance=3000):
        n, s, e, w = ox.utils_geo.bbox_from_point(point, dist=distance)
        rec = ee.Geometry.BBox(w, s, e, n)

        dataset = (ee.ImageCollection('USDA/NAIP/DOQQ')
                .filter(ee.Filter.date('2017-01-01', '2020-12-31'))
                .filter(ee.Filter.bounds(rec))
                .mean())

        task = ee.batch.Export.image.toDrive(
                image=dataset,
                scale=1,
                region=rec,
                fileNamePrefix=f'{name}_{index}'.replace(" ", ""),
                maxPixels=1e13,
                fileFormat= 'GeoTIFF',
                formatOptions = {
                    'cloudOptimized': True
                }
        )

        task.start()


def get_maps_from_point(point, name, index, folder, distance=3000):
    get_streets_from_point(point, name, index, folder, distance)
    get_buildings_from_point(point, name, index, folder, distance)


def get_data_from_points(points, name, folder, get_maps, get_satellite, distance=3000):
    for i, points_y in enumerate(points):
        for j, point in enumerate(points_y):
            index = (i, j)

            if get_maps:
                get_maps_from_point(point, name, index, folder, distance=distance)
            if get_satellite:
                get_satellite_from_point(point, name, index, distance=distance)

def get_data_from_name(name, folder='data', get_maps=True, get_satellite=True, distance_in_kilometers=3):
    points = generate_points_from_name(name, distance=distance_in_kilometers * 2)
    get_data_from_points(points, name, folder, get_maps, get_satellite, distance=distance_in_kilometers * 1000)