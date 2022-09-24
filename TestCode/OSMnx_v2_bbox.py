import osmnx as ox
import matplotlib.pyplot as plt
from pyproj import CRS
import ee
import contextily as cx
from PIL import Image

ee.Initialize()
coord = [-122.20, 37.44, -122.22, 37.46]

place_name = "Harlem, New York, USA"
graph = ox.graph_from_bbox(north=coord[3], south=coord[1], east=coord[2], west=coord[0], network_type='all')
type(graph)

bBox = ee.Geometry.BBox(north=coord[3], south=coord[1], east=coord[2], west=coord[0])

fig, ax = ox.plot_graph(graph)
plt.tight_layout()

#area = ox.geocode_to_gdf(place_name)
buildings = ox.geometries.geometries_from_bbox(north=coord[3], south=coord[1], east=coord[2], west=coord[0], tags={'building': True})
#type(area)
type(buildings)

nodes, edges = ox.graph_to_gdfs(graph)
print(nodes.head())
print(edges.head())

# Set projection
projection = CRS.from_epsg(3067)

# Re-project layers
#area = area.to_crs(projection)
edges = edges.to_crs(projection)
buildings = buildings.to_crs(projection)

fig, ax = plt.subplots(figsize=(12,8))

# Plot the footprint
#area.plot(ax=ax, facecolor='black')

# Plot street edges
edges.plot(ax=ax, linewidth=1, edgecolor='dimgray')

# Plot buildings
buildings.plot(ax=ax, facecolor='silver', alpha=0.7)

plt.show()


leisure = ox.geometries.geometries_from_bbox(north=coord[3], south=coord[1], east=coord[2], west=coord[0], tags={'leisure': True})
print(leisure.head(3))
leisure["leisure"].value_counts()
parks = leisure[leisure["leisure"].isin(["park","playground"])]
#parks.plot(color="green")
parks = parks.to_crs(projection)

fig, ax = plt.subplots(figsize=(12,8))

# Plot the footprint
#area.plot(ax=ax, color='black')
# Plot the parks
parks.plot(ax=ax, color="green")
# Plot street edges
edges.plot(ax=ax, linewidth=1, edgecolor='red')
# Plot buildings
buildings.plot(ax=ax, color='black', alpha=0.7)
plt.axis('off')
fig.tight_layout()
plt.show()
fig.savefig('Test-Label.png', format='png', transparent=True,dpi=300, bbox_inches='tight', pad_inches=0)


# Do the thing with GEE

dataset = ee.ImageCollection('USDA/NAIP/DOQQ').filter(ee.Filter.date('2019-01-01', '2020-12-31'));

task = ee.batch.Export.image.toDrive(image=dataset.mean(),
                                     scale=0.5,
                                     region=bBox,
                                     fileNamePrefix='somewhere',
                                     crs='EPSG:3067',
                                     fileFormat='GEO_TIFF')
task.start()

