import osmnx as ox
import matplotlib.pyplot as plt
from pyproj import CRS
import contextily as cx
from PIL import Image

place_name = "Wallisellen, Zurich, Switzerland"
graph = ox.graph_from_place(place_name, network_type='all')
type(graph)

fig, ax = ox.plot_graph(graph)
plt.tight_layout()

area = ox.geocode_to_gdf(place_name)
buildings = ox.geometries.geometries_from_place(place_name, tags={'building': True})
type(area)
type(buildings)

nodes, edges = ox.graph_to_gdfs(graph)
print(nodes.head())
print(edges.head())

# Set projection
projection = CRS.from_epsg(3067)

# Re-project layers
area = area.to_crs(projection)
edges = edges.to_crs(projection)
buildings = buildings.to_crs(projection)

fig, ax = plt.subplots(figsize=(12,8))

# Plot the footprint
area.plot(ax=ax, facecolor='black')

# Plot street edges
edges.plot(ax=ax, linewidth=1, edgecolor='dimgray')

# Plot buildings
buildings.plot(ax=ax, facecolor='silver', alpha=0.7)

plt.show()


leisure = ox.geometries.geometries_from_place(place_name, tags={'leisure': True})
print(leisure.head(3))
leisure["leisure"].value_counts()
parks = leisure[leisure["leisure"].isin(["park","playground"])]
#parks.plot(color="green")
parks = parks.to_crs(projection)

fig, ax = plt.subplots(figsize=(12,8))

# Plot the footprint
area.plot(ax=ax, color='black')
# Plot the parks
parks.plot(ax=ax, color="green")
# Plot street edges
edges.plot(ax=ax, linewidth=1, edgecolor='red')
# Plot buildings
buildings.plot(ax=ax, color='silver', alpha=0.7)
plt.axis('off')
fig.tight_layout()
plt.show()
fig.savefig('Test-Label.png', format='png', transparent=True,dpi=100, bbox_inches='tight', pad_inches=0)


f, ax = plt.subplots(figsize=(12, 8))
area.plot(alpha=0.5, ax=ax) # must be here for whatever fucking reason... doesnt work without -.-
cx.add_basemap(
    ax,
    crs=area.crs,
    source=cx.providers.Esri.GEE,
    attribution='',
)
ax.set_axis_off()
f.tight_layout()
plt.show()
f.savefig('Test-Data.png', format='png', transparent=False, dpi=100, bbox_inches='tight', pad_inches=0)

