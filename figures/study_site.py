import geopandas as gpd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib import gridspec

b = gpd.read_file("data/Boundaries/chk_water_only.shp").to_crs(epsg=4326)
b_bounds = [x for x in b.bounds.iloc[0]]
extent = (b_bounds[0], b_bounds[2], b_bounds[1], b_bounds[3])

ncol = 1
nrow = 1
fig = plt.figure(figsize=(ncol + 3, nrow + 3))
axes = gridspec.GridSpec(nrow, ncol, wspace=0.0, hspace=0.0, top=1, right=0.9)
ax = plt.subplot(axes[0], xlabel="", projection=ccrs.PlateCarree())
ax.set_extent(extent, ccrs.PlateCarree())
ax.coastlines(resolution="10m", color="black", linewidth=1)

plt.show()