import sys
import xarray as xr
import geopandas as gpd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

sys.path.append(".")
from src import utils


def panel_add(
    i,
    axs,
    title,
    geo_grid,
    diff=False,
    j=None,
    height_frac=0.5,  # larger moves the title up, smaller moves it down
    lon_ticks=[],
    lat_ticks=[],
    vector=False,
    bounds=None,
):
    if j is not None:
        ax = plt.subplot(axs[i, j], xlabel="", projection=ccrs.PlateCarree())
        ax.coastlines(resolution="10m", color="black", linewidth=1)
    else:
        ax = axs[i]
        ax.coastlines(resolution="10m", color="black", linewidth=1)

    if len(lon_ticks) > 0 and len(lat_ticks) > 0:
        ax.set_xticks(lon_ticks, crs=ccrs.PlateCarree())
        ax.set_yticks(lat_ticks, crs=ccrs.PlateCarree())
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)

    if not vector:
        if diff:
            geo_grid.plot.imshow(ax=ax, center=0)
        else:
            geo_grid.plot.imshow(
                ax=ax,
                add_labels=False,
                add_colorbar=False,
            )
    if vector:
        geo_grid = geo_grid.to_crs(ccrs.PlateCarree())
        geo_grid.sample(4000).plot(column="predict", ax=ax, markersize=0.1, legend=False)

    if bounds is not None:
        # minx, miny, maxx, maxy
        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])

    ax.set_title(title, size="small", y=height_frac, x=0.9, rotation="vertical")


# --- rf vs cbofs map
bay_gdf_hires = gpd.read_file("data/Boundaries/chk_water_only.shp").to_crs(epsg=4326)

# get cbofs image
tod = "20220904"
tif_path = "data/cbofs/salt_{date}.tif".format(date=tod)
img_cbofs = xr.open_dataset(tif_path)
img_cbofs = img_cbofs.rio.clip(bay_gdf_hires.geometry)
img_cbofs = img_cbofs["band_data"].sel(band=1)
# img_cbofs.plot.imshow()
# plt.show()

img_rf = utils.get_rf_prediction("2022-09-04", "salinity")

pnts_raw = gpd.read_file("test3.gpkg")

# minx, miny, maxx, maxy
# bounds = pnts_raw.to_crs(ccrs.PlateCarree()).total_bounds
bounds = (-76.44, 36.88, -75.661, 37.952)

plt.close()
fig, axs = plt.subplots(
    ncols=3,
    nrows=1,
    # constrained_layout=True,
    subplot_kw={"projection": ccrs.PlateCarree(), "xlabel": ""},
)
panel_add(
    0,
    axs,
    "Diffuse missing",
    pnts_raw,
    height_frac=0.65,
    lat_ticks=[37, 37.5],
    lon_ticks=[-76.5, -76, -75.5],
    vector=True,
    bounds=bounds,
)


panel_add(
    1,
    axs,
    "NN-interpolation",
    img_rf,
    height_frac=0.6,
    lat_ticks=[37, 38, 39],
    lon_ticks=[-77, -76, -75],
    bounds=bounds
)
axs[1].tick_params(colors="white")

panel_add(
    2,
    axs,
    "Target",
    img_cbofs,
    height_frac=0.8,
    lat_ticks=[37, 38, 39],
    lon_ticks=[-77, -76, -75],
    bounds=bounds
)
axs[2].tick_params(colors="white")

fig.subplots_adjust(top=0.5, bottom=0.1)

# plt.show()
plt.savefig("test.pdf")
# plt.savefig("figures/_rf-vs-cbofs.pdf")
