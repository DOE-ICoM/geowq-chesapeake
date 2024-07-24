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
            )  # , vmax=np.nanmax(img_cbofs.to_numpy()))
    if vector:
        geo_grid = geo_grid.to_crs(ccrs.PlateCarree())
        geo_grid.plot(column="predict", ax=ax, markersize=0.5, legend=False)

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

# breakpoint()

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
    lat_ticks=[37, 38, 39],
    lon_ticks=[-77, -76, -75],
    vector=True,
)


panel_add(1, axs, "NN-interpolation", img_rf, height_frac=0.6, lat_ticks=[37, 38, 39],
    lon_ticks=[-77, -76, -75],)
axs[1].tick_params(colors='white')

panel_add(2, axs, "Target", img_cbofs, height_frac=0.8, lat_ticks=[37, 38, 39],
    lon_ticks=[-77, -76, -75],)
axs[2].tick_params(colors='white')

# fig.delaxes(fig.axes[0])
# fig.delaxes(fig.axes[1])
# fig.delaxes(fig.axes[2])
# plt.tight_layout()
# axs[0].legend().set_visible(False)
# axs[1].legend().set_visible(False)
# plt.legend('',frameon=False)
# fig.legends = []
# axs[1].legend_ = None
# plt.margins(y=1)
fig.subplots_adjust(top=0.5, bottom=0.1)

# plt.show()
plt.savefig("test.pdf")
# plt.savefig("figures/_rf-vs-cbofs.pdf")
