# -*- coding: utf-8 -*-
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from cartopy.feature import NaturalEarthFeature, COLORS

modis_proj4 = '+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs'  # pulled from GEE


def modisLonLat(p_idx):
    # Some MODIS grid info for MYDOCGA
    modis_proj4 = '+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs'  # pulled from GEE
    modis_gt = (
        926.625433056, 0, -20015109.354, 0, -926.625433055, 10007554.677
    )  # geotransform pulled from GEE: (xScale, xShearing, xTranslation, yShearing, yScale, yTranslation)
    modis_shape = (43200, 21600)  # ncols, nrows
    r, c = np.unravel_index(p_idx, modis_shape[::-1])
    lons = modis_gt[2] + modis_gt[0] * c + modis_gt[0] / 2
    lats = modis_gt[5] + modis_gt[4] * r + modis_gt[4] / 2
    return lons, lats


def split_pixelday(df):
    ''' taken from the second (unaccepted) answer at https://stackoverflow.com/questions/14745022/how-to-split-a-dataframe-string-column-into-two-columns'''
    df[['pix_id', 'date']] = df['pixelday'].str.split('_', 1, expand=True)
    df['pix_id'] = df['pix_id'].astype('int')
    return df


def plot_counts(df,
                column,
                startDate='2001-01-01 00:00:00',
                endDate='2021-01-01 00:00:00'):
    dfsub = df.loc[startDate:endDate].groupby('date').count()
    plt.figure(figsize=(15, 10))
    plt.title('# of Pixels Containing Observations: ' + column[:-5],
              fontsize=40)
    plt.plot(dfsub.index, dfsub[column], '.k', markersize=6)
    plt.xlabel('Date', fontsize=32)
    plt.ylabel('Count', fontsize=32)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()


# mean time series of an individual or group of modis pixels
def plot_timeseries(gdf,
                    pix_ids,
                    column,
                    pix_id_col='pix_id',
                    startDate='2001-01-01 00:00:00',
                    endDate='2021-01-01 00:00:00'):
    gdfsub = gdf[gdf[pix_id_col].isin(pix_ids)].groupby([pix_id_col, 'date'
                                                         ])[column].mean()
    fig, ax = plt.subplots(figsize=(15, 10))
    gdfsub.reset_index().pivot('date', pix_id_col, column).plot(ax=ax)
    # plot time series of each point
    plt.xlabel('Date', fontsize=32)
    plt.ylabel(column, fontsize=32)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()


ocean = NaturalEarthFeature(category='physical',
                            name='ocean',
                            scale='10m',
                            facecolor=COLORS['water'])

land = NaturalEarthFeature(category='physical',
                           name='land',
                           scale='10m',
                           facecolor=COLORS['land'])

states_provinces = NaturalEarthFeature(category='cultural',
                                       name='admin_1_states_provinces_lines',
                                       scale='10m',
                                       facecolor='none',
                                       edgecolor='gray')


def map_counts(df,
               column,
               startDate='2001-01-01',
               endDate=None,
               extents=[-77.458841, -74.767094, 36.757802, 39.920274],
               num=5):

    if endDate == None:
        endDate = startDate

    dfsub = df.loc[startDate:endDate].groupby('pix_id').aggregate(
        ['mean', 'count'])

    if len(dfsub) == 0:
        return print('No Data on this day')

    # create dummy plot in order to get information for creating the legend
    plt.figure(figsize=[15, 15])
    ax = plt.scatter(dfsub.longitude['mean'],
                     dfsub.latitude['mean'],
                     s=dfsub[column]['count'],
                     zorder=2)
    handles, labels = ax.legend_elements(prop="sizes",
                                         num=num,
                                         markerfacecolor='none',
                                         markeredgecolor='k')
    plt.close()

    plt.figure(figsize=[15, 15])
    # set up the actual background map
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(extents)
    ax.coastlines(resolution='10m', color='black', linewidth=1)
    ax.add_feature(land)
    ax.add_feature(ocean)
    ax.add_feature(states_provinces, linestyle=':')

    # manuallly add some state labels could probably put in a list for one call
    ax.text(-76.98, 39.56, 'Maryland', transform=ccrs.Geodetic(), color='gray')
    ax.text(-76.98, 39.81, 'New York', transform=ccrs.Geodetic(), color='gray')
    ax.text(-75.1,
            39.56,
            'New Jersey',
            transform=ccrs.Geodetic(),
            color='gray')
    ax.text(-75.5, 38.7, 'Delaware', transform=ccrs.Geodetic(), color='gray')
    ax.text(-77.16, 36.93, 'Virginia', transform=ccrs.Geodetic(), color='gray')

    # plot the count information and add a rough legend
    ax.scatter(dfsub.longitude['mean'],
               dfsub.latitude['mean'],
               s=dfsub[column]['count'],
               zorder=2,
               facecolors='none',
               edgecolors='k')
    ax.legend(handles[::int(num / 4)],
              labels[::int(num / 4)],
              loc="lower right",
              title="Number of \n Observations",
              handletextpad=3.5,
              labelspacing=1.5,
              borderpad=3)
    plt.title('Total Number of Observations: ' + column)
    plt.savefig("figures/" + "".join(map(str.lower, column)).split(" ")[0] +
                "_map_counts.pdf")


def map_variable(df,
                 column,
                 startDate='2008-04-17',
                 endDate=None,
                 num=5,
                 extents=[-77.458841, -74.767094, 36.757802, 39.920274]):

    if endDate == None:
        endDate = startDate

    dfsub = df.loc[startDate:endDate].groupby('pix_id').mean()
    if len(dfsub) == 0:
        return print('No Data on this day')

    # create dummy plot in order to get information for creating the legend
    plt.figure(figsize=[15, 15])
    ax = plt.scatter(dfsub.longitude,
                     dfsub.latitude,
                     c=dfsub[column],
                     zorder=2)
    handles, labels = ax.legend_elements(prop="colors", num=num)
    plt.close()

    plt.figure(figsize=[15, 15])
    # set up the actual background map
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(extents)
    ax.coastlines(resolution='10m', color='black', linewidth=1)
    ax.add_feature(land)
    ax.add_feature(ocean)
    ax.add_feature(states_provinces, linestyle=':')

    # manuallly add some state labels could probably put in a list for one call
    ax.text(-76.98, 39.56, 'Maryland', transform=ccrs.Geodetic(), color='gray')
    ax.text(-76.98, 39.81, 'New York', transform=ccrs.Geodetic(), color='gray')
    ax.text(-75.1,
            39.56,
            'New Jersey',
            transform=ccrs.Geodetic(),
            color='gray')
    ax.text(-75.5, 38.7, 'Delaware', transform=ccrs.Geodetic(), color='gray')
    ax.text(-77.16, 36.93, 'Virginia', transform=ccrs.Geodetic(), color='gray')

    # plot the count information and add a rough legend
    ax.scatter(dfsub.longitude, dfsub.latitude, c=dfsub[column], zorder=2)
    ax.legend(handles, labels, loc="lower right", title=column)
    plt.title('Average ' + column + ': ' + startDate + ' - ' + endDate)
    plt.show()


#def load_dataframe(filepath,filename,datetime_col = 'datetime',pix_id_col = 'pix_id',date_format = '%Y-%m-%d %H:%M:%S'):
#    df = pd.read_csv(filepath + filename)
#    df['datetime'] = pd.to_datetime(df[datetime_col],format = date_format)
#    df.index = df['datetime']
#    df['date'] = pd.to_datetime(df['datetime'].dt.strftime('%Y-%m-%d'))
#    df.longitude,df.latitude = modisLonLat(df[pix_id_col].values)
#    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude),crs = CRS.from_proj4(modis_proj4))
#    gdf = gdf.to_crs(epsg = 4326)
#    return gdf