import numpy as np
import pandas as pd
import os, sys
from icesat2_functions import inpoly
import datetime

#https://global-surface-water.appspot.com/download
#path to folder with extents (or other types) of NDWI
global_surface_water_dir = '/PATH/TO/Global_Surface_Water/'

#Input file with paths to ATL03 files that you want to mask
input_file = '/PATH/TO/Input_NDWI.txt'




df_inputs = pd.read_csv(input_file,header=0,names=['location'],dtype={'location':'str'})


for i in range(len(df_inputs.location)):
    atl03_in = df_inputs.location[i]
    print(atl03_in)
    atl03_out = atl03_in.split('_masked.txt')
    atl03_time_out = atl03_out[0] + '_OSM_NDWI_masked_time.txt'
    atl03_out = atl03_out[0] + '_OSM_NDWI_masked.txt'
    print(atl03_out)

    NDWI_tif_out = atl03_in.split('_ATL03_high_conf_masked.txt')
    NDWI_tif_clipped_out = NDWI_tif_out[0] + '_NDWI_clipped.tif'
    NDWI_tif_out = NDWI_tif_out[0] + '_NDWI.tif'
    print(NDWI_tif_out)

    NDWI_shp_out = NDWI_tif_out.split('.tif')
    NDWI_shp_out = NDWI_shp_out[0] + '.shp'
    print(NDWI_shp_out)

    t_start = datetime.datetime.now()
    df_atl03 = pd.read_csv(atl03_in,header=None,names=['lon','lat','height','time'],dtype={'lon':'float','lat':'float','height':'float','time':'str'})
    t_end = datetime.datetime.now()
    print('Loading done.')
    dt = t_end - t_start
    dt_min, dt_sec = divmod(dt.seconds,60)
    print('It took:')
    print("%d.%d seconds" %(dt_sec,dt.microseconds%1000000))

    lon = np.asarray(df_atl03.lon)
    lat = np.asarray(df_atl03.lat)
    height = np.asarray(df_atl03.height)
    time = np.asarray(df_atl03.time)
    lon_min = np.min(lon)
    lon_max = np.max(lon)
    lat_min = np.min(lat)
    lat_max = np.max(lat)

    lon_min_10 = int(10*np.floor(lon_min/10))
    lon_max_10 = int(10*np.ceil(lon_max/10))
    lat_min_10 = int(10*np.floor(lat_min/10))
    lat_max_10 = int(10*np.ceil(lat_max/10))

    print(lon_min_10)
    print(lon_max_10)
    print(lat_min_10)
    print(lat_max_10)

    extents_list = []
    
    #extents are listed by their NW coordinate, so:
    #lon: min:10:max
    #lat: max:-10:min
    for ii in range(lon_min_10,lon_max_10,10):
        for jj in range(lat_max_10,lat_min_10,-10):
            if ii < 0:
                lon_letter = 'W'
            else:
                lon_letter = 'E'

            if jj < 0:
                lat_letter = 'S'
            else:
                lat_letter = 'N'

            extents_file = 'extent_'+str(ii)+lon_letter+'_'+str(jj)+lat_letter+'_v1_1.tif'
            extents_list.append(extents_file)
    
    t_start = datetime.datetime.now()

    print(extents_list)
    extents_separator = ' ' + global_surface_water_dir
    extents_list_str = extents_separator + extents_separator.join(extents_list)
    lonlat_str = str(lon_min-0.1) + ' ' + str(lat_min-0.1) + ' ' + str(lon_max+0.1) + ' ' + str(lat_max+0.1)
    
    merge_command = 'gdal_merge.py -co compress=LZW -o ' + NDWI_tif_out + extents_list_str
    print(merge_command)
    os.system(merge_command)

    clip_command = 'gdalwarp -te ' + lonlat_str + ' ' + NDWI_tif_out + ' ' + NDWI_tif_clipped_out
    print(clip_command)
    os.system(clip_command)

    polygonize_command = 'gdal_polygonize.py ' + NDWI_tif_clipped_out + ' -f "ESRI Shapefile" ' + NDWI_shp_out
    print(polygonize_command)
    os.system(polygonize_command)


    t_end = datetime.datetime.now()
    print('Subsetting NDWI done.')
    dt = t_end - t_start
    dt_min, dt_sec = divmod(dt.seconds,60)
    print('It took:')
    print("%d minutes, %d.%d seconds" %(dt_min,dt_sec,dt.microseconds%1000000))

    
    t_start = datetime.datetime.now()
    landmask = inpoly(lon,lat,NDWI_shp_out,True)
    t_end = datetime.datetime.now()
    print('Landmask done.')

    dt = t_end - t_start
    dt_min, dt_sec = divmod(dt.seconds,60)
    dt_hour, dt_min = divmod(dt_min,60)
    print('It took:')
    print("%d hours, %d minutes, %d.%d seconds" %(dt_hour,dt_min,dt_sec,dt.microseconds%1000000))

    lon_NDWI_masked = lon[landmask==True]
    lat_NDWI_masked = lat[landmask==True]
    height_NDWI_masked = height[landmask==True]
    time_NDWI_masked = time[landmask==True]

    f1 = open(atl03_out,'w')
    f1a = open(atl03_time_out,'w')

    np.savetxt(f1,np.c_[lon_NDWI_masked,lat_NDWI_masked,height_NDWI_masked],fmt='%10.5f',delimiter=',')
    np.savetxt(f1a,np.c_[time_NDWI_masked],fmt='%s')

    os.system('paste -d , '+atl03_out + ' ' + atl03_time_out + '> tmp_paste.txt')
    os.system('mv tmp_paste.txt ' + atl03_out)
    os.system('rm ' + atl03_time_out)
    
