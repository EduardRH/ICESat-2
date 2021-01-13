import os, sys
import glob
import h5py
import numpy as np
import pandas as pd
import geopandas as gpd
from geopandas.tools import sjoin
from shapely.geometry import Point, Polygon, MultiPoint
import datetime
from datetime import date
from icesat2_functions import strip_gdalinfo_lonlat,cat_str_API,landmask,gps2utc,inpoly
import getpass
from osgeo import gdal, gdalconst

###Written by Eduard Heijkoop, University of Colorado###
###Eduard.Heijkoop@colorado.edu###

#This script will download ICESat-2 ATL03 geolocated photons for a given region.
#The point cloud will be masked with a given shapefile (e.g. a coastline), originally used as ground control points (GCPs)
#Output is a .txt file with ICESat-2 ATL03 data in the format:
#Longitude [deg], Latitude [deg], Height [m above WGS84], Time [UTC]




###################
##Get PW for SRTM##
###################

SRTM_toggle = True
#SRTM is a useful DEM to remove cloud contamination from the ATL03 photon cloud
if SRTM_toggle:
    user = 'YOUR_NASA_EARTHDATA_USERNAME'
    pw = getpass.getpass()
    srtm_threshold = 10 #set your SRTM threshold here

################
##Define paths##
################

input_file = '/YOUR/PATH/TO/INPUT/FILE.txt'
osm_shp_path = '/YOUR/PATH/TO/COASTLINE/SHAPEFILE.shp'
tmp_dir = '/YOUR/PATH/TO/tmp_directory/'
icesat2_dir = '/YOUR/OUTPUT/PATH/FOR/RESULTS/'
egm96_path = '/YOUR/PATH/TO/EGM96.tif' #supplied on github

#############################################
##Check if Token is still valid and load it##
#############################################

os.system('stat Token.txt > tmp_stats_token.txt')
for line in open('tmp_stats_token.txt'):
    if "Modify:" in line:
        date_str = line

date_str = date_str.split(" ")
date_str = date_str[1]

today_datetime = date.today()
token_datetime = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
delta_datetime = today_datetime - token_datetime
if delta_datetime.days > 30:
    print('WARNING!')
    print('Token expired!')
    print('Exiting...')
    sys.exit()

for line in open('Token.txt'):
    token=line
token = token.rstrip('\n')

#"header=None" if you don't want a header in your input file
#"heard=0" if you do want a header, like "Name,lon_min,lon_max,lat_min,lat_max"
df_extents = pd.read_csv(input_file,header=None,names=['city','lon_min','lon_max','lat_min','lat_max'],dtype={'city':'str','lon_min':'float','lon_max':'float','lat_min':'float','lat_max':'float'})

for i in range(len(df_extents.city)):
    ##############
    ##SUBSET OSM##
    ##############

    
    city_name = df_extents.city[i]
    if not os.path.isdir(icesat2_dir+city_name):
        os.system('mkdir ' + icesat2_dir + city_name)
    lon_min_str = str(df_extents.lon_min[i])
    lon_max_str = str(df_extents.lon_max[i])
    lat_min_str = str(df_extents.lat_min[i])
    lat_max_str = str(df_extents.lat_max[i])
    
    #Use GDAL's ogr2ogr to clip shapefile to extents specified (speeds up landmasking)
    extents_str = lon_min_str + ' ' + lat_min_str + ' ' + lon_max_str + ' ' + lat_max_str
    output_shp = icesat2_dir + city_name + '/' + city_name + '.shp'
    subset_shp_path = output_shp
    shp_command = 'ogr2ogr ' + output_shp + ' ' +  osm_shp_path + ' -clipsrc ' + extents_str
    os.system(shp_command)


    #####################
    ##Download ICESat-2##
    #####################
    #See: https://nsidc.org/support/how/how-do-i-programmatically-request-data-services#curl

    token_command = 'token='+token
    site_command = 'https://n5eil02u.ecs.nsidc.org/egi/request?'
    email_command = 'email=false'
    short_name = 'ATL03'
    coverage_command = 'coverage='
    beam_list = ['1l','1r','2l','2r','3l','3r']
    for beam in beam_list:
        coverage_command = coverage_command + cat_str_API(beam)
    coverage_command = coverage_command + '/orbit_info/sc_orient,/ancillary_data/atlas_sdp_gps_epoch,/ancillary_data/data_start_utc,/ancillary_data/data_end_utc'

    #May need to periodically update ICESat-2 version number!
    short_name_command = 'short_name=' + short_name + '&version=003'
    time_command = ''

    bounding_box_command = 'bounding_box='+lon_min_str+','+lat_min_str+','+lon_max_str+','+lat_max_str
    bbox_command = 'bbox='+lon_min_str+','+lat_min_str+','+lon_max_str+','+lat_max_str
    shape_command = bounding_box_command + '&' + bbox_command + '&'

    #API will give at most 10 subsetted .H5 files in a single zip file.
    #Iterating the command page_num=N (where N is page number) allows you to get everything
    #API doesn't say how many files are available, so must check response-header.txt and evaluate code
      #200 is good, download data
      #501 means it's empty
      #404 unknown is a common error in URLs, these numbers are in the same category
    page_number = 1
    #iterate over page numbers
    page_condition = True

    while page_condition:
        page_command = 'page_num='+str(page_number)
        full_command = 'curl -O -J -k --dump-header response-header.txt \"' + site_command + '&' + short_name_command + '&' + token_command + '&' + email_command + '&' + shape_command + time_command + coverage_command + '&' + page_command + '\"'

        #print('Running this command:')
        #print(full_command)
        os.system(full_command)

        with open('response-header.txt','r') as f2:
            response_line = f2.readline().replace('\n','')
        if response_line[9:12] == '200':
            page_number = page_number + 1
        elif response_line[9:12] == '501':
            page_condition = False
        else:
            print('Something bad happened.')
            print('Exiting...')
            page_condition = False
    
    #####################
    ##Unzip & Move Data##
    #####################

    os.system('mv *zip ' + icesat2_dir+city_name + '/')
    os.system('mv *h5 ' + icesat2_dir+city_name + '/')
    os.system('rm *xml')
    os.system('rm response-header.txt')

    os.system('unzip \'' + icesat2_dir+city_name + '/*zip\' -d ' + icesat2_dir+city_name + '/')
    os.system('mv ' + icesat2_dir+city_name + '/*/processed*.h5 ' + icesat2_dir+city_name+ '/')
    os.system('rm -rf ' + icesat2_dir+city_name + '/1*')
    os.system('rm ' + icesat2_dir+city_name + '/5*zip')
    os.system('find ' + icesat2_dir+city_name + '/*h5 -printf "%f\\'+'n" > ' + icesat2_dir+city_name + '/icesat2_list.txt')
    
    icesat2_list = icesat2_dir+city_name + '/icesat2_list.txt'
    
    with open(icesat2_list) as f3:
        file_list = f3.read().splitlines()
    
    beam_list_r = ['gt1r','gt2r','gt3r']
    beam_list_l = ['gt1l','gt2l','gt3l']

    #Initialize arrays and start reading .h5 files
    lon = np.empty([0,1],dtype=float)
    lat = np.empty([0,1],dtype=float)
    h = np.empty([0,1],dtype=float)
    signal_conf = np.empty(shape=[0,5],dtype=float)

    lon_high_conf = np.empty([0,1],dtype=float)
    lat_high_conf = np.empty([0,1],dtype=float)
    h_high_conf = np.empty([0,1],dtype=float)

    delta_time_total_high_conf = np.empty([0,1],dtype=float)

    for h5_file in file_list:

        full_file = icesat2_dir + city_name + '/' + h5_file
        atl03_file = h5py.File(full_file,'r')
        list(atl03_file.keys())

        sc_orient = atl03_file['/orbit_info/sc_orient']
        sc_orient = sc_orient[0]

        if sc_orient == 1:
            beam_list_req = beam_list_r
        elif sc_orient == 0:
            beam_list_req = beam_list_l
        elif sc_orient == 2:
            continue

        for beam in beam_list_req:
            #Some beams don't actually have any height data in them, so this is done to skip those
            heights_check = False
            heights_check = '/'+beam+'/heights' in atl03_file
            if heights_check == False:
                continue

            tmp_lon = np.asarray(atl03_file['/'+beam+'/heights/lon_ph']).squeeze()
            tmp_lat = np.asarray(atl03_file['/'+beam+'/heights/lat_ph']).squeeze()
            tmp_h = np.asarray(atl03_file['/'+beam+'/heights/h_ph']).squeeze()

            tmp_sdp = np.asarray(atl03_file['/ancillary_data/atlas_sdp_gps_epoch']).squeeze()
            tmp_delta_time = np.asarray(atl03_file['/'+beam+'/heights/delta_time']).squeeze()
            tmp_delta_time_total = tmp_sdp + tmp_delta_time

            tmp_signal_conf = np.asarray(atl03_file['/'+beam+'/heights/signal_conf_ph'])

            tmp_high_conf = tmp_signal_conf[:,0] == 4

            if len(tmp_high_conf) < 100:
                continue

            tmp_lon_high_conf = tmp_lon[tmp_high_conf]
            tmp_lat_high_conf = tmp_lat[tmp_high_conf]
            tmp_h_high_conf = tmp_h[tmp_high_conf]
            tmp_delta_time_total_high_conf = tmp_delta_time_total[tmp_high_conf]

            lon = np.append(lon,tmp_lon)
            lat = np.append(lat,tmp_lat)
            h = np.append(h,tmp_h)
            
            #signal_conf = np.concatenate((signal_conf,tmp_signal_conf),axis=0)

            lon_high_conf = np.append(lon_high_conf,tmp_lon_high_conf)
            lat_high_conf = np.append(lat_high_conf,tmp_lat_high_conf)
            h_high_conf = np.append(h_high_conf,tmp_h_high_conf)
            delta_time_total_high_conf = np.append(delta_time_total_high_conf,tmp_delta_time_total_high_conf)
    
    
    print('Running landmask...')
    t_start = datetime.datetime.now()
    #create boolean array of ICESat-2 returns, whether or not they're inside the given shapefile 
    landmask = inpoly(lon_high_conf,lat_high_conf,subset_shp_path,True)
    t_end = datetime.datetime.now()
    print('Landmask done.')

    dt = t_end - t_start
    dt_min, dt_sec = divmod(dt.seconds,60)
    dt_hour, dt_min = divmod(dt_min,60)
    print('It took:')
    print("%d hours, %d minutes, %d.%d seconds" %(dt_hour,dt_min,dt_sec,dt.microseconds%1000000))

    #can easily change this to False if you only want ICESat-2 OUTSIDE the shapefile
    lon_high_conf_masked = lon_high_conf[landmask==True]
    lat_high_conf_masked = lat_high_conf[landmask==True]
    h_high_conf_masked = h_high_conf[landmask==True]
    delta_time_total_high_conf_masked = delta_time_total_high_conf[landmask==True]

    utc_time_high_conf_masked = gps2utc(delta_time_total_high_conf_masked)

    icesat2_file = icesat2_dir + city_name + '/' + city_name + '_ATL03_high_conf_masked.txt'
    icesat2_time_file = icesat2_dir + city_name + '/' + city_name + '_ATL03_high_conf_masked_time.txt'
    f4 = open(icesat2_file,'w')
    f4a = open(icesat2_time_file,'w')

    np.savetxt(f4,np.c_[lon_high_conf_masked,lat_high_conf_masked,h_high_conf_masked],fmt='%10.5f',delimiter=',')
    np.savetxt(f4a,np.c_[utc_time_high_conf_masked],fmt='%s')
    f4.close()
    f4a.close()
    
    ###############
    ##SRTM Filter##
    ###############

    if SRTM_toggle:
        srtm_sampled_file = icesat2_file.strip('.txt')
        srtm_sampled_file = srtm_sampled_file[0] + '_SRTM_sampled.txt'
        srtm_filtered_file = icesat2_file.strip('.txt')
        srtm_filtered_file = srtm_filtered_file + '_SRTM_filtered_threshold_'+str(srtm_threshold) + '_m.txt'
        srtm_filtered_time_file = icesat2_time_file.strip('.txt')
        srtm_filtered_time_file = srtm_filtered_time_file + '_SRTM_filtered_threshold_'+str(srtm_threshold)+'_m_time.txt'

        lon_min_srtm = np.min(lon_high_conf_masked)
        lon_max_srtm = np.max(lon_high_conf_masked)
        lat_min_srtm = np.min(lat_high_conf_masked)
        lat_max_srtm = np.max(lat_high_conf_masked)

        SRTM_list = []
        lon_range = range(int(np.floor(lon_min_srtm)),int(np.floor(lon_max_srtm))+1)
        lat_range = range(int(np.floor(lat_min_srtm)),int(np.floor(lat_max_srtm))+1)

        for jj in range(len(lon_range)):
            for kk in range(len(lat_range)):
                if lon_range[jj] >= 0:
                    lonLetter = 'E'
                else:
                    lonLetter = 'W'
                if lat_range[kk] >= 0:
                    latLetter = 'N'
                else:
                    latLetter = 'S'
                lonCode = f"{int(np.abs(np.floor(lon_range[jj]))):03d}"
                latCode = f"{int(np.abs(np.floor(lat_range[kk]))):02d}"
                SRTMID = latLetter + latCode + lonLetter + lonCode
                SRTM_list.append(SRTMID)
        
        merge_command = 'gdal_merge.py -o ' + icesat2_dir + city_name + '/' + city_name + '_SRTM.tif '
        for jj in range(len(SRTM_list)):
            DL_command = 'wget --user=' + user + ' --password=' + pw + ' https://e4ftl01.cr.usgs.gov//MODV6_Dal_D/SRTM/SRTMGL1.003/2000.02.11/' + SRTM_list[jj] + '.SRTMGL1.hgt.zip'
            os.system(DL_command)
            exists = os.path.isfile(SRTM_list[jj] + '.SRTMGL1.hgt.zip')
            if exists:
                mv_command = 'mv ' + SRTM_list[jj] + '.SRTMGL1.hgt.zip ' + icesat2_dir + city_name + '/'
                os.system(mv_command)
                unzip_command = 'unzip ' + icesat2_dir + city_name + '/' + SRTM_list[jj] + '.SRTMGL1.hgt.zip -d ' + icesat2_dir + city_name + '/'
                os.system(unzip_command)
                delete_command = 'rm ' + icesat2_dir + city_name + '/' + SRTM_list[jj] + '.SRTMGL1.hgt.zip'
                os.system(delete_command)

                merge_command = merge_command + icesat2_dir + city_name + '/' + SRTM_list[jj] + '.hgt '

        os.system(merge_command)
        print('Merged SRTM')
        #SRTM is referenced as heights above/below EGM96, so need EGM96 referenced to WGS84 to get SRMT referenced to WGS84
        #This merges all SRTM tiles together into a single geotiff with the right vertical reference
        src_filename = egm96_path
        src = gdal.Open(src_filename, gdalconst.GA_ReadOnly)
        src_proj = src.GetProjection()
        src_geotrans = src.GetGeoTransform()

        match_filename = icesat2_dir + city_name + '/' + city_name + '_SRTM.tif'
        match_ds = gdal.Open(match_filename,gdalconst.GA_ReadOnly)
        match_proj = match_ds.GetProjection()
        match_geotrans = match_ds.GetGeoTransform()
        wide = match_ds.RasterXSize
        high = match_ds.RasterYSize

        dst_filename = icesat2_dir + city_name + '/EGM96_' + city_name + '.tif'
        dst = gdal.GetDriverByName('GTiff').Create(dst_filename, wide, high, 1, gdalconst.GDT_Float32)
        dst.SetGeoTransform( match_geotrans )
        dst.SetProjection( match_proj)
        gdal.ReprojectImage(src, dst, src_proj, match_proj, gdalconst.GRA_NearestNeighbour)
        del dst
        srtm_wgs84_file = icesat2_dir + city_name + '/' + city_name + '_SRTM_WGS84.tif'
        os.system('gdal_calc.py -A ' + dst_filename + ' -B ' + match_filename + ' --outfile ' + srtm_wgs84_file + ' --calc=A+B')
        os.system('rm ' + match_filename)
        os.system('rm ' + dst_filename)
        for jj in range(len(SRTM_list)):
            os.system('rm ' + icesat2_dir + city_name + '/' + SRTM_list[jj] + '.hgt')
    
        #Sample ICESat-2 over SRTM
        srtm_sampled_file = icesat2_dir + city_name + '/' + city_name + '_sampled_SRTM.txt'
        print('Sampling SRTM...')
        os.system('gmt grdtrack ' + icesat2_file + ' -G' + srtm_wgs84_file + ' > ' + srtm_sampled_file)
        print('Sampled SRTM')
        os.system('rm ' + srtm_wgs84_file)
    
        df_srtm = pd.read_csv(srtm_sampled_file,header=None,names=['lon','lat','h_orig','h_srtm'],dtype={'lon':'float','lat':'float','h_orig':'float','h_srtm':'float'})
    
        h_orig = np.asarray(df_srtm.h_orig)
        h_sampled_srtm = np.asarray(df_srtm.h_srtm)
        srtm_cond = np.abs(h_orig - h_sampled_srtm) < srtm_threshold
        
        lon_filtered_srtm = np.asarray(df_srtm.lon)
        lat_filtered_srtm = np.asarray(df_srtm.lat)
        
        lon_filtered_srtm = lon_filtered_srtm[srtm_cond]
        lat_filtered_srtm = lat_filtered_srtm[srtm_cond]
        h_filtered_srtm = h_orig[srtm_cond]
        delta_time_filtered_srtm = delta_time_total_high_conf_masked[srtm_cond]
        utc_time_filtered_srtm = gps2utc(delta_time_filtered_srtm)

        os.system('rm ' + srtm_sampled_file)
        f5 = open(srtm_filtered_file,'w')
        f5a = open(srtm_filtered_time_file,'w')

        np.savetxt(f5,np.c_[lon_filtered_srtm,lat_filtered_srtm,h_filtered_srtm],fmt='%10.5f',delimiter=',')
        np.savetxt(f5a,np.c_[utc_time_filtered_srtm],fmt='%s')
        f5.close()
        f5a.close()

        os.system('paste -d , '+srtm_filtered_file+' '+srtm_filtered_time_file+ ' > ' + tmp_dir + 'tmp_paste.txt')
        os.system('mv ' + tmp_dir + 'tmp_paste.txt ' + srtm_filtered_file)
        os.system('rm ' + srtm_filtered_time_file)
    
    os.system('paste -d , '+icesat2_file+' '+icesat2_time_file+ ' > ' + tmp_dir + 'tmp_paste.txt')
    os.system('mv ' + tmp_dir + 'tmp_paste.txt ' + icesat2_file)
    os.system('rm ' + icesat2_time_file)

    now = datetime.datetime.now()
    print('Done with ' + city_name + ' at:')
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
