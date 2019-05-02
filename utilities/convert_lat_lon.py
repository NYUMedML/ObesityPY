import os
import sys
import math
import time
import pickle
import difflib
import numpy as np
import pandas as pd
import multiprocessing
from datetime import datetime
from datetime import timedelta
from multiprocessing import Pool
from collections import ChainMap

def get_delim(file_name):
    delim = '\t' if file_name[-4:] == '.txt' else ','
    return delim

def create_lat_long_dictionary(args):
    cols, mrns, nan, no_nan, part, start = args

    mrn_l_ix = cols.index('mrn_l')
    mrn_r_ix = cols.index('mrn_r')
    block_ix = cols.index('BLOCKCE10')
    tract_ix = cols.index('TRACTCE10')
    city_ix = cols.index('City_l')
    county_ix = cols.index('COUNTYFP10')
    xc_ix = cols.index('XCoordinate')
    yc_ix = cols.index('YCoordinate')
    x_ix = cols.index('X')
    y_ix = cols.index('Y')
    zip_ix = cols.index('Zip_l')
    enc_ix = cols.index('enc_date_l')
    add_ix = cols.index('Address_Line_1')

    lat_lon_dic = {}
    for ix,mrn in enumerate(mrns):
        if mrn[0]=='0':
            mrn_int=mrn
        else:
            try:
                mrn_int = int(mrn)
            except:
                mrn_int = mrn
        geocoded = no_nan[(no_nan[:,mrn_l_ix]==mrn)]
        if geocoded.shape[0] == 0:
            continue
        try:
            enc = datetime.strptime(geocoded[0,enc_ix], "%Y-%m-%d %H:%M:%S")
        except:
            try:
                enc = datetime(1899, 12, 30) + timedelta(days=geocoded[0,enc_ix])
            except:
                try:
                    enc = datetime(1899, 12, 30) + timedelta(days=int(geocoded[0,enc_ix]))
                except:
                    continue
        lat_lon_dic[mrn] = {}
        lat_lon_dic[mrn][enc] = {}
        block = geocoded[0,block_ix]
        city = geocoded[0,city_ix]
        county = geocoded[0,county_ix]
        xc = geocoded[0,xc_ix]
        yc = geocoded[0,yc_ix]
        lat = geocoded[0,y_ix]
        long = geocoded[0,x_ix]
        zipcode = geocoded[0,zip_ix]
        if str(geocoded[0,tract_ix])[-2:] == 0 and len(str(geocoded[0,tract_ix])) > 3:
            centrac = int(str(geocoded[0,tract_ix])[:-2])
        else:
            centrac = int(geocoded[0,tract_ix])

        lat_lon_dic[mrn][enc]['censblock'] = block
        lat_lon_dic[mrn][enc]['centrac'] = centrac
        lat_lon_dic[mrn][enc]['city'] = city
        lat_lon_dic[mrn][enc]['county'] = county
        lat_lon_dic[mrn][enc]['easting'] = xc
        lat_lon_dic[mrn][enc]['northing'] = yc
        lat_lon_dic[mrn][enc]['lat'] = lat
        lat_lon_dic[mrn][enc]['long'] = long
        lat_lon_dic[mrn][enc]['zip'] = zipcode

        ungeocoded = nan[(nan[:,mrn_l_ix]==mrn)]
        if ungeocoded.shape[0] > 0:
            geo_add = geocoded[0,add_ix]
            ungeo_add_ix = np.array([difflib.SequenceMatcher(a=geo_add.lower(), b=x.lower()).ratio() > 0.6 for x in ungeocoded[:,add_ix]])

            for enc in ungeocoded[ungeo_add_ix,enc_ix]:
                try:
                    enc = datetime.strptime(enc, "%Y-%m-%d %H:%M:%S")
                except:
                    try:
                        enc = datetime(1899, 12, 30) + timedelta(days=enc)
                    except:
                        enc = datetime(1899, 12, 30) + timedelta(days=int(enc))
                lat_lon_dic[mrn][enc] = {}
                lat_lon_dic[mrn][enc]['censblock'] = block
                lat_lon_dic[mrn][enc]['centrac'] = centrac
                lat_lon_dic[mrn][enc]['city'] = city
                lat_lon_dic[mrn][enc]['county'] = county
                lat_lon_dic[mrn][enc]['easting'] = xc
                lat_lon_dic[mrn][enc]['northing'] = yc
                lat_lon_dic[mrn][enc]['lat'] = lat
                lat_lon_dic[mrn][enc]['long'] = long
                lat_lon_dic[mrn][enc]['zip'] = zipcode

        if ix % 500 == 0:
            print('job name: {0:,d};  ix: {1:,d};  complete: {2:0.2f}%;  elapsed: '.format(part,ix,ix/len(mrns)*100) + str(timedelta(seconds=time.time()-start)))

    return lat_lon_dic

def run_parallel(node_count, geo_data, original_data):
    if node_count == None:
        node_count = math.floor(multiprocessing.cpu_count() * .75)

    df_geo = pd.read_csv(geo_data, sep=get_delim(geo_data), low_memory=False)

    df = pd.read_csv(original_data[0], sep=get_delim(original_data[0]), low_memory=False)
    if len(original_data) > 1:
        for data in original_data[1:]:
            df = df.append(pd.read_csv(data, sep=get_delim(data), low_memory=False))

    final = pd.merge(df, df_geo, how='left', on=['encounterid'] ,suffixes=('_l', '_r'))
    final.mrn_l = final.mrn_l.astype(str)
    final.mrn_r = final.mrn_r.astype(str)
    final.enc_date_l = final.enc_date_l.astype(str)
    final.enc_date_r = final.enc_date_r.astype(str)
    try:
        final.X = final.X.astype(float)
        final.Y = final.Y.astype(float)
    except:
        final['X'] = final.XCoordinate.astype(float)
        final['Y'] = final.YCoordinate.astype(float)
    final.COUNTYFP10 = final.COUNTYFP10.fillna(value=-9999).astype(int)
    final.BLOCKCE10 = final.BLOCKCE10.fillna(value=-9999).astype(int)
    cols = final.columns.tolist()
    finalv = final.values
    del df, df_geo, final
    mrn_l_ix = cols.index('mrn_l')
    mrn_r_ix = cols.index('mrn_r')
    xc_ix = cols.index('XCoordinate')

    mrns = np.unique(finalv[:,mrn_l_ix].astype(str)).tolist()

    nan_filter = np.isnan(finalv[:,xc_ix].astype(float))
    no_nan = finalv[~nan_filter,:]
    nan = finalv[nan_filter,:]

    args = []
    n = len(mrns)
    print(str(n), 'total patients')
    start = time.time()
    for i in range(node_count):
        chunk = mrns[int(n*i/node_count):int(n*(i+1)/node_count)]
        args.append([cols, chunk, nan, no_nan, i, start])
        print('job name: {0:d} assigned {1:,d} patients'.format(i,len(chunk)))

    with Pool(node_count) as p:
        print('processing', node_count, 'parallel jobs to create geocoded patient dictionary')
        outputs = p.map(create_lat_long_dictionary, args)
    lat_lon_dict = ChainMap(*outputs)
    print('{0:,d} patients processed'.format(len([*lat_lon_dict])))

    return lat_lon_dict

def main():
    """
    Converts csv file of lat/long pairs to dictionary of mrns with geocoded address information and
    saves as a pickle file. Run in the terminal with input csv file.
    NOTE: must pass 2 arguments:
        1) s: single entry per mrn. m: multiple entries per mrn
        2) <filepath/filename> for file to convert.
    """
    if sys.argv[1] not in ('s','m'):
        raise ValueError('First argument must be "s" or "m"')

    if sys.argv[1] == 's':
        df = pd.read_csv(sys.argv[2])

        lat_lon_dic = {}
        dfv = df.values
        cols = df.columns.tolist()
        ix_mrn = cols.index('mrn')
        ix_block = cols.index('BLOCKCE10')
        ix_tract = cols.index('TRACTCE10')
        ix_city = cols.index('city')
        ix_county = cols.index('COUNTYFP10')
        ix_xc = cols.index('XCoordinate')
        ix_yc = cols.index('YCoordinate')
        ix_x = cols.index('X')
        ix_y = cols.index('Y')
        ix_zip = cols.index('zip')
        for ix in range(df.shape[0]):
            try:
                mrn = int(dfv[ix,ix_mrn])
            except:
                mrn = dfv[ix,ix_mrn]
            if mrn in lat_lon_dic.keys():
                raise ValueError('patient '+str(mrn)+' already exists in dictionary')

            lat_lon_dic[mrn] = {}
            lat_lon_dic[mrn]['censblock'] = dfv[ix,ix_block]
            if str(dfv[ix,ix_tract])[-2:] == '00' and len(str(dfv[ix,ix_tract])) > 3:
                lat_lon_dic[mrn]['centrac'] = int(str(dfv[ix,ix_tract])[:-2])
            else:
                lat_lon_dic[mrn]['centrac'] = dfv[ix,ix_tract]
            lat_lon_dic[mrn]['city'] = dfv[ix,ix_city]
            lat_lon_dic[mrn]['county'] = [dfv[ix,ix_county]]
            lat_lon_dic[mrn]['easting'] = dfv[ix,ix_xc]
            lat_lon_dic[mrn]['northing'] = dfv[ix,ix_yc]
            lat_lon_dic[mrn]['lat'] = dfv[ix,ix_y]
            lat_lon_dic[mrn]['lon'] = dfv[ix,ix_x]
            lat_lon_dic[mrn]['zip'] = dfv[ix,ix_zip]
    else:
        geo_data = '../../raw_data/DB_Geocoding_Updated_deduped_combined.csv'
        original_data = ['../../raw_data/Address_ZipCode_Part1.txt',
                         '../../raw_data/Address_ZipCode_Part2.txt',
                         '../../raw_data/Address_ZipCode_Part3.txt',
                         '../../raw_data/Address_ZipCode_Part4.txt']
        if not os.path.isfile(geo_data):
            geo_data = '/Volumes/CPO/Environment data/DB_Geocoding_Updated_deduped_combined.csv'
            original_data = ['/Volumes/research/CPO_DataBridge/Address_ZipCode_All_patients_of_age_18_or_less_in_eCW_for_at_least_2_years_II/Address_ZipCode_Part1.txt',
                             '/Volumes/research/CPO_DataBridge/Address_ZipCode_All_patients_of_age_18_or_less_in_eCW_for_at_least_2_years_II/Address_ZipCode_Part2.txt',
                             '/Volumes/research/CPO_DataBridge/Address_ZipCode_All_patients_of_age_18_or_less_in_eCW_for_at_least_2_years_II/Address_ZipCode_Part3.txt',
                             '/Volumes/research/CPO_DataBridge/Address_ZipCode_All_patients_of_age_18_or_less_in_eCW_for_at_least_2_years_II/Address_ZipCode_Part4.txt']

        lat_lon_dic = run_parallel(None, geo_data, original_data)

    dt = time.strftime("%Y%m%d")
    fname = '../python objects/lat_lon_data_'+dt+'.pkl'
    pickle.dump(lat_lon_dic, open(fname, 'wb'))
    print('Processing geocoded data complete. Data saved to: '+ fname)

if __name__ == '__main__':
    main()
