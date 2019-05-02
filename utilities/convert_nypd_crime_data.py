import os

import csv
import time
import pickle
import argparse
from datetime import datetime

import utils

def to_datetime(datestr, timestr):
    if timestr == '':
        if datestr = '':
            return ''
        else:
            return datetime.strftime(datestr, '%m/%d/%Y')
    else:
        return datetime.strftime(' '.join((datestr, timestr)), '%m/%d/%Y %H:%M:%S')S

def create_loc_dict(headers, line, location, lat, lon):
    tract = int(location['TRACT'])
    county = int(location['COUNTY'])
    block = int(location['BLOCK'])
    geoid = location['GEOID'] # STATEFIPS (2) + COUNTYFIPS (3) + TRACT (6) + BLOCK (4) = 15 digit 2010 census identifier
    loc_dict = {
    ['\ufeffCMPLNT_NUM', 'CMPLNT_FR_DT', 'CMPLNT_FR_TM', 'CMPLNT_TO_DT', 'CMPLNT_TO_TM', 'RPT_DT', 'KY_CD', 'OFNS_DESC', 
    'PD_CD', 'PD_DESC', 'CRM_ATPT_CPTD_CD', 'LAW_CAT_CD', 'JURIS_DESC', 'BORO_NM', 'ADDR_PCT_CD', 'LOC_OF_OCCUR_DESC', 
    'PREM_TYP_DESC', 'PARKS_NM', 'HADEVELOPT', 'X_COORD_CD', 'Y_COORD_CD', 'Latitude', 'Longitude', 'Lat_Lon']

        'CMPLNT_NUM': line[headers['CMPLNT_NUM']],
        'owner_type': line[headers['owner type']],
        'use': line[headers['known use']],
        'area': line[headers['area acres']],
        'address': line[headers['address line1']],
        'bbl': line[headers['bbl']],
        'zip': line[headers['postal code']],
        'latitude': lat,
        'longitude': lon,
        'block': block,
        'geoid': geoid
    }
    return county, tract, loc_dict

def read_file(fname):
    locations = []
    with open(fname, 'r') as f:
        reader = csv.reader(f)
        for i, line in enumerate(reader):
            if i == 0:
                headers = {el:ix for ix, el in enumerate(line)}
            else:
                line = [el.strip() for el in line]
                location, lat, lon  = utils.geocode(headers, i, line)
                county, tract, loc_dict = create_loc_dict(headers, line, location, lat, lon)
                # create an arbitrary key for each location to store the dictionary
                # this will be undone to form a list of dicts with matching tracts and counties later
                loc_dict = {tract: {county: {str(lon) + str(lat) + loc_dict['address']: [loc_dict]}}}
                locations.append(loc_dict)

    return locations

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Process ACS files from given folder paths')
    parser.add_argument('-f', '--filename', required=True, type=str, metavar='fname', dest='fname', help='filepath/filename to vacant lots data')
    parser.add_argument('-s', '--savepath', required=False, type=str, metavar='save_path', dest='save_path', help='path to save data', default='.')
    # parser.add_argument('-n', '--nodecount', required=False, type=int, metavar='node_count', dest='node_count', help='number of cpus to use for processing', default=math.ceil(mp.cpu_count() * 0.8))
    args = parser.parse_args()

    if args.save_path == '':
        args.save_path = '/'.join(el for el in args.filename.split('/')[:-1])

    vacant_lots_dict = dict(utils.merge_dicts(read_file(args.fname)))
    vacant_lots_dict = {tract: {county: list(countydict.values())} for tract, tractdict in vacant_lots_dict.items() for county, countydict in tractdict.items()}

    dt = time.strftime("%Y%m%d")
    sp = '/'.join((args.save_path, ''.join(('vacant_lots_dict_',dt,'.pkl'))))
    pickle.dump(vacant_lots_dict, open(sp, 'wb'))
    print('vacant lots dictionary saved to {}'.format(sp))
