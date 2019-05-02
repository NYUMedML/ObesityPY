import os

import csv
import math
import pickle
import argparse
import numpy as np
import pandas as pd
import multiprocessing as mp

from functools import reduce
from collections import defaultdict

import utils

def read_headers(fname):
    """
    Function to return headers from files.
    """
    with open(fname, 'r') as f:
        reader = csv.reader(f)
        for i, line in enumerate(reader):
            if i == 0:
                headers = line
            elif i == 1:
                descriptions = line
            else:
                break
    return headers, descriptions

def read_data(fname, descriptions):
    """
    Function to combine headers with data
    """
    data = utils.set_default()
    record_rows = set([*geo_data])
    with open(fname, 'r') as f:
        reader = csv.reader(f)
        for i, line in enumerate(reader):
            if i not in record_rows:
                continue
            for desc, n in zip(descriptions[6:], line[6:]):
                if n in ('','.'):
                    continue
                try:
                    data[geo_data[i]['tract']][geo_data[i]['county']][desc] = int(n)
                except:
                    data[geo_data[i]['tract']][geo_data[i]['county']][desc] = float(n)

    return utils.dd_to_dict(data)

def geo_parse(fname_data, fname_headers):
    """
    Function to parse the geographic data rows of interest

    Parameters
    ----------
    fname_data : str
        File name of data file.
    fname_headers : str
        File name of headers file.

    Returns
    -------
    geo_data : dict
        Dictionary of logical record numbers, county, and census tracts for
        rows of interest within the data columns.
    """
    imp_fields = ['STUSAB', 'LOGRECNO', 'STATE', 'COUNTY', 'TRACT']
    headers, descriptions = read_headers(fname_headers)
    data = pd.read_csv(fname_data, header=None, names=headers, index_col=False, dtype=str)
    data = data.fillna('')
    data = data.values[:, [headers.index(h) for h in imp_fields]]
    ix_filter = ((data[:, 3] == '005') | (data[:, 3] == '047') | (data[:, 3] == '061') | (data[:, 3] == '081') | (data[:, 3] == '085')) & (data[:, 4] != '')
    record_rows = {int(v[1])-1: {'county': int(v[3]), 'tract': int(v[4])} for v in data[ix_filter, :]} # using LOGRECNO - 1 for proper indexing in Python

    return record_rows

def process_single(args):
    i, hf, df = args
    headers, descriptions = read_headers(hf)
    data_dict = read_data(df, descriptions)
    print('File {} processed'.format(i))
    return data_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Process ACS files from given folder paths')
    parser.add_argument('-p', '--headerpath', required=True, type=str, metavar='header_path', dest='header_path', help='path to template (header) folder')
    parser.add_argument('-d', '--datapath', required=True, type=str, metavar='data_path', dest='data_path', help='path to data folder')
    parser.add_argument('-g', '--geodata', required=True, type=str, metavar='geo_data', dest='geo_data', help='geographic data file')
    parser.add_argument('-s', '--savepath', required=False, type=str, metavar='save_path', dest='save_path', help='path to save data', default='.')
    parser.add_argument('-n', '--nodecount', required=False, type=int, metavar='node_count', dest='node_count', help='number of cpus to use for processing', default=math.ceil(mp.cpu_count() * 0.8))
    args = parser.parse_args()

    args.header_path = args.header_path.rstrip('/')
    args.data_path = args.data_path.rstrip('/')
    args.save_path = args.save_path.rstrip('/')

    # Prepare the file names for reading
    header_files = {int(f.split('.')[0][3:]): '/'.join((args.header_path, f)) for f in os.listdir(args.header_path) if f.startswith('Seq') and f.endswith('.csv')} # only want the converted header files
    data_files = {int(f.split('.')[0][9:12]): '/'.join((args.data_path, f)) for f in os.listdir(args.data_path) if f.startswith('e') and f.endswith('.txt')} # only want estimates (not margin of error)

    geo_header = '/'.join((args.header_path, '2016_SFGeoFileTemplate.csv'))
    geo_data = geo_parse(args.geo_data, geo_header)

    arguments = [[i, header_files[i], data_files[i]] for i in range(1, len(header_files) + 1)]
    

    with mp.Pool(args.node_count) as p:
        outputs = p.map(process_single, arguments)
    print('ACS files processed. Merging into single dictionary.')
    outputs = dict(utils.merge_dicts(outputs))
    sp = '/'.join((args.save_path, 'ACS_2016_5yr_NYC_tracts.pkl'))
    pickle.dump(outputs, open(sp, 'wb'))
    print('Data saved to {}'.format(sp))
    