import os
import sys

import math
import pickle
import argparse
import numpy as np
import pandas as pd
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor

from datetime import datetime

# HELPER FUNCTIONS
def categorize(fname):
    """
    Provides the file category for the data inside.
    
    Parameters
    ----------
    fname : string
        Filename for the file to be loaded
    """
    if fname[:3] == 'BMI':
        return fname[:3]
    elif fname[:3] == 'Hgt':
        return 'Ht'
    elif fname[:3] == 'Wgt':
        return 'Wt'
    else:
        raise ValueError('This is an unsupported filetype in this script')

def date_parse(date_string):
    """
    Returns a parsed datetime.datetime object for a datetime string.
    
    Parameters
    ----------
    date_string : str
        String of date and time representation.
    
    Returns
    -------
    dt : datetime.datetime
        datetime.datetime representation of string object.
    """
    try:
        return datetime.strptime(date_string, '%m/%d/%Y %I:%M:%S %p').date()
    except:
        return datetime.strptime(date_string, '%m/%d/%Y').date()

# DATA FILE/FOLDER PROCESSING FUNCTIONS
def load_single_df(args):
    """
    Load a single file of data into a pandas DataFrame object. Intended to
    be used with a parallel implementation to load many files for processing.
    
    Parameters
    ----------
    fname : string
        Filename for the file to be loaded.
    cat : string
        Type of data within the file.
    
    Returns
    -------
    cat : string
        Type of data within the file.
    df : pandas DataFrame
        DataFrame object representation of the file.
    """
    fname, cat = args
    df = pd.read_csv(fname, low_memory=False)
    return cat, df

def concat_single(args):
    """
    Apply pd.concat in parallel to a list of DataFrame objects
    """
    return args[0], pd.concat(args[1:], ignore_index=True)

def load_data(path, node_count):
    """
    Loads all of the data with a '.txt' extension from a directory.
    
    Parameters
    ----------
    path : string
        Folder destination for where files should be loaded from.
    node_count : int
        Number of cpu nodes to use for loading all the data.
        
    Returns
    -------
    hts : pandas DataFrame
        data frame of height data
    wts : pandas DataFrame
        data frame of weight data
    bmis : pandas DataFrame
        data frame of BMI data
    """
    hts, wts, bmis = (['Ht'], ['Wt'], ['BMI'])
    files = [('/'.join((path, fp, f)), categorize(f)) for fp in os.listdir(path) for f in os.listdir('/'.join((path, fp))) if f.endswith('txt')]
    with ProcessPoolExecutor(max_workers=node_count) as p:
        outputs = p.map(load_single_df, files)
        
    for output in outputs:
        if output[0] == 'Ht':
            hts.append(output[1])
        elif output[0] == 'Wt':
            wts.append(output[1])
        elif output[0] == 'BMI':
            bmis.append(output[1])
            
    with ProcessPoolExecutor(max_workers=node_count) as p:
        outputs = p.map(concat_single, [hts, wts, bmis])
        
    for output in outputs:
        if output[0] == 'Ht':
            hts = output[1]
        elif output[0] == 'Wt':
            wts = output[1]
        elif output[0] == 'BMI':
            bmis = output[1]
            
    return hts, wts, bmis

# DATA PARSING FUNCTIONS
def filter_arr(x, mrn, mrn_ix):
    """
    Get the array where there is a matching MRN or return an empty array.
    
    Parameters
    ----------
    x : np.ndarray
        Data array.
    mrn : int
        Patient MRN.
    mrn_ix : int
        Column index corresponding to mrn.
    
    Returns
    -------
    arr : np.ndarray
        Filtered array for all matches with the provided mrn or an empty array.
    """
    
    return x[x[:, mrn_ix] == mrn]

def get_unique_vals(data, col_ix):
    """
    Get the unique values in a given column among data sets.
    
    Parameters
    ----------
    data : list
        List of data arrays to get patient ids from
    col_ix : int
        Index of focus column.
    
    Returns
    -------
    vals : set
        Set of values in specified column of each array in data.
    """
    
    return set([val.strip() if isinstance(val, str) else val for d in data for val in d[:, col_ix]])

def determine_multiple(vals, mrn, data_type):
    """
    Determine if there are erroneously more than one value for a given data type.
    
    Parameters
    ----------
    vals : set
        Set of values.
    mrn : int
        Patient MRN. Used for logging purposes.
    data_type : string
        Type of data being passed through the field.
    
    Returns
    -------
    vals : unk
        The hopefully singular value inside of parameter vals. Otherwise an error is raised.
    """
    if len(vals) > 1:
        raise ValueError('MRN: {0:d} has more than one {1:s}. Check data and try again.'.format(mrn, data_type))
    else:
        for vals in vals:
            break
    return vals

def get_patient_id(data, mrn, col_ix):
    """
    Get the patient id for a given patient.
    
    Parameters
    ----------
    data : list
        List of data arrays to get patient ids from
    mrn : int
        Patient MRN. Used for logging purposes.
    col_ix : int
        Index of PATIENT_ID column.
    
    Returns
    -------
    pid : int
        Patient ID.
    """
    
    pid = get_unique_vals(data, col_ix)
    if len(pid) > 1:
        print('MRN: {0:d} has multiple matching PATIENT_IDs: {1:s}.\nUsing the PATIENT_ID with the most entries.'.format(mrn, str(pid)))
        inputs = np.sum([[d[d[:, col_ix] == p].shape[0] for p in pid] for d in data], axis=0)
        pid = list(pid)[np.argmax(inputs)]
    else:
        for pid in pid:
            break
    
    return pid
    
def get_measurement_single(data, mrn, col_ix, data_type):
    """
    Get the race for a given patient.
    
    Parameters
    ----------
    data : list
        List of data arrays to get patient ids from.
    mrn : int
        Patient MRN. Used for logging purposes.
    col_ix : int
        Index of patient gender/sex column.
    data_type : string
        Type of data being passed through the field.
    
    Returns
    -------
    val : string
        Patient data point of category data_type.
    """
    val = get_unique_vals(data, col_ix)
    val = determine_multiple(val, mrn, data_type)
    if data_type.lower() == 'gender':
        if val.lower() == 'male':
            return False
        else:
            return True
    elif data_type.lower() in ('bdate', 'birthday', 'birthdate'):
        return date_parse(val)
    else:
    	return val.upper()
        

def get_measurement_multiple(data, mrn, ix_cols):
    """
    Get the list of measurement data for a given patient.
    
    Parameters
    ----------
    data : array
        Data arrays to get value from.
    mrn : int
        Patient MRN. Used for logging purposes.
    ix_cols : int
        List of column indices for the event timestamp and
        the corresponding value.
    
    Returns
    -------
    vals : list
        Nested list of each timestamp and corresponding value 
        measurement: [[datetime.datetime, value], ]
    """
    vals = data[:, ix_cols]
    for i, row in enumerate(vals):
        vals[i] = [date_parse(row[0]), row[1]]
    return vals.tolist()

def process_single_patient(args):
    mrn = args[0]
    
    # global hts, wts, bmis, pat_ix, mrn_ix, sex_ix, race_ix, bdate_ix, evnt_ix, val_ix
    
    ht = filter_arr(hts, mrn, mrn_ix)
    wt = filter_arr(wts, mrn, mrn_ix)
    bmi = filter_arr(bmis, mrn, mrn_ix)
    data = [ht, wt, bmi]
    
    patient_id = get_patient_id(data, mrn, pat_ix)
    patient = {
        'vitals': {
            'Ht': get_measurement_multiple(ht, mrn, [evnt_ix, val_ix]),
            'Wt': get_measurement_multiple(wt, mrn, [evnt_ix, val_ix]),
            'BMI': get_measurement_multiple(bmi, mrn, [evnt_ix, val_ix])
        }, 
        'bdate': get_measurement_single(data, mrn, bdate_ix, 'bdate'),
        'gender': get_measurement_single(data, mrn, sex_ix, 'gender'),
        'ethnicity': np.nan,
        'race': get_measurement_single(data, mrn, race_ix, 'race'),
        'mrn': mrn
    }
    return patient_id, patient

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Process data files from the provided path in parallel')
    parser.add_argument('-p', '--path', 
                        required=True, type=str, metavar='path', dest='path',
                        help='path to data folder')
    parser.add_argument('-n', '--nodecount', default=math.ceil(cpu_count() * 0.8),
                        type=str, metavar='node_count', dest='node_count',
                        help='number of cpu nodes to use for processing in parallel')
    parser.add_argument('-s', '--save', default='.',
                        type=str, metavar='save_path', dest='save_path',
                        help='path to save pickled data')

    args = parser.parse_args()

    # global hts, wts, bmis, pat_ix, mrn_ix, sex_ix, race_ix, bdate_ix, evnt_ix, val_ix

    print('Retrieving files from {}.'.format(args.path))
    hts, wts, bmis = load_data(args.path, args.node_count)
    cols = hts.columns
    hts = hts.values
    wts = wts.values
    bmis = bmis.values

    pat_ix = np.where(cols == 'PATIENT_ID')[0][0]
    mrn_ix = np.where(cols == 'BELLEVUE_MRN')[0][0]
    sex_ix = np.where(cols == 'SEX')[0][0]
    race_ix = np.where(cols == 'RACE')[0][0]
    bdate_ix = np.where(cols == 'BIRTHDATE')[0][0]
    evnt_ix = np.where(cols == 'EVENT_DATE_TIME')[0][0]
    val_ix = np.where(cols == 'VALUE')[0][0]
    field_ix = np.where(cols == 'FIELD')[0][0]

    # convert all heights to inches to match data type of original data set
    # some fields, however, are already inches, so we will ignore those
    field_filter = (hts[:, field_ix] != 'Height (inches numeric)')
    conversion = np.ones(hts.shape[0])
    conversion[field_filter] = 2.54
    hts[:, val_ix] =  np.round_((hts[:, val_ix] / conversion).astype(float), decimals=1)

    # now do the same thing for weight data except for kg to lb
    lbs = ['Admit Wt', 'Wt (lbs)', "Mom's Wt", 'Pre-Pregnancy Weight', 'OB Current Weight', 'Weight (lbs)', 'Pre-Op Weight (Kg)', 'Total Weight']
    field_filter = np.all([wts[:, field_ix] != el for el in lbs], axis=0)
    conversion = np.ones(wts.shape[0])
    conversion[field_filter] = 0.45359237
    wts[:, val_ix] =  np.round_((wts[:, val_ix] / conversion).astype(float), decimals=1)

    print('Creating data with {} nodes.'.format(args.node_count))
    patient_dict = {}
    arguments = [[mrn] for mrn in set().union(*[set(x[:, mrn_ix]) for x in (hts, wts, bmis)])]
    with ProcessPoolExecutor(max_workers=args.node_count) as p:
        outputs = p.map(process_single_patient, arguments)

    for patient_id, patient_data in outputs:
        patient_dict[patient_id] = patient_data

    import time
    timestr = time.strftime("%Y%m%d")
    fname = 'patient_data_2_' + timestr + '.pkl'
    fname = '/'.join((args.save_path, fname))
    pickle.dump(patient_dict, open(fname, 'wb'))
    print('Data saved to {}.'.format(fname))
    print('Done.')
