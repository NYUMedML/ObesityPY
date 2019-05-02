import os

import csv
import censusgeocode as cg

cg.CensusGeocode(benchmark='Public_AR_Census2010', vintage='Census2010_Census2010')

def geocode(headers, i, line):
    """
    Reads the line from the csv and indefinitely tries to get the census geocoder
    to return a valid location. The geocoder seems to arbitrarily send an error
    message. My best guess is this is some weird connection issue.

    Parameters
    ----------
    headers: dict
        Dictionary with column name and index values corresponding to line. This should have
        a latitude and longitude field inside.
    i : int
        Line/list index number. Used for logging purposes. Pass dummy value if not part of a
        larger geocoding process.
    line : list-like
        List of data from corresponding to the row of a file, dataframe, or other object. Data
        should have a latitude and longitude field.

    Returns
    -------
    location : dict
        Dictionary of the 2010 Census Block information returned from
        censusgeocode geocoding.
    lat : float
        Latitude from line.
    lon : float
        Longitude from line.
    """
    lat = float(line[headers['latitude']])
    lon = float(line[headers['longitude']])
    print(i, lat, lon)
    # keep pinging the census geocoder until a valid result is sent
    try:
        location = cg.coordinates(x=lon, y=lat)['2010 Census Blocks'][0]
    except:
        print('Attempting to geocode line {} at x: {}, y: {} again.'.format(i, lon, lat))
        location, lat, lon = geocode(headers, i, line)

    return location, lat, lon

def merge_dicts(dicts):
    """
    Merge a list of nested dictionaries on common keys.
    
    Parameters
    ----------
    dicts : list-like
        List of dictionaries with common nested keys of arbitrary depth.

    Returns
    -------
    generator
        k: keys
        v: values

    merged_dict = dict(merge_dicts([dict1, dict2, ...]))
    """
    for k in set().union(*[set([*d]) for d in dicts]):
        # k_in = np.where(np.array([k in d for d in dicts]))[0]
        k_in = [i for i,d in enumerate(dicts) if k in d]
        if all(isinstance(dicts[i][k], dict) for i in k_in):
            yield(k, dict(merge_dicts([dicts[i][k] for i in k_in])))
        elif all(isinstance(dicts[i][k], list) for i in k_in):
            yield(k, *[dicts[i][k][0] for i in k_in])
        else:
            yield(k, *[dicts[i][k] for i in k_in])

def set_default():
    """
    Create a nested defaultdict object without knowing how deep 
    the end dictionary will be.
    """
    return defaultdict(set_default)

def dd_to_dict(data_dict):
    """
    Convert a set_default object (arbitrarily deep nested defaultdicts) to
    an arbitrarily deep nested dictionary object. defaultdict does not always
    play nicely with other programs such as multiprocessing because pickle
    does not support lambda functions.

    Parameters
    ----------
    data_dict : dict
        Data dictionary to be converted from collections.defaultdict to dict
        at all depths.

    Returns
    -------
    data_dict : dict
        Pure dict object of the passed data_dict.
    """
    if type(data_dict) == defaultdict:
        return dd_to_dict(dict(data_dict))
    elif type(data_dict) == dict:
        for d in data_dict:
            data_dict[d] = dd_to_dict(data_dict[d])
        return data_dict
    else:
        return data_dict

if __name__ == '__main__':
    print('utils.py has nothing to be run.')