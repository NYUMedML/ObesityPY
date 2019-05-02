import re
import time
import pickle
import zscore
import warnings
import datetime
import functools
import numpy as np
import pandas as pd
import config as config_file
import matplotlib.pylab as plt
import outcome_def_pediatric_obesity

from scipy import stats
from datetime import timedelta
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

def if_data_nonexistent(func):
    """
    Decorator function to return an array of zeros if the provided data dictionary does not have
    the desired data categroy within it. Will ensure that incomplete data dictionaries can also be
    used with call_build_function().
    """
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        try:
            res = func(*args, **kwargs)
        except:
            res = np.zeros((len(args[-1])))
        return res
    return wrapper_decorator

@if_data_nonexistent
def build_features_icd(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers):
    res = np.zeros(len(feature_headers), dtype=int)
    for diag in patient_data['diags']:
        # print(diag , diag.replace('.','').strip(), feature_index[diag.replace('.','').strip()])
        for edatel in patient_data['diags'][diag]:
            edate = edatel[0]
            if edate >= reference_date_end or edate <= reference_date_start:
                continue
            try:
                res[feature_index[diag.replace('.','').strip()]] += 1
            except KeyError:
                try:
                    res[feature_index[diag.replace('.','').strip()[0:-2]]] += 1
                except KeyError:
                    pass #print('--->',diag.replace('.','').strip()[0:-1])
            break
    return res

@if_data_nonexistent
def build_features_lab(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers):
    res = np.zeros(len(feature_headers), dtype=float)
    for key1 in patient_data['labs']:
        for edatel in patient_data['labs'][key1]:
            edate = edatel[0]
            if edate >= reference_date_end or edate <= reference_date_start:
                continue
            try:
                res[feature_index[key1.strip()]] = edatel[1]
            except KeyError:
                pass # print('key error lab:', key1)
            break
    return res

@if_data_nonexistent
def build_features_med(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers):
    res = np.zeros(len(feature_headers), dtype=bool)
    for key1 in patient_data['meds']:
        for edatel in patient_data['meds'][key1]:
            edate = edatel[0]
            if edate >= reference_date_end or edate <= reference_date_start:
                continue
            try:
                res[feature_index[key1.strip()]] = True
            except KeyError:
                pass # print ('key error', key1.strip())
            break
    return res

def build_features_gen(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers):
    res = np.zeros(len(feature_headers), dtype=bool)
    code = patient_data['gender']
    res[feature_index[int(code)]] = True
    return res

@if_data_nonexistent
def build_features_vitalLatest(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers):
    res = np.zeros(len(feature_headers), dtype=float)
    bdate = patient_data['bdate']
    for code in patient_data['vitals']:
        for (edate, vitalval) in patient_data['vitals'][code]:
            if edate >= reference_date_end or edate <= reference_date_start:
                continue
            try:
                res[feature_index[code.strip()]] = vitalval
            except:
                pass
    return res

def build_features_vitalAverage_0_0(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers):
    return build_features_vitalAverage(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers, 0, 0)

def build_features_vitalAverage_0_1(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers):
    return build_features_vitalAverage(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers, 0, 1)

def build_features_vitalAverage_1_3(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers):
    return build_features_vitalAverage(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers, 1, 3)

def build_features_vitalAverage_3_5(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers):
    return build_features_vitalAverage(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers, 3, 5)

def build_features_vitalAverage_5_7(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers):
    return build_features_vitalAverage(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers, 5, 7)

def build_features_vitalAverage_7_10(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers):
    return build_features_vitalAverage(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers, 7, 10)

def build_features_vitalAverage_10_13(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers):
    return build_features_vitalAverage(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers, 10, 13)

def build_features_vitalAverage_13_16(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers):
    return build_features_vitalAverage(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers, 13, 16)

def build_features_vitalAverage_16_19(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers):
    return build_features_vitalAverage(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers, 16, 19)

def build_features_vitalAverage_19_24(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers):
    return build_features_vitalAverage(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers, 19, 24)

def build_features_vitalAverage_0_3(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers):
    return build_features_vitalAverage(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers, 0, 3)

def build_features_vitalAverage_3_6(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers):
    return build_features_vitalAverage(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers, 3, 6)

def build_features_vitalAverage_6_9(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers):
    return build_features_vitalAverage(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers, 6, 9)

def build_features_vitalAverage_9_12(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers):
    return build_features_vitalAverage(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers, 9, 12)

def build_features_vitalAverage_12_15(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers):
    return build_features_vitalAverage(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers, 12, 15)

def build_features_vitalAverage_15_18(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers):
    return build_features_vitalAverage(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers, 15, 18)

def build_features_vitalAverage_18_21(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers):
    return build_features_vitalAverage(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers, 18, 21)

def build_features_vitalAverage_18_24(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers):
    return build_features_vitalAverage(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers, 18, 24)

@if_data_nonexistent
def build_features_vitalAverage(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers, frommonth, tomonth):
    res = np.zeros(len(feature_headers), dtype=float)
    res_cnt = np.zeros(len(feature_headers), dtype=float)
    bdate = patient_data['bdate']
    for code in patient_data['vitals']:
        for (edate, vitalval) in patient_data['vitals'][code]:
            if edate > reference_date_end or edate < reference_date_start:
                continue
            try:
                # age_at_vital = (edate - bdate).days / 30
                diff = relativedelta(edate, bdate)
                age_at_vital = diff.years * 12. + diff.months + diff.days / 30.
                if (age_at_vital < frommonth) or (age_at_vital > tomonth) :
                    continue
                res[feature_index[code.strip()]] += vitalval
                res_cnt[feature_index[code.strip()]] += 1
                # print(code, age_at_vital, vitalval)
            except:
                pass
    res_cnt[(res_cnt==0)] = 1.0
    res = res/res_cnt
    return res

def build_features_vitalGain_0_3(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers):
    return build_features_vitalGain(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers, 0, 1, 1, 3)

def build_features_vitalGain_1_5(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers):
    return build_features_vitalGain(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers, 1, 3, 3, 5)

def build_features_vitalGain_3_7(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers):
    return build_features_vitalGain(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers, 3, 5, 5, 7)

def build_features_vitalGain_5_10(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers):
    return build_features_vitalGain(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers, 5, 7, 7, 10)

def build_features_vitalGain_7_13(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers):
    return build_features_vitalGain(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers, 7, 10, 10, 13)

def build_features_vitalGain_10_16(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers):
    return build_features_vitalGain(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers, 10, 13, 13, 16)

def build_features_vitalGain_13_19(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers):
    return build_features_vitalGain(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers, 13, 16, 16, 19)

def build_features_vitalGain_16_24(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers):
    return build_features_vitalGain(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers, 16, 19, 19, 24)

def build_features_vitalGain_0_24(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers):
    return build_features_vitalGain(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers, 0, 0, 19, 24)

@if_data_nonexistent
def build_features_vitalGain(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers, startmonth1, endmonth1, startmonth2, endmonth2):
    """
    Computes the gain between two time periods
    """
    res1 = np.zeros(len(feature_headers), dtype=float)
    res_cnt1 = np.zeros(len(feature_headers), dtype=float)
    res2 = np.zeros(len(feature_headers), dtype=float)
    res_cnt2 = np.zeros(len(feature_headers), dtype=float)
    bdate = patient_data['bdate']
    for code in patient_data['vitals']:
        for (edate, vitalval) in patient_data['vitals'][code]:
            if edate >= reference_date_end or edate <= reference_date_start:
                continue
            try:
                # age_at_vital = (edate - bdate).days / 30
                diff = relativedelta(edate, bdate)
                age_at_vital = diff.years * 12. + diff.months + diff.days / 30.
                if ((age_at_vital < startmonth1) or (age_at_vital > endmonth2)) or ((age_at_vital > endmonth1) and (age_at_vital < startmonth2)):
                    continue
                if (age_at_vital > startmonth1) and (age_at_vital < endmonth1):
                    res1[feature_index[code.strip()]] += vitalval
                    res_cnt1[feature_index[code.strip()]] += 1
                elif (age_at_vital > startmonth2) and (age_at_vital < endmonth2):
                    res2[feature_index[code.strip()]] += vitalval
                    res_cnt2[feature_index[code.strip()]] += 1
            except:
                pass
    res_cnt1[(res_cnt1==0)] = 1.0
    res1 = res1/res_cnt1
    res_cnt2[(res_cnt2==0)] = 1.0
    res2 = res2/res_cnt2
    res = res2-res1
    return res

@if_data_nonexistent
def build_features_ethn(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers):
    res = np.zeros(len(feature_headers), dtype=bool)
    code = patient_data['ethnicity']
    if code in feature_index and pd.notnull(code):
        res[feature_index[code]] = True
    return res

@if_data_nonexistent
def build_features_mat_insurance1(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers):
    res = np.zeros(len(feature_headers), dtype=bool)
    if 'insur1' not in maternal_data:
        # print (maternal_data)
        return res
    code = maternal_data['insur1']
    if code in feature_index and pd.notnull(code):
        res[feature_index[code]] = True
    return res

@if_data_nonexistent
def build_features_mat_insurance2(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers):
    res = np.zeros(len(feature_headers), dtype=bool)
    if 'insur2' not in maternal_data:
        # print (maternal_data)
        return res
    code = maternal_data['insur2']
    if code in feature_index and pd.notnull(code):
        res[feature_index[code]] = True
    return res

@if_data_nonexistent
def build_features_race(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers):
    res = np.zeros(len(feature_headers), dtype=bool)
    code = patient_data['race']
    if code in feature_index and pd.notnull(code):
        res[feature_index[code]] = True
    return res

# def build_features_zipcd(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers):
#     res = np.zeros(len(feature_headers), dtype=bool)
#     if 'zip' in patient_data:
#         code = patient_data['zip'][0][1]
#         if code in feature_index and pd.notnull(code):
#             res[feature_index[code]] = True
#     return res

@if_data_nonexistent
def build_features_zipcd_birth(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers):
    """
    Creates a zip code feature associated with the address closest to the child's birth within 1 year.
    """
    res = np.zeros(len(feature_headers), dtype=bool)
    bdate = patient_data['bdate']
    ix = np.argsort([abs(bdate-z[0]) for z in patient_data['zip'] if abs(bdate-z[0]).days <= 365.])
    if ix.ravel().shape[0] == 0:
        return res
    else:
        code = str(patient_data['zip'][ix[0]][1])
        try:
            res[feature_index[code]] = True
        except:
            code = re.split('[. -]', code)[0]
            res[feature_index[code]] = True
        return res

@if_data_nonexistent
def build_features_zipcd_latest(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers):
    """
    Creates a zip code feature associated with the latest address before reference_date_end.
    """
    res = np.zeros(len(feature_headers), dtype=bool)
    ix = np.argsort([reference_date_end-z[0] for z in patient_data['zip'] if (reference_date_end-z[0]).days >= 0])
    if ix.ravel().shape[0] == 0:
        return res
    else:
        code = str(patient_data['zip'][ix[0]][1])
        try:
            res[feature_index[code]] = True
        except:
            code = re.split('[. -]', code)[0]
            res[feature_index[code]] = True
        return res

# def build_features_census(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers):
#     res = np.zeros(len(feature_headers), dtype=float)
#     if len(lat_lon_data) == 0:
#         return res
#     tract = lat_lon_data['centrac']
#     cntylist = lat_lon_data['county']
#     elem = []
#     for c in cntylist:
#         try:
#             for k in env_data[tract][c]:
#                 res[feature_index[k]]=float(env_data[tract][c][k])
#         except KeyError:
#             continue
#     return res

@if_data_nonexistent
def build_features_census_birth(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers):
    """
    Creates census level features that are associated with the address closest to the child's birth within 1 year.
    """
    res = np.zeros(len(feature_headers), dtype=float)
    bdate = patient_data['bdate']
    if len(lat_lon_data) == 0:
        return res
    diff = [d-bdate for d in lat_lon_data]
    if all(d.days > 365 for d in diff) or all(d.days < 0 for d in diff):
        return res
    if any(d.days < 0 for d in diff):
        for i in range(len(diff)):
            if diff[np.where(np.argsort(diff)==i)[0][0]].days < 0:
                continue
            enc = np.where(np.argsort(diff)==i)[0][0]
    else:
        enc = np.where(np.argsort(diff)==0)[0][0]
    enc_date = [*lat_lon_data][enc]
    tract = lat_lon_data[enc_date]['centrac']
    county = lat_lon_data[enc_date]['county']
    try:
        for k in env_data[tract][county]:
            res[feature_index[k]] = float(env_data[tract][county][k])
    except KeyError:
        pass
    return res

@if_data_nonexistent
def build_features_census_latest(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers):
    """
    Creates census level features that are associated with the latest address before reference_date_end.
    """
    res = np.zeros(len(feature_headers), dtype=float)
    bdate = patient_data['bdate']
    if len(lat_lon_data) == 0:
        return res
    diff = [reference_date_end-d for d in lat_lon_data]
    if all(d < bdate for d in lat_lon_data) or all(d.days < 0 for d in diff):
        return res
    if any(d.days < 0 for d in diff):
        for i in range(len(diff)):
            if diff[np.where(np.argsort(diff)==i)[0][0]].days < 0:
                continue
            enc = np.where(np.argsort(diff)==i)[0][0]
    else:
        enc = np.where(np.argsort(diff)==0)[0][0]
    enc_date = [*lat_lon_data][enc]
    tract = lat_lon_data[enc_date]['centrac']
    county = lat_lon_data[enc_date]['county']
    try:
        for k in env_data[tract][county]:
            res[feature_index[k]] = float(env_data[tract][county][k])
    except KeyError:
        pass
    return res

@if_data_nonexistent
def build_features_numVisits(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers):
    res = np.zeros(len(feature_headers), dtype=int)
    dates = set()
    for key, vals in patient_data.items():
        if key == 'bdate':
            continue
        if isinstance(vals, dict):
            dates.update(el[0] for lst in vals.values() for el in lst if reference_date_start <= el[0] <= reference_date_end)
        # elif isinstance(vals, list):
        #     if isinstance(vals[0], list):
        #         dates.update(el[0] for el in vals if reference_date_start < el[0] < reference_date_end)
        #     else:
        #         dates.update(el for el in vals if reference_date_start < el < reference_date_end)
        elif isinstance(vals, datetime.datetime):
            if reference_date_start <= vals <= reference_date_end:
                dates.update({vals})
        else:
            continue
    res[0] = len(dates)
    return res

@if_data_nonexistent
def build_features_mat_icd(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers):
    res = np.zeros(len(feature_headers), dtype=int)
    if 'diags' not in maternal_data:
        return res
    for diag in maternal_data['diags']:
        # print(diag , diag.replace('.','').strip(), feature_index[diag.replace('.','').strip()])
        try:
            res[feature_index[diag.replace('.','').strip()]] += 1
        except KeyError:
            try:
                res[feature_index[diag.replace('.','').strip()[0:-2]]] += 1
            except KeyError:
                pass #print('--->',diag.replace('.','').strip()[0:-1])
    return res

@if_data_nonexistent
def build_features_nb_icd(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers):
    res = np.zeros(len(feature_headers), dtype=int)
    if 'nbdiags' not in maternal_data:
        return res
    for diag in maternal_data['nbdiags']:
        try:
            res[feature_index[diag.replace('.','').strip()]] += 1
        except KeyError:
            try:
                res[feature_index[diag.replace('.','').strip()[0:-2]]] += 1
            except KeyError:
                pass #print('--->',diag.replace('.','').strip()[0:-1])
    return res

@if_data_nonexistent
def build_features_mat_race(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers):
    res = np.zeros(len(feature_headers), dtype=bool)
    if 'race' not in maternal_data:
        return res
    code = maternal_data['race']
    if code in feature_index and pd.notnull(code):
        res[feature_index[code]] = True
    return res

@if_data_nonexistent
def build_features_mat_ethn(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers):
    res = np.zeros(len(feature_headers), dtype=bool)
    if 'ethnicity' not in maternal_data:
        return res
    code = maternal_data['ethnicity']
    if code in feature_index and pd.notnull(code):
        res[feature_index[code]] = True
    return res

@if_data_nonexistent
def build_features_mat_lang(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers):
    res = np.zeros(len(feature_headers), dtype=bool)
    if 'lang' not in maternal_data:
        return res
    code = maternal_data['lang']
    if code in feature_index and pd.notnull(code):
        res[feature_index[code]] = True
    return res

@if_data_nonexistent
def build_features_mat_natn(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers):
    res = np.zeros(len(feature_headers), dtype=bool)
    if 'nationality' not in maternal_data:
        return res
    code = maternal_data['nationality']
    if code in feature_index and pd.notnull(code):
        res[feature_index[code]] = True
    return res

@if_data_nonexistent
def build_features_mat_marriage(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers):
    res = np.zeros(len(feature_headers), dtype=bool)
    if 'marriage' not in maternal_data:
        return res
    code = maternal_data['marriage']
    if code in feature_index and pd.notnull(code):
        res[feature_index[code]] = True
    return res

@if_data_nonexistent
def build_features_mat_birthpl(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers):
    res = np.zeros(len(feature_headers), dtype=bool)
    if 'birthplace' not in maternal_data:
        return res
    code = maternal_data['birthplace']
    if code in feature_index and pd.notnull(code):
        res[feature_index[code]] = True
    return res

@if_data_nonexistent
def build_features_mat_agedel(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers):
    res = np.zeros(len(feature_headers), dtype=int)
    if 'agedeliv' not in maternal_data:
        return res
    age = int(maternal_data['agedeliv'])
    res[0] = age
    return res


##### FUNCTIONS TO BUILD FEATURES FOR HISTORICAL MATERNAL DATA ####
def mother_child_map(patient_data, maternal_data, maternal_hist_data):
    if maternal_data == {}:
        return {}
    child_mrn = set(np.array([patient_data[k]['mrn'] for k in patient_data.keys()])) & set(np.nan_to_num(np.array([*maternal_data])).astype(int).astype(str))
    mom_mrn = set(maternal_hist_data.keys()) & set([maternal_data[k]['mom_mrn'] for k in maternal_data.keys()])
    keys = [k for k in patient_data.keys() if str(patient_data[k]['mrn']) in child_mrn]
    mother_child_dic = {}
    for k in keys:
        try:
            mother_child_dic[maternal_data[patient_data[k]['mrn']]['mom_mrn']][patient_data[k]['mrn']] = patient_data[k]['bdate']
        except:
            mother_child_dic[maternal_data[patient_data[k]['mrn']]['mom_mrn']] = {patient_data[k]['mrn']: patient_data[k]['bdate']}
    return mother_child_dic

@if_data_nonexistent
def build_features_mat_hist_value(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers, mother_child_data, output_type, measurement, period):
    """
    Function to process maternal doctor visits.
    #### PARAMETERS ####
    output_type:
        "count" - returns count of occurrences
        "average" - returns average measurement value
    measurement:
        "vitals" - vitals
        "labs" - labs
        "diags" - diagnoses
        "procedures" - procedures
    period:
        "pre" - pre-pregnancy
        "post" - post-pregnancy
        "other" - during another Pregnancy
        "trimester1" - first trimester [0,14)
        "trimester2" - second trimester [14,27)
        "trimester3" - third trimester [27,40]
    """
    res = np.zeros(len(feature_headers), dtype=float)
    res_cnt = np.zeros(len(feature_headers), dtype=float)
    if measurement not in maternal_hist_data:
        return res
    else:
        bdate = patient_data['bdate']
        mat_mrn = maternal_data['mom_mrn']

        for code in maternal_hist_data[measurement]:
            for (vital_date, vital_val) in maternal_hist_data[measurement][code]:
                if vital_date >= reference_date_end or vital_date <= reference_date_start:
                    continue
                if period == 'pre':
                    if (bdate - vital_date).days / 7 > 40 and not all([((other_bdate - vital_date).days / 7 >= 0) and ((other_bdate - vital_date).days / 7 <= 40) for (other_mrn,other_bdate) in mother_child_data[mat_mrn].items() if other_mrn != patient_data['mrn']]):
                        if measurement == 'diags':
                            try:
                                res[feature_index[code.replace('.','').strip()]] += float(vital_val)
                                res_cnt[feature_index[code.replace('.','').strip()]] += 1.
                            except KeyError:
                                try:
                                    res[feature_index[code.replace('.','').strip()[0:-2]]] += float(vital_val)
                                    res_cnt[feature_index[code.replace('.','').strip()[0:-2]]] += 1.
                                except KeyError:
                                    pass
                        else:
                            try:
                                res[feature_index[code.strip()]] += float(vital_val)
                                res_cnt[feature_index[code.strip()]] += 1.
                            except:
                                pass
                elif period == 'post':
                    if vital_date > bdate and not all([((other_bdate - vital_date).days / 7 >= 0) and ((other_bdate - vital_date).days / 7 <= 40) for (other_mrn,other_bdate) in mother_child_data[mat_mrn].items() if other_mrn != patient_data['mrn']]):
                        if measurement == 'diags':
                            try:
                                res[feature_index[code.replace('.','').strip()]] += float(vital_val)
                                res_cnt[feature_index[code.replace('.','').strip()]] += 1.
                            except KeyError:
                                try:
                                    res[feature_index[code.replace('.','').strip()[0:-2]]] += float(vital_val)
                                    res_cnt[feature_index[code.replace('.','').strip()[0:-2]]] += 1.
                                except KeyError:
                                    pass
                        else:
                            try:
                                res[feature_index[code.strip()]] += float(vital_val)
                                res_cnt[feature_index[code.strip()]] += 1.
                            except:
                                pass
                elif period == 'other':
                    if any([((other_bdate - vital_date).days / 7 >= 0) and ((other_bdate - vital_date).days / 7 <= 40) for (other_mrn,other_bdate) in mother_child_data[mat_mrn].items() if other_mrn != patient_data['mrn']]):
                        if measurement == 'diags':
                            try:
                                res[feature_index[code.replace('.','').strip()]] += float(vital_val)
                                res_cnt[feature_index[code.replace('.','').strip()]] += 1.
                            except KeyError:
                                try:
                                    res[feature_index[code.replace('.','').strip()[0:-2]]] += float(vital_val)
                                    res_cnt[feature_index[code.replace('.','').strip()[0:-2]]] += 1.
                                except KeyError:
                                    pass
                        else:
                            try:
                                res[feature_index[code.strip()]] += float(vital_val)
                                res_cnt[feature_index[code.strip()]] += 1.
                            except:
                                pass
                elif period == 'trimester1':
                    if 40 - ((bdate-vital_date).days / 7) < 14 and 40 - ((bdate-vital_date).days / 7) >= 0:
                        if measurement == 'diags':
                            try:
                                res[feature_index[code.replace('.','').strip()]] += float(vital_val)
                                res_cnt[feature_index[code.replace('.','').strip()]] += 1.
                            except KeyError:
                                try:
                                    res[feature_index[code.replace('.','').strip()[0:-2]]] += float(vital_val)
                                    res_cnt[feature_index[code.replace('.','').strip()[0:-2]]] += 1.
                                except KeyError:
                                    pass
                        else:
                            try:
                                res[feature_index[code.strip()]] += float(vital_val)
                                res_cnt[feature_index[code.strip()]] += 1.
                            except:
                                pass
                elif period == 'trimester2':
                    if 40 - ((bdate-vital_date).days / 7) < 27 and 40 - ((bdate-vital_date).days / 7) >= 14:
                        if measurement == 'diags':
                            try:
                                res[feature_index[code.replace('.','').strip()]] += float(vital_val)
                                res_cnt[feature_index[code.replace('.','').strip()]] += 1.
                            except KeyError:
                                try:
                                    res[feature_index[code.replace('.','').strip()[0:-2]]] += float(vital_val)
                                    res_cnt[feature_index[code.replace('.','').strip()[0:-2]]] += 1.
                                except KeyError:
                                    pass
                        else:
                            try:
                                res[feature_index[code.strip()]] += float(vital_val)
                                res_cnt[feature_index[code.strip()]] += 1.
                            except:
                                pass
                elif period == 'trimester3':
                    if 40 - ((bdate-vital_date).days / 7) <= 40 and 40 - ((bdate-vital_date).days / 7) >= 27:
                        if measurement == 'diags':
                            try:
                                res[feature_index[code.replace('.','').strip()]] += float(vital_val)
                                res_cnt[feature_index[code.replace('.','').strip()]] += 1.
                            except KeyError:
                                try:
                                    res[feature_index[code.replace('.','').strip()[0:-2]]] += float(vital_val)
                                    res_cnt[feature_index[code.replace('.','').strip()[0:-2]]] += 1.
                                except KeyError:
                                    pass
                        else:
                            try:
                                res[feature_index[code.strip()]] += float(vital_val)
                                res_cnt[feature_index[code.strip()]] += 1.
                            except:
                                pass
        if output_type == 'count':
            return res
        elif output_type == 'average':
            res_cnt[(res_cnt == 0)] = 1.0
            res = res/res_cnt
            return res

def build_features_mat_hist_vitalsAverage_prePregnancy(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers, mother_child_data):
    return build_features_mat_hist_value(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers, mother_child_data, 'average', 'vitals', 'pre')

def build_features_mat_hist_vitalsAverage_firstTri(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers, mother_child_data):
    return build_features_mat_hist_value(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers, mother_child_data, 'average', 'vitals', 'trimester1')

def build_features_mat_hist_vitalsAverage_secTri(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers, mother_child_data):
    return build_features_mat_hist_value(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers, mother_child_data, 'average', 'vitals', 'trimester2')

def build_features_mat_hist_vitalsAverage_thirdTri(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers, mother_child_data):
    return build_features_mat_hist_value(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers, mother_child_data, 'average', 'vitals', 'trimester3')

def build_features_mat_hist_vitalsAverage_postPregnancy(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers, mother_child_data):
    return build_features_mat_hist_value(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers, mother_child_data, 'average', 'vitals', 'post')

def build_features_mat_hist_vitalsAverage_otherPregnancy(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers, mother_child_data):
    return build_features_mat_hist_value(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers, mother_child_data, 'average', 'vitals', 'other')

def build_features_mat_hist_labsAverage_prePregnancy(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers, mother_child_data):
    return build_features_mat_hist_value(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers, mother_child_data, 'average', 'labs', 'pre')

def build_features_mat_hist_labsAverage_firstTri(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers, mother_child_data):
    return build_features_mat_hist_value(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers, mother_child_data, 'average', 'labs', 'trimester1')

def build_features_mat_hist_labsAverage_secTri(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers, mother_child_data):
    return build_features_mat_hist_value(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers, mother_child_data, 'average', 'labs', 'trimester2')

def build_features_mat_hist_labsAverage_thrirdTri(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers, mother_child_data):
    return build_features_mat_hist_value(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers, mother_child_data, 'average', 'labs', 'trimester3')

def build_features_mat_hist_labsAverage_postPregnancy(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers, mother_child_data):
    return build_features_mat_hist_value(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers, mother_child_data, 'average', 'labs', 'post')

def build_features_mat_hist_labsAverage_otherPregnancy(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers, mother_child_data):
    return build_features_mat_hist_value(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers, mother_child_data, 'average', 'labs', 'other')

def build_features_mat_hist_proceduresCount_prePregnancy(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers, mother_child_data):
    return build_features_mat_hist_value(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers, mother_child_data, 'count', 'procedures', 'pre')

def build_features_mat_hist_proceduresCount_firstTri(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers, mother_child_data):
    return build_features_mat_hist_value(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers, mother_child_data, 'count', 'procedures', 'trimester1')

def build_features_mat_hist_proceduresCount_secTri(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers, mother_child_data):
    return build_features_mat_hist_value(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers, mother_child_data, 'count', 'procedures', 'trimester2')

def build_features_mat_hist_proceduresCount_thrirdTri(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers, mother_child_data):
    return build_features_mat_hist_value(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers, mother_child_data, 'count', 'procedures', 'trimester3')

def build_features_mat_hist_proceduresCount_postPregnancy(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers, mother_child_data):
    return build_features_mat_hist_value(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers, mother_child_data, 'count', 'procedures', 'post')

def build_features_mat_hist_proceduresCount_otherPregnancy(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers, mother_child_data):
    return build_features_mat_hist_value(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers, mother_child_data, 'count', 'procedures', 'other')

def build_features_mat_hist_icdCount_prePregnancy(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers, mother_child_data):
    return build_features_mat_hist_value(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers, mother_child_data, 'count', 'diags', 'pre')

def build_features_mat_hist_icdCount_firstTri(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers, mother_child_data):
    return build_features_mat_hist_value(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers, mother_child_data, 'count', 'diags', 'trimester1')

def build_features_mat_hist_icdCount_secTri(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers, mother_child_data):
    return build_features_mat_hist_value(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers, mother_child_data, 'count', 'diags', 'trimester2')

def build_features_mat_hist_icdCount_thrirdTri(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers, mother_child_data):
    return build_features_mat_hist_value(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers, mother_child_data, 'count', 'diags', 'trimester3')

def build_features_mat_hist_icdCount_postPregnancy(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers, mother_child_data):
    return build_features_mat_hist_value(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers, mother_child_data, 'count', 'diags', 'post')

def build_features_mat_hist_icdCount_otherPregnancy(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers, mother_child_data):
    return build_features_mat_hist_value(patient_data, maternal_data, maternal_hist_data, lat_lon_data, env_data, reference_date_start, reference_date_end, feature_index, feature_headers, mother_child_data, 'count', 'diags', 'other')


##### FEATURE INDEX FUNCTIONS ####
def build_feature_matlang_index():
    try:
        codesNnames = [l.strip().decode('utf-8') for l in open(config_file.BM_Language, 'rb').readlines()]
    except:
        codesNnames = [l.strip().decode('latin-1') for l in open(config_file.BM_Language, 'rb').readlines()]
    feature_index = {}
    feature_headers = []
    for (ix, codeline) in enumerate(codesNnames):
        code = codeline.strip()
        if code in feature_index:
            feature_index[code].append(ix)
            # print('double!!', icd_code)
        else:
            feature_index[code] = [ix]
    feature_headers = ['Maternal-Language:'+ i for i in codesNnames]
    return feature_index, feature_headers

def build_feature_matinsurance1_index():
    try:
        codesNnames1 = [l.strip().decode('utf-8') for l in open(config_file.BM_Prim_Ins, 'rb').readlines()]
    except:
        codesNnames1 = [l.strip().decode('latin-1') for l in open(config_file.BM_Prim_Ins, 'rb').readlines()]
    feature_index = {}
    feature_headers = []
    for (ix, codeline) in enumerate(codesNnames1):
        code = codeline.strip()
        if code in feature_index:
            feature_index[code].append(ix)
            # print('double!!', icd_code)
        else:
            feature_index[code] = [ix]
    feature_headers = ['Prim_Insur:'+ i for i in codesNnames1]
    return feature_index, feature_headers

def build_feature_matinsurance2_index():
    try:
        codesNnames2 = [l.strip().decode('utf-8') for l in open(config_file.BM_Second_Ins, 'rb').readlines()]
    except:
        codesNnames2 = [l.strip().decode('latin-1') for l in open(config_file.BM_Second_Ins, 'rb').readlines()]
    feature_index = {}
    feature_headers = []
    for (ix, codeline) in enumerate(codesNnames2):
        code = codeline.strip()
        if code in feature_index:
            feature_index[code].append(ix)
            # print('double!!', icd_code)
        else:
            feature_index[code] = [ix]
    feature_headers = ['Second_Insur:'+ i for i in codesNnames2]
    return feature_index, feature_headers

def build_feature_matethn_index():
    try:
        codesNnames = [l.strip().decode('utf-8') for l in open(config_file.BM_EthnicityList, 'rb').readlines()]
    except:
        codesNnames = [l.strip().decode('latin-1') for l in open(config_file.BM_EthnicityList, 'rb').readlines()]
    feature_index = {}
    feature_headers = []
    for (ix, codeline) in enumerate(codesNnames):
        code = codeline.strip()
        if code in feature_index:
            feature_index[code].append(ix)
            # print('double!!', icd_code)
        else:
            feature_index[code] = [ix]
    feature_headers = ['Maternal-ethnicity:'+ i for i in codesNnames]
    return feature_index, feature_headers

def build_feature_matrace_index():
    try:
        codesNnames = [l.strip().decode('utf-8') for l in open(config_file.BM_RaceList, 'rb').readlines()]
    except:
        codesNnames = [l.strip().decode('latin-1') for l in open(config_file.BM_RaceList, 'rb').readlines()]
    feature_index = {}
    feature_headers = []
    for (ix, codeline) in enumerate(codesNnames):
        code = codeline.strip()
        if code in feature_index:
            feature_index[code].append(ix)
            # print('double!!', icd_code)
        else:
            feature_index[code] = [ix]
    feature_headers = ['Maternal-race:'+ i for i in codesNnames]
    return feature_index, feature_headers

def build_feature_matnatn_index():
    try:
        codesNnames = [l.strip().decode('utf-8') for l in open(config_file.BM_NationalityList, 'rb').readlines()]
    except:
        codesNnames = [l.strip().decode('latin-1') for l in open(config_file.BM_NationalityList, 'rb').readlines()]
    feature_index = {}
    feature_headers = []
    for (ix, codeline) in enumerate(codesNnames):
        code = codeline.strip()
        if code in feature_index:
            feature_index[code].append(ix)
            # print('double!!', icd_code)
        else:
            feature_index[code] = [ix]
    feature_headers = ['Maternal-nationality:'+ i for i in codesNnames]
    return feature_index, feature_headers

def build_feature_matmarriage_index():
    try:
        codesNnames = [l.strip().decode('utf-8') for l in open(config_file.BM_Marital_StatusList, 'rb').readlines()]
    except:
        codesNnames = [l.strip().decode('latin-1') for l in open(config_file.BM_Marital_StatusList, 'rb').readlines()]
    feature_index = {}
    feature_headers = []
    for (ix, codeline) in enumerate(codesNnames):
        code = codeline.strip()
        if code in feature_index:
            feature_index[code].append(ix)
            # print('double!!', icd_code)
        else:
            feature_index[code] = [ix]
    feature_headers = ['Maternal-marriageStatus:'+ i for i in codesNnames]
    return feature_index, feature_headers
def build_feature_matbirthpl_index():
    try:
        codesNnames = [l.strip().decode('utf-8') for l in open(config_file.BM_BirthPlace, 'rb').readlines()]
    except:
        codesNnames = [l.strip().decode('latin-1') for l in open(config_file.BM_BirthPlace, 'rb').readlines()]
    feature_index = {}
    feature_headers = []
    for (ix, codeline) in enumerate(codesNnames):
        code = codeline.strip()
        if code in feature_index:
            feature_index[code].append(ix)
            # print('double!!', icd_code)
        else:
            feature_index[code] = [ix]
    feature_headers = ['Maternal-birthplace:'+ i for i in codesNnames]
    return feature_index, feature_headers
def build_feature_agedeliv_index():
    feature_index={}
    feature_headers=['MatDeliveryAge']
    return feature_index, feature_headers

def build_feature_Mat_ICD_index():
    try:
        icd9 = [l.strip().decode("utf-8")  for l in open(config_file.icd9List, 'rb').readlines()]
    except:
        icd9 = [l.strip().decode('latin-1')  for l in open(config_file.icd9List, 'rb').readlines()]
    try:
        icd10 = [l.strip().decode("utf-8")  for l in open(config_file.icd10List, 'rb').readlines()]
    except:
        icd10 = [l.strip().decode('latin-1')   for l in open(config_file.icd10List, 'rb').readlines()]
    feature_index = {}
    feature_headers = []
    for (ix, icd) in enumerate(icd9 + icd10):
        icd_codes = icd.split('|')[0].strip().split(' ')
        icd_codes_desc = icd.split('|')[1].strip()
        # print(icd_codes_desc)
        feature_headers.append('Maternal Diagnosis:'+icd_codes_desc)
        for icd_code in icd_codes:
            if icd_code in feature_index:
                feature_index[icd_code].append(ix)
                # print('warning - double icd in 9&10:', icd_code)
            else:
                feature_index[icd_code] = [ix]
    # feature_headers = ['Diagnosis:' + i for i in  (icd9 + icd10)]
    return feature_index, feature_headers
def build_feature_NB_ICD_index():
    try:
        icd9 = [l.strip().decode("utf-8")  for l in open(config_file.icd9List, 'rb').readlines()]
    except:
        icd9 = [l.strip().decode('latin-1')  for l in open(config_file.icd9List, 'rb').readlines()]
    try:
        icd10 = [l.strip().decode("utf-8")  for l in open(config_file.icd10List, 'rb').readlines()]
    except:
        icd10 = [l.strip().decode('latin-1')   for l in open(config_file.icd10List, 'rb').readlines()]
    feature_index = {}
    feature_headers = []
    for (ix, icd) in enumerate(icd9 + icd10):
        icd_codes = icd.split('|')[0].strip().split(' ')
        icd_codes_desc = icd.split('|')[1].strip()
        # print(icd_codes_desc)
        feature_headers.append('Newborn Diagnosis:'+icd_codes_desc)
        for icd_code in icd_codes:
            if icd_code in feature_index:
                feature_index[icd_code].append(ix)
                # print('warning - double icd in 9&10:', icd_code)
            else:
                feature_index[icd_code] = [ix]
    # feature_headers = ['Diagnosis:' + i for i in  (icd9 + icd10)]
    return feature_index, feature_headers

def build_feature_gender_index():
    try:
        codesNnames = [l.strip().decode('utf-8') for l in open(config_file.genderList, 'rb').readlines()]
    except:
        codesNnames = [l.strip().decode('latin-1') for l in open(config_file.genderList, 'rb').readlines()]
    feature_index = {}
    feature_headers = []
    for (ix, codeline) in enumerate(codesNnames):
        code = int(codeline.split(' ')[0].strip())
        if code in feature_index:
            feature_index[code].append(ix)
            # print('double!!', icd_code)
        else:
            feature_index[code] = [ix]
    feature_headers = ['Gender:'+ i for i in codesNnames]
    return feature_index, feature_headers
def build_feature_ethn_index():
    try:
        codesNnames = [l.strip().decode('utf-8') for l in open(config_file.ethnicityList, 'rb').readlines()]
    except:
        codesNnames = [l.strip().decode('latin-1') for l in open(config_file.ethnicityList, 'rb').readlines()]
    feature_index = {}
    feature_headers = []
    for (ix, codeline) in enumerate(codesNnames):
        code = codeline.strip()
        if code in feature_index:
            feature_index[code].append(ix)
            # print('double!!', icd_code)
        else:
            feature_index[code] = [ix]
    feature_headers = ['Ethnicity:'+ i for i in codesNnames]
    return feature_index, feature_headers

def build_feature_vitallatest_index():
    try:
        codesNnames = [l.strip().decode('utf-8') for l in open(config_file.vitalsList, 'rb').readlines()]
    except:
        codesNnames = [l.strip().decode('latin-1') for l in open(config_file.vitalsList, 'rb').readlines()]
    feature_index = {}
    feature_headers = []
    for (ix, codeline) in enumerate(codesNnames):
        codes = codeline.strip().split('#')[0]
        descr = codeline.strip().split('#')[1]
        for code in codes.split(' | '):
            if code in feature_index:
                feature_index[code.strip()].append(ix)
            else:
                feature_index[code.strip()] = [ix]
        feature_headers.append('Vital:'+ descr)
    return feature_index, feature_headers

def build_feature_vital_gains_index():
    try:
        codesNnames = [l.strip().decode('utf-8') for l in open(config_file.vitalsGainsList, 'rb').readlines()]
    except:
        codesNnames = [l.strip().decode('latin-1') for l in open(config_file.vitalsGainsList, 'rb').readlines()]
    feature_index = {}
    feature_headers = []
    for (ix, codeline) in enumerate(codesNnames):
        codes = codeline.strip().split('#')[0]
        descr = codeline.strip().split('#')[1]
        for code in codes.split(' | '):
            if code in feature_index:
                feature_index[code.strip()].append(ix)
            else:
                feature_index[code.strip()] = [ix]
        feature_headers.append('Vital:'+ descr)
    return feature_index, feature_headers

def build_feature_zipcd_index():
    try:
        codesNnames = [l.strip().decode('utf-8') for l in open(config_file.zipList, 'rb').readlines()]
    except:
        codesNnames = [l.strip().decode('latin-1') for l in open(config_file.zipList, 'rb').readlines()]
    feature_index = {}
    feature_headers = []
    for (ix, codeline) in enumerate(codesNnames):
        code = codeline.strip()
        if code in feature_index:
            feature_index[code].append(ix)
            # print('double!!', icd_code)
        else:
            feature_index[code] = [ix]
    feature_headers = ['Zipcode:'+ i for i in codesNnames]
    return feature_index, feature_headers

def build_feature_census_index(env_dic):
    feature_index = {}
    feature_headers = []
    counter = 0
    for item in env_dic:
        for item2 in env_dic[item]:
            for k in env_dic[item][item2]:
                if k not in feature_index:
                    feature_index[k] = counter
                    counter += 1
    feature_headers = [''.join(('Census:', i)) for i in feature_index]
    return feature_index, feature_headers

def build_feature_race_index():
    try:
        codesNnames = [l.strip().decode('utf-8') for l in open(config_file.raceList, 'rb').readlines()]
    except:
        codesNnames = [l.strip().decode('latin-1') for l in open(config_file.raceList, 'rb').readlines()]
    feature_index = {}
    feature_headers = []
    for (ix, codeline) in enumerate(codesNnames):
        codes = codeline.strip().split('#')[0]
        descr = codeline.strip().split('#')[1].strip()
        for code in codes.split(' | '):
            if code in feature_index:
                feature_index[code.strip()].append(ix)
            else:
                feature_index[code.strip()] = [ix]
        feature_headers.append(''.join(('Race:',descr)))
    return feature_index, feature_headers

def build_feature_lab_index():
    try:
        labsfile = [l.strip().decode("utf-8")  for l in open(config_file.labslist, 'rb').readlines()]
    except:
        labsfile = [l.strip().decode('latin-1')  for l in open(config_file.labslist, 'rb').readlines()]
    feature_index = {}
    feature_headers = []
    for (ix, labcd) in enumerate(labsfile):
        lab_codes = labcd.split('|')[0].strip().split('#')
        lab_codes_desc = labcd.split('|')[1].strip()
        feature_headers.append('Lab:'+lab_codes_desc)
        for lab_code in lab_codes:
            if lab_code in feature_index:
                feature_index[lab_code].append(ix)
            else:
                feature_index[lab_code] = [ix]
    return feature_index, feature_headers

def build_feature_med_index():
    try:
        medsfile = [l.strip().decode("utf-8")  for l in open(config_file.medslist, 'rb').readlines()]
    except:
        medsfile = [l.strip().decode('latin-1')  for l in open(config_file.medslist, 'rb').readlines()]

    feature_index = {}
    feature_headers = []
    for (ix, medcd) in enumerate(medsfile):
        med_codes = medcd.split('|')[0].strip().split('#')
        med_codes_desc = medcd.split('|')[1].strip()
        feature_headers.append('Medication:'+med_codes_desc)
        for med_code in med_codes:
            if med_code in feature_index:
                feature_index[med_code].append(ix)
            else:
                feature_index[med_code] = [ix]
    return feature_index, feature_headers


def build_feature_ICD_index():
    try:
        icd9 = [l.strip().decode("utf-8")  for l in open(config_file.icd9List, 'rb').readlines()]
    except:
        icd9 = [l.strip().decode('latin-1')  for l in open(config_file.icd9List, 'rb').readlines()]
    try:
        icd10 = [l.strip().decode("utf-8")  for l in open(config_file.icd10List, 'rb').readlines()]
    except:
        icd10 = [l.strip().decode('latin-1')   for l in open(config_file.icd10List, 'rb').readlines()]
    feature_index = {}
    feature_headers = []
    for (ix, icd) in enumerate(icd9 + icd10):
        icd_codes = icd.split('|')[0].strip().split(' ')
        icd_codes_desc = icd.split('|')[1].strip()
        # print(icd_codes_desc)
        feature_headers.append('Diagnosis:' + icd_codes_desc)
        for icd_code in icd_codes:
            if icd_code in feature_index:
                feature_index[icd_code].append(ix)
                # print('warning - double icd in 9&10:', icd_code)
            else:
                feature_index[icd_code] = [ix]
    # feature_headers = ['Diagnosis:' + i for i in  (icd9 + icd10)]
    return feature_index, feature_headers

def build_feature_mat_hist_labs_index():
    try:
        codesNnames = [l.strip().decode('utf-8') for l in open(config_file.BM_Labs, 'rb').readlines()]
    except:
        codesNnames = [l.strip().decode('latin-1') for l in open(config_file.BM_Labs, 'rb').readlines()]
    feature_index = {}
    feature_headers = []
    for (ix, codeline) in enumerate(codesNnames):
        codes = codeline.strip().split('#')[0]
        descr = codeline.strip().split('#')[1]
        for code in codes.split(' | '):
            if code in feature_index:
                feature_index[code.strip()].append(ix)
            else:
                feature_index[code.strip()] = [ix]
        feature_headers.append('Maternal Lab History:'+ descr)
    return feature_index, feature_headers

def build_feature_mat_hist_meds_index():
    try:
        medsfile = [l.strip().decode("utf-8")  for l in open(config_file.BM_Meds, 'rb').readlines()]
    except:
        medsfile = [l.strip().decode('latin-1')  for l in open(config_file.BM_Meds, 'rb').readlines()]

    feature_index = {}
    feature_headers = []
    for (ix, medcd) in enumerate(medsfile):
        med_codes = medcd.split('|')[0].strip().split('#')
        med_codes_desc = medcd.split('|')[1].strip()
        feature_headers.append('Maternal Medication History:' + med_codes_desc)
        for med_code in med_codes:
            if med_code in feature_index:
                feature_index[med_code].append(ix)
            else:
                feature_index[med_code] = [ix]
    return feature_index, feature_headers

def build_feature_mat_hist_procedures_index():
    try:
        procsfile = [l.strip().decode("utf-8")  for l in open(config_file.BM_Procedures, 'rb').readlines()]
    except:
        procsfile = [l.strip().decode('latin-1')  for l in open(config_file.BM_Procedures, 'rb').readlines()]

    feature_index = {}
    feature_headers = []
    for (ix, proccd) in enumerate(procsfile):
        feature_headers.append('Maternal Procedure History:' + proccd)
        if proccd in feature_index:
            feature_index[proccd].append(ix)
        else:
            feature_index[proccd] = [ix]
    return feature_index, feature_headers

def build_feature_mat_hist_icd_index():
    try:
        icd9 = [l.strip().decode("utf-8")  for l in open(config_file.icd9List, 'rb').readlines()]
    except:
        icd9 = [l.strip().decode('latin-1')  for l in open(config_file.icd9List, 'rb').readlines()]
    try:
        icd10 = [l.strip().decode("utf-8")  for l in open(config_file.icd10List, 'rb').readlines()]
    except:
        icd10 = [l.strip().decode('latin-1')   for l in open(config_file.icd10List, 'rb').readlines()]
    feature_index = {}
    feature_headers = []
    for (ix, icd) in enumerate(icd9 + icd10):
        icd_codes = icd.split('|')[0].strip().split(' ')
        icd_codes_desc = icd.split('|')[1].strip()
        # print(icd_codes_desc)
        feature_headers.append('Maternal Diagnosis History:'+icd_codes_desc)
        for icd_code in icd_codes:
            if icd_code in feature_index:
                feature_index[icd_code].append(ix)
                # print('warning - double icd in 9&10:', icd_code)
            else:
                feature_index[icd_code] = [ix]
    # feature_headers = ['Diagnosis:' + i for i in  (icd9 + icd10)]
    return feature_index, feature_headers

def build_feature_num_visits_index():
    feature_index = {0}
    feature_headers = ['Number of Visits']
    return feature_index, feature_headers

def get_obesity_label_bmi(pct, bmi, age, gender):
    """
    Returns the obesity label as underweight, normal, overweight, obese, class I or class II severe obesity

    Parameters
    ----------
    pct : float or int
        Percentile for patient's BMI given their age and gender.
    bmi : float or int
        Patient's BMI.
    age : float or int
        Patient's age in years.
    gender : int
        0 : Male
        1 : Female

    Returns
    -------
    label : str
        Outcome label for a patient's BMI given their age and gender.
    """
    if 0 <= pct < 0.05:
        return 'underweight'
    elif 0.05 <= pct < 0.85:
        return 'normal'
    elif 0.85 <= pct < 0.95:
        return 'overweight'
    elif 0.95 <= pct < 1:
        if zscore.severe_obesity_bmi(gender, age, bmi, unit='years', severity=2):
            return 'class II severe obesity'
        elif zscore.severe_obesity_bmi(gender, age, bmi, unit='years', severity=1):
            return 'class I severe obesity'
        else:
            return 'obese'
    else:
        return 'none'

def get_obesity_label_wfl(pct, ht, wt, gender, units='usa'):
    """
    Returns the obesity label as underweight, normal, overweight, obese, class I or class II severe obesity

    Parameters
    ----------
    pct : int or float
        Percentile for patient's weight for length.
    ht : int or float
        Patient's height.
    wt : int or float
        Patient's weight.
    gender : int
        0 : Male.
        1 : Female.
    Units : str, default 'usa'
        'usa' : uses inches and pounds for height and weight, respectively.
        'metric' : uses centimeters and kilograms for height and weigth, respectively.

    Returns
    -------
    label : str
        CDC classifcation for a patient's weight for length percentile.
    """
    if 0 <= pct < 0.05:
        return 'underweight'
    elif 0.05 <= pct < 0.85:
        return 'normal'
    elif 0.85 <= pct < 0.95:
        return 'overweight'
    elif 0.95 <= pct < 1:
        if zscore.severe_obesity_wfl(gender, ht, wt, units=units, severity=2):
            return 'class II severe obesity'
        elif zscore.severe_obesity_wfl(gender, ht, wt, units=units, severity=1):
            return 'class I severe obesity'
        else:
            return 'obese'
    else:
        return 'none'

def get_final_bmi(data_dic, agex_low, agex_high, mrnsForFilter=[], filter=True):
    """
    Function to get the distinct BMI percentile readings for predictions.
    
    Parameters
    ----------
    data_dic : dict
        Dictionary of patient data
    agex_low : int or float
        Lower bound for prediction window.
    agex_high : int or float
        Upper bound for prediction window.
    mrnsForFilter : list
        List of MRNs for which data will be produced.
    filter : bool, default True
        If True returns mrn filtered data only, otherwise returns all data with either a 0 or '' for patient's with no data available.
    
    Returns
    -------
    outcome : array, shape (n_samples, ), dtype float
        Median BMI reading for all readings in the prediction window.
    outcome_pct : array, shape (n_samples, ), dtype float
        CDC percentile corresponding to the BMI, age, and gender of patient.
    outcome_labels : array, shape (n_samples, ), dtype str
        CDC label for a patient's BMI percentile.
    """
    outcome = np.zeros(len(data_dic), dtype=float)
    outcome_pct = np.zeros(len(data_dic), dtype=float)
    outcome_labels = ['none'] * len(data_dic)
    indices = np.zeros(len(data_dic))
    for (ix, k) in enumerate(data_dic):
        if (len(mrnsForFilter) > 0) & (str(data_dic[k]['mrn']) not in mrnsForFilter):
            continue
        bmi, pct, label = get_final_bmi_single(data_dic[k], agex_low, agex_high)
        if pct == 0 and label == 'none':
            continue
        outcome[ix] = bmi
        outcome_pct[ix] = pct
        outcome_labels[ix] = label
        indices[ix] = 1
    if filter:
        indices = (indices == 1)
        return outcome[indices], outcome_pct[indices], np.array(outcome_labels)[indices]
    else:
        return outcome, outcome_pct, np.array(outcome_labels)

def get_latest_reading(data_dic, months_from, months_to, mrnsForFilter=[], zero_filter=True):
    """
    Function to get the distinct bmi percentile readings for predictions.

    Parameters
    ----------
    data_dic : dict
        Data dictionary of patient data
    months_from : int
        Lower bound on data consideration window.
    months_to : int
        Upper bound on data consideration window.
    mrnsForFilter : list 
        List of mrns that will be skipped.
    zero_filter : bool, default True
        Indicator to return mrn filtered data only or all data.

    Returns
    -------
    outcome : array, shape (len(data_dic),)
        Weight for length or BMI if months_to is less than or equal to or greather than 24, respectively.
        If zero_filter, then shape is smaller.
    outcome_pct : array, shape (len(data_dic),)
        Percentile for outcome data. If zero_filter, then shape is smaller.
    outcome_labels : array, shape (len(data_dic),)
        String output for label class ranging from underweight to class II severe obesity.
    """
    outcome = np.zeros(len(data_dic), dtype=float) if months_to > 24 else np.zeros((len(data_dic), 2), dtype=float)
    outcome_pct = np.zeros(len(data_dic), dtype=float)
    outcome_labels = [''] * len(data_dic)
    indices = np.zeros(len(data_dic))
    for (ix, k) in enumerate(data_dic):
        if (len(mrnsForFilter) > 0) & (str(data_dic[k]['mrn']) not in mrnsForFilter):
            continue
        age, reading, pct, label = get_latest_label_single(data_dic[k], months_from, months_to)
        if pct == 0 and label == '':
            continue
        outcome[ix] = reading
        outcome_pct[ix] = pct
        outcome_labels[ix] = label
        indices[ix] = 1
    if zero_filter:
        indices = (indices == 1)
        return outcome[indices,:], outcome_pct[indices], np.array(outcome_labels)[indices]
    else:
        return outcome, outcome_pct, np.array(outcome_labels)

def get_final_bmi_single(patient_data, agex_low, agex_high):
    """
    Function to get the BMI percentile and outcome label for an individual patient.

    Parameters
    ----------
    patient_data : dict
        Data dictionary for a single patient.
    agex_low : int or float
        Lower bound on age for prediction window.
    agex_high : int or float
        Upper bound on age for prediction window.

    Returns
    -------
    BMI : float
        Median BMI value for all BMI readings between agex_low and agex_low. 0 if there is no valid data.
    BMI_pct : float
        Percentile for the BMI value with respect to the median age for all readings in the data. 0 if there is no valid data.
    obese_label : string
        Label ranging from underweight to class II severe obesity for a patient's BMI and age. '' if there is no valid data.
    """
    bdate = patient_data['bdate']
    gender = patient_data['gender']
    BMI_pct_list = []
    BMI_label_list = []
    BMI_list = []
    age_list = []
    if ('vitals' in patient_data) and ('BMI' in patient_data['vitals']):
        for (edate, bmi) in patient_data['vitals']['BMI']:
            age = (edate - bdate).days / 365.0
            if (age >= agex_low) and (age < agex_high):
                age_list.append(age)
                BMI_list.append(bmi)
                BMI_pct_list.append(stats.norm.cdf(zscore.zscore_bmi(gender, age, bmi, unit='years'))) # function takes age as months
    if len(BMI_pct_list) > 1:
        pct_med = np.median(np.array(BMI_pct_list))
        age_med = np.median(np.array(age_list))
        bmi_med = np.median(np.array(BMI_list))
        return bmi_med, pct_med, get_obesity_label_bmi(pct_med, bmi_med, age_med, gender)
    elif BMI_pct_list == []:
        return 0, 0, 'none'
    else:
        return BMI_list[0], BMI_pct_list[0], get_obesity_label_bmi(BMI_pct_list[0], BMI_list[0], age_list[0], gender)

def get_latest_label_single(patient_data, months_from, months_to):
    """
    Function to get the BMI or WFL percentile and outcome label for an individual patient
    
    Parameters
    ----------
    patient_data : dict
        Data dictionary for patient.
    months_from : int
        Lower bound on data validity window.
    months_to : int
        Upper bound on data validity window.

    Returns
    -------
    age : float.
        Age of patient at reading in years
    percentile : float
        BMI or wfl percentile for patient a patitent under or over 2 years of age, respectively.
    label : 
        CDC label for underweight, normal, overweight, obese, class I severe obesity, and class II severe obesity coresponding with percentile.
    """
    bdate = patient_data['bdate']
    gender = patient_data['gender']

    start_date = bdate + relativedelta(months=months_from)
    end_date = bdate + relativedelta(months=months_to)
    age_final = 0
    label_final = 'none'
    if months_to > 24:
        bmi_final = 0
        bmi_pct_final = 0
        if ('vitals' in patient_data) and ('BMI' in patient_data['vitals']):
            for (edate, bmi) in patient_data['vitals']['BMI']:
                if (edate >= start_date) and (edate < end_date):
                    age_final = (edate - bdate).days / 365.0
                    bmi_final = bmi
                    bmi_pct_final = stats.norm.cdf(zscore.zscore_bmi(gender, age_final, bmi_final, unit='years'))
                    label_final = get_obesity_label_bmi(bmi_pct_final, bmi_final, age_final, gender)
            return age_final, bmi_final, bmi_pct_final, label_final
    else:
        ht_final = 0
        wt_final = 0
        wfl_pct_final = 0
        if ('vitals' in patient_data) and ('Ht' in patient_data['vitals']) and ('Wt' in patient_data['vitals']):
            for (edate, ht) in patient_data['vitals']['Ht']:
                if (edate >= start_date) and (edate < end_date):
                    for (wdate, wt) in patient_data['vitals']['Wt']:
                        if edate == wdate:
                            age_final = age_final = (edate - bdate).days / 365.0
                            ht_final = ht
                            wt_final = wt
                            wfl_pct_final = stats.norm.cdf(zscore.zscore_wfl(gender, ht_final, wt_final, units='usa'))
                            label_final = get_obesity_label_wfl(wfl_pct_final, ht_final, wt_final, gender)
        return age_final, (ht_final, wt_final), wfl_pct_final, label_final

def call_build_function(data_dic, data_dic_moms, data_dic_hist_moms, lat_lon_dic, env_dic, agex_low, agex_high, months_from, months_to, percentile, prediction='obese', mrnsForFilter=[]):
    """
    Creates the base data set to be used in analysis for obesity prediction.

    Parameters
    ----------
    data_dic : dict
        Data dictionary of children's EHR.
    data_dic_moms : dict
        Data dictionary of of mother's EHR at time of child's birth.
    data_dic_hist_moms : dict
        Data dictionary of mother's EHR that is within the same hospital system.
    lat_lon_dic : dict
        Data dictionary of the child's or mother's geocoded address information.
    env_dic : dict
        Data dictionary of census features. --------- DOUBLE CHECK THE FORMAT OF THIS DATA!!!!
    agex_low : int or float
        Lower bound on prediction window.
    agex_high : int or float
        Upper bound on prediction window.
    months_from : int
        Lower bound on data validity window.
    months_to int
        Upper bound on data validity window.
    percentile : bool, default False
        Filter to ensure certain types of features exist for each data point.
    prediction : str, default 'obese'
        Obesity threshold for bmi/age percentile for outcome class.
        Source: https://www.cdc.gov/obesity/childhood/defining.html
        'underweight': 0.0 <= bmi percentile < 0.05
        'normal': 0.05 <= bmi percentile < 0.85
        'overweight': 0.85 <= bmi percentile < 0.95
        'obese': 0.95 <= bmi percentile <= 1.0
        'class I severe obesity': class I severe obesity; 120% of the 95th percentile
        'class II severe obesity': class II severe obesity; 140% of the 95th percentile
        'multi': multiclass label for columns ['underweight','normal','overweight','obese','class I severe obesity','class II severe obesity']
            NOTE: will return redundant labels for obese and severe obese classes as they are a subset
    mrnsForFilter : list, default []
        Filter for valid MRNs. All other data will be skipped.

    Returns
    -------
    features : array, shape (n_samples, n_features)
        Data array.
    outcome : array, shape (n_samples, )
        Target data.
    outcomelabels : array, shape (n_samples, )
        Target labels.
    headers : list, shape (n_features)
        Feature names corresponding to columns of features.
    mrns : array, shape (n_samples, )
        MRNs for patients with created data.
    """

    multi_ix = {
        'underweight': 0,
        'normal': 1,
        'overweight': 2,
        'obese': 3,
        'class I severe obesity': [3, 4], 
        'class II severe obesity': [3, 5],
        'none': 6
    }
    outcome = np.zeros(len(data_dic.keys()), dtype=float)
    if prediction != 'multi':
        outcomelabels = np.zeros(len(data_dic.keys()), dtype=float)
        multi = False
    else:
        outcomelabels = np.zeros((len(data_dic.keys()), len(multi_ix)), dtype=float)
        multi = True
    if prediction not in multi_ix.keys() and not multi:
        warnings.warn('Invalid prediction parameter. Using default "obese" thresholds.')
        prediction = 'obese'

    feature_index_gen, feature_headers_gen = build_feature_gender_index()
    feature_index_icd, feature_headers_icd = build_feature_ICD_index()
    feature_index_lab, feature_headers_lab = build_feature_lab_index()
    feature_index_med, feature_headers_med = build_feature_med_index()
    feature_index_ethn, feature_headers_ethn = build_feature_ethn_index()
    feature_index_race, feature_headers_race = build_feature_race_index()
    feature_index_zipcd, feature_headers_zipcd = build_feature_zipcd_index()
    feature_index_census, feature_headers_census = build_feature_census_index(env_dic)
    feature_index_vitalLatest, feature_headers_vitalsLatest = build_feature_vitallatest_index()
    feature_index_vitalGains, feature_headers_vitalsGains = build_feature_vital_gains_index()
    feature_index_numVisits, feature_headers_numVisits = build_feature_num_visits_index()

    feature_index_mat_ethn, feature_headers_mat_ethn = build_feature_matethn_index()
    feature_index_mat_race, feature_headers_mat_race = build_feature_matrace_index()
    feature_index_mat_marriage, feature_headers_mat_marriage = build_feature_matmarriage_index()
    feature_index_mat_natn, feature_headers_mat_natn = build_feature_matnatn_index()
    feature_index_mat_birthpl, feature_headers_mat_birthpl = build_feature_matbirthpl_index()
    feature_index_mat_lang, feature_headers_mat_lang = build_feature_matlang_index()
    feature_index_mat_agedeliv, feature_headers_age_deliv = build_feature_agedeliv_index()
    feature_index_mat_icd, feature_headers_mat_icd = build_feature_Mat_ICD_index()
    feature_index_nb_icd, feature_headers_nb_icd = build_feature_NB_ICD_index()
    feature_index_mat_insurance1, feature_headers_mat_insurance1 = build_feature_matinsurance1_index()
    feature_index_mat_insurance2, feature_headers_mat_insurance2 = build_feature_matinsurance2_index()

    feature_index_mat_hist_labsAverage, feature_headers_mat_hist_labs = build_feature_mat_hist_labs_index()
    # feature_index_mat_hist_medsAverage, feature_headers_mat_hist_meds = build_feature_mat_hist_meds_index()
    feature_index_mat_hist_procsAverage, feature_headers_mat_hist_procs = build_feature_mat_hist_procedures_index()
    feature_index_mat_hist_icd, feature_headers_mat_hist_icd = build_feature_Mat_ICD_index()

    mother_child_dic = mother_child_map(data_dic, data_dic_moms, data_dic_hist_moms)

    funcs = [
        (build_features_icd, [ feature_index_icd, feature_headers_icd ]), #
        (build_features_lab, [ feature_index_lab, feature_headers_lab ]),
        (build_features_med, [ feature_index_med, feature_headers_med ]),
        (build_features_gen, [ feature_index_gen, feature_headers_gen ]), #
        (build_features_ethn, [ feature_index_ethn, feature_headers_ethn]),
        (build_features_race, [ feature_index_race, feature_headers_race]),
        (build_features_vitalLatest, [ feature_index_vitalLatest, [h+'-latest' for h in feature_headers_vitalsLatest]]),
        # (build_features_vitalAverage_0_3, [ feature_index_vitalLatest, [h+'-avg0to3' for h in feature_headers_vitalsLatest]]),
        # (build_features_vitalAverage_3_6, [ feature_index_vitalLatest, [h+'-avg3to6' for h in feature_headers_vitalsLatest]]),
        # (build_features_vitalAverage_6_9, [ feature_index_vitalLatest, [h+'-avg6to9' for h in feature_headers_vitalsLatest]]),
        # (build_features_vitalAverage_9_12, [ feature_index_vitalLatest, [h+'-avg9to12' for h in feature_headers_vitalsLatest]]),
        # (build_features_vitalAverage_12_15, [ feature_index_vitalLatest, [h+'-avg12to15' for h in feature_headers_vitalsLatest]]),
        # (build_features_vitalAverage_15_18, [ feature_index_vitalLatest, [h+'-avg15to18' for h in feature_headers_vitalsLatest]]),
        # # (build_features_vitalAverage_18_21, [ feature_index_vitalLatest, [h+'-avg18to21' for h in feature_headers_vitalsLatest]]),
        # (build_features_vitalAverage_18_24, [ feature_index_vitalLatest, [h+'-avg18to24' for h in feature_headers_vitalsLatest]]),
        (build_features_vitalAverage_0_0, [ feature_index_vitalLatest, [h+'-AtBirth' for h in feature_headers_vitalsLatest]]),
        (build_features_vitalAverage_0_1, [ feature_index_vitalLatest, [h+'-avg0to1' for h in feature_headers_vitalsLatest]]),
        (build_features_vitalAverage_1_3, [ feature_index_vitalLatest, [h+'-avg1to3' for h in feature_headers_vitalsLatest]]),
        (build_features_vitalAverage_3_5, [ feature_index_vitalLatest, [h+'-avg3to5' for h in feature_headers_vitalsLatest]]),
        (build_features_vitalAverage_5_7, [ feature_index_vitalLatest, [h+'-avg5to7' for h in feature_headers_vitalsLatest]]),
        (build_features_vitalAverage_7_10, [ feature_index_vitalLatest, [h+'-avg7to10' for h in feature_headers_vitalsLatest]]),
        (build_features_vitalAverage_10_13, [ feature_index_vitalLatest, [h+'-avg10to13' for h in feature_headers_vitalsLatest]]),
        (build_features_vitalAverage_13_16, [ feature_index_vitalLatest, [h+'-avg13to16' for h in feature_headers_vitalsLatest]]),
        (build_features_vitalAverage_16_19, [ feature_index_vitalLatest, [h+'-avg16to19' for h in feature_headers_vitalsLatest]]),
        (build_features_vitalAverage_19_24, [ feature_index_vitalLatest, [h+'-avg19to24' for h in feature_headers_vitalsLatest]]),
        (build_features_vitalGain_0_3, [ feature_index_vitalGains, [h+'-gain0to3' for h in feature_headers_vitalsGains]]),
        (build_features_vitalGain_1_5, [ feature_index_vitalGains, [h+'-gain1to5' for h in feature_headers_vitalsGains]]),
        (build_features_vitalGain_3_7, [ feature_index_vitalGains, [h+'-gain3to7' for h in feature_headers_vitalsGains]]),
        (build_features_vitalGain_5_10, [ feature_index_vitalGains, [h+'-gain5to10' for h in feature_headers_vitalsGains]]),
        (build_features_vitalGain_7_13, [ feature_index_vitalGains, [h+'-gain7to13' for h in feature_headers_vitalsGains]]),
        (build_features_vitalGain_10_16, [ feature_index_vitalGains, [h+'-gain10to16' for h in feature_headers_vitalsGains]]),
        (build_features_vitalGain_13_19, [ feature_index_vitalGains, [h+'-gain13to19' for h in feature_headers_vitalsGains]]),
        (build_features_vitalGain_16_24, [ feature_index_vitalGains, [h+'-gain16to24' for h in feature_headers_vitalsGains]]),
        (build_features_vitalGain_0_24, [ feature_index_vitalGains, [h+'-gain0to24' for h in feature_headers_vitalsGains]]),
        (build_features_numVisits, [feature_index_numVisits, feature_headers_numVisits]),
        # environment
        (build_features_zipcd_birth, [ feature_index_zipcd, [h+'-birth' for h in feature_headers_zipcd]]),
        (build_features_zipcd_latest, [ feature_index_zipcd, [h+'-latest' for h in feature_headers_zipcd]]),
        (build_features_census_birth, [ feature_index_census, [h+'-birth' for h in feature_headers_census]]),
        (build_features_census_latest, [ feature_index_census, [h+'-latest' for h in feature_headers_census]]),
        # maternal features
        (build_features_mat_icd, [ feature_index_mat_icd, feature_headers_mat_icd]), #
        (build_features_nb_icd, [ feature_index_nb_icd, feature_headers_nb_icd]),
        # (build_features_del_icd, [ feature_index_mat_deldiag, feature_headers_mat_deldiag ]),
        (build_features_mat_ethn, [ feature_index_mat_ethn, feature_headers_mat_ethn]), #
        (build_features_mat_insurance1, [ feature_index_mat_insurance1, feature_headers_mat_insurance1]), #
        (build_features_mat_insurance2, [ feature_index_mat_insurance2, feature_headers_mat_insurance2]),
        (build_features_mat_race, [ feature_index_mat_race, feature_headers_mat_race]),
        (build_features_mat_lang, [ feature_index_mat_lang, feature_headers_mat_lang]),
        (build_features_mat_natn, [ feature_index_mat_natn, feature_headers_mat_natn]),
        (build_features_mat_marriage, [ feature_index_mat_marriage, feature_headers_mat_marriage ]), #
        (build_features_mat_birthpl, [ feature_index_mat_birthpl, feature_headers_mat_birthpl]),
        (build_features_mat_agedel, [ feature_index_mat_agedeliv, feature_headers_age_deliv]),
        #historical maternal features
        (build_features_mat_hist_vitalsAverage_prePregnancy, [feature_index_vitalLatest, ['Maternal ' + h + '-prePregnancy' for h in feature_headers_vitalsLatest], mother_child_dic]),
        (build_features_mat_hist_vitalsAverage_firstTri, [feature_index_vitalLatest, ['Maternal ' + h + '-firstTrimester' for h in feature_headers_vitalsLatest], mother_child_dic]),
        (build_features_mat_hist_vitalsAverage_secTri, [feature_index_vitalLatest, ['Maternal ' + h + '-secondTrimester' for h in feature_headers_vitalsLatest], mother_child_dic]),
        (build_features_mat_hist_vitalsAverage_thirdTri, [feature_index_vitalLatest, ['Maternal ' + h + '-thirdTrimester' for h in feature_headers_vitalsLatest], mother_child_dic]),
        (build_features_mat_hist_vitalsAverage_postPregnancy, [feature_index_vitalLatest, ['Maternal ' + h + '-postPregnancy' for h in feature_headers_vitalsLatest], mother_child_dic]),
        (build_features_mat_hist_vitalsAverage_otherPregnancy, [feature_index_vitalLatest, ['Maternal ' + h + '-otherPregnancy' for h in feature_headers_vitalsLatest], mother_child_dic]),
        (build_features_mat_hist_labsAverage_prePregnancy, [feature_index_mat_hist_labsAverage, [h+'-prePregnancy' for h in feature_headers_mat_hist_labs], mother_child_dic]),
        (build_features_mat_hist_labsAverage_firstTri, [feature_index_mat_hist_labsAverage, [h+'-firstTrimester' for h in feature_headers_mat_hist_labs], mother_child_dic]),
        (build_features_mat_hist_labsAverage_secTri, [feature_index_mat_hist_labsAverage, [h+'-secondTrimester' for h in feature_headers_mat_hist_labs], mother_child_dic]),
        (build_features_mat_hist_labsAverage_thrirdTri, [feature_index_mat_hist_labsAverage, [h+'-thirdTrimester' for h in feature_headers_mat_hist_labs], mother_child_dic]),
        (build_features_mat_hist_labsAverage_postPregnancy, [feature_index_mat_hist_labsAverage, [h+'-postPregnancy' for h in feature_headers_mat_hist_labs], mother_child_dic]),
        (build_features_mat_hist_labsAverage_otherPregnancy, [feature_index_mat_hist_labsAverage, [h+'-otherPregnancy' for h in feature_headers_mat_hist_labs], mother_child_dic]),
        (build_features_mat_hist_icdCount_prePregnancy, [feature_index_mat_hist_icd, [h+'-prePregnancy' for h in feature_headers_mat_hist_icd], mother_child_dic]),
        (build_features_mat_hist_icdCount_firstTri, [feature_index_mat_hist_icd, [h+'-firstTrimester' for h in feature_headers_mat_hist_icd], mother_child_dic]),
        (build_features_mat_hist_icdCount_secTri, [feature_index_mat_hist_icd, [h+'-secondTrimester' for h in feature_headers_mat_hist_icd], mother_child_dic]),
        (build_features_mat_hist_icdCount_thrirdTri, [feature_index_mat_hist_icd, [h+'-thirdTrimester' for h in feature_headers_mat_hist_icd], mother_child_dic]),
        (build_features_mat_hist_icdCount_postPregnancy, [feature_index_mat_hist_icd, [h+'-postPregnancy' for h in feature_headers_mat_hist_icd], mother_child_dic]),
        (build_features_mat_hist_icdCount_otherPregnancy, [feature_index_mat_hist_icd, [h+'-otherPregnancy' for h in feature_headers_mat_hist_icd], mother_child_dic]),
        (build_features_mat_hist_proceduresCount_prePregnancy, [feature_index_mat_hist_procsAverage, [h+'-prePregnancy' for h in feature_headers_mat_hist_procs], mother_child_dic]),
        (build_features_mat_hist_proceduresCount_firstTri, [feature_index_mat_hist_procsAverage, [h+'-firstTrimester' for h in feature_headers_mat_hist_procs], mother_child_dic]),
        (build_features_mat_hist_proceduresCount_secTri, [feature_index_mat_hist_procsAverage, [h+'-secondTrimester' for h in feature_headers_mat_hist_procs], mother_child_dic]),
        (build_features_mat_hist_proceduresCount_thrirdTri, [feature_index_mat_hist_procsAverage, [h+'-thirdTrimester' for h in feature_headers_mat_hist_procs], mother_child_dic]),
        (build_features_mat_hist_proceduresCount_postPregnancy, [feature_index_mat_hist_procsAverage, [h+'-postPregnancy' for h in feature_headers_mat_hist_procs], mother_child_dic]),
        (build_features_mat_hist_proceduresCount_otherPregnancy, [feature_index_mat_hist_procsAverage, [h+'-otherPregnancy' for h in feature_headers_mat_hist_procs], mother_child_dic])
        # (build_features_mat_hist_medsAverage_prePregnancy, [feature_index_mat_hist_medsAverage, [h+'-prePregnancy' for h in feature_headers_mat_hist_meds], mother_child_dic]),
        # (build_features_mat_hist_medsAverage_firstTri, [feature_index_mat_hist_medsAverage, [h+'-firstTrimester' for h in feature_headers_mat_hist_meds], mother_child_dic]),
        # (build_features_mat_hist_medsAverage_secTri, [feature_index_mat_hist_medsAverage, [h+'-secondTrimester' for h in feature_headers_mat_hist_meds], mother_child_dic]),
        # (build_features_mat_hist_medsAverage_thrirdTri, [feature_index_mat_hist_medsAverage, [h+'-thirdTrimester' for h in feature_headers_mat_hist_meds], mother_child_dic]),
        # (build_features_mat_hist_medsAverage_postPregnancy, [feature_index_mat_hist_medsAverage, [h+'-postPregnancy' for h in feature_headers_mat_hist_meds], mother_child_dic]),
        # (build_features_mat_hist_medsAverage_otherPregnancy, [feature_index_mat_hist_medsAverage, [h+'-otherPregnancy' for h in feature_headers_mat_hist_meds], mother_child_dic])
    ]

    features = np.zeros((len(data_dic.keys()), sum([len(f[1][1]) for f in funcs ]) ), dtype=float)
    mrns = [0]*len(data_dic.keys())

    headers = []
    for (pos, f ) in enumerate(funcs):
        headers += f[1][1]

    num = '{:,d}'.format(len(data_dic))
    mom_keys = np.nan_to_num(np.array([*data_dic_moms])).astype(int).astype(str)
    for (ix, k) in tqdm(enumerate(data_dic), desc='Processing ' + num + ' patients'):
        if (len(mrnsForFilter) > 0) & (str(data_dic[k]['mrn']) not in mrnsForFilter):
            continue
        flag=False
        bmi, pct, label = get_final_bmi_single(data_dic[k], agex_low, agex_high)
        if pct == 0 and label ==  'none':
            outcomelabels[ix] = 0
        outcome[ix] = bmi
        if not multi:
            if prediction == 'obese':
                outcomelabels[ix] = 1 if label in ('obese','class I severe obesity','class II severe obesity') else 0
            else:
                outcomelabels[ix] = 1 if prediction == label else 0
        else:
            outcomelabels[ix, multi_ix[label]] = 1


        bdate = data_dic[k]['bdate']
        mrns[ix] = data_dic[k]['mrn']
        if mrns[ix] in data_dic_moms:
            maternal_data = data_dic_moms[mrns[ix]]
            if data_dic_moms[mrns[ix]]['mom_mrn'] in data_dic_hist_moms:
                maternal_hist_data = data_dic_hist_moms[data_dic_moms[mrns[ix]]['mom_mrn']]
                try:
                    mother_child_data = mother_child_dic[data_dic_moms[k]['mom_mrn']]
                except:
                    mother_child_data = {}
            else:
                maternal_hist_data = {}
        else:
            maternal_data = {}
            maternal_hist_data = {}
            mother_child_data = {}
        try:
            lat_lon_item = lat_lon_dic[str(mrns[ix])]
        except:
            try:
                lat_lon_item = lat_lon_dic[mrns[ix]]
            except:
                lat_lon_item = {}
        ix_pos_start = 0
        ix_pos_end = len(funcs[0][1][1])
        for (pos, f) in enumerate(funcs):
            func = f[0]
            features[ix, ix_pos_start:ix_pos_end] = func(
                data_dic[k],
                maternal_data,
                maternal_hist_data,
                lat_lon_item,
                env_dic,
                bdate + relativedelta(months=months_from), # timedelta(days=months_from*30)
                bdate + relativedelta(months=months_to), # timedelta(days=months_to*30)
                *f[1])
            ix_pos_start += len(f[1][1])
            try:
                ix_pos_end += len(funcs[pos+1][1][1])
            except IndexError:
                ix_pos_end = features.shape[1]

    # Calculate the Z-Scores for each of the vital periods and the gain between them
    zscore_headers = ['Vital: Wt for Length ZScore-AtBirth','Vital: Wt for Length ZScore-avg0to1','Vital: Wt for Length ZScore-avg1to3','Vital: Wt for Length ZScore-avg3to5','Vital: Wt for Length ZScore-avg5to7','Vital: Wt for Length ZScore-avg7to10','Vital: Wt for Length ZScore-avg10to13','Vital: Wt for Length ZScore-avg13to16','Vital: Wt for Length ZScore-avg16to19','Vital: Wt for Length ZScore-avg19to24','Vital: Wt for Length ZScore-latest']
    zscore_gain_headers = ['Vital: Wt for Length ZScore-gain0to3','Vital: Wt for Length ZScore-gain1to5','Vital: Wt for Length ZScore-gain3to7','Vital: Wt for Length ZScore-gain5to10','Vital: Wt for Length ZScore-gain7to13','Vital: Wt for Length ZScore-gain10to16','Vital: Wt for Length ZScore-gain13to19','Vital: Wt for Length ZScore-gain16to24']
    headers += zscore_headers + zscore_gain_headers

    zscores = np.zeros((len(mrns),len(zscore_headers)))
    zscores_gain = np.zeros((len(mrns),len(zscore_gain_headers)))
    for ix, age in enumerate(['-AtBirth','-avg0to1','-avg1to3','-avg3to5','-avg5to7','-avg7to10','-avg10to13','-avg13to16','-avg16to19','-avg19to24', '-latest']):
        wts = features[:,headers.index('Vital: Wt'+age)]
        hts = features[:,headers.index('Vital: Ht'+age)]
        genders = features[:,headers.index('Gender:1 female')]
        zscores[:,ix] = zscore.zscore_wfl(genders, hts, wts, units='usa')
    for ix in range(len(zscore_gain_headers)):
        zscores_gain[:,ix] = zscores[:,ix+1] - zscores[:,ix]

    headers += ['Vital: Wt for Length ZScore-gain0to24']
    zscore_gain_0_24 = zscores[:,-1] - zscores[:,0]

    features = np.hstack((features, zscores, zscores_gain, zscore_gain_0_24.reshape(-1,1)))
    return features, outcome, outcomelabels, headers, np.array(mrns)
