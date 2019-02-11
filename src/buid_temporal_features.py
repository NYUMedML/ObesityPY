import config as config_file
import pickle 
import re
import matplotlib.pylab as plt
import time
from datetime import timedelta
from dateutil import parser
import numpy as np
import CDC_percentiles
from dateutil import relativedelta
import pickle
Vital_Percentile = {}

def init():
    CDC_bmi, unitbmi, plevelsbmi = CDC_percentiles.load_CDC_refs('../auxdata/CDC_BMIs.txt')
    CDC_hc, unitHC, plevs = CDC_percentiles.load_CDC_refs('../auxdata/CDC_HeadCirc.txt')
    CDC_wei, unitwei , plevs = CDC_percentiles.load_CDC_refs('../auxdata/CDC_Weights.txt')
    CDC_len, unitlen , plevs = CDC_percentiles.load_CDC_refs('../auxdata/CDC_Length.txt')
    Vital_Percentile['BMI']=(CDC_bmi, unitbmi, plevelsbmi)
    Vital_Percentile['HC']=(CDC_hc, unitHC, plevs)
    Vital_Percentile['Wt']=(CDC_wei, unitwei , plevs)
    Vital_Percentile['Ht']=(CDC_len, unitlen , plevs)


def built_vital_features(bdate, datalist, gender, datatype='HC'):
    data = np.zeros((12*18,), dtype=float)
    data_percentile = np.zeros((12*18,), dtype=float)	
    for (edate, vitalval) in datalist:
        # age_at_vital_month = int((edate - bdate).month / 30.0) #which month is it.
        age_at_vital = relativedelta.relativedelta(edate, bdate)
        months_since_birth = int(age_at_vital.years*12 + age_at_vital.months)
        data[months_since_birth] = vitalval
        try:
            (CDC_per, unit, plevels) = Vital_Percentile[datatype]
        except:
            continue
        p = CDC_percentiles.percentile(vitalval, gender, months_since_birth, CDC_per, unit, plevels )
        if (p == 0.05) or (p == 0.97) :
            oldvitalval = vitalval*1.0
            if datatype in ['HC', 'Ht']:
                vitalval *= 2.54 #inch urg convert to CM

            if datatype in ['Wt']:
                vitalval *= 0.45 #convert to kg and try again

            p = CDC_percentiles.percentile(vitalval, gender, months_since_birth, CDC_per, unit, plevels)
            if (p == 0.97) or (p == 0.05): #revert back. didn't work out.
                p = CDC_percentiles.percentile(oldvitalval, gender, months_since_birth, CDC_per, unit, plevels)

        # print('vital measured at month:', age_at_vital, 'val:', vitalval, 'percentile:', p)
        data_percentile[months_since_birth] = p
    # print(data)
    # print(data_percentile)
    return data, data_percentile

def build_data(datadic):
    vitals = ['BMI', 'HC', 'Ht Percentile', 'Wt Percentile']
    data = np.zeros((len(datadic), len(vitals), 12*18), dtype=float)
    data_percentile = np.zeros((len(datadic), len(vitals), 12*18), dtype=float)
    datakeys = []
    datagenders = []
    dataEthn = []
    dataRace = []
    dataDisease = []
    for (ix, k) in enumerate(datadic.keys()):
        print('user id:', ix, k)
        if 'vitals' in datadic[k]:
            datakeys.append(k)
            datagenders.append(datadic[k]['gender'])
            dataEthn.append(datadic[k]['ethnicity'])	
            dataRace.append(datadic[k]['race'])
            for (jx, vtype) in enumerate(vitals):
                if vtype in datadic[k]['vitals'].keys():
                    # print(datadic[k]['bdate'], datadic[k]['vitals'][vtype], datadic[k]['gender'], vtype)
                    d, dp = built_vital_features(datadic[k]['bdate'], datadic[k]['vitals'][vtype], datadic[k]['gender'], vtype)
                    data[ix, jx, :] = d.copy()
                    data_percentile[ix, jx, :] = dp.copy()
    timestr = time.strftime("%Y%m%d-%H%M%S")	
    pickle.dump(file=open('timeseries_data'+timestr+'.pkl','wb'), obj=(data, data_percentile, datakeys, datagenders, dataEthn, dataRace), protocol=-1)
    return "(data, data_percentile, datakeys, datagenders, dataEthn, dataRace) was stored in pickle file:" + "timeseries_data" + timestr + ".pkl"

init()



