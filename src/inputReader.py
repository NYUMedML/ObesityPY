import config as config_file
import pandas as pd
import pickle 
import re
import matplotlib.pylab as plt
import time
from datetime import timedelta
from dateutil import parser
import numpy as np

def load_csv_input():
    print('loading data:', config_file.input_csv)
    data1 = pd.read_csv(config_file.input_csv[0], delimiter=config_file.input_csv_delimiter)
    data2 = pd.read_csv(config_file.input_csv[1], delimiter=config_file.input_csv_delimiter)
    data = pd.concat([data1, data2])
    print('done')
    return (data)

def load_mom_csv_input():
    return pd.read_csv(config_file.mom_input_csv[0],delimiter=config_file.input_csv_delimiter)

def load_lat_lon_csv_input():
    return pd.read_csv(config_file.lat_lon_csv, delimiter=config_file.input_csv_delimiter)

def analyse_ages(data):
    birth = pd.to_datetime(data[config_file.input_csv_birth_colname])
    order = pd.to_datetime(data[config_file.input_csv_order_colname])
    diff = (order - birth).apply(lambda l: l.days)
    diffmx = (diff[diff>0]).as_matrix()
    #import matplotlib.pylab as plt
    #plt.hist(diffmax, bins=100)
    #plt.show
    return (diffmx, birth, order)

def parse_lat_lon_data(data):
    '''col_mrn_latlon = 'mrn'
    col_lat = 'Lat'
    col_lon = 'Long'
    col_censustract = 'WA2_2010CensusTract'
    col_censusblock = 'WA2_2010CensusBlock'
    col_census_city = 'City
    col_census_zip = 'zip'
    '''
    ziptocounty_np = np.loadtxt(config_file.zip_to_county, delimiter=',')
    county_cds = {}
    for ix, row in enumerate(ziptocounty_np):
        if str(int(row[1])) in county_cds:
            county_cds[str(int(row[1]))].append(int(row[0]))
        else:
            county_cds[str(int(row[1]))] = [int(row[0])]

    db = {}
    for ix, item in data.iterrows():
        try:
            mrn = int(item[config_file.col_mrn_latlon])
        except ValueError:
            continue
        db[mrn]={}
        try:
            lat = float(item[config_file.col_lat])
        except:
            print('lat is not float!', item[config_file.col_lat])
        try:
            lon = float(item[config_file.col_lon])
        except:
            print('lon is not float!', item[config_file.col_lon])
        try:
            centrac = int(item[config_file.col_censustract])
        except:
            print('cencus tract is not int!', item[config_file.col_censustract])
        try:
            censblock = int(item[config_file.col_censusblock])
        except:
            print('census block is not int!', item[config_file.col_censusblock])
        try:
            censcity = item[config_file.col_census_city]
        except:
            print('census city invalid', item[config_file.col_census_city])
        try:
            censzip = str(item[config_file.col_census_zip])
        except:
            print('census zip invalid', item[config_file.col_census_zip])
        
        db[mrn]['lat'] = lat
        db[mrn]['lon'] = lon
        db[mrn]['centrac'] = centrac
        db[mrn]['censblock'] = censblock
        db[mrn]['city'] = censcity
        db[mrn]['zip'] = censzip
        try:
            db[mrn]['county'] = county_cds[censzip]        
        except KeyError:
            print(censzip)
            db[mrn]['county'] = []

    pickle.dump(file=open('lat_lon_data_20170920.pkl', 'wb'), obj=db, protocol=2)

def parse_data(data):
    db = {}
    for ix, item in data.iterrows():
        try:
            mid = item[config_file.input_csv_mid_colname]
        except ValueError:
            pass
        try:
            mrn = item[config_file.input_csv_mrn_colname]
        except ValueError:
            pass    
        try:
            bdate = parser.parse(item[config_file.input_csv_birth_colname])
        except ValueError:
            pass
        try:
            odate = parser.parse(item[config_file.input_csv_order_colname])
        except ValueError:
            pass
        try:
            gender = item[config_file.input_csv_gender_colname].startswith('F')
        except:
            pass
        try:
            address = item[config_file.input_csv_addr_colname]
        except ValueError:
            pass
        try:    
            email = item[config_file.input_csv_email_colname]
        except ValueError:
            pass
        try:    
            zipcode = item[config_file.input_csv_zip_colname]
        except ValueError:
            pass
        try:    
            vitals = item[config_file.input_csv_vitals_colname]
        except ValueError:
            pass
        try:    
            vitals_dic = parse_vitals_dic(vitals)
        except ValueError:
            pass
        try:    
            diags = item[config_file.input_csv_diag_colname]
            diags_dic = parse_diag_dic(diags)
        except ValueError:
            pass
        try:    
            labs = item[config_file.input_csv_labs_colname]
        except ValueError:
            pass
        try:    
            lab_vals = item[config_file.input_csv_labres_colname]
            labs_dic = parse_labs_dic(labs, lab_vals)
        except ValueError:
            pass
        try:    
            meds = item[config_file.input_csv_med_colname]
            meds_dic = parse_medications(meds)
        except ValueError:
            pass
        try:    
            vaccines = item[config_file.input_csv_vac_colname]
        except ValueError:
            pass
        try:    
            ethnicity = item[config_file.input_csv_eth_colname]
        except ValueError:
            pass
        try:    
            race = item[config_file.input_csv_race_colname]
        except ValueError:
            pass

        if mid not in db:
            db[mid] = {}
            db[mid]['diags']={}
            db[mid]['vitals']={}
            db[mid]['labs']={}    
            db[mid]['meds']={}        

        db[mid]['bdate'] = bdate
        db[mid]['gender'] = gender
        db[mid]['ethnicity'] = ethnicity
        db[mid]['race'] = race
        db[mid]['mrn'] = mrn
        
        if 'odate' in db[mid]:
            db[mid]['odate'].append(odate)
        else:
            db[mid]['odate'] = [odate]

        if 'address' in db[mid]:
            db[mid]['address'].append([odate, address])
        else:
            db[mid]['address'] = [[odate, address]]
        
        if 'email' in db[mid]:    
            db[mid]['email'].append([odate, email])
        else:
            db[mid]['email'] = [[odate, email]]

        if 'zip' in db[mid]:
            db[mid]['zip'].append([odate, zipcode])
        else:
            db[mid]['zip'] = [[odate, zipcode]]

        for k in vitals_dic.keys():
            if k in db[mid]['vitals']:
                db[mid]['vitals'][k].append([odate, vitals_dic[k]])
            else:
                db[mid]['vitals'][k] = [[odate, vitals_dic[k]]]

        for k in diags_dic.keys():
            if k in db[mid]['diags']:
                db[mid]['diags'][k].append([odate, diags_dic[k]])
            else:
                db[mid]['diags'][k] = [[odate, diags_dic[k]]]

        for k in meds_dic.keys():
            if k in db[mid]['meds']:
                db[mid]['meds'][k].append([odate, meds_dic[k]])
            else:
                db[mid]['meds'][k] = [[odate, meds_dic[k]]]

        for k in labs_dic.keys():
            if k in db[mid]['labs']:
                db[mid]['labs'][k].append([odate, labs_dic[k]])
            else:
                db[mid]['labs'][k] = [[odate, labs_dic[k]]]
    return db
def parse_mother_data(data):
    db = {}
    for ix, item in data.iterrows():
        try:
            mid = item[config_file.input_csv_newborn_MRN]
        except ValueError:
            pass
        try:
            mom_mrn = item[config_file.input_csv_mothers_MRN]
        except ValueError:
            pass    
        try:
            agedeliv = int(item[config_file.input_csv_mothers_agedeliv])
        except ValueError:
            pass
        try:    
            diags = str(item[config_file.input_csv_mothers_diags])
            if pd.notnull(diags): diags_dic = diags.split(';')
        except ValueError:
            pass
        try:    
            nbdiags = str(    item[config_file.input_csv_mothers_NB_diags])
            if pd.notnull(nbdiags): nbdiags_dic = nbdiags.split(';')
        except ValueError:
            pass
        try:    
            deldiags = str(item[config_file.input_csv_mothers_deliv_diags])
            if pd.notnull(deldiags): deldiags_dic = deldiags.split(';')
        except ValueError:
            pass
        except AttributeError:
            print(deldiags)
        try:    
            ethnicity = item[config_file.input_csv_mothers_ethn]
        except ValueError:
            pass
        try:    
            race = item[config_file.input_csv_mothers_race]
        except ValueError:
            pass
        try:    
            nationality = item[config_file.input_csv_mothers_national]
        except ValueError:
            pass
        try:    
            marriage = item[config_file.input_csv_mothers_marriage]
        except ValueError:
            pass
        try:    
            birthplace = item[config_file.input_csv_mothers_birthplace]
        except ValueError:
            pass
        try:    
            lang = item[config_file.input_csv_mothers_lang]
        except ValueError:
            pass
        try:    
            insur1 = item[config_file.input_csv_mothers_insur1]
        except ValueError:
            pass
        try:    
            insur2 = item[config_file.input_csv_mothers_insur2]
        except ValueError:
            pass

        if mid not in db:
            db[mid] = {}
            db[mid]['diags']={}
            db[mid]['nbdiags']={}
            db[mid]['deldiags']={}

        db[mid]['mom_mrn'] = mom_mrn
        db[mid]['ethnicity'] = ethnicity
        db[mid]['race'] = race
        db[mid]['nationality'] = nationality
        db[mid]['marriage'] = marriage
        db[mid]['birthplace'] = birthplace
        db[mid]['lang'] = lang
        db[mid]['insur1'] = insur1
        db[mid]['insur2'] = insur2
        db[mid]['agedeliv'] = agedeliv

        for k in diags_dic:
            db[mid]['diags'][k]=True
        for k in nbdiags_dic:
            db[mid]['nbdiags'][k]=True
        for k in deldiags_dic:
            db[mid]['deldiags'][k]=True
    return db

def parse_vitals_dic(str1):
    vitals = {}
    ws = re.split('\|+ ', str1.strip())
    if len(ws) == 0:
        return vitals

    for w in ws:
        if w.strip() == '':
            continue

        ks = re.split('(-*\d*\.*\d+)', w.strip())
        if len(ks) < 2: 
            continue
        try:
            vitals[ks[0].strip(' ')] = float(ks[1])
        except ValueError:
            try:
                vitals[ks[0].strip(' ')] = [float(ks[1].split('/')[0]), float(ks[1].split('/')[1])]
            except:
                pass        
    return vitals

def parse_diag_dic(str1):
    diag = {}
    ws = re.split('\|+ ', str1.strip())
    if len(ws) == 0:
        return diag

    for w in ws:
        if w.strip() == '':
            continue
        ks = re.split(' ', w.strip())
        if len(ks[0]) > 6:
            #print(w)
            continue
        diag[ks[0]] = 1
    return diag

def parse_medications(str1):
    meds = {}
    ws = re.split('\|+ ', str1.strip())
    
    if len(ws) == 0:
        return meds

    for w in ws:
        if w.strip() == '':
            continue
        # ks = re.split(' ', w.strip())
        # print(w)
        meds[w.strip()] = 1

    return meds

def parse_labs_dic(str1, str2):
    d1 = {}
    ws1 = re.split('\|+ ', str1.strip()) #lab codes
    try:
        str2 = str2.lower().replace('positive', '1').replace('pos','1').replace('negative','-1').replace('neg','-1')
        ws2 = re.split('\|+ ', str2.strip()) #results
    except:
        # print ('str2', str2)
        return d1

    if len(ws1) == 0 or len(ws1) != len(ws2):
        return d1

    for i, w in enumerate(ws1):
        if w.strip() == '':
            continue
        ks = re.split(' ', w.strip())
        try:
            d1[w.strip()] = float(ws2[i])
        except ValueError:
            pass #print('value error in parsing', ws1[i], ws2[i])
    return d1

def percentile(ht, wt, genderref):
    #Height    L    M    S    P01    P1    P3    P5    P10    P15    P25    P50    P75    P85    P90    P95    P97    P99    P999
    plist = [0.01, 1, 3, 5 , 10, 15, 25, 50, 75, 85, 90, 95, 97, 99, 99.9]
    perc = -1
    ht = float("{0:4.1f}".format(ht*2.54))
    try:
        ix = 0
        while (wt*0.453592 >= genderref[ht][ix]):
            ix += 1
            if ix == len(genderref[ht]):
                break
        return plist[ix-1]
    except:
        return -1

# wght4leng_girl , wght4leng_boy
def append_weight_for_length(data):
    ref_girl, ref_boy = np.loadtxt(config_file.wght4leng_girl), np.loadtxt(config_file.wght4leng_boy)
    ref_girl_db, ref_boy_db = {}, {}
    for i in ref_girl:
        ref_girl_db[i[0]] = i[4:]
    for i in ref_boy:
        ref_boy_db[i[0]] = i[4:]

    f = open("tmpout.csv", 'w')

    for k in data.keys():
        if ('Wt' in data[k]['vitals']) & ('Ht' in data[k]['vitals']):
            listwt = data[k]['vitals']['Wt']
            listht = data[k]['vitals']['Ht']
            gender = data[k]['gender']
            bdate = data[k]['bdate']
            weight_for_length = []
            for (date1, ht1) in listht:
                for (date2, wt1) in listwt:
                    if date1 == date2:
                        if (date1 - bdate).days <= 365 * 3 :
                            f.write(str(ht1) + ' ' + str(wt1) + (' f' if gender else ' m') + '\n')
                            p = percentile(ht1, wt1, ref_girl_db if gender == True else ref_boy_db)
                            if p != -1:
                                weight_for_length.append([date1, p])
                            f.flush()
        data[k]['vitals']['Wt for Ht Percentile']=weight_for_length
        # import pdb
        # pdb.set_trace()
    f.close()
    return data

def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1       # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def run_builddata():
    (data) = load_csv_input()
    db = parse_data(data)
    return db

if __name__=='__main__':
    run_builddata()