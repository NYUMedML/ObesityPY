import pandas as pd
import numpy as np
import csv
import config as config_file
import pickle

def load_csv_files(filelist=[]):
    if len(filelist) == 0:
        filelist = config_file.census_file_list
    for f in filelist:
        data = pd.read_csv(f, delimiter=config_file.census_csv_delimiter)
        return data

def parse_csv_census(data):
    census_dict = {}
    for i in range(1, data.shape[0]):
        for j in range(data.shape[1]):
            if data.iloc[0,j] == 'Id2':
                try:
                    tract = int(str(data.iloc[i,j])[5:])
                    if tract % 100 == 0:
                        tract = int(tract/100)
                except:
                    break
                county = int(str(data.iloc[i,j])[2:5])
                state = str(data.iloc[i,j])[0:2]
                if tract not in census_dict:
                    census_dict[tract]={}
                if county not in census_dict[tract]:
                    census_dict[tract][county] = {}
            else:
                try:
                    census_dict[tract][county][data.iloc[0,j]] = float(data.iloc[i,j])
                except ValueError:
                    continue
    print('done')
    pickle.dump(file=open('census_data_20170920.pkl', 'wb'), obj=census_dict, protocol=2)
    return census_dict