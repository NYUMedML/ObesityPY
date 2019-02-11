input_csv = ["/Volumes/R/obesity/data/All_patients_of_age_18_or_less_in_eCW_for_at_least_2_years_II.csv","/Volumes/R/obesity/data/All_patients_of_age_18_or_less_in_eCW_for_at_least_2_years_II_sheet2.csv"]
mom_input_csv = ['/Volumes/R/obesity/data/luth_mat_nbrn_revis.csv']
lat_lon_csv  = '/Volumes/R/obesity/dataorig/DB_Geocoded_Latlong_results.csv'
zip_to_county = '/Volumes/R/obesity/dataorig/zip_to_county.csv'
input_csv_delimiter = ","
census_file_list = ['/Volumes/R/obesity/dataorig/census_clean.csv']
census_csv_delimiter = ','

input_csv_mid_colname = 'patientid'
input_csv_mrn_colname = 'mrn'
input_csv_zip_colname = 'zip'
input_csv_email_colname = 'email'
input_csv_birth_colname = 'date_of_birth'
input_csv_order_colname = 'order_date'
input_csv_vitals_colname = 'vitals'
input_csv_vitals_colname_bmi = 'BMI'
input_csv_gender_colname = 'gender'
input_csv_addr_colname = 'address'
input_csv_diag_colname = 'diagnosis'
input_csv_labs_colname = 'labs'
input_csv_labres_colname = 'lab_results'
input_csv_med_colname = 'meds'
input_csv_vac_colname = 'vaccines'
input_csv_eth_colname = 'ethnicity'
input_csv_race_colname = 'race'
col_mrn_latlon = 'mrn'
col_lat = 'Lat'
col_lon = 'Long'
col_censustract = 'WA2_2010CensusTract'
col_censusblock = 'WA2_2010CensusBlock'
col_census_city = 'City'
col_census_zip = 'zip'
#RecordId,mrn,City,zip,Long,Lat,WA2_2010CensusTract,WA2_2010CensusBlock
vital_keys = {'Temp', 'Ht', 'Wt', 'BMI', 'BP', 'HR', 'Oxygen', 'Pulse', 'Hearing', 'Vision', 'RR', 'PEF', 'Pre-gravid', 'Repeat', 'Pain', 'HC', 'Fundal', 'Education', 'Insulin', 'HIV', 'BMI Percentile', 'Ht Percentile', 'Wt Percentile', 'Wt Change', 'Oxygen sat', 'Pulse sitting', 'Vision (R) CC', 'Vision (L) CC', ''}

shelve_database_file = '/shelveDB.shlv'
icd9List = '../auxdata/icd9listccs.txt'
icd10List = '../auxdata/icd10listccs.txt'
genderList = '../auxdata/genderlist.txt'
ethnicityList = '../auxdata/ethnicityList.txt'
raceList = '../auxdata/raceList.txt'
zipList = '../auxdata/zipList.txt'
vitalsList = '../auxdata/vitals_subset.txt'
vitalsGainsList = '../auxdata/vitals_subset_gains.txt'
vitalsZScoreList = '../auxdata/vitals_wt_bmi_zscore.txt'
labslist = '../auxdata/labs.txt'
medslist = '../auxdata/meds.txt'

CDC_BMI_Ref = '../auxdata/CDC_BMIs.txt'
CDC_BMI_95 = '../auxdata/CDC_BMI_95.txt'
wght4leng_girl = '../auxdata/WHO_wfl_girls_p_exp.txt'
wght4leng_boy = '../auxdata/WHO_wfl_boys_p_exp.txt'
bmi_girl = '../auxdata/CDC_bmi_age_girls.csv'
bmi_boy = '../auxdata/CDC_bmi_age_boys.csv'

timeList = '../auxdata/time_list.txt'  #ROB

#maternal info
input_csv_mothers = ['']
input_csv_mothers_delim = ','

input_csv_mothers_MRN = 'BM_MRN'
input_csv_newborn_MRN = 'NB_MRN'
input_csv_mothers_agedeliv = 'BM_Age_at_Delivery' #int
input_csv_mothers_marriage = 'BM_Marital_Status'
input_csv_mothers_race = 'BM_Race'
input_csv_mothers_ethn = 'BM_Ethnicity'
input_csv_mothers_national = 'BM_Nationality'
input_csv_mothers_birthplace = 'BM_BirthPlace'
input_csv_mothers_lang = 'BM_Language'
input_csv_mothers_insur1 = 'BM_Prim_Ins'
input_csv_mothers_insur2 = 'BM_Second_Ins'
input_csv_mothers_diags = 'MAT_ACCT_DAIG_LIST'# (icd9/10 list separated by ';')
input_csv_mothers_NB_diags = 'NEW ACCT DIAG List'# (icd9/10 list separated by ';')
input_csv_mothers_deliv_diags = 'Delv DIAG'# (icd9/10 list separated by ';')


BM_Marital_StatusList = '../auxdata/BM_Marital_Status.txt'
BM_RaceList = '../auxdata/BM_Race.txt'
BM_EthnicityList = '../auxdata/BM_Ethnicity.txt'
BM_NationalityList = '../auxdata/BM_Nationality.txt'
BM_BirthPlace = '../auxdata/BM_BirthPlace.txt'
BM_Language = '../auxdata/BM_Language.txt'
BM_Prim_Ins = '../auxdata/BM_Prim_Ins.txt'
BM_Second_Ins = '../auxdata/BM_Second_Ins.txt'

# historical maternal info
BM_Labs = '../auxdata/BM_Labs.txt'  #ROB
# BM_Meds = '../auxdata/BM_Meds.txt'  #ROB
BM_Procedures = '../auxdata/BM_Procedures.txt'  #ROB
