import numpy as np
import pandas as pd
from cuts import *
from z_score import *
 

cdcref_d = pd.read_table('../sas/cdcref_d.csv',sep = ',',na_values = "")
mydata = pd.read_table('./mydata.csv',sep = ',',na_values = "",dtype = {'agemos':np.float64})
mydata['id'] = mydata.index + 1
mydata.ix[(mydata['agemos'].notnull()) & (mydata['agemos'] < 24), 'length'] = mydata['height']
mydata.ix[(mydata['agemos'].notnull()) & (mydata['agemos'] >= 24), 'stand_ht'] = mydata['height']
_mydata = mydata[mydata.agemos < 240]
_old = mydata[mydata.agemos >= 240]
_orig = mydata

_cinage = _mydata
_cinage.ix[_cinage.agemos >= 0 & _cinage.agemos < 0.5,'_agecat'] = 0
_cinage.ix[_cinage.agemos < 0 | _cinage.agemos >= 0.5,'_agecat'] = np.floor(_cinage.agemos + 0.5) - 0.5
_cinage.ix[_cinage.bmi < 0 & _cinage.weight > 0 & _cinage.height > 0 & _cinage.agemos >= 24,'weight'] = _cinage.bmi/ (_cinage.height / 100) ** 2

_cinlen = _cinage[_cinage.length.notnull()]
_cinlen.ix[_cinlen.length >= 45,'_htcat'] = np.floor(_cinlen.length + 0.5) - 0.5
_cinlen.ix[(_cinlen.length >= 45) & (_cinlen.length < 45.5),'_htcat'] = 45

_cinht = _cinage[_cinage.stand_ht.notnull()]
_cinht.ix[_cinht.stand_ht >= 77.5,'_htcat'] = np.floor(_cinht.stand_ht + 0.5) - 0.5
_cinht.ix[(_cinht.stand_ht >= 77) & (_cinht.stand_ht < 77.5),'_htcat'] = 77

###### begin the for-age calscs
crefage = cdcref_d[cdcref_d.denom == "age"].sort(['sex', '_agecat'], ascending=[True,True])
_cinage = _cinage.sort(['sex', '_agecat'], ascending=[True,True])
finfage = pd.merge(_cinage, crefage, on = ['sex','_agecat'], how='left')
finfage['ageint'] = finfage._agemos2 - finfage._agemos1
finfage['dage'] = finfage.agemos - finfage._agemos1

### array "do over l0"
def finfage_array(df,out_varname,var_name1,var_name2):
    df[out_varname] = df[var_name1] + (df['dage'] * (df[var_name2] - df[var_name1])) / df['ageint']
    return df

array_lst = ['_llg','_mlg','_slg','_lht','_mht','_sht','_lwt','_mwt' ,'_swt','_lhc','_mhc','_shc','_lbmi','_mbmi','_sbmi']

for i in range(len(array_lst)):
     tmp_var = array_lst[i]
     tmp_var_name1 = tmp_var + '1'
     tmp_var_name2 = tmp_var + '2'
     df = finfage_array(finfage,tmp_var,tmp_var_name1,tmp_var_name2)

finfage = df

finfage.ix[finfage.agemos < 24, '_mbmi'] = None
## apply zscore and cuts funciton
finfage = zscore(finfage,'length','_llg','_mlg','_slg','lgz','lgpct','_flenz')
finfage = cuts(finfage,'_flenz','_bivlg',-5,4) 

finfage = zscore(finfage,'stand_ht','_lht','_mht','_sht','stz','stpct','_fstatz')
finfage = cuts(finfage,'_fstatz', '_bivst', -5, 4)

finfage = zscore(finfage,'weight', '_lwt', '_mwt', '_swt', 'waz', 'wapct', '_fwaz')
finfage = cuts(finfage,'_fwaz', '_bivwt', -5, 8)    

finfage = zscore(finfage,'headcir', '_lhc', '_mhc', '_shc', 'headcz', 'headcpct', '_fheadcz')
finfage = cuts(finfage,'_fheadcz', '_bivhc', -5, 5)

finfage = zscore(finfage,'bmi', '_lbmi', '_mbmi', '_sbmi', 'bmiz', 'bmipct', '_fbmiz')
finfage = cuts(finfage,'_fbmiz', '_bivbmi', -4, 8)

finfage['bmi95'] = finfage._mbmi * ((1 + finfage._lbmi * finfage._sbmi * norm.ppf(0.95)) ** (1 / finfage._lbmi))
finfage['bmipct95'] = 100 * (finfage.bmi / finfage.bmi95)  # % of 95th percentile: m * ((1+l*s*z)^(1/l))
finfage['bmidif95'] = finfage.bmi - finfage.bmi95
finfage['bmi50'] = finfage._mbmi * ((1 + finfage._lbmi * finfage._sbmi * norm.ppf(0.5)) ** (1 / finfage._lbmi))
drop_col = ['_llg1','_mlg1','_slg1','_lht1','_mht1','_sht1','_lwt1','_mwt1','_swt1','_lhc1','_mhc1','_shc1','_lbmi1','_mbmi1','_sbmi1','_llg2','_mlg2','_slg2','_lht2','_mht2','_sht2','_lwt','_mwt2','_swt2','_lhc2','_mhc2','_shc2','_lbmi2','_mbmi2','_sbmi2','_lwht1','_mwht1','_swht1','_lwht2','_mwht2', '_swht2', '_lwlg1', '_mwlg1', '_swlg1','_lwlg2', '_mwlg2', '_swlg2']
finfage = finfage.drop(drop_col,axis = 1)

### begin for-length and for -stand_ht calcs
### begin for-length calcs, birth to 36 mos
_cinlen = _cinlen.sort(['sex','_htcat'],ascending=[True,True])
creflg = cdcref_d.iloc[:,[0,1] + list(range(35,43))] #select enom sex _lg1--_swlg2
creflg['_htcat'] = creflg['_lg1']
creflg = creflg[creflg.denom == 'length'].sort(['sex','_htcat'],ascending=[True,True])
finflg = pd.merge(_cinlen, creflg, on = ['sex','_htcat'], how='left')
finflg = finflg[(finflg.length < 104) & (finflg.length > 43)]
finflg['lenint'] = finflg._lg2 - finflg._lg1
finflg['dlen'] = finflg.length - finflg._lg1
def finflg_array(df,out_varname,var_name1,var_name2):
    df[out_varname] = df[var_name1] + (df['dlen'] * (df[var_name2] - df[var_name1])) / df['lenint']

    return df

array_lst_len = ['_lwl','_mwl','_swl']

for i in range(len(array_lst_len)):
     tmp_var = array_lst_len[i]
     tmp_var_name1 = tmp_var + 'g1'
     tmp_var_name2 = tmp_var + 'g2'
     df = finflg_array(finflg,tmp_var,tmp_var_name1,tmp_var_name2)
    
finflg = df
finflg = zscore(finflg,'weight', '_lwl', '_mwl', '_swl', 'wlz', 'wlpct', '_fwlz') 
finflg = cuts(finflg,'_fwlz','_bivwlg', -4, 8); 
finflg = finflg[['id','sex','_agecat','agemos','weight','_fwlz','_bivwlg','wlz','wlpct']]


### begin for- stand_ht calcs
_cinht = _cinht.sort(['sex','_htcat'],ascending=[True,True])
crefht = cdcref_d[['denom','sex', '_ht1', '_ht2', '_lwht1', '_lwht2', '_mwht1', '_mwht2', '_swht1','_swht2']]
crefht['_htcat'] = crefht['_ht1']
crefht = crefht[crefht.denom == 'height'].sort(['sex','_htcat'],ascending=[True,True])
finfht = pd.merge(_cinht, crefht, on = ['sex','_htcat'], how='left')
finfht = finfht[(finfht.height < 122) & (finfht.length > 77)]
finfht['htint'] = finfht._ht2 - finfht._ht1
finfht['dht'] = finfht.height - finfht._ht1

def finfht_array(df,out_varname,var_name1,var_name2):
    df[out_varname] = df[var_name1] + (df['dht'] * (df[var_name2] - df[var_name1])) / df['htint']
    return df

array_lst_ht = ['_lwh','_mwh','_swh']
for i in range(len(array_lst_ht)):
     tmp_var = array_lst_ht[i]
     tmp_var_name1 = tmp_var + 't1'
     tmp_var_name2 = tmp_var + 't2'
     df_final = finfht_array(finfht,tmp_var,tmp_var_name1,tmp_var_name2)


finfht = df_final
del df_final

finfht = zscore(finfht,'weight', '_lwh', '_mwh', '_swh', 'wstz', 'wstpct', '_fwstz')
finfht = cuts(finfht,'_fwstz','_bivwst', -4, 8); 
finfht = finfht[['id','sex','_agecat','agemos','weight','_fwstz','_bivwst','wstz','wstpct']]


## combine the for-age for-length,for-height
_outdata = pd.merge(finflg, finfht, finfage,on = 'id')
# define height vars as max of standing height and length vars

def outdata_array(df,out_varname,var_name1,var_name2):
    df.loc[[df.agemos >= 24,out_varname]] = df[[var_name1]]
    df.loc[[df.agemos < 24, out_varname]] = df[[var_name2]]
    
    
outdata_array(_outdata,'haz','stz','lgz')
outdata_array(_outdata,'hapct','stpct','lgpct')
outdata_array(_outdata,'_bivht','_bivst','_bivlg')
outdata_array(_outdata,'_Fhaz','_fstat','_flenz')
outdata_array(_outdata,'whz','wstz','wlz')
outdata_array(_outdata,'whpct','wstpct','wlpct')
outdata_array(_outdata,'_bivwh','_bivwst','bivwlg')
outdata_array(_outdata,'_Fwhz','_fwstz','_fwlz')

_outdata.ix[(_outdata.weight.notnull()) & (_outdata.weight < 0.01), ['waz','wapct','bmiz','bmipct','whz','whpct']] = None
_outdata.ix[(_outdata.height.notnull()) & (_outdata.height < 0.01), ['waz','wapct','bmiz','bmipct','whz','whpct']] = None
_outdata.ix[(_outdata.headcir.notnull()) & (_outdata.headcir < 0.01), ['headcz','headcpct']] = None
_outdata['min'] = _outdata[['_bivht','_bivwt','_bivbmi','_bivhc','_bivwh']].min(axis = 1)
_outdata['max'] = _outdata[['_bivht','_bivwt','_bivbmi','_bivhc','_bivwh']].max(axis = 1)
_outdata.ix[(_outdata.max == 0) | (_outdata.max == -1),'_bivhigh'] = 0
_outdata.ix[(_outdata.max == 1),'_bivhigh'] = 1
_outdata.ix[(_outdata.min == 0) | (_outdata.min == -1),'_bivhigh'] = 0
_outdata.ix[(_outdata.min == 1),'_bivhigh'] = 1

_outdata[['_fbmiz','_fhaz','_fheadcz','_flenz','_fstatz','_fwaz','_fwhz','_fwstz','_fwlz','_bivbmi','_bivhc','_bivhigh','_bivht','_bivlg','_bivlow','_bivst','_bivwh','_bivwlg','_bivwst','_bivwt','agemos','bmi','bmipct',
'bmipct95','bmiz','bmi50','bmi95','bmidif95','height','haz','stz','lgz','headcir','headcpct','headcz','id','lgpct','sex','stpct','wapct','waz','whpct','whz','hapct','wstz','wstpct','_fstatz']]

# _outdata.rename(columns={'waz':'weight-for-age Z','_bivbmi':'BIV BMI-for-age' ,'_bivhc' :'BIV head_circ' ,'_bivht':'BIV height-for-age' ,'_bivwh':'BIV weight-for-height' ,'_bivwt':'BIV weight-for-age','_bivlow':'any low BIV' ,'_bivhigh:''any high BIV','_fbmiz:''modified BMI-for-age Z','_fhaz:''modified height-for-age Z','_Fheadcz':'modified head_circ Z','_fwaz':'modified weight-for-age Z','_fwhz':'modified weight-for-height Z','bmi50' : 'CDC median BMI-for-age','bmi95' : 'CDC 95th pctl BMI-for-age','bmipct':'BMI-for-age percentile','bmipct95':'percent of 95th BMI percentile','bmidif95': 'difference from the 95th BMI percentile','bmiz':'BMI-for-age Z','hapct':'height-for-age percentile','haz':'height-for-age Z','headcz':'head_circ-for-age Z','headcpct':'head_circ-for age perc','wapct':'weight-for-age percentile','whpct':'weight-for-height percentile','whz':'weight-for-height Z'})
_cdcdata = pd.merge(_outdata,_orig, on = "id", how = "inner")
_cdcdata = _cdcdata.drop[['_id','stz','stpct','_bivst', '_fstatz','wstz','wstpct','_fwstz','lgz','lgpct','_bivlg','_flenz','_fwlz','length','stand_ht','_bivwst','_bivwlg' ]]


