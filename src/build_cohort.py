import numpy as np

def load_ccs_dic(ccsfile='aux/ccsIcd10.csv'):
     hash2, hash3, hash4 = {}, {}, {}
     f = csv.reader(open(ccsfile, 'r'), delimiter=',')
     for row in f:
         if len(row) == 1:
             continue
         icd, ccs, ccs_desc = (row[0].replace('\'', '').strip(), row[1].replace('\'', '').strip(), row[3].replace('\'', '').strip())
         hash2[icd] = ccs
         if ccs in hash3:
             hash3[ccs].append(icd)
         else:
             hash3[ccs]=[icd]
         hash4[ccs] = ccs_desc
     return hash2, hash3, hash4

def make_ccs_file():
	h2 = load_ccs_dic('/Volumes/R/hcc/data/icd10_ccs.csv')
	f = open('../auxdata/icd10listccs.txt', 'w')
	for hcc in h2[1]:                                    
          for icd in h2[1][hcc]:
              f.write(icd + ' ')
          hccstr = '10ccs'+hcc             
          try:
              hccstr += ':' + h1[2][hcc]  
          except:                     
              pass
          f.write('|' + hccstr + '\n') 
    f.flush(); f.close();