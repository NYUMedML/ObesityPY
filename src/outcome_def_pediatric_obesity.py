import config
import numpy as np
CDC_Ranges_95th_percentile = {}

def init():
	global CDC_Ranges_95th_percentile
	if CDC_Ranges_95th_percentile=={}:
		load_CDC_refs()

def load_CDC_refs(inputfile='None'):
	if inputfile =='None':
		inputfile = config.CDC_BMI_95
	global CDC_Ranges_95th_percentile
	rawdata = np.loadtxt(inputfile, delimiter=',')
	for i in range(0, rawdata.shape[0]):
		CDC_Ranges_95th_percentile[int(rawdata[i][0])] = (rawdata[i][2],rawdata[i][1]) #data is of format: lowrangemonth, f95th, m95th

def outcome(bmi, gender, agex):
	global CDC_Ranges_95th_percentile
	age_range_low = int((agex - (agex%0.5)) * 12)
	if bmi > CDC_Ranges_95th_percentile[age_range_low][gender]:
		return 1.0
	return 0.0

# def linear_interpolation(bmi, x_1, x_2, y_1, y_2):
# 	# print(bmi, x_1, x_2, y_1, y_2)
# 	return y_1 + ((y_2 - y_1) * (bmi - x_1) / (x_2 - x_1))

# def percentile(bmi, gender, agex):
# 	global CDC_Ranges_95th_percentile
# 	age_range_low = agex - (agex%0.5)

# 	for i, p_l in enumerate(p_levels):
# 		if i == len(p_levels) -1 and bmi >= CDC_Ranges_95th_percentile[gender][age_range_low][i]:
# 			return 0.97
# 		if bmi < CDC_Ranges_95th_percentile[gender][age_range_low][0]:
# 			return 0.05

# 		if (bmi >= CDC_Ranges_95th_percentile[gender][age_range_low][i]) & (bmi < CDC_Ranges_95th_percentile[gender][age_range_low][i+1]):
# 			return linear_interpolation(bmi, CDC_Ranges_95th_percentile[gender][age_range_low][i], CDC_Ranges_95th_percentile[gender][age_range_low][i+1], p_levels[i], p_levels[i+1])

# 	return 0.0

init()
# print(CDC_Ranges_95th_percentile)