import numpy as np

def load_CDC_refs(inputfile):
	print('loading CDC reference ranges from:', inputfile)
	rawdata = np.loadtxt(inputfile, delimiter=',')
	# print(len(rawdata[0]))
	unit = 1.0
	if len(rawdata[0]) == 14:
		p_levels = [0.05,0.10,0.25,0.50,0.75,0.90,0.95,0.97,1]
	else:
		p_levels  = [0.05,0.10,0.25,0.50,0.75,0.85,0.90,0.95,0.97,1]
	CDC_Ranges_percentile = [{},{}]
	for i in range(0,rawdata.shape[0]):	
		CDC_Ranges_percentile[int(rawdata[i][0])][str(rawdata[i][1])] = [rawdata[i][5], rawdata[i][6], rawdata[i][7], rawdata[i][8], rawdata[i][9], rawdata[i][10], rawdata[i][11], rawdata[i][12], rawdata[i][13]]
		if len(rawdata[0]) == 15:
			CDC_Ranges_percentile[int(rawdata[i][0])][str(rawdata[i][1])].append(rawdata[i][14])
		if (float(rawdata[i][1]) % 1) != 0:
			unit = 0.5
	return CDC_Ranges_percentile, unit, p_levels

def linear_interpolation(val1, x_1, x_2, y_1, y_2):
	return y_1 + ((y_2 - y_1) * (val1 - x_1) / (x_2 - x_1))

def percentile(val1, gender, agex, CDC_Ranges_percentile, unit, p_levels):
	age_range_low = str(agex - (agex%unit))
	try:
		tmp = CDC_Ranges_percentile[gender][age_range_low]
	except KeyError:
		return 0.0

	for i, p_l in enumerate(p_levels):
		if i == (len(p_levels) -2) and (val1 >= CDC_Ranges_percentile[gender][age_range_low][i]):
			return 0.97
		if val1 < CDC_Ranges_percentile[gender][age_range_low][0]:
			return 0.05
		if (val1 >= CDC_Ranges_percentile[gender][age_range_low][i]) & (val1 < CDC_Ranges_percentile[gender][age_range_low][i+1]):
			return linear_interpolation(val1, CDC_Ranges_percentile[gender][age_range_low][i], CDC_Ranges_percentile[gender][age_range_low][i+1], p_levels[i], p_levels[i+1])
		
			
	return 0.0