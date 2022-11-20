# create and evaluate an autoregressive (AR1) model for perturb precipitation
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from statsmodels.tsa.ar_model import AutoReg
import pandas as pd

# adding noise to the initial conditions
PATH = 'D:/Mission_research/AR1/'
filename = 'precip_half_pert.nc'

# Read .NC file:
ncfile  = Dataset (PATH+filename, 'r')
precip  = ncfile.variables['precip'][:]
ncfile.close()

nensembles = 50

precip_mean=np.mean(precip, axis =1)

# First order regressive model (AR1), 24 hr
# total number of points  = 888
# X= Yt-1 i.e. 888 - 24 = 864

# Create training and testing data
train_data = precip [0:864,   :]
test_data  = precip [24:888, :]

# train autoregression
predict = np.zeros ((865, nensembles), dtype= np.float64)
for iensemble in range (nensembles):
	model = AutoReg(train_data[:, iensemble], lags=[1, 24])
	model_fit = model.fit()
	# make predictions
	predictions = model_fit.predict(start=24, end=888, dynamic=False)
	predict [:, iensemble] = predictions

#print('Coefficients: %s' % model_fit.params)
predict_mean = np.mean(predict, axis =1)


# plot precipitation data before Autoregression:
fig= plt.figure(figsize= (16,8))
nrows =2
ncols =1

date = pd.date_range('2019-05-09 00:00:00', periods=888, freq='60min', tz= 'UTC')
date_form = DateFormatter("%d-%m")

# Subplot 1
ax1 = plt.subplot(nrows, ncols, 1)
ax1.set_title("Perturb precip before Autoregression (AR1)")
for i in range(nensembles):
	ax1.plot(date, precip[:, i], '*', color= 'cornflowerblue', markersize= 2)
ax1.plot( date, precip_mean, color ='red',linewidth= 0.8, label='mean of ensembles')
ax1.set_ylabel("precip(mm/day)")
ax1.set_xticklabels('')
ax1.xaxis.set_major_formatter(date_form)
ax1.set_ylim ([0,0.0008])
ax1.legend(loc = 'upper right')

date = pd.date_range('2019-05-10 00:00:00', periods=865, freq='60min', tz= 'UTC')
# Subplot 2
ax2 = plt.subplot(nrows, ncols, 2)
ax2.set_title("Perturb precip after Autoregression (AR1), 24 hour")
for i in range(nensembles):
	ax2.plot(date, predict[:, i], '*', color= 'cornflowerblue', markersize= 2)
ax2.plot( date, predict_mean, color ='red',linewidth= 0.8, label='mean of ensembles')
ax2.set_ylabel("precip(mm/day)")
ax2.set_xticklabels('')
ax2.xaxis.set_major_formatter(date_form)
ax2.set_ylim ([0,0.0008])
ax2.legend(loc = 'upper right')
plt.savefig(PATH+ 'perturb_precip_AR_24hr.png', dpi=300, bbox_inches='tight')

plt.show()
