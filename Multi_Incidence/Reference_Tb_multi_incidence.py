import pandas as pd
import numpy as np
import bible
import pylab as plt
from netCDF4 import Dataset
import Jules_extract_nc
#  -----------------------------------------------
print(" Wrapper for Reference Tb with roughness")
#-------------------------------------------------
# ---------------------------------------------------
# Import In-situ Observations:
#---------------------------------------------------
PATH =  "/media/rohit/Data/Ensemble_kalman_filter/Wrapper_tau_bible_reference/"

# --------------------------------------------------
# Run Jules ModeL:
# --------------------------------------------------
JULES_OUT = '/media/rohit/Data/Ensemble_kalman_filter/Wrapper_tau_bible_reference/'
file_SM = 'Jules.soil moisture.nc'
file_ST = 'Jules.soil temp.nc'
SM_J, ST_J, day, Hour_J, Minutes_J, nlayers_SM, nlayers_ST, ndays, nrows, ini_time =Jules_extract_nc.Jules_extract_nc(JULES_OUT, file_SM, file_ST)
print ("Jules extraction done !")

#-----------------------------------------------------------------
# Calling Tau-Omega RTM operator
#----------------------------------------------------------------
print ("Calling bible RTM Operator")

# Select Band:
# select the band:
freq = 1.4e9  # l-band
Band_ID =2    # l-band
#freq = 0.7e9  # p-band
#Band_ID =1    # p-band

#---------------------------------------------------------#
# for p-band : incidence = 46 and L-band: incidence = 38
#---------------------------------------------------------#

# Coustomized incidence angle on Jules rows
incidence_list = [15, 30, 45]

for incidence in incidence_list:
    print (incidence)
    Tb_Simulated = np.zeros((nrows,2) , dtype= np.float64)
       
    for i in range (nrows):
        mm = i
    
        if Band_ID ==1:
                  
            #for p-band (Average of top 2 layers)
            st = np.mean (ST_J[i, 0:3], axis=0)
            sm = np.mean (SM_J[i, 0:3], axis=0)
            T2 = ST_J[i, 1]
        
        else:
            # for L-band (consider top layer only)
            st = ST_J[i, 0]   # Ts [0-5 cm]
            sm = SM_J[i, 0] 
            T2 = ST_J[i, 1]  # [5-10 cm]
    
        # Input : Soil Moisture, Soil Temperature, roughness, incidence, number of layers
        Output = bible.bible(sm, st,T2, freq, incidence)
        Tb_Simulated[mm,0] = Output [0, 0]
        Tb_Simulated[mm,1] = Output [0, 1]

    # ----------------------------------------------------------------------------------
    print ("Successfully Tb Simulation Done ! ")
    # ----------------------------------------------------------------------------------
    # Unperturb Tb
    PATH_OUT  = PATH
    if Band_ID ==1:
        FILEOUT = PATH_OUT + 'Tb_Ref_P'+str(incidence)+'.nc'
    else:
        FILEOUT = PATH_OUT + 'Tb_Ref_L'+str(incidence)+'.nc'
    print(FILEOUT)
    ncfile = Dataset (FILEOUT, 'w')
    ncfile.createDimension('ndays',size= nrows)
    ncfile.createDimension('pol',size= 2)

    datanc    = ncfile.createVariable('Tb_Simulated_unperturb',np.float64,('ndays','pol'))
    datanc[:, :] = np.float64(Tb_Simulated)
    ncfile.close()

    #----------------------------------------------------------------------------
    print ('Reference Output saved for incidence angle = '+ str(incidence))
    #----------------------------------------------------------------------------
