import numpy as np
import Jules_extract_nc
import Jules_extract_RTM
from netCDF4 import Dataset
from numpy import dot
from scipy.linalg import inv
import os
import shutil
import bible

#  -----------------------------------------------
print(" Welcome to Jules-Nujoko-Ensemble Kalman filter Assimilation Framework")
#-------------------------------------------------
# Initialization
ndays   = 36
ncycles = 35
nbands  = 2
index= 0
ini_date_start = 9  # # Starting  Assimilation date = 13th May 2019
ini_date_end   = 10
ini_month_start = 5
ini_month_end = 5
npol = 4
count_ref = 95
nrows= 3402
count1 =95
obs_error = 1.5
incidence_list = [15, 30, 45]
nensembles = 50

# --------------------------------------------------
# Select Jules Ensemble folder
# --------------------------------------------------

# copy all files from ens_50 folder to Jules_run/file folder
source_folder      = '/media/rohit/Data/Ensemble_kalman_filter/Jules_run/file_ens_'+str(nensembles)+'/'
destination_folder = '/media/rohit/Data/Ensemble_kalman_filter/Jules_run/file/'

# fetch all files
for file_name in os.listdir(source_folder):
    # construct full file path
    source = source_folder + file_name
    destination = destination_folder + file_name
    # copy only files
    if os.path.isfile(source):
        shutil.copy(source, destination)
        print('copied', file_name)


# copy grid.nml file to nml folder:
destination_folder = '/media/rohit/Data/Ensemble_kalman_filter/Jules_run/nmls/'
file_nml = 'model_grid.nml'
source = source_folder + file_nml
destination = destination_folder + file_nml
shutil.copy(source, destination)
print ('nml copied to destimation folder')


for icycle in range (ncycles):

    
    JULES_OUT = '/media/rohit/Data/Ensemble_kalman_filter/Jules_run/testout/'              # Path to Jules 
    cmd = '$JULES_ROOT/build/bin/jules.exe $NAMELIST'
    os.system(cmd)
    print ('Jules output successfully generated')
    
    file_SM = 'Jules.soil moisture.nc'
    file_ST = 'Jules.soil temp.nc'
    
    # Jules extract for RTM Input
    SM_J, ST_J, day_J, Hour_J, Minutes_J , nlayers_SM, nlayers_ST, nrows, nensembles = Jules_extract_RTM.Jules_extract_RTM(JULES_OUT, file_SM, file_ST)

    print("SM_F and ST_F contains JULES output with perturbed forcings (nensembles)")
    print("JULES run for EnKF")

    ntimesteps = np.size (SM_J, axis=0)
    # Dimensions:
    # SM_J : 96 x 7x 50
    print (" Successfully extracted information from Jules Output")

    
    PATH_OBS = '/media/rohit/Data/Ensemble_kalman_filter/Wrapper_bible/'

    nincidence = len(incidence_list)

    Tb_Simulated_P  = np.zeros((2, nensembles, nincidence),  dtype=np.float64)
    Tb_Simulated_L  = np.zeros((2, nensembles, nincidence), dtype=np.float64) 
    count_inci = 0 
    for incidence in incidence_list:

        print (incidence)

        # Extract Reference output for multiple incidence angles:
        # Import Reference Tb (JULES run with unperturbed forcings and simulating Tb)

        PATH_ref = '/media/rohit/Data/Ensemble_kalman_filter/Wrapper_tau_bible_reference/'
        filename_P = 'Tb_Ref_P'+str(incidence)+'.nc'
        filename_L = 'Tb_Ref_L'+str(incidence)+'.nc'

        # Read Reference file:
        ncfile_P = Dataset(PATH_ref+filename_P, 'r')
        Tb_Ref_P = ncfile_P.variables['Tb_Simulated_unperturb'][:]
        ncfile_P.close()

        ncfile_L = Dataset(PATH_ref+filename_L, 'r')
        Tb_Ref_L = ncfile_L.variables['Tb_Simulated_unperturb'][:]
        ncfile_L.close()
        print("Reference Tb exported")

        #----------------------------------------------------
        # Tb simulation loop for both bands :
        #----------------------------------------------------
        
        for iband in range(nbands):
            if iband == 0:
                # Import P-band observations :
                filename = 'P_2_46_2_126_R.xlsx'
            
                # SM, ST  at last time step (7, nensembles)---> transpose to (nensembles, 7)

                SM_P = SM_J [count_ref, :, :]
                SM_P = np.transpose (SM_P)

                ST_P = ST_J [count_ref, :, :]
                ST_P = np.transpose (ST_P)

                # Obs_simulated with same structure, Tb simulated from JULES with refrence simulation

                # -----------------------------------------------------------------
                # Calling bible RTM operator for P-band
                # ----------------------------------------------------------------
                print("Calling bible RTM Operator for P-band")
                # Select Band:
                Band_ID = 1  # 1: P-Band
                freq= 0.7e9
                for iensemble in range(nensembles):
                    mm =iensemble
                                
                    sm = np.mean (SM_P[mm, 0:3], axis=0)
                    st = np.mean (ST_P[mm, 0:3], axis=0)
                    T2 = ST_P[mm, 1]
                
                    # Input : Soil Moisture, Soil Temperature, roughness, incidence, number of layers
                    Output = bible.bible(sm, st, T2, freq, incidence)
                    Tb_Simulated_P[0, iensemble, count_inci ] = Output[0, 0]
                    Tb_Simulated_P[1, iensemble, count_inci] = Output[0, 1]
                print("P-band Tb Simulation Done")
            
            
            else:
                # Import L-band observations :
                filename = 'L_3_38_2_125_R.xlsx'
                        
            
                # SM, ST  at last time step (7, nensembles)---> transpose to (nensembles, 7)

                SM_L = SM_J [count_ref, :, :]
                SM_L = np.transpose (SM_L)

                ST_L = ST_J [count_ref, :, :]
                ST_L = np.transpose (ST_L)
            
                # -----------------------------------------------------------------
                # Calling Njoku RTM operator for L-band
                # ----------------------------------------------------------------
                print("Calling Njoku RTM Operator for L-band")

                # Select Band:
                Band_ID = 2  # 2: L-Band
                freq = 1.4e9
                for iensemble in range(nensembles):
                    mm =iensemble
                
                    st = ST_L[mm, 0]
                    sm = SM_L[mm, 0]
                    T2 = ST_L[mm, 1]  
                
                    # Input : Soil Moisture, Soil Temperature, roughness, incidence, number of layers
                    Output = bible.bible(sm, st, T2, freq, incidence)
                    Tb_Simulated_L[0, iensemble, count_inci] = Output[0, 0]
                    Tb_Simulated_L[1, iensemble, count_inci] = Output[0, 1]
                
                print("L-band Tb Simulation Done")
        count_inci = count_inci +1

    error

    # ----------------------------------------------------------------------------------
    print ("Successfully Tb Simulation Done ! ") 
    # ----------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------
    print('Generating State Matrix')
    #-----------------------------------------------------------------------------------------
    # Generate State Martrix:
    #------------X Matrix---------------------------
    
    X= SM_J[count_ref, :, :]                         # (nlayers,   nensembles)
    
    X1 = SM_J                                 # (ntimesteps, nlayers,   nensembles)    

    X_mod_space  = np.zeros ((nlayers_SM, nensembles), dtype= np.float64)
    X_mod_square = np.zeros ((nlayers_SM, nensembles), dtype= np.float64)
    X_mod_mean   = np.zeros (nlayers_SM,  dtype= np.float64)

    for ilayer in range (nlayers_SM):
        X_mean = np.mean(X[ilayer, :])
        X_mod_mean[ilayer] = X_mean
        for iensemble in range(nensembles):
            X_mod_space[ilayer, iensemble] = X[ilayer, iensemble] - X_mean
            X_mod_square[ilayer, iensemble] = (X[ilayer, iensemble] - X_mean) ** 2

    X_mod_space=X_mod_space/np.sqrt(nensembles-1)
    print ('X and X_Mod_space matrix Generated')

    # Forecast Variance and Standard deviation (X matrix)
    X_mod_var    = np.zeros (nlayers_SM, dtype= np.float64)
    X_mod_std    = np.zeros (nlayers_SM, dtype= np.float64)
    
    for ilayer in range (nlayers_SM):
        X_mod_var[ilayer] = np.sum(X_mod_square[ilayer, :])/(nensembles-1)
        X_mod_std[ilayer] = np.sqrt(X_mod_var[ilayer])

    print ('Forecast Mean, std and variance for X matrix computed')
    
    #-------------------Y Matrix---------------------------#

    Y =  np.zeros ((npol, nensembles), dtype =np.float64)     # (npol(P,L) , nesembles)
    Y [0, :] = Tb_Simulated_P[0, :]      # PH
    Y [1, :] = Tb_Simulated_P[1, :]      # PV
    Y [2, :] = Tb_Simulated_L[0, :]      # LH 
    Y [3, :] = Tb_Simulated_L[1, :]      # LV 
    
    Y_obs_space   = np.zeros ((npol, nensembles), dtype= np.float64)  #declaring 
    Y_obs_square  = np.zeros ((npol, nensembles), dtype= np.float64)
    Y_obs_mean    = np.zeros (npol , dtype= np.float64)

    for ipol in range (npol):
        Y_mean = np.mean(Y[ipol, :])
        Y_obs_mean[ipol] = Y_mean
        for iensemble in range(nensembles):
            Y_obs_space[ipol, iensemble] = Y[ipol, iensemble] - Y_mean
            Y_obs_square[ipol, iensemble] = (Y[ipol, iensemble] - Y_mean) ** 2  #not been used in innovations

    Y_obs_space= Y_obs_space/np.sqrt(nensembles-1)
    print ('Y and Y_obs_space Matrix Generated')

    # Forecast Variance and Standard deviation (X matrix)
    Y_obs_var = np.zeros(npol, dtype=np.float64)
    Y_obs_std = np.zeros(npol, dtype=np.float64)

    for ipol in range(npol):
        Y_obs_var[ipol] = np.sum(Y_obs_square[ipol, :]) / (nensembles - 1)
        Y_obs_std[ipol] = np.sqrt(Y_obs_var[ipol])

    print ('Observed Mean, std and variance for Y matrix computed')

    #------Kalman matrix generation----------------------
    
    R=  (obs_error **2)* np.identity(npol) #calibration accuracy (Shen et al., 2021)
    print ("Observation error covariance matrix Generated: R")
    

    F1 = dot(X_mod_space, Y_obs_space.T) #elements of this matrix used for Jacobians
    F2 = inv(dot(Y_obs_space, Y_obs_space.T)+ R)
    K = dot (F1, F2)
    print ("Kalman Gain Generated")

    # ------------------Compute Jacobians---------------------#

    Jacobians = np.zeros ((nlayers_SM, npol), dtype = np.float64)
    for ilayer in range (nlayers_SM):
        Jacobians [ilayer, :] = F1 [ilayer, :] /  X_mod_var [ilayer]
    
    #-------- Reference Tb -----------
    Tb_ref_P_cycle =  Tb_Ref_P[count1, :] #last timesetep of Tb for assim (assimilation window is 24hrs)
    Tb_ref_L_cycle  = Tb_Ref_L[count1, :]
    count1 = count1+96
    
    #print ('count_ref =', count_ref)
    
    Tb_ref = np.zeros( npol,  dtype = np.float64)
    Tb_ref [0] = Tb_ref_P_cycle[0]  # PH
    Tb_ref [1] = Tb_ref_P_cycle[1]  # PV
    Tb_ref [2] = Tb_ref_L_cycle[0]  # LH
    Tb_ref [3] = Tb_ref_L_cycle[1]  # LV

    
    # Gaussian Noise (4x1) matrix:
    #noise = np.random.normal(0, 1.5, npol)

    # extract diagnoal element R matrix:
    # Ey
    Ey  = np.zeros ((npol, nensembles), dtype = np.float64)
    for iensemble in range (nensembles):
            Ey [ :, iensemble ] = np.random.normal(0, obs_error, npol)

    

    # Multiplication term (Innovations calculation)
    Innovations = np.zeros ((npol, nensembles), dtype = np.float64)
    for iadd in range(npol):
        for jadd in range(nensembles):
            Innovations[iadd, jadd] = Tb_ref[iadd] - Y[iadd, jadd] + Ey[iadd, jadd]
    
    
        
    if index< 0:                        		# (9/05 to 12/05)
        X_updated = X                       # X= SM_F[95,:,:]
        print("i am here before assimilation")
        ST_A = ST_J[count_ref, :, :] 
        
    else:      
        X_updated = X  + dot(K, Innovations)                   # (7x50) with K
        print("i am here after assimilation")
                                                # (7x50)  -> goes to update intiail conditions
    
        ST_A = ST_J[count_ref, :, :] 

    #Analysis inflation, alpha=1
    
    X_mean_updated = np.zeros (nlayers_SM, dtype = np.float64)  
    X_infl_updated = np.zeros ((nlayers_SM, nensembles), dtype = np.float64)
    alpha= 1.06

    for ilayer in range (nlayers_SM): 
        X_mean_updated[ilayer]   = np.mean(X_updated[ilayer, :])
        X_infl_updated[ilayer,:] =  X_mean_updated[ilayer] + ( alpha * ( X_updated[ilayer, :]- X_mean_updated[ilayer] )) 
    				     
    X_updated = X_infl_updated      #alpha =1 same updated , alpha larger than 1, increases the spread around mean  
        
    # Final output saved: [initializing in first cycle, after that updated in else]
    if icycle ==0:
    
        X_mod_mean_final = np.zeros ((ncycles, nlayers_SM), dtype= np.float64)
        X_mod_var_final  = np.zeros ((ncycles, nlayers_SM), dtype= np.float64)
        X_mod_std_final  = np.zeros ((ncycles, nlayers_SM), dtype= np.float64)
        
        Y_obs_mean_final = np.zeros ((ncycles, npol), dtype= np.float64)
        Y_obs_std_final  = np.zeros ((ncycles, npol), dtype= np.float64)
        Y_obs_var_final  = np.zeros ((ncycles, npol), dtype= np.float64)
    
        Tb_simulated_final   = np.zeros ((ncycles, npol, nensembles), dtype= np.float64)
        X_updated_final      = np.zeros ((ncycles, ntimesteps, nlayers_SM, nensembles), dtype= np.float64)
        Tb_simulated_final_P = np.zeros ((ncycles, 2, nensembles), dtype= np.float64)
        Tb_simulated_final_L = np.zeros ((ncycles, 2, nensembles), dtype= np.float64)
        K_final              = np.zeros ((ncycles, nlayers_SM, npol), dtype = np.float64)

        K_final [icycle, :, :] = K  
        
        X_mod_mean_final [icycle, :] = X_mod_mean
        X_mod_var_final  [icycle, :] = X_mod_var
        X_mod_std_final  [icycle, :] = X_mod_std

        Y_obs_mean_final [icycle, :] = Y_obs_mean
        Y_obs_std_final  [icycle, :] = Y_obs_std
        Y_obs_var_final  [icycle, :] = Y_obs_var

        
        X_updated_final [icycle, :, :, :]  = X1 
        X_updated_final [icycle, count_ref, :, :] = X_updated

        # Saving Simulated Tbs
        Tb_simulated_final_P [icycle, :, :] = Tb_Simulated_P        
        Tb_simulated_final_L [icycle, :, :] = Tb_Simulated_L

        Innovations_final = np.zeros ((ncycles, npol, nensembles), dtype = np.float64)
        Innovations_final [icycle, :, :] = Innovations

        Jacobians_final = np.zeros ((ncycles, nlayers_SM, npol), dtype = np.float64)
        Jacobians_final [icycle, :, :] = Jacobians
        
    else:
         
        X_mod_mean_final [icycle, :] = X_mod_mean
        X_mod_var_final  [icycle, :] = X_mod_var
        X_mod_std_final  [icycle, :] = X_mod_std

        Y_obs_mean_final [icycle, :] = Y_obs_mean
        Y_obs_std_final  [icycle, :] = Y_obs_std
        Y_obs_var_final  [icycle, :] = Y_obs_var
        
        X_updated_final [icycle, :, :, :]  = X1         # assign all 96 soil moisture values  
        X_updated_final [icycle, count_ref, :, :] = X_updated  # replace last values in the cycle

        Tb_simulated_final_P [icycle, :, :] = Tb_Simulated_P
        Tb_simulated_final_L [icycle, :, :] = Tb_Simulated_L
        
        Innovations_final [icycle, :, :] = Innovations
        
        K_final [icycle, :, :] = K  
        Jacobians_final [icycle, :, :] = Jacobians
        
    #K                          # (7 x 4)
    # y-H(x)
    #X_updated= X + K*(Tb obs_sim (ref_1)+ Ey - Y_simulated(all))
    # X_updated= X  to check the loop for initial conditions

    PATH_Analysis = '/media/rohit/Data/Ensemble_kalman_filter/Create_Analysis/'
    #filename_Analysis = 'Analysis_'+str(icycle+1)+'.nc'
    #ncfile = Dataset(PATH_Analysis+filename_Analysis, 'w')
    #ncfile.createDimension('nlayers', size=7)
    #ncfile.createDimension('nensembles', size=50)
    #ncfile.createDimension('npol', size=2)
    #datanc = ncfile.createVariable('SM_Analysis', np.float64, ('nlayers', 'nensembles'))
    #datanc[:, :] = np.float64(X_updated) #analysis state of SM
    #datanc = ncfile.createVariable('ST_Analysis', np.float64, ('nlayers', 'nensembles'))
    #datanc[:, :] = np.float64(ST_A)
    #ncfile.close()

    SW_updated = X_updated/0.45
    
    #---------------------------------------------------------------
    #Saving  Updated Initial Conditions
    PATH_INC = '/media/rohit/Data/Ensemble_kalman_filter/Jules_run/file/'
    filename_INC = 'initial_conditions.nc'
    updated_file = 'updated_initial_conditions.nc'
    ncfile = Dataset (PATH_INC+updated_file, 'w')
    ncfile.createDimension('z',size= nlayers_SM)
    ncfile.createDimension('land',size= nensembles)
    datanc    = ncfile.createVariable('sthuf',np.float64,('z', 'land'))
    datanc[:,:] = np.float64(SW_updated)
    datanc    = ncfile.createVariable('t_soil',np.float64,('z', 'land'))
    datanc[:,:] = np.float64(ST_A)
    ncfile.close()

    # Copy Initial Conditions to Anaysis Path
    shutil.copyfile(PATH_INC + updated_file, PATH_Analysis+ 'initial_conditions_'+str(icycle+1)+'.nc')

    # Remove orignal file
    os.remove(PATH_INC+filename_INC)
    # Rename Updated file to as same as orignal file
    os.rename(PATH_INC+updated_file, PATH_INC+filename_INC)

    print("Initial Conditions Updated")

    #----------------------------------------------------------------------
    # Saving Update timesteps.nml
    PATH_NML = '/media/rohit/Data/Ensemble_kalman_filter/Jules_run/nmls/'
    filename_time_steps =  'timesteps.nml'
    updated_timesteps =  'timesteps2.nml'
    fin  = open(PATH_NML+filename_time_steps, "rt")
    fout = open(PATH_NML+updated_timesteps, "wt")

    # Reading End time stamps
    end_old_date = str(ini_month_end).zfill(2) + '-' + str(ini_date_end).zfill(2)
    end_new_date = str(ini_month_end).zfill(2) + '-' + str(ini_date_end + 1).zfill(2)
    # Reading Start time stamp:
    start_old_date = str(ini_month_start).zfill(2) + '-' + str(ini_date_start).zfill(2)
    start_new_date = str(ini_month_start).zfill(2) + '-' + str(ini_date_start + 1).zfill(2)
    if ini_date_end == 31:
        ini_month_end = 6
        ini_date_end = 0
        end_new_date = str(ini_month_end).zfill(2) + '-' + str(ini_date_end + 1).zfill(2)

    if ini_date_start == 31:
        ini_month_start = 6
        ini_date_start = 0
        start_new_date = str(ini_month_start).zfill(2) + '-' + str(ini_date_start + 1).zfill(2)
    # Update End time stamp:
    for line in fin:
        fout.write(line.replace(end_old_date, end_new_date))
    fin.close()
    fout.close()
    os.remove(PATH_NML+filename_time_steps)
    os.rename(PATH_NML+updated_timesteps, PATH_NML+filename_time_steps)
    fin = open(PATH_NML + filename_time_steps, "rt")
    fout = open(PATH_NML + updated_timesteps, "wt")
    # Update Start time stamp:
    for line in fin:
        fout.write(line.replace(start_old_date, start_new_date))
    fin.close()
    fout.close()
    os.remove(PATH_NML+filename_time_steps)
    os.rename(PATH_NML+updated_timesteps, PATH_NML+filename_time_steps)

    print("Timestamp Updated")

    print('index', index) 
    index = index +1
    

    print (" Assimilation cycle completed for the period: ", '  Start date = '+str(ini_date_start)+ ' '+ '&' + ' ' + ' Start Month = ', str(ini_month_start)
            +' '+'End date = ' + str(ini_date_end) + ' ' + 'End Month =' + str (ini_month_end))

    ini_date_start = ini_date_start + 1
    ini_date_end   = ini_date_end + 1

    
PATH_Analysis = '/media/rohit/Data/Ensemble_kalman_filter/Wrapper_bible/Output/test_1/'

filename_Analysis_final = 'Forecast+Analysis_Oct2_P_L_mean_3_index0_inc30.nc'
ncfile = Dataset(PATH_Analysis+filename_Analysis_final, 'w')

ncfile.createDimension('npol', size=npol)
ncfile.createDimension('nensembles', size=nensembles)
ncfile.createDimension('ncycles', size=ncycles)
ncfile.createDimension('nlayers', size=7)
ncfile.createDimension('ntimesteps', size=96)
ncfile.createDimension('nbands', size=2)
ncfile.createDimension('pol', size=2)
ncfile.createDimension('nrows',  size=3402)

datanc = ncfile.createVariable('Tb_ref_P', np.float64, ('nrows', 'pol'))
datanc[:, :] = np.float64(Tb_Ref_P)

datanc = ncfile.createVariable('Tb_ref_L', np.float64, ('nrows', 'pol'))
datanc[:, :] = np.float64(Tb_Ref_L)                              

datanc = ncfile.createVariable('X_updated_final', np.float64, ('ncycles', 'ntimesteps','nlayers', 'nensembles'))
datanc[:, :, :, :] = np.float64(X_updated_final)

datanc = ncfile.createVariable('X_mod_mean_final', np.float64, ('ncycles', 'nlayers'))
datanc[:, :] = np.float64(X_mod_mean_final)

datanc = ncfile.createVariable('X_mod_std_final', np.float64, ('ncycles', 'nlayers'))
datanc[:, :] = np.float64(X_mod_std_final)

datanc = ncfile.createVariable('X_mod_var_final', np.float64, ('ncycles', 'nlayers'))
datanc[:, :] = np.float64(X_mod_var_final)

datanc = ncfile.createVariable('Y_obs_mean_final', np.float64, ('ncycles', 'npol'))
datanc[:, :] = np.float64(Y_obs_mean_final)

datanc = ncfile.createVariable('Y_obs_std_final', np.float64, ('ncycles', 'npol'))
datanc[:, :] = np.float64(Y_obs_std_final)

datanc = ncfile.createVariable('Y_obs_var_final', np.float64, ('ncycles', 'npol'))
datanc[:, :] = np.float64(Y_obs_var_final)

datanc = ncfile.createVariable('Tb_simulated_final_P', np.float64, ('ncycles', 'nbands', 'nensembles'))  
datanc[:, :, :] = np.float64(Tb_simulated_final_P)

datanc = ncfile.createVariable('Tb_simulated_final_L', np.float64, ('ncycles', 'nbands',  'nensembles'))  
datanc[:, :, :] = np.float64(Tb_simulated_final_L)

datanc = ncfile.createVariable('Innovations_final', np.float64, ('ncycles', 'npol',  'nensembles'))  
datanc[:, :, :] = np.float64(Innovations_final)

datanc = ncfile.createVariable('K_final', np.float64, ('ncycles', 'nlayers', 'npol'))
datanc [:, :, :] = np.float64 (K_final)

datanc = ncfile.createVariable('Jacobians_final', np.float64, ('ncycles', 'nlayers', 'npol'))
datanc [:, :, :] = np.float64 (Jacobians_final)

ncfile.close()

PATH_def = source_folder  #'/media/rohit/Data/Ensemble_kalman_filter/'

# Copy Initial Conditions and timestep.nml to Orignal Jules Path
shutil.copyfile(PATH_def+ filename_INC, PATH_INC + filename_INC)
shutil.copyfile(PATH_def+ filename_time_steps, PATH_NML + filename_time_steps)

#----------------------------------------------------------------------------
print ("Thanks for using Wrapper, Have a good time ! ")
print ("This wrapper was developed under an open-source license in collaboration with Meteo-France, Monash University, and IIT Bombay")
print ("Please cite the below work for using this framework:")
print("Prajapati et al. 2023: Development of Tb Assimilation Framework for both P and L band in Jules Land"
      " Surface Model: PART 1: Initial trial results on Synthetic datasets")
#----------------------------------------------------------------------------
