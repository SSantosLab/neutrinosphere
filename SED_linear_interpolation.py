import h5py
import numpy as np
from scipy.interpolate import griddata

# lists of masses, velocities, Xlans
mass_options_str=np.array(['0.001','0.0025','0.005','0.010','0.020','0.025','0.030','0.040','0.050','0.075','0.1'])
vk_options_str=np.array(['0.03','0.05','0.10','0.20','0.30'])
Xlan_options_str=np.array(['1e-9.0','1e-5.0','1e-4.0','1e-3.0','1e-2.0','1e-1.0'])
mass_options=mass_options_str.astype(np.float)
vk_options=vk_options_str.astype(np.float)
Xlan_options=[10**(-float(x.split('-')[1])) for x in Xlan_options_str]

# coordinate axes for interpolator
data_coordinates=[]
for i in range(len(mass_options)):
    for j in range(len(vk_options)):
        for k in range(len(Xlan_options)):
            data_coordinates+=[[mass_options[i],vk_options[j],Xlan_options[k]]]
data_coordinates=np.array(data_coordinates)

# creates new 5D SED grid interpolated to the needed test_times [days]
# "SEDs" axes are mass, vk, Xlan, time, wavelength
# test_times is 1D array of days, SED_folder has the SEDs in it
def new_SED_grid(test_times,SED_folder):
    
    # get the times & wavelengths of the original SED grid
    fname=SED_folder+'knova_d1_n10_m'+str(mass_options_str[0])+'_vk'+str(vk_options_str[0])+'_fd1.0_Xlan'+str(Xlan_options_str[0])+'.h5'
    fin=h5py.File(fname,'r')
    nu=np.array(fin['nu'],dtype='d')
    c_diff_units=2.99e10
    SED_wavelengths=c_diff_units/nu*1e8
    times = np.array(fin['time'])/86400.
    
    SEDs=np.empty((len(mass_options),len(vk_options),len(Xlan_options),len(test_times),SED_wavelengths.shape[0]))
    for i in range(len(mass_options)):
        for j in range(len(vk_options)):
            for k in range(len(Xlan_options)):
                # one is missing from Kasen's models
                if mass_options[i]==0.1 and vk_options[j]==0.3 and Xlan_options[k]==1.0e-1:
                    SEDs[i,j,k,:,:]=0
                    continue
                
                fname=SED_folder+'knova_d1_n10_m'+str(mass_options_str[i])+'_vk'+str(vk_options_str[j])+'_fd1.0_Xlan'+str(Xlan_options_str[k])+'.h5'
                fin=h5py.File(fname,'r')
                
                # specific luminosity (ergs/s/Hz) 
                # this is a 2D array, Lnu[times,nu]
                Lnu_all   = np.array(fin['Lnu'],dtype='d')
            
                for m in range(len(test_times)):
                    equal=np.where(times==test_times[m])
                    
                    # if SED has that actual time
                    if len(equal[0])!=0:
                        SEDs[i,j,k,m,:]=Lnu_all[equal[0],:]
                        continue
                
                    # interpolate between neighboring ones
                    lower=np.where(times<test_times[m])[0][-1]
                    higher=np.where(times>test_times[m])[0][0]
                    SEDs[i,j,k,m,:]=(Lnu_all[lower,:]*(times[higher]-test_times[m]) + Lnu_all[higher,:]*(test_times[m]-times[lower]))/(times[higher]-times[lower])
    return SEDs #axes are mass [solar], vk [c], Xlan, time [day], wavelength [angstrom]


# SEDs is 5D grid of m,vk,xlan,time,wavelength
# returns a 2D grid of fluxes at (times, wavelengths)
def interpolate_SED(SEDs,M,vk,Xlan): # M in solar masses, vk in c
    slice_shape=SEDs[0,0,0,:,:].shape
    flat_SEDs=SEDs.reshape(-1,slice_shape[0],slice_shape[1])
    return griddata(data_coordinates,flat_SEDs,(M,vk,Xlan))


# EXAMPLE:
test_times=[0.5,1.5,2.5]
path='/data/des40.a/data/jmetzger/Kasen_Kilonova_Models_2017/systematic_kilonova_model_grid/'
SED_grid=new_SED_grid(test_times,path)
SED_slice=interpolate_SED(SED_grid,0.0225,0.15,1e-6)