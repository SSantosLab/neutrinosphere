import numpy as np
import matplotlib.pyplot as plt
import os
import time
from scipy import interpolate

# Neutrino luminosities & energies for 2 possible viewing angles: polar and equatorial
# polar influences shock-heated ejecta; equatorial influences tidal ejecta

# constants
c=3e8 #m/s
G=6.67e-11 #SI
Msun=2e30
M=2.75*Msun #kg
sb_const=1.028e-5 #10^51 erg/s, km^-2, MeV^-4
M_n=939.6 #neutron mass, MeV
M_p=938.3 #proton mass, MeV
D=1.293 # neutron proton mass difference, MeV
kconst=9.3e-48 #m^2 / MeV^2
eng=6.24151e56 #MeV per 10^51 erg/s
dt=2.5e-6 #time increments of Y_e integration in s
Ye0=.06 #initial electron fraction determined from cold beta-equilibrium value for neutron stars (Wanajo et al. 2014)

# in MeV - from scaled up Perego+2014 data
T_neut_disk=4.3
T_neut_NS=7
T_anti_disk=5.4
T_anti_NS=8.7

# in km - disk geometry from Perego+2014
R_neut=40
R_anti=27
H=15

# in km^2, projected areas for each viewing angle
A_neut_disk_pol=np.pi*(R_neut**2)
A_neut_disk_equ=4*R_neut*H
A_anti_disk_pol=np.pi*(R_anti**2)
A_anti_disk_equ=4*R_anti*H
A_NS=np.pi*(H**2)

# in MeV, luminosities for each viewing angle
L_neut_disk_pol=sb_const*4*(A_neut_disk_pol*T_neut_disk**4)*eng
L_neut_NS_pol=sb_const*4*(A_NS*T_neut_NS**4)*eng
L_neut_pol=L_neut_disk_pol+L_neut_NS_pol
L_neut_equ=sb_const*4*(A_neut_disk_equ*T_neut_disk**4)*eng
L_anti_disk_pol=sb_const*4*(A_anti_disk_pol*T_anti_disk**4)*eng
L_anti_NS_pol=sb_const*4*(A_NS*T_anti_NS**4)*eng
L_anti_pol=L_anti_disk_pol+L_anti_NS_pol
L_anti_equ=sb_const*4*(A_anti_disk_equ*T_anti_disk**4)*eng

# in MeV, average (anti-)neutrino energies for each viewing angle
E_neut_pol=3.1514*T_neut_disk*T_neut_NS*(L_neut_pol)/(L_neut_disk_pol*T_neut_NS + L_neut_NS_pol*T_neut_disk)
E_neut_equ=3.1514*T_neut_disk
E_anti_pol=3.1514*T_anti_disk*T_anti_NS*(L_anti_pol)/(L_anti_disk_pol*T_anti_NS + L_anti_NS_pol*T_anti_disk)
E_anti_equ=3.1514*T_anti_disk

# Integrate Y_e for dynamical ejecta
def get_Ye(vf,t0,comp,dt):
    # vf [c], t0 [ms], comp=0 for shock and 1 for tidal, dt [s]
    
    # different components get different luminosities & energies
    if comp==0: #shock-heated
        L_n=L_neut_pol
        E_n=E_neut_pol
        L_a=L_anti_pol
        E_a=E_anti_pol
    elif comp==1: #tidal
        L_n=L_neut_equ
        E_n=E_neut_equ
        L_a=L_anti_equ
        E_a=E_anti_equ

    # (anti-)neutrino capture cross sections
    W_n=1+1.02*1.2*1.14*E_n/M_n
    W_a=1-7.22*1.2*1.14*E_a/M_p
    sigma_n=kconst*(1.2*E_n**2 + 2*D*E_n + D**2)*W_n
    sigma_a=kconst*(1.2*E_a**2 - 2*D*E_a + D**2)*W_a
    
    # (anti-)neutrino capture reaction rates
    k_n=L_n*sigma_n/(4*np.pi*E_n)
    k_a=L_a*sigma_a/(4*np.pi*E_a)
    k1=k_n
    k2=k_n+k_a
    
    # ejected from ~surface of neutron star
    r=15.*1000 #m
    t=t0*.001 #s
    v=((vf*c)**2. + 2*G*M/r)**.5 #m/s - starting velocity so it ends up with final velocity
    
    reactions=0 #sum up reactions
    
    for n in range(5000):
        #(anti-)neutrino luminosity (and thus reaction rates) increases linearly through first 5ms
        if t < .005:
            k2_new=k2*(max(0,t)/.005)
        else: k2_new=k2
        
        #integrate forward reactions, a, v, r, t
        reactions+=(k2_new/(r**2))*dt
        a=-G*M/(r**2)
        v += a*dt
        r += v*dt + .5*a*(dt**2)
        t += dt
        if r<0: #so it can restart with a smaller timestep if it falls in - prob won't happen
            return 0
    
    Ye_eq=k1/k2
    Ye_answer=Ye_eq-(Ye_eq-Ye0)*(np.e**(-reactions))
    return Ye_answer

#to go from "transition velocity" to "kinetic velocity"
velocity_factor=((14/9.)*(1/4. + 1/5. - 1/(50*(10**.5))))**.5

#table giving lanthanide fraction for a given Y_e, s, Msph, and vcoast
lrdata=open('LRdata_modified.txt','r').readlines()[1:]
lrdata=np.array([x.rstrip().split(',') for x in lrdata]).astype(np.float)

#exclude data that def won't be needed
lrdata=lrdata[np.where((lrdata[:,1] >= 7.5) & (lrdata[:,1] <= 32))]
masses=np.unique(lrdata[:,2])

# The one MOSFiT will call over and over:
# start with very beginning parameters, return parameters which will be used to choose SED (and weights)
def get_good_params(Mej, vcoast, Ye_or_angle, comp):
    # ejecta mass in solar masses
    # coasting velocity in c (corresponds to Kasen's "transition velocity")
    # Ye only if disk wind; otherwise opening polar angle of shock-heated ejecta in degrees
    # comp = 0 for shock-heated, comp = 1 for tidal, comp = 2 for wind

    if comp==2: #wind
        Ye=Ye_or_angle
        entropy=25
        weight=1
        
    else: #dynamical
        #mass fractions (or volume fractions, same thing)
        shock_opening_ang=Ye_or_angle*np.pi/180.
        shock_mass_fraction=.5*((2+np.cos(shock_opening_ang))*(1-np.cos(shock_opening_ang))**2 + (np.sin(shock_opening_ang)**2)*np.cos(shock_opening_ang))

        if comp==0: #shock-heated
            entropy=30
            weight=shock_mass_fraction
            Ye=get_Ye(vcoast,2,comp,dt)
            if Ye==0: #in case integration failed, repeat with smaller timestep
                Ye=get_Ye(vcoast,2,comp,1.0e-6)
                
        elif comp==1: #tidal
            entropy=10
            weight=1-shock_mass_fraction
            Ye=get_Ye(vcoast,0,comp,dt)
            if Ye==0: #in case integration failed, repeat with smaller timestep
                Ye=get_Ye(vcoast,0,comp,1.0e-6)
    
    Msph=Mej/weight
    v_k=vcoast*velocity_factor
    
    #cut out unneeded masses, Ye's from Xla table
    lower_mass=0
    upper_mass=1
    if Msph>.001:
        try: lower_mass=masses[np.where(masses<Msph)[0][-1]]
        except IndexError: lower_mass=0
    if Msph<.1:
        try: upper_mass=masses[np.where(masses>Msph)[0][0]]
        except IndexError: upper_mass=1
    lrdatatemp=lrdata[np.where((lrdata[:,0] >= Ye-.04) &(lrdata[:,0] <= Ye+.04) & (lrdata[:,2] >= lower_mass) & (lrdata[:,2] <= upper_mass))]
    
    #interpolate grid for Xla. If outside grid, it'll return Xla=0
    Xla=interpolate.griddata((lrdatatemp[:,0],lrdatatemp[:,1],lrdatatemp[:,2],lrdatatemp[:,3]),lrdatatemp[:,4],(Ye,entropy,Msph,vcoast),method='linear',fill_value=0)*1.
    #exclude ones out of range of SED grid
    if Xla>.1: Xla=.1
    if Xla<1.0e-9: Xla=1.0e-9
    
    #spherical mass [Msun], kinetic velocity [c], lanthanide mass fraction, SED weight
    return [Msph,v_k,Xla,weight]
