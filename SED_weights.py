# returns 4Aproj/A, where A = the component's fractional surface area of a sphere
def Aproj_weight_s(theta_s,theta_obs):
    x=0
    if theta_s+theta_obs > np.pi/2.: x=(np.sin(theta_s)**2. - np.cos(theta_obs)**2.)**.5/np.sin(theta_obs)
    num=(np.pi*(np.sin(theta_s)**2.)*(np.cos(theta_obs)) + 2*(1-np.cos(theta_obs))*(np.arcsin(x) - x*(1 - x**2.)**.5))
    den=np.pi*(1-np.cos(theta_s))
    return num/den
def Aproj_weight_t(theta_s,theta_obs):
    x=0
    if theta_s+theta_obs > np.pi/2.: x=(np.sin(theta_s)**2. - np.cos(theta_obs)**2.)**.5/np.sin(theta_obs)
    num=np.pi - (np.pi*(np.sin(theta_s)**2.)*(np.cos(theta_obs)) + 2*(1-np.cos(theta_obs))*(np.arcsin(x) - x*(1 - x**2.)**.5))
    den=np.pi*np.cos(theta_s)
    return num/den

# volume fraction of a sphere
def mass_weight_s(theta_s):
    return .5*((2+np.cos(theta_s))*(1-np.cos(theta_s))**2 + (np.sin(theta_s)**2)*np.cos(theta_s))
def mass_weight_t(theta_s):
    return 1 - .5*((2+np.cos(theta_s))*(1-np.cos(theta_s))**2 + (np.sin(theta_s)**2)*np.cos(theta_s))

def shock_factor(theta_s,theta_obs):
    return Aproj_weight_s(theta_s,theta_obs)*mass_weight_s(theta_s)
def tidal_factor(theta_s,theta_obs):
    return Aproj_weight_t(theta_s,theta_obs)*mass_weight_t(theta_s)


# # EXAMPLE
# # if you want the new SED in ergs /s /ang or ergs /s /Hz
# shock_opening_ang=60*np.pi/180.
# viewing_ang=45*np.pi/180.
# weighted_sum = SED_0*shock_factor(shock_opening_ang,viewing_ang) + SED_1*tidal_factor(shock_opening_ang,viewing_ang)


