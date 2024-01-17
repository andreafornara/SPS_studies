import numpy as np

# %%
# The emittance is defined by taking into account both the longitudinal position and delta
# A linear interpolation is performed to obtain the crabbing function for the correction of the emittance
def emittance(x,px,zeta,delta,dx,dpx,interp_x,interp_px):
    x_np = np.array(x)
    px_np= np.array(px)
    x_np = np.array(x - delta*dx-interp_x(zeta))
    px_np = np.array(px - delta*dpx-interp_px(zeta))

    x_mean_2 = np.dot(x_np,x_np)/len(x)
    px_mean_2 = np.dot(px_np,px_np)/len(x)
    x_px_mean = np.dot(x_np,px_np)/len(x)

    emitt = np.sqrt(x_mean_2*px_mean_2-(x_px_mean)**2)

    return emitt

# We convert the measured PSD to Gyy
def measured_psd_2_Gyy(psd_meas):
    # see notes_on_PSD.png
    L = 10**(psd_meas/10)
    Gyy = 2*L
    return Gyy

# We convert the measured Gyy to PSD
def Gyy_2_measured_psd(Gyy):
    # see notes_on_PSD.png
    L = Gyy/2
    psd_meas = 10*np.log10(L)
    return psd_meas

# Simple function to convert from rad to deg
def from_rad_to_deg(phi_rad):
    phi_deg = phi_rad*180/np.pi
    return phi_deg

# Simple linear fit
def linear_fit(x, slope, intercept):
    return slope*x + intercept

