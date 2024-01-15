# %%
import os
import xpart as xp
import xtrack as xt
xp.enable_pyheadtail_interface()
import matplotlib.pyplot as plt
import numpy as np
import json
import xobjects as xo
from scipy.interpolate import interp1d
from PyHEADTAIL.particles.slicing import UniformBinSlicer
from PyHEADTAIL.particles.generators import generate_Gaussian6DTwiss 
from PyHEADTAIL.trackers.detuners import Chromaticity, AmplitudeDetuning
from PyHEADTAIL.trackers.transverse_tracking import TransverseMap
from PyHEADTAIL.trackers.simple_long_tracking import RFSystems, LinearMap
from PyHEADTAIL.monitors.monitors import BunchMonitor, SliceMonitor, ParticleMonitor
from PyHEADTAIL.feedback.transverse_damper import TransverseDamper
from PyHEADTAIL.impedances.wakes import CircularResonator, WakeTable, WakeField
from scipy.optimize import curve_fit
import time
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
def measured_psd_2_Gyy(psd_meas):
    # see notes_on_PSD.png
    L = 10**(psd_meas/10)
    Gyy = 2*L
    return Gyy

def Gyy_2_measured_psd(Gyy):
    # see notes_on_PSD.png
    L = Gyy/2
    psd_meas = 10*np.log10(L)
    return psd_meas

def from_rad_to_deg(phi_rad):
    phi_deg = phi_rad*180/np.pi
    return phi_deg

def linear_fit(x, slope, intercept):
    return slope*x + intercept
# %%

# Load the lattice
fname_line = '/afs/cern.ch/work/a/afornara/public/New_Crab_Cavities_MD/sps-md-ccnoise/MD_scripts/sps_madx/'
with open(fname_line+'line_SPS_Q26.json', 'r') as fid:
    input_data = json.load(fid)
# print('Context is set to CPU as default')
ctx = xo.ContextCupy(device = 2)
ctx = xo.ContextCpu()
line = xt.Line.from_dict(input_data)
# We rotate the line to measure everything at the WS
line = line.cycle('bwsrc.51637')
line.build_tracker(_context=ctx)
particle_ref = xp.Particles(p0c=270e9, q0=1, mass0=xp.PROTON_MASS_EV)
change_octupole = True
klod = 30.0
if(change_octupole):
    print(f'Changing LOD octupole currents to {klod}/m^4')
    for elem in line.element_dict :
        if isinstance(line.element_dict[elem],xt.beam_elements.elements.Multipole): 
            if((line.element_dict[elem].order==3) and (elem.startswith('lod'))):
                # print(elem)
                # print(line.element_dict[elem])
                # print(line.element_dict[elem].knl[3])
                # #find elem in twiss['name']
                # index = np.where(twiss['name']==elem)
                # print(twiss['s'][index])
                line.element_dict[elem].knl[3] = klod
twiss = line.twiss(particle_ref)

# %%
circumference = twiss['s'][-1]
c_light = 299792458
frev = c_light/circumference
print('frev = ', frev)
tau = 1.83e-9
sigma_z = c_light*tau/4
normal_emitt_x = 2.0e-6
normal_emitt_y = 2.0e-6
h = 4620

# %%
# This section is used to control the RF frequency and voltage of the SPS
# Main RF Systems and Crab Cavity
Pilot_Accelerating_Cavities = False
if(Pilot_Accelerating_Cavities):
    MV_distribution = 7.6e6/18
    my_dict = line.to_dict()
    for ii in line.to_dict()['elements']:
        if(my_dict['elements'][ii]["__class__"] == 'Cavity'):
            if(ii.startswith('act')):
                    line.element_dict[ii].frequency = h*frev
                    line.element_dict[ii].voltage = MV_distribution
                    line.element_dict[ii].lag = 180
            if(ii.startswith('acl')):
                    line.element_dict[ii].frequency = 4*h*frev
                    line.element_dict[ii].voltage = 0#0.3e6
                    line.element_dict[ii].lag = 180
    twiss = line.twiss(particle_ref)
Pilot_Crab_Cavity = False
MV_CC = 1.0
E0_MeV = particle_ref.p0c[0]*1e-6
ksl = MV_CC/E0_MeV
if(Pilot_Crab_Cavity):
    line.element_dict['cravity.1'].frequency = 2*h*frev
    line.element_dict['cravity.1'].voltage = MV_CC*1e6
    line.element_dict['cravity.1'].lag = 0
    line.element_dict['cravity.1'].ksl = ksl
    line.element_dict['cravity.1'].knl = 0
    line.element_dict['cravity.1'].ps = 90
    line.element_dict['cravity.1'].pn = 0
    twiss = line.twiss(particle_ref)
#Check the RF frequency and voltage
print(f'RF frequency = {line.element_dict["cravity.1"].frequency} Hz')
print(f'RF ksl = {line.element_dict["cravity.1"].ksl}')

# %%
activate_wakefields = True
if(activate_wakefields):
    print('Activating wakefields')
    line.discard_tracker()
    n_slices = 500 # 500
    slicer_for_wakefields = UniformBinSlicer(n_slices, z_cuts=(-3.*sigma_z, 3.*sigma_z))#,circumference=circumference, h_bunch=h1)
    n_turns_wake = 1 # for the moment we consider that the wakefield decays after 1 turn
    #wakefile0 = ('/afs/cern.ch/work/n/natriant/private/pyheadtail_example_crabcavity/wakefields/SPS_complete_wake_model_2018_Q26.txt')
    path2wakefields = 'wakefields/'
    wakefile1 = (f'{path2wakefields}step_analytical_wake.txt')
    wakefile2 = (f'{path2wakefields}Analytical_Wall_Q26_270GeV.txt')
    wakefile3 = (f'{path2wakefields}SPS_wake_model_with_EF_BPH_BPV_RF200_RF800_kicker_ZS_2018_Q26.txt')
    ww1 = WakeTable(wakefile1, ['time', 'dipole_y'], n_turns_wake=n_turns_wake) #step transition
    ww2 = WakeTable(wakefile2, ['time', 'dipole_x', 'dipole_y', 'quadrupole_x', 'quadrupole_y'], n_turns_wake=43) #wall impedance
    ww3 = WakeTable(wakefile3, ['time', 'dipole_x', 'dipole_y', 'quadrupole_x', 'quadrupole_y'], n_turns_wake=n_turns_wake) #everything else
    # multiply with a factor 2
    ww1.wake_table['dipole_y'] = 2*ww1.wake_table['dipole_y'] # for the analytical step wake
    ww2.wake_table['dipole_y'] = 1.034*ww2.wake_table['dipole_y'] # for the analytical step wake
    ww2.wake_table['dipole_x'] = 1.034*ww2.wake_table['dipole_x'] # for the analytical step wake
    ww2.wake_table['quadrupole_y'] = 1.034*ww2.wake_table['quadrupole_y'] # for the analytical step wake
    ww2.wake_table['quadrupole_x'] = 1.034*ww2.wake_table['quadrupole_x'] # for the analytical step wake
    #wake_field_0 = WakeField(slicer_for_wakefields, ww0)#, beta_x=beta_x, beta_y=beta_y)
    wake_field_1 = WakeField(slicer_for_wakefields, ww1)#, beta_x=beta_x, beta_y=beta_y)
    wake_field_2 = WakeField(slicer_for_wakefields, ww2)#, beta_x=beta_x, beta_y=beta_y)
    wake_field_3 = WakeField(slicer_for_wakefields, ww3)#, beta_x=beta_x, beta_y=beta_y)
    line.append_element(wake_field_1, 'wake_field_1')
    line.append_element(wake_field_2, 'wake_field_2')
    line.append_element(wake_field_3, 'wake_field_3')
    line.build_tracker(_context=ctx)
    twiss = line.twiss(particle_ref)
# %%
dx = twiss['dx'][0]
dpx = twiss['dpx'][0]
dy = twiss['dy'][0]
dpy = twiss['dpy'][0]
sigma_x = np.sqrt(twiss['betx'][0]*normal_emitt_x/(particle_ref.gamma0*particle_ref.beta0))[0]
sigma_y = np.sqrt(twiss['bety'][0]*normal_emitt_x/(particle_ref.gamma0*particle_ref.beta0))[0]

# %%
N_turns = 100000
noise_level = -104.0 # dBc/Hz
Gyy = measured_psd_2_Gyy(noise_level)
Syy = Gyy/2 # rad^2/Hz
# print(f'Noise level: {Syy} rad^2/Hz')
amp_noise = Syy*frev
print(f'Amplitude noise: {amp_noise} rad^2')
# amp_noise_deg = from_rad_to_deg(np.sqrt(amp_noise))
#generate a normal distribution of amplitude noise for N_turns
amp_noise_list = np.random.normal(0, np.sqrt(amp_noise), N_turns)
# %%
#Calculating the Crab Dispersion at the BWS location via 4D Twiss
# First factor in the analytical expression
at_point = 0
iterall = 200
xs = np.zeros(iterall)
dxs = np.zeros(iterall)
ys = np.zeros(iterall)
dys = np.zeros(iterall)
pxs = np.zeros(iterall)
pys = np.zeros(iterall)
deltas = np.zeros(iterall)
zetas = np.zeros(iterall)
y_disps = np.zeros(iterall)
delta_values = np.linspace(-0.00010, 0.00010, iterall)
zeta_values = np.linspace(-8*sigma_z, 8*sigma_z, iterall)
for i in range(iterall):
    if((i%10 == 0)):
        print(f'Xsuite working on {i} of {len(delta_values)}, zeta = {np.round(zeta_values[i],2)} ')
    tr4d = line.twiss(particle_ref,method='4d',zeta0=zeta_values[i])
    xs[i] = tr4d['x'][at_point] 
    dxs[i] = tr4d['dx'][at_point]
    ys[i] = tr4d['y'][at_point]
    dys[i] = tr4d['dy'][at_point]
    pxs[i] = tr4d['px'][at_point]
    pys[i] = tr4d['py'][at_point]
    zetas[i] = tr4d['zeta'][at_point]
    deltas[i] = tr4d['delta'][at_point]

# %%
interp_xs = interp1d(zetas, xs, kind='cubic')
interp_pxs = interp1d(zetas, pxs, kind='cubic')
interp_ys = interp1d(zetas, ys, kind='cubic')
interp_pys = interp1d(zetas, pys, kind='cubic')

# %%
# Prepare the Gaussian Distribution
bunch_intensity = 3.0e10
N_particles = 20000
particles = xp.generate_matched_gaussian_bunch(
                num_particles=N_particles, total_intensity_particles=bunch_intensity,
                nemitt_x=normal_emitt_x, nemitt_y=normal_emitt_y, sigma_z=sigma_z,
                particle_ref=particle_ref, line=line)
# %%
#Print out the time of the simulation importing time

start_time = time.time()
exs = np.zeros(N_turns)
eys = np.zeros(N_turns)
for ii in range(N_turns):
    line.track(particles)
    line.element_dict['cravity.1'].ksl = ksl*(1 + amp_noise_list[ii])
    # Check the emittance
    xs_0 = ctx.nparray_from_context_array(particles.x)
    pxs_0 = ctx.nparray_from_context_array(particles.px)
    ys_0 = ctx.nparray_from_context_array(particles.y)
    pys_0 = ctx.nparray_from_context_array(particles.py)
    delta_0 = ctx.nparray_from_context_array(particles.delta)
    zeta_0 = ctx.nparray_from_context_array(particles.zeta)
    cut = (np.abs(zeta_0)<2*sigma_z)
    ex = emittance(xs_0[cut],pxs_0[cut],zeta_0[cut],delta_0[cut],dx,dpx,interp_xs,interp_pxs)*particle_ref.gamma0*particle_ref.beta0
    ey = emittance(ys_0[cut],pys_0[cut],zeta_0[cut],delta_0[cut],dy,dpy,interp_ys,interp_pys)*particle_ref.gamma0*particle_ref.beta0
    exs[ii] = ex[0]
    eys[ii] = ey[0]
    if(ii%10000 == 0):
        #print minutes
        print(f'Turn number {ii} of {N_turns}')
        print("--- %s ---" % (time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))))
print("Simulation ended in :")
print("--- %s ---" % (time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))))

            
# %%
#plot the emittance


# fig, ax = plt.subplots(figsize=(10, 10))
turns = np.arange(N_turns)
# plt.plot(turns, exs*1e6, label=r'$\epsilon_{x}$')
# plt.plot(turns, eys*1e6, label=r'$\epsilon_{y}$')
#fit a line to the emittance
popt, pcov = curve_fit(linear_fit, turns, exs*1e6)

# plt.plot(turns, linear_fit(turns, *popt), 'k-', label=r'$\frac{d\epsilon_{x}}{dt}}$'+f' = {popt[0]*60*60*frev:.2f} um/hour')
print(f'Ex growth rate = {popt[0]*60*60*frev:.2f} um/hour')
popt, pcov = curve_fit(linear_fit, turns, eys*1e6)
# plt.plot(turns, linear_fit(turns, *popt), 'r-', label=r'$\frac{d\epsilon_{y}}{dt}}$'+f' = {popt[0]*60*60*frev:.2f} um/hour')
print(f'Ey growth rate = {popt[0]*60*60*frev:.2f} um/hour')
# plt.xlabel('Turns')
# plt.ylabel(r'Emittance [$\mu$m]')
# plt.grid()
# plt.title('Emittance Growth in both planes, no noise')
# plt.legend()

# %%
# import NAFFlib
# start = 3000
# end = 5000
# omega_x = NAFFlib.get_tunes(exs[start:end],5)[0]
# omega_y = NAFFlib.get_tunes(eys[start:end],5)[0]
# print('Frequency x = ', omega_x)
# print('Frequency y =', omega_y)
# # %%
# NAFFlib.get_tunes(exs[start:end],5)[0]
# NAFFlib.get_tunes(eys[start:end],5)[0]
# %%

# %%
