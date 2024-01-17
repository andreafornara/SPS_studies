# Basic example of the simulation with wakefields
# %%
import h5py
import yaml
import os
import argparse
import xtrack as xt
import xpart as xp
xp.enable_pyheadtail_interface()
import matplotlib.pyplot as plt
import numpy as np
import json
import xobjects as xo
from scipy.interpolate import interp1d
from PyHEADTAIL.particles.slicing import UniformBinSlicer
from PyHEADTAIL.trackers.detuners import Chromaticity, AmplitudeDetuning
from PyHEADTAIL.trackers.transverse_tracking import TransverseMap
from PyHEADTAIL.trackers.simple_long_tracking import RFSystems, LinearMap
from PyHEADTAIL.monitors.monitors import BunchMonitor, SliceMonitor, ParticleMonitor
from PyHEADTAIL.feedback.transverse_damper import TransverseDamper
from PyHEADTAIL.impedances.wakes import CircularResonator, WakeTable, WakeField
from scipy.optimize import curve_fit
import time
from functions import *

parser = argparse.ArgumentParser()
parser.add_argument("--yaml_path", type=str, required=True)

args = parser.parse_args()

# open the yaml file and get the parameters of the simulation
with open(args.yaml_path, "r", encoding="utf-8") as f:
    parameters = yaml.safe_load(f)
# %%
# Load the lattice, this is an SPS in the conditions of the MD
fname_line = '/afs/cern.ch/work/a/afornara/public/New_Crab_Cavities_MD/sps-md-ccnoise/MD_scripts/sps_madx/'
with open(fname_line+'line_SPS_Q26.json', 'r') as fid:
    input_data = json.load(fid)
# We choose the context
GPU_flag = parameters['GPU']
if(GPU_flag):
    device = parameters['GPU_device']
    ctx = xo.ContextCupy(device = device)
    print(f'Using GPU {device}')
else:
    ctx = xo.ContextCpu()
    print('Using CPU, warning, this is VERY slow!')
line = xt.Line.from_dict(input_data)
# We rotate the line to measure everything at the WS
line = line.cycle('bwsrc.51637')
line.build_tracker(_context=ctx)
# MD was performed at 270 GeV
particle_ref = xp.Particles(p0c=270e9, q0=1, mass0=xp.PROTON_MASS_EV)
# If we want to change the octupole current we just need to activate this flag
change_octupole = parameters['change_octupole']
# Desired octupole current to be set
klod = parameters['klod']
if(change_octupole):
    print(f'Changing LOD octupole currents to {klod}/m^4')
    for elem in line.element_dict :
        if isinstance(line.element_dict[elem],xt.beam_elements.elements.Multipole): 
            if((line.element_dict[elem].order==3) and (elem.startswith('lod'))):
                line.element_dict[elem].knl[3] = klod
twiss = line.twiss(particle_ref)

# %%
circumference = twiss['s'][-1]
c_light = 299792458
frev = c_light/circumference
# From MD info
tau = 1.83e-9
normal_emitt_x = 2.0e-6
normal_emitt_y = 2.0e-6
h = 4620
ksl = line.element_dict['cravity.1'].ksl
ps = line.element_dict['cravity.1'].ps
# %%
dx = twiss['dx'][0]
dpx = twiss['dpx'][0]
dy = twiss['dy'][0]
dpy = twiss['dpy'][0]
sigma_z = c_light*tau/4
sigma_x = np.sqrt(twiss['betx'][0]*normal_emitt_x/(particle_ref.gamma0*particle_ref.beta0))[0]
sigma_y = np.sqrt(twiss['bety'][0]*normal_emitt_x/(particle_ref.gamma0*particle_ref.beta0))[0]

# %%
activate_wakefields = parameters['activate_wakefields']
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
    wake_field_1.needs_cpu = True
    wake_field_2.needs_cpu = True
    wake_field_3.needs_cpu = True

    line.append_element(wake_field_1, 'wake_field_1')
    line.append_element(wake_field_2, 'wake_field_2')
    line.append_element(wake_field_3, 'wake_field_3')
    line.build_tracker(_context=ctx)
    twiss = line.twiss(particle_ref)

# %%
N_turns = parameters['N_turns']
print(f'Number of turns: {N_turns}')
noise_level = parameters['noise_level'] # dBc/Hz
Gyy = measured_psd_2_Gyy(noise_level)
Syy = Gyy/2 # rad^2/Hz
print(f'Noise level: {Syy} rad^2/Hz')
amplitude_noise_flag = parameters['amplitude_noise']
phase_noise_flag = parameters['phase_noise']
if(amplitude_noise_flag):
    print('Amplitude noise is ON')
    amp_noise = Syy*frev
    print(f'Amplitude noise: {amp_noise} rad^2')
    amp_noise_list = np.random.normal(0, np.sqrt(amp_noise), N_turns)
    print(f'Amplitude noise list: {amp_noise_list}')
    print('sigma: ', np.std(amp_noise_list), 'rad')
if(phase_noise_flag):
    print('Phase noise is ON')
    ph_noise = Syy*frev
    print(f'Phase noise: {ph_noise} rad^2')
    ph_noise_deg = from_rad_to_deg(np.sqrt(ph_noise))
    ph_noise_list = np.random.normal(0, ph_noise_deg, N_turns)
    print(f'Phase noise list: {ph_noise_list}')
    print('sigma: ', np.std(ph_noise_list), 'deg')
# %%
# Calculating the Crab Dispersion at the BWS location via 4D Twiss
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
zeta_values = np.linspace(-8*sigma_z, 8*sigma_z, iterall)
# We iterate on the zeta values to obtain the 4D Twiss parameters
for i in range(iterall):
    if((i%10 == 0)):
        print(f'Xsuite working on {i} of {len(deltas)}, zeta = {np.round(zeta_values[i],2)} ')
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
# We interpolate the 4D Twiss parameters to obtain the crabbing function in both planes
interp_xs = interp1d(zetas, xs, kind='cubic')
interp_pxs = interp1d(zetas, pxs, kind='cubic')
interp_ys = interp1d(zetas, ys, kind='cubic')
interp_pys = interp1d(zetas, pys, kind='cubic')

# %%
# Emittance arrays to be filled with turn by turn data
EXS = []
EYS = []

# Prepare the Gaussian Distribution
bunch_intensity = 3.0e10
N_particles = parameters['N_particles']

if(activate_wakefields):
    particles_gauss = xp.generate_matched_gaussian_bunch(
                    num_particles=N_particles, total_intensity_particles=bunch_intensity,
                    nemitt_x=normal_emitt_x, nemitt_y=normal_emitt_y, sigma_z=sigma_z,
                    particle_ref=particle_ref, line=line, _context=ctx)
    
    n_macroparticles = 20000
    particles = xp.Particles(
        _context=ctx,
        circumference=circumference,
        particlenumber_per_mp=bunch_intensity / n_macroparticles,
        q0=1,
        mass0= xp.PROTON_MASS_EV,
        gamma0=particle_ref.gamma0[0],
        x= ctx.nparray_from_context_array(particles_gauss.x),
        px= ctx.nparray_from_context_array(particles_gauss.px),
        y= ctx.nparray_from_context_array(particles_gauss.y),
        py= ctx.nparray_from_context_array(particles_gauss.py),
        zeta= ctx.nparray_from_context_array(particles_gauss.zeta),
        delta= ctx.nparray_from_context_array(particles_gauss.delta))
else:
    particles = xp.generate_matched_gaussian_bunch(
                    num_particles=N_particles, total_intensity_particles=bunch_intensity,
                    nemitt_x=normal_emitt_x, nemitt_y=normal_emitt_y, sigma_z=sigma_z,
                    particle_ref=particle_ref, line=line, _context=ctx)

# %%
#Print out the time of the simulation importing time
start_time = time.time()
exs = np.zeros(N_turns)
eys = np.zeros(N_turns)
for ii in range(N_turns):
    line.track(particles)
    if(amplitude_noise_flag):
        line.element_dict['cravity.1'].ksl = ksl*(1 + amp_noise_list[ii])
    if(phase_noise_flag):
        line.element_dict['cravity.1'].ps = ps + ph_noise_list[ii]
    # Check the emittance
    xs_0 = ctx.nparray_from_context_array(particles.x)
    pxs_0 = ctx.nparray_from_context_array(particles.px)
    ys_0 = ctx.nparray_from_context_array(particles.y)
    pys_0 = ctx.nparray_from_context_array(particles.py)
    delta_0 = ctx.nparray_from_context_array(particles.delta)
    zeta_0 = ctx.nparray_from_context_array(particles.zeta)
    # We select only the particles within 2 sigma_z to avoid non-linear effects
    cut = (np.abs(zeta_0)<2*sigma_z)
    ex = emittance(xs_0[cut],pxs_0[cut],zeta_0[cut],delta_0[cut],dx,dpx,interp_xs,interp_pxs)*particle_ref.gamma0*particle_ref.beta0
    ey = emittance(ys_0[cut],pys_0[cut],zeta_0[cut],delta_0[cut],dy,dpy,interp_ys,interp_pys)*particle_ref.gamma0*particle_ref.beta0
    exs[ii] = ex[0]
    eys[ii] = ey[0]
    if(ii%1000== 0):
        #print minutes
        print(f'Turn number {ii} of {N_turns}')
        print("--- %s ---" % (time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))))
print("Simulation ended in :")
print("--- %s ---" % (time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))))

turns = np.arange(N_turns)
popt, pcov = curve_fit(linear_fit, turns, exs*1e6)
print(f'Ex growth rate = {popt[0]*60*60*frev:.2f} um/hour')
popt, pcov = curve_fit(linear_fit, turns, eys*1e6)
print(f'Ey growth rate = {popt[0]*60*60*frev:.2f} um/hour')

# %%
