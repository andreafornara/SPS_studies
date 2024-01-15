# %%
import os
import xtrack as xt
import xpart as xp
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
# %%
def emittance(x,px,delta,dx,dpx):
    x_np = np.array(x)
    px_np= np.array(px)
    x_np = np.array(x - delta*dx)
    px_np = np.array(px - delta*dpx)

    x_mean_2 = np.dot(x_np,x_np)/len(x)
    px_mean_2 = np.dot(px_np,px_np)/len(x)
    x_px_mean = np.dot(x_np,px_np)/len(x)

    emitt = np.sqrt(x_mean_2*px_mean_2-(x_px_mean)**2)

    return emitt

# %%
# Load the lattice
fname_line = '/afs/cern.ch/work/a/afornara/public/New_Crab_Cavities_MD/sps-md-ccnoise/MD_scripts/sps_madx/'
with open(fname_line+'line_SPS_Q26.json', 'r') as fid:
    input_data = json.load(fid)
ctx = xo.ContextCpu()
line = xt.Line.from_dict(input_data)
# line = line.cycle('bph.40808')
# line = line.cycle('drift_3504')
line = line.cycle('bwsrc.51637')
line.build_tracker()
particle_ref = xp.Particles(p0c=270e9, q0=1, mass0=xp.PROTON_MASS_EV)
twiss = line.twiss(particle_ref)

# %%
circumference = twiss['s'][-1]
c_light = 299792458
frev = c_light/circumference
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
# %%
# Define the bunch 
c_light = 299792458
tau = 1.83e-9
normal_emitt_x = 2.0e-6
normal_emitt_y = 2.0e-6
sigma_z = c_light*tau/4
sigma_x = np.sqrt(twiss['betx'][0]*normal_emitt_x/(particle_ref.gamma0*particle_ref.beta0))[0]
sigma_y = np.sqrt(twiss['bety'][0]*normal_emitt_x/(particle_ref.gamma0*particle_ref.beta0))[0]


# %%
#Calculating the Crab Dispersion at the BWS location via 4D Twiss
betas = twiss['bety'][0]*twiss['bety','cravity.1']
f_crab = line.element_dict['cravity.1'].frequency
Delta_phi = (twiss['muy','cravity.1'] - twiss['muy'][0])*np.pi*2
Qy = twiss['qy']
# First factor in the analytical expression
first_factor = np.sqrt(betas)/(2*np.sin(np.pi*Qy))
at_point = 0
iterall = 60
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
sigma_z = c_light*tau/4
zeta_values = np.linspace(-6*sigma_z, 6*sigma_z, iterall)
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
    Dp1 = np.sin((2*np.pi)*f_crab*zeta_values[i]/c_light)*ksl
    # Second factor in the analytical expression, depends on the zeta value via Dp1
    second_factor = Dp1*np.cos(Delta_phi-np.pi*Qy)
    y_disps[i] = first_factor*second_factor

# %%
fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(zeta_values, ys*1000, 'b', label=r'y[mm] from 4D Twiss')
plt.plot(zeta_values, twiss['dy_zeta'][at_point]*zeta_values*1000, 'green', label=r'$D_{y_{cc}}$(z)*z[mm], 6D Twiss ')
plt.plot(zeta_values, y_disps*1000, '--', label='Analytical Crabbing', color = 'violet')
# plt.plot(np.sort(zeta_0), 
#          (ys_0[np.argsort(zeta_0)]-twiss['dy'][0]*delta_0[np.argsort(zeta_0)])*1000, 
#          '.', color = 'orange', alpha = 0.2, label=r'y[mm] from Xtrack')
plt.axvline(x=sigma_z, color='r', linestyle='--', label=r'$+1\sigma_{z}$')
plt.axvline(x=-sigma_z, color='r', linestyle='--', label=r'$-1\sigma_{z}$')
plt.axvline(x=3*sigma_z, color='k', linestyle='--', label=r'$+3\sigma_{z}$')
plt.axvline(x=-3*sigma_z, color='k', linestyle='--', label=r'$-3\sigma_{z}$')
plt.legend(loc='upper right', prop={'size': 16})
plt.xlabel(r'zeta [m]')
plt.ylabel(r'y [mm]')
plt.grid()
plt.title('Crabbing Dispersion in the SPS at the BWS location, 270 GeV', fontsize=15)

# %%
# %%
fig, ax = plt.subplots(figsize=(10, 10))
plt.plot(twiss['muy'], twiss['dy_zeta']/np.sqrt(twiss['bety']), 'green', label=r'$D_{y_{cc}}$(z), 6D Twiss ')


plt.xlabel(r'$\mu_{y}$')
plt.ylabel(r'$\frac{dy}{dz}\times\frac{1}{\sqrt{\beta_{y}}}$',rotation=0)
plt.axvline(twiss['muy','cravity.1'], label = 'Crab Cavity Location')
plt.grid()
plt.legend(loc='upper right', prop={'size': 10})
plt.title('Crabbing Dispersion in the SPS, 270 GeV', fontsize=15)

# %%
# Generate the bunch

# particles = xp.generate_matched_gaussian_bunch(
#                 num_particles=N_particles, total_intensity_particles=bunch_intensity,
#                 nemitt_x=normal_emitt_x, nemitt_y=normal_emitt_y, sigma_z=sigma_z,
#                 particle_ref=particle_ref, line=line)

# xs_0 = ctx.nparray_from_context_array(particles.x)
# pxs_0 = ctx.nparray_from_context_array(particles.px)
# ys_0 = ctx.nparray_from_context_array(particles.y)
# pys_0 = ctx.nparray_from_context_array(particles.py)
# delta_0 = ctx.nparray_from_context_array(particles.delta)
# zeta_0 = ctx.nparray_from_context_array(particles.zeta)
# states_0 = ctx.nparray_from_context_array(particles.state)
# p0_0 = ctx.nparray_from_context_array(particles.p0c[0])


# %%
interp_xs = interp1d(zetas, xs, kind='cubic')
interp_pxs = interp1d(zetas, pxs, kind='cubic')
interp_ys = interp1d(zetas, ys, kind='cubic')
interp_pys = interp1d(zetas, pys, kind='cubic')



# %%
# Define the bunch
N_particles = 50
x_in_sigmas = 0.
px_in_sigmas = 0.
y_in_sigmas = 0.
py_in_sigmas = 0.
delta_0 = 0.
zeta_0 = np.linspace(-5*sigma_z, 5*sigma_z, N_particles)
y0s= interp_ys(zeta_0)
py0s = interp_pys(zeta_0)
# y0s = np.ones(N_particles)*sigma_y

# interpolate pys



particles = line.build_particles(
                               y = y0s, py=py0s,
                               zeta=zeta_0, delta=delta_0,
                               particle_ref=particle_ref)


# particles = line.build_particles(
#             zeta=zeta_0, delta=delta_0,
#             x_norm=x_in_sigmas, px_norm=px_in_sigmas,
#             y_norm=y_in_sigmas, py_norm=py_in_sigmas,
#             nemitt_x=normal_emitt_x, nemitt_y=normal_emitt_y, particle_ref=particle_ref)

xs_0 = ctx.nparray_from_context_array(particles.x)
pxs_0 = ctx.nparray_from_context_array(particles.px)
ys_0 = ctx.nparray_from_context_array(particles.y)
pys_0 = ctx.nparray_from_context_array(particles.py)
delta_0 = ctx.nparray_from_context_array(particles.delta)
zeta_0 = ctx.nparray_from_context_array(particles.zeta)
states_0 = ctx.nparray_from_context_array(particles.state)
p0_0 = ctx.nparray_from_context_array(particles.p0c[0])

line.track(particles, num_turns=10, freeze_longitudinal=True)
# %%
dy = twiss['dy'][0]
dpy = twiss['dpy'][0]
fig, ax = plt.subplots(figsize=(10, 10))

plt.plot(np.sort(zeta_0), (ys_0)*1000, 'o', color = 'black', alpha = 0.5, label=r'y[mm] from Xtrack')

plt.plot(zeta_values, ys*1000, 'cyan', label=r'y[mm] from 4D Twiss')

plt.plot(zeta_values, y_disps*1000, '--', label='Analytical Dispersion', color = 'violet')

plt.plot(zeta_values, twiss['dy_zeta'][0]*zeta_values*1000, 'green', label=r'$D_{y_{cc}}$(z)*z[mm], 4D Twiss ')

plt.axvline(x=sigma_z, color='r', linestyle='--', label=r'$+1\sigma_{z}$')
plt.axvline(x=-sigma_z, color='r', linestyle='--', label=r'$-1\sigma_{z}$')
plt.axvline(x=3*sigma_z, color='k', linestyle='--', label=r'$+3\sigma_{z}$')
plt.axvline(x=-3*sigma_z, color='k', linestyle='--', label=r'$-3\sigma_{z}$')
plt.legend(loc='upper right', prop={'size': 16})
plt.xlabel(r'zeta [m]')
plt.ylabel(r'y [mm]')
plt.grid()
plt.title('Crabbing Dispersion in the SPS at the BWS location, 270 GeV', fontsize=15)


# %%

bunch_intensity = 3.0e10
N_particles = 100000
# x = np.random.normal(0, sigma_x, N_particles)
# px = np.random.normal(0, sigma_x, N_particles)
# y = np.random.normal(0, sigma_y, N_particles)
# py = np.random.normal(0, sigma_y, N_particles)
# zeta,delta = xp.generate_longitudinal_coordinates(
#         num_particles=N_particles, distribution='gaussian',
#         sigma_z = sigma_z, line=line, particle_ref=particle_ref)

particles = xp.generate_matched_gaussian_bunch(
                num_particles=N_particles, total_intensity_particles=bunch_intensity,
                nemitt_x=normal_emitt_x, nemitt_y=normal_emitt_y, sigma_z=sigma_z,
                particle_ref=particle_ref, line=line)
line.track(particles, num_turns=1)
# %%
xs_0 = ctx.nparray_from_context_array(particles.x)
pxs_0 = ctx.nparray_from_context_array(particles.px)
ys_0 = ctx.nparray_from_context_array(particles.y)
pys_0 = ctx.nparray_from_context_array(particles.py)
delta_0 = ctx.nparray_from_context_array(particles.delta)
zeta_0 = ctx.nparray_from_context_array(particles.zeta)
states_0 = ctx.nparray_from_context_array(particles.state)
p0_0 = ctx.nparray_from_context_array(particles.p0c[0])

# y_new = ys_0-interp_ys(zeta_0)
# py_new = pys_0-interp_pys(zeta_0)

# particles = line.build_particles(
#                                x=xs_0, px=pxs_0,
#                                y = y_new, py=py_new,
#                                zeta=zeta_0, delta=delta_0,
#                                particle_ref=particle_ref, 
#                                nemitt_x=normal_emitt_x,
#                                nemitt_y=normal_emitt_y,)

# %%

# %%
xs_1 = ctx.nparray_from_context_array(particles.x)
pxs_1 = ctx.nparray_from_context_array(particles.px)
ys_1 = ctx.nparray_from_context_array(particles.y)
pys_1 = ctx.nparray_from_context_array(particles.py)
delta_1 = ctx.nparray_from_context_array(particles.delta)
zeta_1 = ctx.nparray_from_context_array(particles.zeta)
dx = twiss['dx'][0]
dpx = twiss['dpx'][0]
dy = twiss['dy'][0]
dpy = twiss['dpy'][0]
states_1 = ctx.nparray_from_context_array(particles.state)
p0_1 = ctx.nparray_from_context_array(particles.p0c[0])

# %%
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

ex = emittance(xs_1,pxs_1,zeta_1,delta_1,dx,dpx,interp_xs,interp_pxs)*particle_ref.gamma0*particle_ref.beta0
ey = emittance(ys_1,pys_1,zeta_1,delta_1,dy,dpy,interp_ys,interp_pys)*particle_ref.gamma0*particle_ref.beta0
print(f'Ex = {ex}', f'Ey = {ey}')

# %%



# %%

















# %
particles = xp.generate_matched_gaussian_bunch(
                num_particles=N_particles, total_intensity_particles=bunch_intensity,
                nemitt_x=normal_emitt_x, nemitt_y=normal_emitt_y, sigma_z=sigma_z,
                particle_ref=particle_ref, line=line)
xs_ = ctx.nparray_from_context_array(particles.x)
pxs_ = ctx.nparray_from_context_array(particles.px)
ys_ = ctx.nparray_from_context_array(particles.y)
pys_ = ctx.nparray_from_context_array(particles.py)
delta_ = ctx.nparray_from_context_array(particles.delta)
zeta_ = ctx.nparray_from_context_array(particles.zeta)
states_ = ctx.nparray_from_context_array(particles.state)

line.track(particles, num_turns=10)
# %%

#Normalize particles in the SPS at 270 GeV
part_on_co = line.find_closed_orbit(particle_ref=particle_ref)
gemitt_x = normal_emitt_x/part_on_co.beta0/part_on_co.gamma0
gemitt_y = normal_emitt_y/part_on_co.beta0/part_on_co.gamma0
ew_given = 0.6124693221123673
c_light = 299792458
converted_ew = ew_given/(part_on_co.gamma0[0]/part_on_co.beta0[0]/1e3*1e12)*c_light/(4*np.pi)
gemitt_z = converted_ew
v = np.array([xs_/np.sqrt(gemitt_x), pxs_/np.sqrt(gemitt_x), ys_/np.sqrt(gemitt_y), pys_/np.sqrt(gemitt_y), zeta_/np.sqrt(gemitt_z), delta_/np.sqrt(gemitt_z)])
W = twiss['W_matrix'][0]
W_inv = np.linalg.inv(W)
norm_coord = np.dot(W_inv, v)

#figure size 
plt.figure(figsize=(10,10))
plt.plot(norm_coord[0],norm_coord[1],'o')
plt.plot(norm_coord[2],norm_coord[3],'o')
plt.plot(norm_coord[4],norm_coord[5],'o')
plt.xlabel('x,y,zeta')
plt.ylabel('px,py,delta')
plt.title('Normalized Coordinates')
plt.grid()

# %%
#Going back to physical coordinates and check again
phys_coord = np.dot(W, norm_coord)
plt.figure(figsize=(10,10))
phys_coord[0] = phys_coord[0]*np.sqrt(gemitt_x)
phys_coord[1] = phys_coord[1]*np.sqrt(gemitt_x)
phys_coord[2] = phys_coord[2]*np.sqrt(gemitt_y)
phys_coord[3] = phys_coord[3]*np.sqrt(gemitt_y)
phys_coord[4] = phys_coord[4]*np.sqrt(gemitt_z)
phys_coord[5] = phys_coord[5]*np.sqrt(gemitt_z)

plt.plot(phys_coord[4],phys_coord[5],'o')
plt.plot(phys_coord[0],phys_coord[1],'o')
plt.plot(phys_coord[2],phys_coord[3],'o')
plt.grid()




# %%





# %%





norm_coord = np.dot(W_inv, v)

# %% 
#create a 6x6 Symplectic matrix
S = np.array([[0., 1., 0., 0., 0., 0.],
              [-1., 0., 0., 0., 0., 0.],
              [ 0., 0., 0., 1., 0., 0.],
              [ 0., 0.,-1., 0., 0., 0.],
              [ 0., 0., 0., 0., 0., 1.],
              [ 0., 0., 0., 0.,-1., 0.]])

#put to zero all the elements of the matrix that are smaller than 1e-15
wt_s_w = np.matmul(W.transpose(),np.matmul(S,W))
wt_s_w[np.abs(wt_s_w) < 1e-6] = 0

print(np.matmul(W_inv,v))



# %%




# %%
zzz = np.linspace(-5*sigma_z, 5*sigma_z, 100)
first_factor = np.sqrt(betas)/(2*np.sin(2*np.pi*Qy))
Dp1 = np.sin(2*np.pi*f_crab*zzz/c_light)*ksl
second_factor = Dp1*np.cos(Delta_phi-2*np.pi*Qy)
plt.plot(zzz, first_factor*second_factor, '-')
plt.grid()


# %%


names = line.element_names
# find the elements containing 'bws'
bws_names = [name for name in names if 'bws' in name]

# theta_cc = ksl*np.sin(f_crab*zeta_1/c_light+np.pi/2)
# Dp1 = np.sin(f_crab*zeta_1/c_light)*ksl
# second_factor = Dp1*np.cos(Delta_phi-2*np.pi*Qy)
# y_disp = first_factor*second_factor


# %%


# %%

def emittance(x,px,delta,dx,dpx):
    x_np = np.array(x)
    px_np= np.array(px)
    x_np = np.array(x - delta*dx)
    px_np = np.array(px - delta*dpx)
    
    x_mean_2 = np.dot(x_np,x_np)/len(x)
    px_mean_2 = np.dot(px_np,px_np)/len(x)
    x_px_mean = np.dot(x_np,px_np)/len(x)

    emitt = np.sqrt(x_mean_2*px_mean_2-(x_px_mean)**2)

    return emitt

ex = emittance(xs_1,pxs_1,delta_1,dx,dpx)*particle_ref.gamma0*particle_ref.beta0
ey = emittance(ys_1,pys_1,delta_1,dy,dpy)*particle_ref.gamma0*particle_ref.beta0
print(f'Ex = {ex}', f'Ey = {ey}')



# %%

# %%
# %%
N_particles = 20000
bunch_intensity = 3.0e10
particles = xp.generate_matched_gaussian_bunch(
                num_particles=N_particles, total_intensity_particles=bunch_intensity,
                nemitt_x=normal_emitt_x, nemitt_y=normal_emitt_y, sigma_z=sigma_z,
                particle_ref=particle_ref, line=line)
# %%

betx = twiss['betx'][0]
alfx = twiss['alfx'][0]
bety = twiss['bety'][0]
alfy = twiss['alfy'][0]
dx = twiss['dx'][0]
dpx = twiss['dpx'][0]
dy = twiss['dy'][0]
dpy = twiss['dpy'][0]
dy_dzeta = twiss['dy_zeta'][0]
dmuy_dzeta = twiss['dmuy'][0]
alphy_CC = twiss['alfy','cravity.1']
bety_CC = twiss['bety','cravity.1']


# bpm = 'bph.40808'
# %%

xs_ = ctx.nparray_from_context_array(particles.x)
pxs_ = ctx.nparray_from_context_array(particles.px)
ys_ = ctx.nparray_from_context_array(particles.y)
pys_ = ctx.nparray_from_context_array(particles.py)
delta_ = ctx.nparray_from_context_array(particles.delta)
zeta_ = ctx.nparray_from_context_array(particles.zeta)
states_ = ctx.nparray_from_context_array(particles.state)
p0_ = ctx.nparray_from_context_array(particles.p0c[0])

xs_1 = np.array(xs_)
pxs_1 = np.array(pxs_)
ys_1= np.array(ys_)
pys_1 = np.array(pys_)
zeta_1 = np.array(zeta_)
delta_1 = np.array(delta_)
states_1 = np.array(states_)
part_on_co = line.find_closed_orbit(particle_ref=particle_ref)

betas = twiss['bety'][0]*bety_CC
Qy = twiss['qy']
Delta_phi = (twiss['muy','cravity.1'] - twiss['muy'][0])*np.pi*2
first_factor = np.sqrt(betas)/(2*np.sin(2*np.pi*Qy))
f_crab = line.element_dict['cravity.1'].frequency
R11 = np.sqrt(twiss['bety'][0]/bety_CC)*(np.cos(Delta_phi)+alphy_CC*np.sin(Delta_phi))
R12 = np.sqrt(bety*bety_CC)*np.sin(Delta_phi)
R21 = ((alphy_CC-alfy)*np.cos(Delta_phi)/np.sqrt(bety*bety_CC)-(1+alfy*alphy_CC)*np.sin(Delta_phi)/(np.sqrt(bety*bety_CC)))
R22 = np.sqrt(bety/bety_CC)*(np.cos(Delta_phi)-alfy*np.sin(Delta_phi))
R = np.array([[R11, R12], [R21, R22]])
theta_cc = ksl*np.sin(2*np.pi*f_crab*zeta_1/c_light+np.pi/2)
Crabbing = R11*theta_cc + R22*theta_cc*zeta_1
#plot the y_disp along the machine
#increase the size of the plot
plt.figure(figsize=(10,8))
# plt.plot(twiss['s'], y_disp, '-', label='Analytical')
plt.plot(twiss['s'], twiss['dy'], '-', label='Xsuite, simple dispersion')
plt.plot(twiss['s'], twiss['dy_zeta'], '-', label='Xsuite, CC dispersion')
plt.xlabel('s [m]')
plt.ylabel(r'$X_{D_{cc}}$ [rad]', rotation=0)
plt.title('Crabbing Dispersion along the lattice')
plt.legend()
plt.grid()
plt.xlim(1000,4500)

# %%
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
tracker = line.build_tracker()
twiss = line.twiss(particle_ref)

# %%
N_particles = 20000
bunch_intensity = 3.0e10
particles_gauss = xp.generate_matched_gaussian_bunch(
                num_particles=N_particles, total_intensity_particles=bunch_intensity,
                nemitt_x=normal_emitt_x, nemitt_y=normal_emitt_y, sigma_z=sigma_z,
                particle_ref=particle_ref, line=line)
dx = twiss['dx'][0]
dpx = twiss['dpx'][0]
dy = twiss['dy'][0]
dpy = twiss['dpy'][0]

# %%
ctx = xo.ContextCpu()
xs_ = ctx.nparray_from_context_array(particles_gauss.x)
pxs_ = ctx.nparray_from_context_array(particles_gauss.px)
ys_ = ctx.nparray_from_context_array(particles_gauss.y)
pys_ = ctx.nparray_from_context_array(particles_gauss.py)
delta_ = ctx.nparray_from_context_array(particles_gauss.delta)
states_ = ctx.nparray_from_context_array(particles_gauss.state)
p0_ = ctx.nparray_from_context_array(particles_gauss.p0c[0])

xs_1 = np.array(xs_)
pxs_1 = np.array(pxs_)
ys_1= np.array(ys_)
pys_1 = np.array(pys_)
delta_1 = np.array(delta_)
states_1 = np.array(states_)

ex = emittance(xs_1,pxs_1,delta_1,dx,dpx)*particle_ref.gamma0*particle_ref.beta0
ey = emittance(ys_1,pys_1,delta_1,dy,dpy)*particle_ref.gamma0*particle_ref.beta0
print(f'Ex = {ex}', f'Ey = {ey}')

# %%
plt.plot(ys_1, pys_1, '.')

# %%
y_rms = np.std(ctx.nparray_from_context_array(particles.y))
py_rms = np.std(ctx.nparray_from_context_array(particles.py))
x_rms = np.std(ctx.nparray_from_context_array(particles.x))
px_rms = np.std(ctx.nparray_from_context_array(particles.px))
delta_rms = np.std(ctx.nparray_from_context_array(particles.delta))
zeta_rms = np.std(ctx.nparray_from_context_array(particles.zeta))
part_on_co = line.find_closed_orbit(particle_ref=particle_ref)
part_on_co.move(_context=xo.ContextCpu())
gemitt_x = normal_emitt_x/part_on_co.beta0/part_on_co.gamma0
gemitt_y = normal_emitt_y/part_on_co.beta0/part_on_co.gamma0
assert np.isclose(zeta_rms, sigma_z, rtol=1e-2, atol=1e-15)
assert np.isclose(x_rms,
                    np.sqrt(twiss['betx'][0]*gemitt_x + twiss['dx'][0]**2*delta_rms**2),
                    rtol=1e-2, atol=1e-15)
assert np.isclose(y_rms,
                    np.sqrt(twiss['bety'][0]*gemitt_y + twiss['dy'][0]**2*delta_rms**2),
                    rtol=1e-2, atol=1e-15)

# %%
