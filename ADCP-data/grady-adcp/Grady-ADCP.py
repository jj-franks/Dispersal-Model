#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 12:08:43 2025

@author: jessicafranks

Description: ADCP SWC Data - Grady --> Determime average current speeds based on
height above the bottom
"""
#%% Essentials 
import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime, timedelta

#%% Plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors
from matplotlib.colors import ListedColormap, BoundaryNorm, TwoSlopeNorm
import matplotlib.ticker as ticker
from matplotlib import patches
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from scipy.interpolate import RegularGridInterpolator


#%%

adcp1 = xr.open_dataset("/Users/jessicafranks/Desktop/grad_school/grad_thesis/thesis-python/grady-adcp/adcp1_final.nc")

#%% Only use data with a flag == 1, meaning the data is useable and not uncertain or suspicious (2,3,4)
adcp1_qc = adcp1.where(adcp1.Flag == 1) ## .where subsets the data using logical array

#%%
adcp1_qc.bindist

#%% Monitor full vector profile colorplots to check that remaining data is good quality
ds = adcp1_qc.copy(deep=True)

plt.figure(figsize = (15,10))

plt.subplot(211)
bounds = np.array([1, 2, 3, 4])
norm = matplotlib.colors.BoundaryNorm(boundaries=bounds, ncolors=256)
plt.title('Flag')
plt.pcolormesh(ds.time, ds.bindist, ds.Flag.values,norm=norm)
plt.plot(ds.time, ds.Depth, '-k')
plt.colorbar(label='Flag')
#plt.ylim(1.25,12)
plt.ylabel('Depth [MOB]')
plt.xticks(rotation= 15)

plt.subplot(212)
plt.title('Eastern Velocity')
plt.pcolormesh(ds.time, ds.bindist, ds.East.where(ds.Flag <= 4),vmin=-.25, vmax = .25)
plt.plot(ds.time, ds.Depth, '-k')
plt.colorbar(label='Eastern velocity [m/s]')
#plt.ylim(1.25,12)
plt.ylabel('Depth [MOB]')
plt.xlabel('Datetime')
plt.xticks(rotation= 15)

#%% get current speeds from east and north velocities
adcp1_qc['currents'] = np.sqrt((adcp1_qc.East**2) + (adcp1_qc.North**2))
# this uses the equation: current speed = sqrt(east velocity^2 + north velocity ^2)


#%% attempt to get current values for binned depths

adcp1_qc['currentsTA'] = adcp1_qc.currents.where(adcp1_qc.currents.notnull()).mean(dim='time')
adcp1_qc.currentsTA

#%% Plot depth(bindist) vs time averaged currents 
ds = adcp1_qc

plt.figure(figsize = (10,10))
plt.title('Time Averaged currents at binned depth\n Shallow site- Grady ADCP')
plt.plot(ds.currentsTA, ds.bindist)
plt.ylabel('Height above the bottom [m]')
plt.xlabel('Current speeds [m/s]')

#%%
# == Calculate reynold's numbers for largest plants in deep and in shallow ==
kv = 1.22*10^-6 #m^2/s^-1
# deep
re_deep = ( max_current * diameter ) / kv #kv = kinematic velocity of seawater

#%%

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

# === Parameters (Customize These) ===
n_particles = 1000
release_height = 3.0  # meters above bottom
plant_depth = 4.0 # depth
sinking_speed = 0.000356  # m/s
bottom_roughness = 0.08  # m
Kz = 1e-4  # vertical diffusivity m^2/s
dt = 1000  # seconds
settle_fraction = 0.9  # stop when 90% of particles have settled

# === Wave parameters ===
wave_height = 0.5  # H, in meters
wave_period = 8.0  # T, in seconds
plant_depth = 12.75  # m (deepest bindist, water column depth)
g = 9.81  # gravity

# Estimate wavelength using dispersion relation
L = (g * wave_period**2) / (2 * np.pi) * np.tanh((2 * np.pi * plant_depth) / (wave_period**2 * g))
k = 2 * np.pi / L  # wavenumber

# Surface Stokes velocity
U_s = (np.pi * wave_height**2) / (2 * wave_period * np.sinh(k * plant_depth)**2)

def stokes_drift(z):
    """Stokes drift decays with depth."""
    return U_s * np.exp(-2 * k * z)

# === Load ADCP ===
ds = xr.open_dataset("/Users/jessicafranks/Desktop/grad_school/grad_thesis/thesis-python/grady-adcp/adcp1_final.nc")

# Filter by QC
valid = ds.Flag == 1
east = ds.East.where(valid)
north = ds.North.where(valid)

# Get coordinate values
bindist = ds.bindist.values  # m above bottom
time = ds.time.values
time_sec = (time - time[0]) / np.timedelta64(1, 's')  # seconds since start

# Transpose to (time, bindist)
u_data = east.transpose("time", "bindist").values
v_data = north.transpose("time", "bindist").values

# Interpolators (time, height above bottom)
u_interp = RegularGridInterpolator((time_sec, bindist), u_data, bounds_error=False, fill_value=0)
v_interp = RegularGridInterpolator((time_sec, bindist), v_data, bounds_error=False, fill_value=0)

# === Particle Class ===
class Particle:
    def __init__(self, z=release_height):
        self.x = 0.0 # initial x position
        self.y = 0.0 # initial y position
        self.z = z  # initial Z height above bottom
        self.settled = False #initially not settled

# === Initialize Particles ===
particles = [Particle() for _ in range(n_particles)]
positions = []

# === Simulation Loop ===
t = 0
settled_count = 0
max_time_steps = len(time_sec)  # to avoid running past ADCP data

def run_simulation(release_height, plant_depth):
    particles = [Particle(z=release_height) for _ in range(n_particles)]
    positions = []
    t = 0
    settled_count = 0
    max_time_steps = len(time_sec)  # ensure within ADCP time range
    
    print (f"Max time steps: {max_time_steps}")
    
    while settled_count < settle_fraction * n_particles and t < max_time_steps - 1:
        t_sec = time_sec[t]
        settled_count = 0
        current_positions = []
        
        print(f"Time step {t}: Settled count = {settled_count}, Total particles = {n_particles}")

        for p in particles:
            if not p.settled: # only track non-settled particles
                # Debugging: Track vertical movement
                print(f"Particle z before sinking: {p.z}")
                
                # Horizontal movement (advection)
                u = u_interp((t_sec, p.z))
                v = v_interp((t_sec, p.z))

                # add stokes drift to eastward velocity
                u += stokes_drift(p.z)

                # Horizontal movement (advection)
                p.x += u * dt
                p.y += v * dt

                # Vertical movement (sinking + diffusion)
                dz_sink = sinking_speed * dt
                dz_diff = np.random.normal(0, np.sqrt(2 * Kz * dt))
                p.z -= dz_sink  # sinking reduces height above bottom
                p.z += dz_diff
                
                print(f"Particle z after sinking: {p.z}")

                # Bottom interaction
                if p.z <= bottom_roughness:
                    p.z = 0 # it settles at the bottom
                    p.settled = True # mark as settled
                    settled_count += 1  #increases the settled count

            # Record the particle's position (whether settled or not)
            current_positions.append((p.x, p.y, p.z))

        # Add positions for this time step
        positions.append(current_positions)
        t += 1 # Move to the next time step
        
    # Ensure positions have been populated before accessing
    if len(positions) > 0:
        final_positions = np.array(positions[-1])
    else:
        print("No positions recorded!")
        final_positions = np.array([]) # Handle case with empty positions
        
    # return final positions or trajectories
    return final_positions
    
release_heights = [1, 3, 5]  # meters above bottom
plant_depths = [12.0, 12.75]  # m depth

results = {}

for r in release_heights:
    for pd in plant_depths:
        label = f"r{r}_pd{pd}"
        # run_simulation is your full particle model (refactored into a function)
        trajectories = run_simulation(release_height=r, plant_depth=pd)
        results[label] = trajectories

# === Plot Final Positions ===
final = np.array(positions[-1])
x, y, z = final[:, 0], final[:, 1], final[:, 2]

plt.figure(figsize=(8,6))
plt.scatter(x, y, c=z, cmap='viridis')
plt.colorbar(label="Height above bottom (m)")
plt.xlabel("Eastward Displacement (m)")
plt.ylabel("Northward Displacement (m)")
plt.title("Final Particle Positions")
plt.grid()
plt.show()



#%%
# == Dispersal kernel stuff ==
from scipy.stats import gaussian_kde

# Get final positions
final_x = np.array([p.x for p in particles])
final_y = np.array([p.y for p in particles])

# Compute KDE (kernel density estimate)
xy = np.vstack([final_x, final_y])
kde = gaussian_kde(xy)

# Grid for plotting
xgrid, ygrid = np.meshgrid(
    np.linspace(final_x.min()-10, final_x.max()+10, 100),
    np.linspace(final_y.min()-10, final_y.max()+10, 100)
)
positions = np.vstack([xgrid.ravel(), ygrid.ravel()])
density = kde(positions).reshape(xgrid.shape)

# Plot
plt.figure(figsize=(8,6))
plt.contourf(xgrid, ygrid, density, cmap='viridis')
plt.colorbar(label="Kernel density")
plt.scatter(final_x, final_y, c='red', s=5, label='Particles')
plt.title("Dispersal Kernel")
plt.xlabel("East (m)")
plt.ylabel("North (m)")
plt.legend()
plt.grid()
plt.show()









