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
plt.title('Time Averaged currents at binned depth (ADCP)')
plt.plot(ds.currentsTA, ds.bindist)
plt.ylabel('Depth [m]')
plt.xlabel('Current speeds [m/s]')
