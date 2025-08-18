#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 12:13:31 2025

@author: jessicafranks

Description: ADCP SWC Data - PISCO --> Determime average current speeds based on
height above the bottom
"""

#%% Essentials 
import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime, timedelta
import urllib

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


#%% Import the data 
adcp = pd.read_csv('/Users/jessicafranks/Desktop/grad_school/grad_thesis/thesis-python/PISCO-adcp/PISCO-SWC-ADCP.txt', sep ='\s+', header = 0)

#%% Use only good data AKA where flag == 1
adcp_qc = adcp.where(
    (adcp.flag == 0) & (adcp.eatward != 9999) & (adcp.northward != 9999)
)

#%% Check eastward and northward currents
adcp_qc.eatward

#%% get current speeds from east and north velocities
adcp_qc['currents'] = np.sqrt((adcp_qc.eatward**2) + (adcp_qc.northward**2))
# this uses the equation: current speed = sqrt(east velocity^2 + north velocity ^2)

#%% attempt to get current values for binned depths

# Get time-averaged currents per depth bin and keep the depth column
adcp_qc_avg = adcp_qc.groupby('height')['currents'].mean().reset_index()

# Check the result (this should be a DataFrame with 'height' and 'currents')
print(adcp_qc_avg.head())

#%% Plot depth(bindist) vs time averaged currents 
ds = adcp_qc_avg

plt.figure(figsize = (10,10))
plt.title('Time Averaged currents at binned depth\n Deep site- PISCO ADCP')
plt.plot(ds.currents, ds.height) #plot currents vs height
plt.ylabel('Height above the bottom [m]')
plt.xlabel('Current speeds [m/s]')
plt.show()
