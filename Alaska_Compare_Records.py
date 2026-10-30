#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 10:24:27 2026

@author: kellymckeon
"""

import time
code_start = time.time() #begin timing code

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42 
rng = np.random.default_rng()
from scipy.stats import variation

mac = r"/Users/kellymckeon/OneDrive - Woods Hole Oceanographic Institution/Documents/WHOI/Alaska/Codes/Coherent_Codes/Compilation_Codes"
pc = r"C:/Users/kelly/OneDrive - Woods Hole Oceanographic Institution/Documents/WHOI/Alaska/Codes/Coherent_Codes/Compilation_Codes"

# CHANGE DIRECTORY TO WHICHEVER COMPUTER YOU ARE ON
os.chdir(mac)


#%% Load Grainsize Data

#### JPC05
#raw data
JPC05 = np.genfromtxt(r"JPC05_CoreData/JPC05C_GS.csv", delimiter=",", skip_header=1)
JPC05_GS_depth = JPC05[:, 0]
JPC05_d50 = JPC05[:,2]
JPC05_d90 = JPC05[:,3]

#grainsize threshold
JPC05_thresh = np.genfromtxt(r"JPC05_CoreData/threshold.csv", delimiter=",", skip_header=0)
JPC05_movmean = np.genfromtxt(r"JPC05_CoreData/moving_mean.csv", delimiter=",", skip_header=0)

#events
JPC05_event_gs = np.genfromtxt(r"JPC05_CoreData/event_gs.csv", delimiter=",", skip_header=0)
JPC05_event_depth = np.genfromtxt(r"JPC05_CoreData/event_depth.csv", delimiter=",", skip_header=0)


#################################################

#### VC23
#raw data
VC23 = np.genfromtxt(r"VC23_CoreData/VC23_Composite_GS.csv", delimiter=",", skip_header=1)
VC23_GS_depth = VC23[:, 2]
VC23_d50 = VC23[:,4]
VC23_d90 = VC23[:,5]

#grainsize threshold
VC23_thresh = np.genfromtxt(r"VC23_CoreData/threshold.csv", delimiter=",", skip_header=0)
VC23_movmean = np.genfromtxt(r"VC23_CoreData/moving_mean.csv", delimiter=",", skip_header=0)

#events
VC23_event_gs = np.genfromtxt(r"VC23_CoreData/event_gs.csv", delimiter=",", skip_header=0)
VC23_event_depth = np.genfromtxt(r"VC23_CoreData/event_depth.csv", delimiter=",", skip_header=0)


#################################################

#### VC11
#raw data
VC11 = np.genfromtxt(r"VC11_CoreData/VC11_Composite_GS_71cm.csv", delimiter=",", skip_header=1)
VC11_GS_depth = VC11[:, 3]
VC11_d90 = VC11[:,6]
VC11_d50 = VC11[:,5]

#grainsize threshold
VC11_thresh = np.genfromtxt(r"VC11_CoreData/threshold.csv", delimiter=",", skip_header=0)
VC11_movmean = np.genfromtxt(r"VC11_CoreData/moving_mean.csv", delimiter=",", skip_header=0)

#events
VC11_event_gs = np.genfromtxt(r"VC11_CoreData/event_gs.csv", delimiter=",", skip_header=0)
VC11_event_depth = np.genfromtxt(r"VC11_CoreData/event_depth.csv", delimiter=",", skip_header=0)

#################################################




#%% Load Age Data

#### JPC05
#median ages
JPC05_ages_med = np.genfromtxt(r"JPC05_CoreData/median_ages.csv", delimiter=",", skip_header=0)
JPC05_event_times = np.genfromtxt(r"JPC05_CoreData/event_times.csv", delimiter=",", skip_header=0)
#distributions
age_dist = np.genfromtxt(r"JPC05_CoreData/JPC05_Age_Distributions.csv", delimiter=",", skip_header=1)
age_dist = age_dist[:,1:]

#full age model
ages = np.genfromtxt(r"JPC05_CoreData/JPC05C_Ages.txt", skip_header=1)
ages_depth = ages[:,0]
ind = np.isin(ages_depth, JPC05_GS_depth)
#get ages 
JPC05_ages_mean = ages[ind,4]
JPC05_ages_min = ages[ind,1]
JPC05_ages_max = ages[ind,2]

#c14
JPC05_c14 = np.genfromtxt(r"JPC05_CoreData/JPC05_Calibrated.csv", delimiter=",", skip_header=1)
JPC05_c14_depth = JPC05_c14[:,0]
JPC05_c14_med = JPC05_c14[:,1]
JPC05_c14_min = JPC05_c14[:,2]
JPC05_c14_max = JPC05_c14[:,3]

#pb210
JPC05_pb210 = np.genfromtxt(r"JPC05_CoreData/JPC05_Pb210.csv", delimiter=",", skip_header=1)
JPC05_pb210_depth = JPC05_pb210[:,0]
JPC05_pb210_val = JPC05_pb210[:,1]
JPC05_pb210_err = JPC05_pb210[:,2]

#################################################

#### VC23
#median ages
VC23_ages_med = np.genfromtxt(r"VC23_CoreData/median_ages.csv", delimiter=",", skip_header=0)
VC23_event_times = np.genfromtxt(r"VC23_CoreData/event_times.csv", delimiter=",", skip_header=0)
#distributions
age_dist = np.genfromtxt(r"VC23_CoreData/VC23_Age_Distributions.csv", delimiter=",", skip_header=1)
age_dist = age_dist[:,1:]

#full age model
ages = np.genfromtxt(r"VC23_CoreData/VC23_ages.txt", skip_header=1)
ages_depth = ages[:,0]
ind = np.isin(ages_depth, VC23_GS_depth)
#get ages 
VC23_ages_mean = ages[ind,4]
VC23_ages_min = ages[ind,1]
VC23_ages_max = ages[ind,2]

#c14
VC23_c14 = np.genfromtxt(r"VC23_CoreData/VC23_Calibrated.csv", delimiter=",", skip_header=1)
VC23_c14_depth = VC23_c14[:,0]
VC23_c14_med = VC23_c14[:,1]
VC23_c14_min = VC23_c14[:,2]
VC23_c14_max = VC23_c14[:,3]

#pb210
VC23_pb210 = np.genfromtxt(r"VC23_CoreData/VC23_Pb210.csv", delimiter=",", skip_header=1)
VC23_pb210_depth = VC23_pb210[:,0]
VC23_pb210_val = VC23_pb210[:,1]
VC23_pb210_err = VC23_pb210[:,2]

#################################################

#### VC11
#median ages
VC11_ages_med = np.genfromtxt(r"VC11_CoreData/median_ages.csv", delimiter=",", skip_header=0)
VC11_event_times = np.genfromtxt(r"VC11_CoreData/event_times.csv", delimiter=",", skip_header=0)
#distributions
age_dist = np.genfromtxt(r"VC11_CoreData/VC11_Age_Distributions.csv", delimiter=",", skip_header=1)
age_dist = age_dist[:,1:]

#full age model
ages = np.genfromtxt(r"VC11_CoreData/VC11_Composite_Ages_71.txt", skip_header=1)
ages_depth = ages[:,0]
ind = np.isin(ages_depth, VC11_GS_depth)
#get ages 
VC11_ages_mean = ages[ind,4]
VC11_ages_min = ages[ind,1]
VC11_ages_max = ages[ind,2]

#c14
VC11_c14 = np.genfromtxt(r"VC11_CoreData/VC11_Calibrated.csv", delimiter=",", skip_header=1)
VC11_c14_depth = VC11_c14[:,0]
VC11_c14_med = VC11_c14[:,1]
VC11_c14_min = VC11_c14[:,2]
VC11_c14_max = VC11_c14[:,3]

#pb210
VC11_pb210 = np.genfromtxt(r"VC11_CoreData/VC11_Pb210.csv", delimiter=",", skip_header=1)
VC11_pb210_depth = VC11_pb210[:,0]
VC11_pb210_val = VC11_pb210[:,1]
VC11_pb210_err = VC11_pb210[:,2]

#################################################

#%% Load Density Data

#### JPC05
JPC05_Density = np.genfromtxt(r"JPC05_CoreData/JPC05_Density.csv", delimiter=",", skip_header=1)
JPC05_density_depth = JPC05_Density[:, 2]
JPC05_wet_density = JPC05_Density[:,3]
JPC05_dry_density = JPC05_Density[:,4]
#get ages of density depths
ind = np.isin(JPC05_GS_depth, JPC05_density_depth)
JPC05_density_ages = JPC05_ages_med[ind]


#### VC23
VC23_Density = np.genfromtxt(r"VC23_CoreData/VC23_Density.csv", delimiter=",", skip_header=1)
VC23_density_depth = VC23_Density[:, 2]
VC23_wet_density = VC23_Density[:,3]
VC23_dry_density = VC23_Density[:,4]
#get ages of density depths
ind = np.isin(VC23_GS_depth, VC23_density_depth)
VC23_density_ages = VC23_ages_med[ind]


#### VC11
VC11_Density = np.genfromtxt(r"VC11_CoreData/VC11_Density.csv", delimiter=",", skip_header=1)
VC11_density_depth = VC11_Density[:, 2]
VC11_wet_density = VC11_Density[:,3]
VC11_dry_density = VC11_Density[:,4]
#get ages of density depths
ind = np.isin(VC11_GS_depth, VC11_density_depth)
VC11_density_ages = VC11_ages_med[ind]


#### Get density moving averages

#JPC05
JPC05_wd_series = pd.Series(JPC05_wet_density)
JPC05_wet_density10 = JPC05_wd_series.rolling(window=5, min_periods=1, center=True).mean().to_numpy()
JPC05_dd_series = pd.Series(JPC05_dry_density)
JPC05_dry_density10 = JPC05_dd_series.rolling(window=5, min_periods=1, center=True).mean().to_numpy()

#VC23
VC23_wd_series = pd.Series(VC23_wet_density)
VC23_wet_density10 = VC23_wd_series.rolling(window=5, min_periods=1, center=True).mean().to_numpy()
VC23_dd_series = pd.Series(VC23_dry_density)
VC23_dry_density10 = VC23_dd_series.rolling(window=5, min_periods=1, center=True).mean().to_numpy()

#VC11
VC11_wd_series = pd.Series(VC11_wet_density)
VC11_wet_density10 = VC11_wd_series.rolling(window=5, min_periods=1, center=True).mean().to_numpy()
VC11_dd_series = pd.Series(VC11_dry_density)
VC11_dry_density10 = VC11_dd_series.rolling(window=5, min_periods=1, center=True).mean().to_numpy()

#%% Plot Grainsize Distributions

fig,ax = plt.subplots(2,3, figsize=(7,4), constrained_layout=True, dpi=150)
ax0 = ax[0,0]
ax1 = ax[0,1]
ax2 = ax[0,2]
ax3 = ax[1,0]
ax4 = ax[1,1]
ax5 = ax[1,2]

#### JPC05 d50
ax0.hist(JPC05_d50, bins=50, histtype='bar', ec='k', density=True)
ax0.set_xlabel(r'd50 ($\mu$m)', fontname="Arial", fontweight='bold')
ax0.set_ylabel('Density', fontname="Arial", fontweight='bold')
ax0.set_title('JPC05', fontname="Arial", fontweight='bold')
ax0.grid(color='lightgray', ls=':', lw=0.7, axis='both', which='major', zorder=0)

#### VC23 d50
ax1.hist(VC23_d50, bins=50, histtype='bar', ec='k', density=True)
ax1.set_xlabel(r'd50 ($\mu$m)', fontname="Arial", fontweight='bold')
ax1.set_ylabel('Density', fontname="Arial", fontweight='bold')
ax1.set_title('VC23', fontname="Arial", fontweight='bold')
ax1.grid(color='lightgray', ls=':', lw=0.7, axis='both', which='major', zorder=0)

#### VC11 d50
ax2.hist(VC11_d50, bins=50, histtype='bar', ec='k', density=True)
ax2.set_xlabel(r'd50 ($\mu$m)', fontname="Arial", fontweight='bold')
ax2.set_ylabel('Density', fontname="Arial", fontweight='bold')
ax2.set_title('VC11', fontname="Arial", fontweight='bold')
ax2.grid(color='lightgray', ls=':', lw=0.7, axis='both', which='major', zorder=0)

#### JPC05 d90
ax3.hist(JPC05_d90, bins=50, histtype='bar', ec='k',density=True)
ax3.set_xlabel(r'd90 ($\mu$m)', fontname="Arial", fontweight='bold')
ax3.set_ylabel('Density', fontname="Arial", fontweight='bold')
ax3.grid(color='lightgray', ls=':', lw=0.7, axis='both', which='major', zorder=0)
# ax3.set_title('JPC05_d90')

#### VC23 d90
ax4.hist(VC23_d90, bins=50, histtype='bar', ec='k',density=True)
ax4.set_xlabel(r'd90 ($\mu$m)', fontname="Arial", fontweight='bold')
ax4.set_ylabel('Density', fontname="Arial", fontweight='bold')
ax4.grid(color='lightgray', ls=':', lw=0.7, axis='both', which='major', zorder=0)
# ax4.set_title('VC23_d90')

#### VC11 d90
ax5.hist(VC11_d90, bins=50, histtype='bar',ec='k', density=True)
ax5.set_xlabel(r'd90 ($\mu$m)', fontname="Arial", fontweight='bold')
ax5.set_ylabel('Density', fontname="Arial", fontweight='bold')
ax5.grid(color='lightgray', ls=':', lw=0.7, axis='both', which='major', zorder=0)
# ax5.set_title('VC11_d90')

#%% Plot Raw Grainsize Timeseries of d50 

fig, ax = plt.subplots(3,1, figsize=(6,6), constrained_layout=True, dpi=150)
ax0 = ax[0]
ax1 = ax[1]
ax2 = ax[2]

#### JPC05
ax0.grid(color='lightgray', linestyle=':', linewidth=0.7, axis='both', which='major', zorder=0)
ax0.plot(JPC05_ages_med, JPC05_d50, label=r"d50 ($\mu$m)",zorder=1)
ax0.set_ylabel(r'd50 ($\mu$m)')
# ax0.set_yscale('log')
ax0.legend()
ax0.set_title('JPC05')
ax0.set_xlim(-150,8100)
ax0.set_ylim(-1, 350)

#### VC23
ax1.grid(color='lightgray', linestyle=':', linewidth=0.7, axis='both', which='major', zorder=0)
ax1.plot(VC23_ages_med, VC23_d50, label=r"d50 ($\mu$m)",zorder=1)
ax1.set_ylabel(r'd50 ($\mu$m)')
# ax1.set_yscale('log')
ax1.legend()
ax1.set_title('VC23')
ax1.set_xlim(-150,8100)
ax1.set_ylim(-1, 350)

#### VC11
ax2.grid(color='lightgray', linestyle=':', linewidth=0.7, axis='both', which='major', zorder=0)
ax2.plot(VC11_ages_med, VC11_d50, label=r'd50 ($\mu$m)',zorder=1)
ax2.set_xlabel('Age (YBP)')
ax2.set_ylabel(r'd50 ($\mu$m)')
# ax2.set_yscale('log')
ax2.legend()
ax2.set_title('VC11')
ax2.set_xlim(-150,8100)
ax2.set_ylim(-1, 350)


#%% Plot Raw Grainsize Timeseries of d90 

fig, ax = plt.subplots(3,1, figsize=(6,6), constrained_layout=True, dpi=150)
ax0 = ax[0]
ax1 = ax[1]
ax2 = ax[2]

#### JPC05
ax0.grid(color='lightgray', linestyle=':', linewidth=0.7, axis='both', which='major', zorder=0)
ax0.plot(JPC05_ages_med, JPC05_d90, label=r"d90 ($\mu$m)",zorder=1)
ax0.set_ylabel(r'd90 ($\mu$m)')
# ax0.set_yscale('log')
ax0.legend()
ax0.set_title('JPC05')
ax0.set_xlim(-150,8100)
ax0.set_ylim(-1, 1250)

#### VC23
ax1.grid(color='lightgray', linestyle=':', linewidth=0.7, axis='both', which='major', zorder=0)
ax1.plot(VC23_ages_med, VC23_d90, label=r"d90 ($\mu$m)",zorder=1)
ax1.set_ylabel(r'd90 ($\mu$m)')
# ax1.set_yscale('log')
ax1.legend()
ax1.set_title('VC23')
ax1.set_xlim(-150,8100)
ax1.set_ylim(-1, 1250)

#### VC11
ax2.grid(color='lightgray', linestyle=':', linewidth=0.7, axis='both', which='major', zorder=0)
ax2.plot(VC11_ages_med, VC11_d90, label=r'd90 ($\mu$m)',zorder=1)
ax2.set_xlabel('Age (YBP)')
ax2.set_ylabel(r'd50 ($\mu$m)')
# ax2.set_yscale('log')
ax2.legend()
ax2.set_title('VC11')
ax2.set_xlim(-150,8100)
ax2.set_ylim(-1, 1250)

#%% Make Event Threshold Plot

fig, ax = plt.subplots(3,1, figsize=(10,7), constrained_layout=True, dpi=150)
ax0 = ax[0]
ax1 = ax[1]
ax2 = ax[2]

#### JPC05
ax0.grid(color='lightgray', linestyle=':', linewidth=0.7, axis='both', which='major', zorder=0)
ax0.plot(JPC05_ages_med, JPC05_d50, label=r"d50 ($\mu$m)",zorder=1)
ax0.plot(JPC05_ages_med, JPC05_thresh, linestyle='--', lw=1, color='k', label='threshold', zorder=2)
ax0.plot(JPC05_ages_med, JPC05_movmean, lw=1, color='k', label='moving mean', zorder=3)
ax0.scatter(JPC05_event_times, JPC05_event_gs, marker='*', s=50, color='k', label='events', zorder=4)
# ax0.axvline(2000, color='k')
# ax0.axvline(3000, color='k')
ax0.set_ylabel(r'd50 ($\mu$m)')
ax0.set_yscale('log')
ax0.legend()
ax0.set_title('JPC05')
ax0.set_xlim(-150,8100)

#### VC23
ax1.grid(color='lightgray', linestyle=':', linewidth=0.7, axis='both', which='major', zorder=0)
ax1.plot(VC23_ages_med, VC23_d50, label=r"d50 ($\mu$m)",zorder=1)
ax1.plot(VC23_ages_med, VC23_thresh, linestyle='--', lw=1, color='k', label='threshold', zorder=2)
ax1.plot(VC23_ages_med, VC23_movmean, lw=1, color='k', label='moving mean', zorder=3)
ax1.scatter(VC23_event_times, VC23_event_gs, marker='*', s=50, color='k', label='events', zorder=4)
# ax1.axvline(2000, color='k')
# ax1.axvline(3000, color='k')
ax1.set_ylabel(r'd50 ($\mu$m)')
ax1.set_yscale('log')
ax1.legend()
ax1.set_title('VC23')
ax1.set_xlim(-150,8100)


#### VC11
ax2.grid(color='lightgray', linestyle=':', linewidth=0.7, axis='both', which='major', zorder=0)
ax2.plot(VC11_ages_med, VC11_d90, label=r'd90 ($\mu$m)',zorder=1)
ax2.plot(VC11_ages_med, VC11_thresh, linestyle='--', lw=1, color='k', label='threshold', zorder=2)
ax2.plot(VC11_ages_med, VC11_movmean, lw=1, color='k', label='moving median', zorder=3)
ax2.scatter(VC11_event_times, VC11_event_gs, marker='*', s=50, color='k', label='events', zorder=4)
# ax2.axvline(2000, color='k')
# ax2.axvline(3000, color='k')
ax2.set_xlabel('Age (YBP)')
ax2.set_ylabel(r'd90 ($\mu$m)')
ax2.set_yscale('log')
ax2.legend()
ax2.set_title('VC11')
ax2.set_xlim(-150,8100)



#%% Make Age Model Plot

fig, ax = plt.subplots(1,3, figsize=(10,6),constrained_layout=True, dpi=150)
ax0 = ax[0]
ax1 = ax[1]
ax2 = ax[2]


#### JPC05
#radiocarbon
ax0.plot(JPC05_ages_med, JPC05_GS_depth,color='k', label='Median Age')
ax0.fill_betweenx(JPC05_GS_depth, JPC05_ages_min, JPC05_ages_max,color='k', alpha=0.1, linewidth=1)
ax0.set_ylabel('Depth (cm)')
ax0.set_xlabel('YBP')
ax0.grid(color='lightgray', linestyle=':', linewidth=0.7, axis='both', which='major')
ax0.tick_params(axis='both', direction='in')
xerror = np.vstack((JPC05_c14_min-JPC05_c14_med, JPC05_c14_med-JPC05_c14_max))
ax0.errorbar(JPC05_c14_med, JPC05_c14_depth, xerr=xerror, fmt='o', color='tab:blue', markersize=3.5, capsize=3,label='Radiocarbon Date')
ax0.set_ylim(-50,2100)
ax0.set_xlim(-500,8500)
ax0.invert_yaxis()
#pb210
ax00 = ax0.twiny()
ax00.errorbar(JPC05_pb210_val, JPC05_pb210_depth, xerr=JPC05_pb210_err, fmt='o', color="tab:green", markersize=2, capsize=1, label='Pb210 Measurement', zorder=0)
# ax00.invert_yaxis()
#legend
data1, label1 = ax0.get_legend_handles_labels()
data2, label2 = ax00.get_legend_handles_labels()
ax0.legend(data1+data2, label1+label2, loc='lower left')
ax0.set_title('JPC05')

#### VC23
ax1.plot(VC23_ages_med, VC23_GS_depth,color='k', label='Median Age')
ax1.fill_betweenx(VC23_GS_depth, VC23_ages_min, VC23_ages_max,color='k', alpha=0.1, linewidth=1)
ax1.set_ylabel('Depth (cm)')
ax1.set_xlabel('YBP')
ax1.grid(color='lightgray', linestyle=':', linewidth=0.7, axis='both', which='major')
ax1.tick_params(axis='both', direction='in')
xerror = np.vstack((VC23_c14_min-VC23_c14_med, VC23_c14_med-VC23_c14_max))
ax1.errorbar(VC23_c14_med, VC23_c14_depth, xerr=xerror, fmt='o', color='tab:blue', markersize=3.5, capsize=3,label='Radiocarbon Date')
ax1.set_ylim(-50,2100)
ax1.set_xlim(-500,8500)
ax1.invert_yaxis()
#pb210
ax01 = ax1.twiny()
ax01.errorbar(VC23_pb210_val, VC23_pb210_depth, xerr=VC23_pb210_err, fmt='o', color="tab:green", markersize=2, capsize=1, label='Pb210 Measurement',zorder=0)
# ax01.invert_yaxis()
#legend
data1, label1 = ax1.get_legend_handles_labels()
data2, label2 = ax01.get_legend_handles_labels()
ax1.legend(data1+data2, label1+label2, loc='lower left')
ax1.set_title('VC23')


#### VC11
ax2.plot(VC11_ages_med, VC11_GS_depth,color='k', label='Median Age')
ax2.fill_betweenx(VC11_GS_depth, VC11_ages_min, VC11_ages_max,color='k', alpha=0.1, linewidth=1)
ax2.set_ylabel('Depth (cm)')
ax2.set_xlabel('YBP')
ax2.grid(color='lightgray', linestyle=':', linewidth=0.7, axis='both', which='major')
ax2.tick_params(axis='both', direction='in')
xerror = np.vstack((VC11_c14_min-VC11_c14_med, VC11_c14_med-VC11_c14_max))
ax2.errorbar(VC11_c14_med, VC11_c14_depth, xerr=xerror, fmt='o', color='tab:blue', markersize=3.5, capsize=3,label='Radiocarbon Date')
ax2.set_ylim(-50,2100)
ax2.set_xlim(-500,8500)
ax2.invert_yaxis()
#pb210
ax02 = ax2.twiny()
# ax02.errorbar(VC11_pb210_val, VC11_pb210_depth, xerr=VC11_pb210_err, fmt='o', color="tab:blue", markersize=2, capsize=1, label='Pb210 Measurement')
# ax02.invert_yaxis()
#legend
data1, label1 = ax2.get_legend_handles_labels()
data2, label2 = ax02.get_legend_handles_labels()
ax2.legend(data1+data2, label1+label2, loc='lower left')
ax2.set_title('VC11')



#%% Plot Densities

fig, ax = plt.subplots(3,1, figsize=(6,6), constrained_layout=True, dpi=150)
ax0 = ax[0]
ax1 = ax[1]
ax2 = ax[2]

#### JPC05
ax0.grid(color='lightgray', linestyle=':', linewidth=0.7, axis='both', which='major', zorder=0)
ax0.plot(JPC05_density_ages, JPC05_wet_density, label=r"wet density", color='tab:blue', alpha=0.5, lw=1, zorder=1)
ax0.plot(JPC05_density_ages, JPC05_wet_density10, label=r"wet density avg", color='tab:blue', lw=1.5, zorder=1)
ax0.plot(JPC05_density_ages, JPC05_dry_density, label=r"dry density", color='tab:red', alpha=0.5, lw=1, zorder=1)
ax0.plot(JPC05_density_ages, JPC05_dry_density10, label=r"dry density avg", color='tab:red', lw=1.5, zorder=1)
ax0.set_ylabel(r'g/cc')
ax0.legend()
ax0.set_title('JPC05')
ax0.set_xlim(-150,8100)

#### VC23
ax1.grid(color='lightgray', linestyle=':', linewidth=0.7, axis='both', which='major', zorder=0)
ax1.plot(VC23_density_ages, VC23_wet_density, label=r"wet density", color='tab:blue', alpha=0.5, lw=1, zorder=1)
ax1.plot(VC23_density_ages, VC23_wet_density10, label=r"wet density avg", color='tab:blue', lw=1.5, zorder=1)
ax1.plot(VC23_density_ages, VC23_dry_density, label=r"dry density", color='tab:red', alpha=0.5, lw=1, zorder=1)
ax1.plot(VC23_density_ages, VC23_dry_density10, label=r"dry density avg", color='tab:red', lw=1.5, zorder=1)
ax1.set_ylabel(r'g/cc')
ax1.legend()
ax1.set_title('VC23')
ax1.set_xlim(-150,8100)

#### VC11
ax2.grid(color='lightgray', linestyle=':', linewidth=0.7, axis='both', which='major', zorder=0)
ax2.plot(VC11_density_ages, VC11_wet_density, label=r"wet density", color='tab:blue', alpha=0.5, lw=1, zorder=1)
ax2.plot(VC11_density_ages, VC11_wet_density10, label=r"wet density avg", color='tab:blue', lw=1.5, zorder=1)
ax2.plot(VC11_density_ages, VC11_dry_density, label=r"dry density", color='tab:red', alpha=0.5, lw=1, zorder=1)
ax2.plot(VC11_density_ages, VC11_dry_density10, label=r"dry density avg", color='tab:red', lw=1.5, zorder=1)
ax2.set_ylabel(r'g/cc')
ax2.legend()
ax2.set_title('VC11')
ax2.set_xlim(-150,8100)


#%% GS Statistics Before and After 2000 YBP

#### JPC05
### 2000 years
ind = np.argwhere(JPC05_ages_med >= 2000)
ind2000 = np.squeeze(ind[0])

JPC05_pre2000_gsavg = np.nanmean(JPC05_d50[:ind2000])
JPC05_post2000_gsavg = np.nanmean(JPC05_d50[ind2000:])
JPC05_pre2000_gscv = variation(JPC05_d50[:ind2000])
JPC05_post2000_gscv = variation(JPC05_d50[ind2000:])

### 3000 years
ind = np.argwhere(JPC05_ages_med >= 3000)
ind3000 = np.squeeze(ind[0])

JPC05_pre3000_gsavg = np.nanmean(JPC05_d50[:ind3000])
JPC05_post3000_gsavg = np.nanmean(JPC05_d50[ind3000:])
JPC05_pre3000_gscv = variation(JPC05_d50[:ind3000])
JPC05_post3000_gscv = variation(JPC05_d50[ind3000:])



#### VC23
ind = np.argwhere(VC23_ages_med >= 2000)
ind2000 = np.squeeze(ind[0])

VC23_pre2000_gsavg = np.nanmean(VC23_d50[:ind2000])
VC23_post2000_gsavg = np.nanmean(VC23_d50[ind2000:])
VC23_pre2000_gscv = variation(VC23_d50[:ind2000])
VC23_post2000_gscv = variation(VC23_d50[ind2000:])

### 3000 years
ind = np.argwhere(VC23_ages_med >= 3000)
ind3000 = np.squeeze(ind[0])

VC23_pre3000_gsavg = np.nanmean(VC23_d50[:ind3000])
VC23_post3000_gsavg = np.nanmean(VC23_d50[ind3000:])
VC23_pre3000_gscv = variation(VC23_d50[:ind3000])
VC23_post3000_gscv = variation(VC23_d50[ind3000:])


#### VC11
ind = np.argwhere(VC11_ages_med >= 2000)
ind2000 = np.squeeze(ind[0])

VC11_pre2000_gsavg = np.nanmean(VC11_d90[:ind2000])
VC11_post2000_gsavg = np.nanmean(VC11_d90[ind2000:])
VC11_pre2000_gscv = variation(VC11_d90[:ind2000])
VC11_post2000_gscv = variation(VC11_d90[ind2000:])

### 3000 years
ind = np.argwhere(VC11_ages_med >= 3000)
ind3000 = np.squeeze(ind[0])

VC11_pre3000_gsavg = np.nanmean(VC11_d90[:ind3000])
VC11_post3000_gsavg = np.nanmean(VC11_d90[ind3000:])
VC11_pre3000_gscv = variation(VC11_d90[:ind3000])
VC11_post3000_gscv = variation(VC11_d90[ind3000:])

#%% Density Statistics Before and After 2000 YBP

#### JPC05

### 2000 years
ind = np.argwhere(JPC05_density_ages >= 2000)
ind2000 = np.squeeze(ind[0])
#wet density
JPC05_pre2000_wdavg = np.nanmean(JPC05_wet_density[:ind2000])
JPC05_post2000_wdavg = np.nanmean(JPC05_wet_density[ind2000:])
JPC05_pre2000_wdcv = variation(JPC05_wet_density[:ind2000])
JPC05_post2000_wdcv = variation(JPC05_wet_density[ind2000:])
#dry density
JPC05_pre2000_ddavg = np.nanmean(JPC05_dry_density[:ind2000])
JPC05_post2000_ddavg = np.nanmean(JPC05_dry_density[ind2000:])
JPC05_pre2000_ddcv = variation(JPC05_dry_density[:ind2000])
JPC05_post2000_ddcv = variation(JPC05_dry_density[ind2000:])


### 3000 years
ind = np.argwhere(JPC05_density_ages >= 3000)
ind3000 = np.squeeze(ind[0])
#wet density
JPC05_pre3000_wdavg = np.nanmean(JPC05_wet_density[:ind3000])
JPC05_post3000_wdavg = np.nanmean(JPC05_wet_density[ind3000:])
JPC05_pre3000_wdcv = variation(JPC05_wet_density[:ind3000])
JPC05_post3000_wdcv = variation(JPC05_wet_density[ind3000:])
#dry density
JPC05_pre3000_ddavg = np.nanmean(JPC05_dry_density[:ind3000])
JPC05_post3000_ddavg = np.nanmean(JPC05_dry_density[ind3000:])
JPC05_pre3000_ddcv = variation(JPC05_dry_density[:ind3000])
JPC05_post3000_ddcv = variation(JPC05_dry_density[ind3000:])




#### VC23

### 2000 years
ind = np.argwhere(VC23_density_ages >= 2000)
ind2000 = np.squeeze(ind[0])
#wet density
VC23_pre2000_wdavg = np.nanmean(VC23_wet_density[:ind2000])
VC23_post2000_wdavg = np.nanmean(VC23_wet_density[ind2000:])
VC23_pre2000_wdcv = variation(VC23_wet_density[:ind2000])
VC23_post2000_wdcv = variation(VC23_wet_density[ind2000:])
#dry density
VC23_pre2000_ddavg = np.nanmean(VC23_dry_density[:ind2000])
VC23_post2000_ddavg = np.nanmean(VC23_dry_density[ind2000:])
VC23_pre2000_ddcv = variation(VC23_dry_density[:ind2000])
VC23_post2000_ddcv = variation(VC23_dry_density[ind2000:])


### 3000 years
ind = np.argwhere(VC23_density_ages >= 3000)
ind3000 = np.squeeze(ind[0])
#wet density
VC23_pre3000_wdavg = np.nanmean(VC23_wet_density[:ind3000])
VC23_post3000_wdavg = np.nanmean(VC23_wet_density[ind3000:])
VC23_pre3000_wdcv = variation(VC23_wet_density[:ind3000])
VC23_post3000_wdcv = variation(VC23_wet_density[ind3000:])
#dry density
VC23_pre3000_ddavg = np.nanmean(VC23_dry_density[:ind3000])
VC23_post3000_ddavg = np.nanmean(VC23_dry_density[ind3000:])
VC23_pre3000_ddcv = variation(VC23_dry_density[:ind3000])
VC23_post3000_ddcv = variation(VC23_dry_density[ind3000:])



#### VC11

### 2000 years
ind = np.argwhere(VC11_density_ages >= 2000)
ind2000 = np.squeeze(ind[0])
#wet density
VC11_pre2000_wdavg = np.nanmean(VC11_wet_density[:ind2000])
VC11_post2000_wdavg = np.nanmean(VC11_wet_density[ind2000:])
VC11_pre2000_wdcv = variation(VC11_wet_density[:ind2000])
VC11_post2000_wdcv = variation(VC11_wet_density[ind2000:])
#dry density
VC11_pre2000_ddavg = np.nanmean(VC11_dry_density[:ind2000])
VC11_post2000_ddavg = np.nanmean(VC11_dry_density[ind2000:])
VC11_pre2000_ddcv = variation(VC11_dry_density[:ind2000])
VC11_post2000_ddcv = variation(VC11_dry_density[ind2000:])


### 3000 years
ind = np.argwhere(VC11_density_ages >= 3000)
ind3000 = np.squeeze(ind[0])
#wet density
VC11_pre3000_wdavg = np.nanmean(VC11_wet_density[:ind3000])
VC11_post3000_wdavg = np.nanmean(VC11_wet_density[ind3000:])
VC11_pre3000_wdcv = variation(VC11_wet_density[:ind3000])
VC11_post3000_wdcv = variation(VC11_wet_density[ind3000:])
#dry density
VC11_pre3000_ddavg = np.nanmean(VC11_dry_density[:ind3000])
VC11_post3000_ddavg = np.nanmean(VC11_dry_density[ind3000:])
VC11_pre3000_ddcv = variation(VC11_dry_density[:ind3000])
VC11_post3000_ddcv = variation(VC11_dry_density[ind3000:])
