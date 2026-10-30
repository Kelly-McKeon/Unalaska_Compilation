#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 15:09:12 2025

@author: kellymckeon
"""

import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
import pandas as pd
import time
import random

mac = r"/Users/kellymckeon/Library/CloudStorage/OneDrive-WoodsHoleOceanographicInstitution/Documents/WHOI/Alaska/Codes/Coherent_Codes/Compilation_Codes/For_Publication"
pc = r"C:/Users/kelly/OneDrive - Woods Hole Oceanographic Institution/Documents/WHOI/Alaska/Codes/Coherent_Codes/Compilation_Codes/For_Publication"

# CHANGE DIRECTORY TO WHICHEVER COMPUTER YOU ARE ON
os.chdir(mac)

code_start = time.time() #begin timing code


#%% Load Data

#### grainsize data 
JPCC = np.genfromtxt(r"JPC05_CoreData/JPC05C_GS.csv", delimiter=",", skip_header=1)
GS_depth = JPCC[:, 0] 
d10 = JPCC[:, 1]
d50 = JPCC[:, 2]
d90 = JPCC[:, 3]

#### age data
#choose age model slumps based on the threshold set in the next section
#if you have not run the age model yet, comment out this section and return to it after running the section below
ages = np.genfromtxt(r"JPC05_CoreData/JPC05C_Ages.txt", skip_header=1) 
ages_depth = ages[:,0]
ind = np.isin(ages_depth, GS_depth) #get rid of duplicate ages from slumps

# ages are in YBP
ages_med = ages[ind,3] 
ages_mean = ages[ind,4] 
ages_min = ages[ind,1] 
ages_max = ages[ind,2] 

#%% Visualize Grainsize Data

#### plot d50 histogram to get lay of the land
plt.figure()
plt.hist(d50, bins=40, histtype='bar', ec='k')
plt.xlabel('grainsize ($\mu$m)')
plt.ylabel('counts')
plt.title('JPCC d50')
plt.show()

#save the histogram output 
count, bins_count = np.histogram(d50, bins=20)


#### plot d50 PDF and CDF

#calculate PDF
pdf = count / sum(count)
#calculate CDF
cdf = np.cumsum(pdf)

#plot d50 distribution
plt.figure(dpi=150)
plt.plot(bins_count[1:], pdf, color="red", label="PDF")
plt.plot(bins_count[1:], cdf, label="CDF")
plt.axvline(np.std(d50) + np.median(d50), color='k', linestyle='dashed', label=r'd50 of 1$\sigma$')
plt.axvline(2*np.std(d50) + np.median(d50), color='gray', linestyle='dashed', label=r'd50 of 2$\sigma$')
plt.axvline(63, color='k', label=r'63 $\mu$m')
plt.ylabel('Distribution Function')
plt.xlabel(r'd50 ($\mu$m)')
plt.title('JPC05 d50 Distribution')
plt.legend()
plt.show()

#%% Create Slump Threshold
sd50 = np.std(d50) #d50 standard deviation
med50 = np.median(d50) #d50 median
avg50 =np.mean(d50) #d50 mean
thr50 = 63 #use threshold for sand

# get points of indices greater than the threshold
tind = np.argwhere(d50 >= thr50) #save indices greater than the threshold
d50_2 = np.copy(d50) #make a new array because python back changes your array if you save it as the same name
d50_2[tind] = med50 #replace values above the threshold with the timeseries median


#### Get Event Depths to slump in Age Model
event_depths = GS_depth[tind]
event_gs = d50[tind]


#### Plot d50 with Slumps Marked (Sanity Check)
fig,ax = plt.subplots(figsize=(4,8),dpi=150)
ax.plot(d50, GS_depth, label = 'd50') #grainsize
ax.scatter(event_gs, event_depths, marker = '*', color='k', label='slumps') #slumps
ax.plot(d50_2, GS_depth, color='k', label= r"d50 < 63$\mu$m") #grainsize below threshold
plt.axvline(x=thr50, label=str(int(thr50)),color='k', linestyle = 'dashed') #threshold
ax.invert_yaxis()
ax.set_ylabel('depth (cm)')
ax.set_xlabel(r'd50 ($\mu$m)')
ax.set_title('JPC05 Slump Depths for Age Model')
ax.legend()
plt.show()

#### Save Slump Starts and Slump Ends
# open these variables and type them into the age model program
iterator = np.arange(np.size(event_depths))
starts = np.zeros(np.size(iterator)) * np.nan
ends = np.zeros(np.size(iterator)) * np.nan
slump_start = []
slump_end = []

#get cm difference between event depths
for i in iterator[:-1]:
    starts[i] = event_depths[i+1] - event_depths[i] 

for i in iterator:
    if i == 0: #skip first deposit
        pass
    else: 
        #if distance between this event and the last event is greater than 1, it's a new slump start
        if starts[i] == 1 and starts[i-1] > 1:
           slump_start.append(event_depths[i])
        #if distance between this event and the next event is greater than 1, it's a new slump end
        if starts[i] == 1 and starts[i+1] > 1:
            slump_end.append(event_depths[i+1])
        #the core ends with an event
        #in this case just append the last event depth to the end 
        if i == len(event_depths)-1:
            slump_end.append(event_depths[i])
            
###############################################################################           

##### Now go in rPlum and add slumps at all the event depths that are over the threshold for more than 1 cm

###############################################################################

#%% Make Grainsize Anomaly Timeseries

# 25cm window is centennial avg assuming 4 yr/cm accumulation rate
window_size = 25 

# Convert array of integers to pandas series because rolling windows is easier in Pandas
d50_series = pd.Series(d50_2) #use the timeseries with the outliers removed so they do not skew the moving mean
  
# Get the windows
windows = d50_series.rolling(window_size)
  
# Create a series of moving averages of each window
mov = windows.mean()

#fill the nans left out of the moving average with the ts mean
moving_averages = mov.fillna(avg50)
  
# Convert pandas series back to numpy 
d50_mov = moving_averages.to_numpy()

#### Get Anomaly
anomaly50 = d50 - d50_mov #use the full timeseries not the one with outliers removed

#### Grainsize Anomaly (Sanity Check)
fig, ax = plt.subplots(1,2, figsize=(6,8),dpi=150,constrained_layout=True)
ax0=ax[0]
ax1=ax[1]

# moving mean
ax0.plot(d50, GS_depth, label = 'JPCC d50') #grainsize
ax0.plot(d50_mov, GS_depth, label = '25pt mov mean') #100yr moving mean
ax0.invert_yaxis()
ax0.set_ylabel('depth (cm)')
ax0.set_xlabel(r'd50 ($\mu$m)')
ax0.set_title('JPCC d50')
ax0.legend()

# anomaly with optional thresholds
ax1.plot(anomaly50, GS_depth) #anomaly
ax1.axvline(0, color='k', label='zero') #plot zero line
ax1.axvline(np.median(anomaly50)+np.std(anomaly50), color='k', linestyle = 'dashed', label=r'1$\sigma$')
ax1.axvline(np.median(anomaly50)+2*np.std(anomaly50), color='k', linestyle = 'dotted', label=r'2$\sigma$')
ax1.invert_yaxis()
ax1.set_ylabel('depth (cm)')
ax1.set_xlabel(r'd50 Anomaly ($\mu$m)')
ax1.set_title(r'JPCC d50 Anomaly')
ax1.legend()

            
#%% Create Event Threshold

#### Visualize Anomalies


#### Get Anomaly Summary Statistics
sd50a = np.std(anomaly50)
med50a = np.median(anomaly50)
avg50a =np.mean(anomaly50)
uppera = np.percentile(anomaly50, 90)
lowera = np.percentile(anomaly50, 83) #vary this one to choose event threshold
athresh = anomaly50 >= lowera


#### Plot Anomaly with Threshold Options Marked 
fig, ax = plt.subplots(1,2,constrained_layout=True,dpi=150)
ax0=ax[0]
ax1=ax[1]

ax0.plot(d50, GS_depth, label='d50')#grainsize plotted against depth
ax0.plot(d50_mov, GS_depth, label = 'd50 filter') #moving mean plotted with depth
ax0.scatter(event_gs, event_depths, marker = '*', color='k', label='slump depths')
ax0.invert_yaxis()
ax0.set_ylabel('depth (cm)')
ax0.set_xlabel(r'd50 ($\mu$m)')
ax0.legend()

ax1.plot(anomaly50, ages_med, label = 'anomaly') #anomaly plotted against age
ax1.scatter(anomaly50[athresh], ages_med[athresh], marker = '*', color='k', label='events') #depths above the event threshold
ax1.invert_yaxis()
ax1.axvline(lowera, color = 'k', linestyle = 'dashed', linewidth=1, label= '85th percentile anomaly')
ax1.set_xlabel(r'coarse anomaly ($\mu$m)')
ax1.set_ylabel('median age (YBP)')
ax1.legend()
plt.suptitle('JPC05 Events')
plt.xlim(0,300)
plt.show()


#%% Make Booelan Event Index
#this block removes consecutive depths over the threshold as individual events

new_array = np.ones(np.size(anomaly50)) #make array that will populate with Boolean index

event_depth_array = [] #make empty depth array to save event depths in
event_gs_array = [] #make empty gs array to save event grainsizes in
event_anom_array = [] #make empty anomaly array
event_time_array = [] #make empty time array

iterator = np.arange(0,np.size(anomaly50))

#### Set Each Depth as Event or Not
#use the lower threshold defined above

for i in iterator:
    if anomaly50[i] < lowera:
        new_array[i] = 0
        
    #only code it as an event if the cm before was NOT an event
    elif anomaly50[i] >= lowera and anomaly50[i-1] < lowera: 
        new_array[i] = 1
        event_depth_array.append(GS_depth[i])
        event_gs_array.append(d50[i])
        event_anom_array.append(anomaly50[i])
        event_time_array.append(ages_med[i])
   
    #if both consecutive cm are events, only code the first one as an event
    elif anomaly50[i] >= lowera and anomaly50[i-1] >= lowera: 
        new_array[i] = 0
        
    bool_anom = new_array #make boolean event array


#%% Remove Tsunamis from Event Depths 

del event_depth_array[19] #deposit at 518cm
del event_depth_array[55] #deposit at 1415cm

del event_gs_array[19] #deposit at 518cm
del event_gs_array[55] #deposit at 1415cm

del event_anom_array[19] #deposit at 518cm
del event_anom_array[55] #deposit at 1415cm

del event_time_array[19] #deposit at 518cm
del event_time_array[55] #deposit at 1415cm


#set tsunamis to non-events
bool_anom[517] = 0 
bool_anom[1415] = 0


#%% Plot Event Threshold with Raw d50

mov_anomaly = d50_mov + lowera

fig,ax = plt.subplots(1,1, figsize=(8,4), dpi=150)
ax.plot(ages_med, d50, label='d50', zorder=1)
ax.plot(ages_med[25:], mov_anomaly[25:], linestyle='--', linewidth=1, color='k',label='threshold', zorder=2)
ax.scatter(event_time_array, event_gs_array, marker='*', color='k', label='events', zorder=3)
ax.set_xlabel('median age (YBP)')
ax.set_ylabel('d50 ($\mu$m)')
plt.grid(color='lightgray', linestyle=':', linewidth=0.7, axis='both', which='major')
ax.legend()
ax.set_title('JPC05')

#%% Save Data

#### Data for Comparison Plot 
np.savetxt(r"JPC05_CoreData/threshold.csv", mov_anomaly, delimiter=',')
np.savetxt(r"JPC05_CoreData/moving_mean.csv", d50_mov, delimiter=',')
np.savetxt(r"JPC05_CoreData/median_ages.csv", ages_med, delimiter=',')
np.savetxt(r"JPC05_CoreData/event_times.csv", event_time_array, delimiter=',')
np.savetxt(r"JPC05_CoreData/event_gs.csv", event_gs_array, delimiter=',')    

#### Event Depths For Export to R
np.savetxt(r"JPC05_CoreData/event_depth.csv", event_depth_array, delimiter=',')
np.savetxt(r"JPC05_CoreData/event_gs.csv", event_gs_array, delimiter=',')

#once you have event depths go to rPlum and export age distributions for each of these depths

#%% Load Age Distributions from Bacon
#ages are in YBP
age_dist = np.genfromtxt(r"JPC05_CoreData/JPC05_Age_Distributions.csv", delimiter=",", skip_header=1)
age_dist = age_dist[:,1:] #remove first index column of nans


#%% Make Dataset Annual

eventyrs=[]
#populate new list with event years based on median age
for i in range(0, len(bool_anom)):
    if bool_anom[i] == True:
        eventyrs.append(ages_med[i]) 
        
eventyrs = np.array(eventyrs) #convert to array
eventyrs = eventyrs.astype(int) #round to nearest year (no decimals)

#make new array with evently spaced years
allyrs = np.arange(min(ages_med), max(ages_med), 1, dtype=int) #boundaries are the minimum and maximum median ages, stepping by 1 year at a time
strmyrs = np.isin(allyrs, eventyrs) #this creates a binary index of the same size as allyrs, where if the year in allyrs is a storm year (in yrs) it's coded as 1

#### Stem Plot of Storm Years
plt.figure(figsize=(6,4), dpi=150)
plt.stem(allyrs,strmyrs, 'k', markerfmt =" ", basefmt='k-')
plt.xlabel('years before present')
plt.ylabel('storm')
plt.xlim(-100,3000) #cut it off at 3000ybp because of the sea level transgression


# save storm years (binary)
np.savetxt(r"JPC05_CoreData/storm_years.csv", strmyrs, delimiter=',')
# save allyrs
np.savetxt(r"JPC05_CoreData/all_years.csv", allyrs, delimiter=',')

#%% Kernel Rate Estimation Using Median Ages of Storms
import time
start_time = time.time()
#set window size
h = 50 #h represents the standard deviation of the sliding bell curve so make it half of your important time period

#### Calculate rate parameter
#create a function for easy bootstrapping later

# input parameters are all the years in your dataset, the event years, and the window size 
def rate_parameter(allyrs, eventyrs, h):
    #initialize the rate parameter
    l = np.nan*(np.zeros(len(allyrs))) #l is lambda as in Mudelsee Eq 3.16
    
    for i in range(0,len(allyrs)): #loop through every year of the dataset
        l[i] = 0 
        for j in range(0, len(eventyrs)):
            y = (allyrs[i] - eventyrs[j]) / h
            l[i] = l[i] + (np.power((2*np.pi), -0.5) * np.exp((-y**2)/2))/h
    return(l*100)

#experiment with some different window sizes
l25 = rate_parameter(allyrs, eventyrs, 25)
l50 = rate_parameter(allyrs, eventyrs, 50)
l100 = rate_parameter(allyrs, eventyrs, 100)

print('time to run kernel rate estimation is ', (time.time() - start_time)/60, ' minutes')  


#%% Plot Rate Parameters With Different Bandwidths

plt.figure(dpi=150)
plt.stem(allyrs,strmyrs, 'k', markerfmt =" ", basefmt=" ")
plt.plot(allyrs, l25, color='tab:blue', linewidth=2, label='25')
plt.plot(allyrs, l50, color='tab:green', linewidth=2, label='50')
plt.plot(allyrs, l100, color='tab:orange',linewidth=2, label='100')
plt.xlabel('Year (CE)')
plt.ylabel(r'Storm Frequency ($\lambda$)')

plt.grid(color='lightgray', linestyle=':', linewidth=0.7, axis='both', which='major', zorder=0)

plt.legend()


#%% Bootstrap Gaussian Kernel Rate Estimation with Age Uncertainties (~30 min)


start_time = time.time()

#set number of iterations
iterations = 1000
#set window size
h = 50
#initialize empty matrix
l_matrix = np.zeros((iterations, len(allyrs)))
event_matrix_bool = np.nan * np.zeros((iterations,len(allyrs)))

for ii in range(0,iterations):
    
    #draw a random event age from the age probability distributions
    amax = len(age_dist) #maximum index
    amin = 0 #minimum index
    age_ind = np.random.randint(amin, high=amax, size=1) #get random index
    
    #save alternative set of event years from the random age model iteration
    eventyrs_temp = age_dist[age_ind,:]
    eventyrs_int = np.squeeze(eventyrs_temp.astype(int)) #convert to integer
    
    #create a new set of event years with replacement from the intermediate event years 
    #keeping the same total number of events
    #preserving stratigraphic order
    rmax = len(eventyrs) #maximum index
    rmin = 0 #minimum index
    event_ind = np.random.randint(rmin, high=rmax, size=len(eventyrs)) #rmax is exclusive of the final element so add 1
    eventyrs_boot = eventyrs_int[event_ind]
    
    strmyrs_boot = np.isin(allyrs, eventyrs_boot) #boolean index of event years
    
    #perform rate parameter analysis on the new timeseries of event years
    l_temp = rate_parameter(allyrs, eventyrs_boot, h)
    
    l_matrix[ii,:] = l_temp
    event_matrix_bool[ii,:] = strmyrs_boot

print('time to run bootstrapping is ', (time.time() - start_time)/60, ' minutes')


#%% Plot Rate Parameter with Age Uncertainty 

#### Get Error Bars
mean_rate = np.median(l_matrix, axis=0)
low_rate2 = np.percentile(l_matrix, 5, axis=0)
low_rate1 = np.percentile(l_matrix, 16, axis=0)
high_rate2 = np.percentile(l_matrix, 95, axis=0)
high_rate1 = np.percentile(l_matrix, 84, axis=0)

#### Plot 
fig,ax = plt.subplots(figsize=(12,4), dpi=150)
ax.plot(allyrs, mean_rate, color='k', label = 'bootstrapped mean')
ax.fill_between(allyrs, low_rate2, high_rate2, color = 'k', alpha=0.3, label=r'2$\sigma$')
ax.fill_between(allyrs, low_rate1, high_rate1, color='k', alpha=0.5, label = r'1$\sigma$')
ax.plot(allyrs, l50, label='median event age')
ax.legend()
ax.set_xlabel('Years Before Present (YBP)')
ax.set_ylabel(r'Storm Frequency ($\lambda$)')
ax.set_title('JPC05')
ax.set_xlim(-150,8150)

#%% Save kernel data for comparison later 
np.savetxt(r"JPC05_CoreData/l_matrix.csv", l_matrix, delimiter=',')
np.savetxt(r"JPC05_CoreData/event_matrix_bool.csv", event_matrix_bool, delimiter=',')


print('time to run code is ', (time.time() - code_start)/60, ' minutes')  
