#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 13:33:37 2026

@author: kellymckeon
"""

import time
code_start = time.time() #begin timing code

import numpy as np
import os

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42 
rng = np.random.default_rng()

import statsmodels.api as sm


import random
import networkx as nx          # graph library: nodes = deposits, edges = overlaps
from collections import defaultdict  # dictionary that auto-initializes missing keys

from scipy import stats
from scipy.stats import gaussian_kde, skewnorm

mac = r"/Users/kellymckeon/OneDrive - Woods Hole Oceanographic Institution/Documents/WHOI/Alaska/Codes/Coherent_Codes/Compilation_Codes/For_Publication"
pc = r"C:/Users/kelly/OneDrive - Woods Hole Oceanographic Institution/Documents/WHOI/Alaska/Codes/Coherent_Codes/Compilation_Codes/For_Publication"

# CHANGE DIRECTORY TO WHICHEVER COMPUTER YOU ARE ON
os.chdir(mac)

#%% Load Age Data

#### JPC05
#median ages
JPC05_ages_med = np.genfromtxt(r"JPC05_CoreData/median_ages.csv", delimiter=",", skip_header=0)
JPC05_event_times = np.genfromtxt(r"JPC05_CoreData/event_times.csv", delimiter=",", skip_header=0)
#distributions
age_dist = np.genfromtxt(r"JPC05_CoreData/JPC05_Age_Distributions.csv", delimiter=",", skip_header=1)
age_dist = age_dist[:,1:]
JPC05_age_dist = age_dist
#annual years
JPC05_allyears = np.genfromtxt(r"JPC05_CoreData/all_years.csv", delimiter=",", skip_header=0)

#################################################

#### VC23
#median ages
VC23_ages_med = np.genfromtxt(r"VC23_CoreData/median_ages.csv", delimiter=",", skip_header=0)
VC23_event_times = np.genfromtxt(r"VC23_CoreData/event_times.csv", delimiter=",", skip_header=0)
#distributions
age_dist = np.genfromtxt(r"VC23_CoreData/VC23_Age_Distributions.csv", delimiter=",", skip_header=1)
age_dist = age_dist[:,1:]
VC23_age_dist = age_dist
#annual years
VC23_allyears = np.genfromtxt(r"VC23_CoreData/all_years.csv", delimiter=",", skip_header=0)

#################################################

#### VC11
#median ages
VC11_ages_med = np.genfromtxt(r"VC11_CoreData/median_ages.csv", delimiter=",", skip_header=0)
VC11_event_times = np.genfromtxt(r"VC11_CoreData/event_times.csv", delimiter=",", skip_header=0)
#distributions
age_dist = np.genfromtxt(r"VC11_CoreData/VC11_Age_Distributions.csv", delimiter=",", skip_header=1)
age_dist = age_dist[:,1:]
VC11_age_dist = age_dist
#annual years
VC11_allyears = np.genfromtxt(r"VC11_CoreData/all_years.csv", delimiter=",", skip_header=0)

#################################################



#%% Get Event Age Percentiles

#### JPC05
JPC05_events5 = np.percentile(JPC05_age_dist, 5, axis=0)
JPC05_events95 = np.percentile(JPC05_age_dist, 95, axis=0)
JPC05_events50 = np.median(JPC05_age_dist, axis=0)
#to plot error bars the error has to be in relation to the center value
#so subtract the median value from the percentiles to be able to plot percentiles w/error
lowerr = JPC05_events50 - JPC05_events5
higherr = JPC05_events95 - JPC05_events50
xerr= np.stack((lowerr, higherr), axis=1)
xerr_JPC05 = xerr.T


#### VC23
VC23_events5 = np.percentile(VC23_age_dist, 5, axis=0)
VC23_events95 = np.percentile(VC23_age_dist, 95, axis=0)
VC23_events50 = np.median(VC23_age_dist, axis=0)
#to plot error bars the error has to be in relation to the center value
#so subtract the median value from the percentiles to be able to plot percentiles w/error
lowerr = VC23_events50 - VC23_events5
higherr = VC23_events95 - VC23_events50
xerr= np.stack((lowerr, higherr), axis=1)
xerr_VC23 = xerr.T

#### VC11
VC11_events5 = np.percentile(VC11_age_dist, 5, axis=0)
VC11_events95 = np.percentile(VC11_age_dist, 95, axis=0)
VC11_events50 = np.median(VC11_age_dist, axis=0)
#to plot error bars the error has to be in relation to the center value
#so subtract the median value from the percentiles to be able to plot percentiles w/error
lowerr = VC11_events50 - VC11_events5
higherr = VC11_events95 - VC11_events50
xerr= np.stack((lowerr, higherr), axis=1)
xerr_VC11 = xerr.T



#%% Add Up Age Pdfs

#sort by number of deposits
record_dists = []
record_dists.append(VC23_age_dist)
record_dists.append(JPC05_age_dist)
record_dists.append(VC11_age_dist)

 
#%% Define Functions for Excluding Indices

#excludes multiple indexes
def indices_except(data, excluders):
    return [item for i, item in enumerate(data) if i not in excluders]

#excludes a single index
def indices_except1(data, excluded_index):
    return[i for i in range(len(data)) if i != excluded_index]

#%% Get Event Bounds
JPC05_eventbounds = np.nan * np.zeros((np.size(JPC05_age_dist, axis=1),2))
for i in range(0,len(JPC05_eventbounds)):
    #get minimum and maximum ages of each deposit from the distributions
    JPC05_eventbounds[i,0] = int(np.min(JPC05_age_dist[:,i])) #minimum age
    JPC05_eventbounds[i,1] = int(np.max(JPC05_age_dist[:,i])) #maximum age

VC23_eventbounds = np.nan * np.zeros((np.size(VC23_age_dist, axis=1),2))
for i in range(0,len(VC23_eventbounds)):
    #get minimum and maximum ages of each deposit from the distributions
    VC23_eventbounds[i,0] = int(np.min(VC23_age_dist[:,i])) #minimum age
    VC23_eventbounds[i,1] = int(np.max(VC23_age_dist[:,i])) #maximum age
    
VC11_eventbounds = np.nan * np.zeros((np.size(VC11_age_dist, axis=1),2))
for i in range(0,len(VC11_eventbounds)):
    #get minimum and maximum ages of each deposit from the distributions
    VC11_eventbounds[i,0] = int(np.min(VC11_age_dist[:,i])) #minimum age
    VC11_eventbounds[i,1] = int(np.max(VC11_age_dist[:,i])) #maximum age  
    

    
#%% Define Function to Get Overlapping Coordinates

def get_overlap_coordinates(interval1, interval2):
    start_overlap = max(interval1[0], interval2[0])
    end_overlap = min(interval1[1], interval2[1])
    
    if start_overlap < end_overlap:
        return (start_overlap, end_overlap)
    else:
        return None


#%% Determing Independent Events

start = time.time()
sim_len = 5000
perc_thresh = 0.95
record_names = ["VC23", "JPC05", "VC11"] #sort by record length

#going to be 5 loops here
#outer 2 loops loop through the matrix of cores (setting which 2 cores to compare on each loop)
#all deposits in each core will be compared individually to all deposits in the other cores

#this loop sets the anchor core
#each item in record_dists represents a core, 
# within the item in the list is a matrix of the annually binned PDFs of each deposit in that core
for i in range(0, len(record_dists)): 
    
    #makes a list called VC23, JPC05, etc "overlap matrix" that has numpy arrays embedded of
    #the indices of overlaps with the 3 other records that are not the title record
    varname = record_names[i] + '_overlap_matrix'
    locals()[varname] = [None] * len(record_dists)
    
    
    #get the record you will compare all the other records to
    #search through every deposit in Record 1 (the title record)
    record1 = record_dists[i] #matrix of ages of each deposit, size is 5000 x # of events
    
    #compare every deposit in Record 1 with all other deposits in the other records
    #no need to compare it to itself, so cut that index out
    #VC23 is 0, JPC05 is 1, VC11 is 2
    records_to_compare = indices_except1(np.arange(0,len(record_dists)), i)
    
    #this loop sets the core to compare to
    for ii in range(0, len(records_to_compare)):
        
        #index of record to compare to (this way ensures that you're never comparing a record to itself)
        record2_ind = records_to_compare[ii]
        #record to compare to
        record2 = record_dists[record2_ind] #matrix of ages of each deposit, size is 5000 x # of events
    
    
    ####### OK NOW WE HAVE OUR 2 MATRIXES TO COMPARE SO LETS NEST A COUPLE MORE LOOPS
    #the inner 3 loops will loop through every event in the anchor core and compare it to every event in the other 3 cores
    
        #where to save the indices to exclude
        #this will be a vector of indexes in the records that are NOT the title record
        #the loop will create 3 of these
        #indices to exclude variable is all the indices that are overlapping with the associated index in the title record
        #(it's not identifying overlaps per se, just identifying those not flagged as keepers)
        #in this case keepers means they are an independent distribution to be kept as is
        #keepers are only flagged when one age is older more than 95% of the iterations
        r2_exclude = []
        
        
        #now get all the deposits in the first record that will be individually compared with all the deposits in the other 3
        
        for j in range(0, np.size(record1,1)):
            r2_keep = [] #indices from record 2 to keep will be appended here 
            #keepers will be appended here only if there is no age overlap with r1
            
            for jj in range(0, np.size(record2,1)):
                r1_dist = record1[:,j]
                r2_dist = record2[:,jj]
                
                r1_older = []
                r2_older = []
                yrs_tied = []
                
                #permute distributions to compare ages randomly
                r1_distp = rng.permutation(r1_dist)
                r2_distp = rng.permutation(r2_dist)
                
                #compare each possible age to see which is older
                for p in range(0, sim_len):
                    #get the year associated with the iteration you're on
                    r1_yr = r1_distp[p]
                    r2_yr = r2_distp[p]
                    
                    #test which year is older
                    if r1_yr > r2_yr:
                        r1_older.append(p)
                    elif r2_yr > r1_yr:
                        r2_older.append(p)
                    elif r1_yr == r2_yr:
                        yrs_tied.append(p)
                        
                        
                #if one deposit is older than the other more than 95% of the time, it's kept as an independent ('keeper')
                #if the deposits are the same age, you'd expect a 50/50 split on which random draws are older        
                if len(r1_older) > sim_len*perc_thresh or len(r2_older) > sim_len*perc_thresh:
                    r2_keep.append(jj)                   
                
            #now get the indexes to "exclude" from record 2 by taking all the Record 2 inds that are NOT keepers
            r2_allinds = np.arange(0, np.size(record2,1))
            r2_ind = indices_except(r2_allinds, r2_keep)
            
            # if the number of indexes that overlap with this event in record one is NONzero, 
            # append those indexes to the record2 exclude list (which remember has the shape of record 1 because the format is:
            # every deposit in record 2 is looped past every deposit in record 1, and those that overlap are saved in the row of
            # whichever record 1 deposit the loop is on)
            # this will append once for each depsosit in record one
            # output is a list, of the size of record 1, with the indexes of each record 2 deposit that overlap with each record 1 deposit
            if len(r2_ind) > 0:
                r2_exclude.append(np.array(r2_ind))
                
            elif len(r2_ind) == 0:
                r2_exclude.append([])
                
            
        #now we have all the indices of the second record that overlap with the first record going through the loop
        #add these to a matrix that we will nest called "corenameX_overlap_matrix"
        #these overlapping indices will be used in the next step to determine which distributions get compiled
        locals()[varname][record2_ind]=r2_exclude

print('time to compare event beds is ', (time.time() - start)/60, ' minutes')


#%% Identifying Events to Compile
sorder_start = time.time()

# this section is going to create a "network graph" using the networkx python package
# it will map all of the overlapping deposits by setting each deposit as a "node"
# every deposit to compile will be connected as an "edge" - graphically this is drawing a line between them
# this code will loop through every set of overlapping deposits and either preserve deposits as they are, or combine their distributions
# the outcome is as follows:
    # 1) stratigraphic order is preserved (if four deposits from one core overlap with three deposits from another core, 
    # four distributions are used. The distributions of the three deposits are merged with the 3 deposits of the four that are closest in age, 
    # the fourth deposit that is farthest in age from the overlapping three will be kept independent)
    # 2) no distributions are discarded, they are either merged or kept as is
    # 3) no "edge" in the graph may ever be connected to two deposits from the same core because these are stratigraphically distinct
    # 4) no transitive overlaps, all merged distributions must overlap with each other
    
# first create mega list of all the event bounds
# event bounds defined above as the range of the ENTIRE event distribution (min to max)
# format is a list with 3 elements for each core, each list item has a np.array inside of it
# dimensions of the np arrays are # deposits in core x 2 where the first column is minimum age and second column is max age

eventbounds = [VC23_eventbounds, JPC05_eventbounds,
               VC11_eventbounds]#sort by # of deposits

# create mega list of all deposit medians
# output in same format as eventbounds
core_medians = [[float(np.median(record_dists[ci][:, di]))
     for di in range(np.size(record_dists[ci], axis=1))]  # loop over all deposits in core ci
    for ci in range(len(record_dists))]  # loop over all cores


# make set of directly overlapping pairs to be checked later to make sure no transitive overlaps are made
# If A overlaps B and B overlaps C but C doesn't overlap A, C won't be combined with B
direct_overlap_set = set()

for ci in range(len(record_dists)): #ci is first core
    for cj in range(ci + 1, len(record_dists)): #cj is second core
        rname_ci  = record_names[ci] + '_overlap_matrix'
        rname_cj  = record_names[cj] + '_overlap_matrix'
        matrix_ci = locals()[rname_ci]
        matrix_cj = locals()[rname_cj]
        
        #get overlapping pairs in both directions
        ovlp = set()
        if matrix_ci[cj] is not None:
            for di, olist in enumerate(matrix_ci[cj]):
                for dj in olist:
                    ovlp.add((di, int(dj))) #list all the overlaps associated with this deposit for this core
        if matrix_cj[ci] is not None:
            for dj, olist in enumerate(matrix_cj[ci]):
                for di in olist:
                    ovlp.add((int(di), dj)) #list them backwards now to make sure they are symmetrical
        
        #make every di,dj pair into a "frozenset" which is like a set but can't be changed
        #later in the code we will sort through the "sets" of overlapping pairs and eliminate sets based on priorities for distribtuion merging
        #will cross check the final sets to this frozen set to make sure they all have direct overlaps
        for di, dj in ovlp:
            direct_overlap_set.add(frozenset([(ci, di), (cj, dj)])) 

#make empty network graph object
tie_graph = nx.Graph()

#add every deposit as a node
for ci in range(len(record_dists)):
    for di in range(np.size(record_dists[ci], axis=1)):
        tie_graph.add_node((ci, di))

#### Loop Through All Pairs 

#outer loop is going to go through every possible pair of cores
#keep ci < cj to make sure that pairs are not going both ways (don't want to match 0-1 and 1-0 becuase it will duplicate everything)
# so pairs will be (0,1) (0,2) (1,2) 

for ci in range(len(record_dists)): # ci: index of first core in pair
    for cj in range(ci + 1, len(record_dists)): # cj: index of second core in pair
    
    #get the overlap matrixes for each core pair (created in the section above)
        rname_ci  = record_names[ci] + '_overlap_matrix'
        rname_cj  = record_names[cj] + '_overlap_matrix'
        matrix_ci = locals()[rname_ci]
        matrix_cj = locals()[rname_cj]
        
        #get the overlapping deposit pairs from each core pair
        #use python set data type because it prevents duplicates
        #i.e. (0,2) only appears once instead of also as (2,0)
        #make empty set
        overlap_set = set()

        #do it in the first direction (core ci is the anchor core)
        #loop through every deposit in the overlap matrix
        if matrix_ci[cj] is not None: #make sure there are overlaps with that core
        # enumerate returns index and values, so "overlapping" is the numpy array of deposits that overlap in the
        #second core with the deposit you're on in the anchor core 
            for di, overlapping in enumerate(matrix_ci[cj]):
                for dj in overlapping:
                    overlap_set.add((di, int(dj))) #store overlaps as tuple   

        #loop through in other direction to make sure there are no assymetric pairs produced from the montecarlo sequence
        #this would only happen if deposits were very borderline overlapping such that a few random comparisons made 
        #the difference in meeting the threshold or not
        if matrix_cj[ci] is not None: #core 2 is outer loop now
            for dj, overlapping in enumerate(matrix_cj[ci]):
                for di in overlapping:
                    overlap_set.add((int(di), dj))
                    
        #if no overlaps, skip this pair
        if not overlap_set:
            print(f"\n{record_names[ci]} vs {record_names[cj]}: no overlaps")
            continue

        #sort overlapping pairs by distance between them (closest in age first)
        #a lambda function repeats the same step for every item in the array so is just an alternative to a loop
        #this bit makes it so that overlaps between deposits closest in age are consolidated first
        overlap_pairs_sorted = sorted(overlap_set,
            key=lambda p: ( #p is like the iterator
                abs(core_medians[ci][p[0]] - core_medians[cj][p[1]]),  # get distance between ages
                p[0],# deposit 1 (di)
                p[1]))  # deposit 2 (dj)
        
        #now this loop is going to match the deposits that need to be merged
        #so like if 3 overlap in core1 with 3 in core2, it is going to pair the youngest, middle, and oldest from each core
        #make emtpy sets, these will be reset in each outer loop 
        #so basically new sets for every core-pair so that 
        #deposits can be matched with a different core as well even if they match in this core-pair set
        #this will populate as each deposit is looped through for each core
        matched_ci = set() 
        matched_cj = set() 

        #print which core set we're on for sanity
        print(f"\n{record_names[ci]} vs {record_names[cj]} "
              f"({len(overlap_set)} overlapping pairs):")
        
        #di is the first number of the tuple, dj is the second number of the tuple
        for di, dj in overlap_pairs_sorted:

            #each deposit is only allowed one partner in this specific core pair comparison
            #so if either deposit is already in the matched list, skip it
            if di in matched_ci or dj in matched_cj:
                continue

            #make the node labels
            node1 = (ci, di) #core, deposit
            node2 = (cj, dj)

            #make sure deposits are not circularly connecting within the same core (they're separate!! stratigraphic order!!)
            #nx.node_connect_component is built into network graphing function to return node labels
            #here it will be a list of all the deposits that would be merged with node1 (deposit we're on in this loop)
            #if the edge of node2 is added to the graph (so like, if node2 is JPC05 deposit1 and it has already been connected to VC23 deposit1,
            #if we connect it to VC23 deposit 2 as well, now we're merging two distributions from the same core) -- we don't want this
            # rule here is that each deposit can only have one deposit pair in each CORE-PAIR
            cluster1 = set(nx.node_connected_component(tie_graph, node1)) #set of tuples (core #, dep#) that are connected to node1
            # if the deposit being compared is already connected to the node (i.e. in the cluster), skip (prevent circular mega clustering)
            if node2 in cluster1:
                continue
            cluster2 = set(nx.node_connected_component(tie_graph, node2)) #
            
            #merge the components to see if it would make any bad combos
            merged = cluster1 | cluster2 #this is a python "set" variable "union" which is combining the sets

            #check here if the merged component would contain two deposits from the same core
            #if true, skip this pair
            conflict = False #defalult is conflict is false
            core_count = defaultdict(int) #function to count the deposits per core
            for c, d in merged:  #core labels
                core_count[c] += 1
                if core_count[c] > 1: #if more than one c label in this count, means deposit being double matched
                    conflict = True #label this pair as a conflict, exit loop
                    break
                
            #if conflict labeled true, skip this pair with continue
            if conflict:
                print(f"  SKIP  {record_names[ci]}[{di}]"
                      f"({core_medians[ci][di]:.0f}) ↔ "
                      f"{record_names[cj]}[{dj}]"
                      f"({core_medians[cj][dj]:.0f})"
                      f"  ← same-core conflict")
                continue

            #check here to see if any transitive overlap would be created 
            #this is if A-B overlap and B-C overlap but A-C don't overlap
            merged_list = list(merged)
            for ia in range(len(merged_list)): #loop through all possible merges associated with this deposit1
                for ib in range(ia + 1, len(merged_list)): #loop through all possible merges associated with deposit we're potentially connecting to
                    na = merged_list[ia]
                    nb = merged_list[ib]
                    if frozenset([na, nb]) not in direct_overlap_set: #compare to the frozen set of overlaps at the beginning
                        conflict = True #if not a direct match skip this pair
                        break
                if conflict:
                    break

            #print transitive overlaps
            if conflict:
                print(f"  SKIP  {record_names[ci]}[{di}]"
                      f"({core_medians[ci][di]:.0f}) ↔ "
                      f"{record_names[cj]}[{dj}]"
                      f"({core_medians[cj][dj]:.0f})"
                      f"  ← transitive overlap (not all members overlap directly)")
                continue

            #if there's no conflict, add the edge and label the deposits as matched
            tie_graph.add_edge(node1, node2)
            matched_ci.add(di) # add to already matched list
            matched_cj.add(dj) 
            
            #get distance in median ages between matched pairs
            age_gap = abs(core_medians[ci][di] - core_medians[cj][dj])  
            
            #print all the overlapping pairs to be merged and their age distances
            print(f"  MATCH {record_names[ci]}[{di}]"
                  f"({core_medians[ci][di]:.0f}) ↔ "
                  f"{record_names[cj]}[{dj}]"
                  f"({core_medians[cj][dj]:.0f})"
                  f"  [Δ = {age_gap:.0f} yr]")

print('time to determine overlapping distributions is ',
      (time.time() - sorder_start) / 60, ' minutes')
#%% Create Compiled Distributions

#empty array to put the new distributions in
combined_dists = []  
             
#make an empty list of each core name that will fill with the deposit indexes to be kept from each core 
#on each iteration of the loop below
core_keepers   = defaultdict(list) 

for comp in nx.connected_components(tie_graph):
    nodes = list(comp)

    # if there is no matching deposit, let this distribution stay as is
    if len(nodes) == 1:
        ci, di = nodes[0]         
        core_keepers[ci].append(di)
        continue

    #if there is a matching deposit, combine distributions
    
    #ci is core iterator, di is deposit iteratior
    temp_dists = np.nan * np.zeros(sim_len)
    for ci, di in nodes:                            
        temp_dists = np.vstack((temp_dists, record_dists[ci][:, di])) #get distribution of the specific deposit we're on                    
    cmean = np.nanmean(temp_dists, axis=0)

    # Concatenate
    combined_dists.append(cmean)
    
    #print which deposits combined and what the median ages were
    node_labels = [(record_names[ci], di) for ci, di in sorted(nodes)]  
    node_meds   = [f"{core_medians[ci][di]:.0f}" for ci, di in sorted(nodes)]  
    print(f"Combined ({len(nodes)} deposits): {node_labels}  "
          f"medians: {node_meds}")

            
#%% Save Distributions for Rate Parameter Analysis      
VC23_keepers = np.sort(np.array(core_keepers.get(0, []), dtype=int))
JPC05_keepers = np.sort(np.array(core_keepers.get(1, []), dtype=int))
VC11_keepers = np.sort(np.array(core_keepers.get(2, []), dtype=int))

#print how many it's keeping from each analysis
print(f"JPC05 keepers ({len(JPC05_keepers)}): {JPC05_keepers}")
print(f"VC23 keepers ({len(VC23_keepers)}): {VC23_keepers}")
print(f"VC11 keepers ({len(VC11_keepers)}): {VC11_keepers}")


#all distributions have 5000 members
iterations=5000

#make sure there are combinations
if combined_dists:
    tied_matrix = np.zeros((iterations, len(combined_dists))) #size of 5000 x # combined distributions
    
    for i in range(0, len(combined_dists)):
        tied_matrix[:, i] = combined_dists[i]
        
else: #if no combinations just make this empty and we'll ignore it for the final compilation
    tied_matrix = np.empty((iterations, 0))
    
#### make sure there is data to keep
#keeper matrixes can be empty if all deposits from that core get combined, or there are no combinations

#define function to keep empty array if no deposits kept from the core
def safe_slice(age_dist, keepers):
    if len(keepers)==0:
        return np.empty((age_dist.shape[0],0)) #make empty array
    return age_dist[:,keepers]

#get final distributions
JPC05_dist_keep = safe_slice(JPC05_age_dist, JPC05_keepers)
VC23_dist_keep = safe_slice(VC23_age_dist, VC23_keepers)
VC11_dist_keep = safe_slice(VC11_age_dist, VC11_keepers)


#stack combined distributions
composite_distributions = np.hstack((tied_matrix, JPC05_dist_keep, VC23_dist_keep, VC11_dist_keep))
np.savetxt(r"Compilation_Data/composite_dist.csv", composite_distributions, delimiter=',')

n_kept = (len(JPC05_keepers) + len(VC23_keepers) +
          len(VC11_keepers))

#### print summary of what was kept
print(f"\nComposite : {composite_distributions.shape[1]} total events")
print(f"  {len(combined_dists)} combined  +  {n_kept} kept unchanged")
print(f"    JPC05={len(JPC05_keepers)}, VC23={len(VC23_keepers)}, "
      f"VC11={len(VC11_keepers)}")
print(f"\nTotal runtime : {(time.time() - code_start)/60:.2f} min")

#%% Get Percentiles For Plotting


#### JPC05
#percentiles
JPC05_keepers5 = np.percentile(JPC05_dist_keep, 5, axis=0)
JPC05_keepers95 = np.percentile(JPC05_dist_keep, 95, axis=0)
JPC05_keepers50 = np.median(JPC05_dist_keep, axis=0)
#errors
lowerr = JPC05_keepers50 - JPC05_keepers5
higherr = JPC05_keepers95 - JPC05_keepers50
xerr = np.stack((lowerr, higherr), axis=1)
xerr_JPC05_keep = xerr.T

#### VC23
#percentiles
VC23_keepers5 = np.percentile(VC23_dist_keep, 5, axis=0)
VC23_keepers95 = np.percentile(VC23_dist_keep, 95, axis=0)
VC23_keepers50 = np.median(VC23_dist_keep, axis=0)
#errors
lowerr = VC23_keepers50 - VC23_keepers5
higherr = VC23_keepers95 - VC23_keepers50
xerr = np.stack((lowerr, higherr), axis=1)
xerr_VC23_keep = xerr.T

#### VC11
#percentiles
VC11_keepers5 = np.percentile(VC11_dist_keep, 5, axis=0)
VC11_keepers95 = np.percentile(VC11_dist_keep, 95, axis=0)
VC11_keepers50 = np.median(VC11_dist_keep, axis=0)
#errors
lowerr = VC11_keepers50 - VC11_keepers5
higherr = VC11_keepers95 - VC11_keepers50
xerr = np.stack((lowerr, higherr), axis=1)
xerr_VC11_keep = xerr.T

#### Ties
#percentiles
ties50 = np.median(tied_matrix, axis=0)
ties5 = np.percentile(tied_matrix, 5, axis=0)
ties95 = np.percentile(tied_matrix, 95, axis=0)
#errors
lowerr = ties50 - ties5
higherr = ties95 - ties50
xerr = np.stack((lowerr, higherr), axis=1)
xerr_ties = xerr.T

#%% Plot Median Compilations

#get y axis values
JPC05_y = 3 * np.ones(len(JPC05_events50))
VC23_y = 2 * np.ones(len(VC23_events50))
VC11_y = 1 * np.ones(len(VC11_events50))


JPC05_y2 = 0 * np.ones(len(JPC05_keepers50))
VC23_y2 = 0 * np.ones(len(VC23_keepers50))
VC11_y2 = 0 * np.ones(len(VC11_keepers50))
ties_y2 = 0 * np.ones(len(ties50))


plt.figure(figsize=(20,5), dpi=150)

plt.errorbar(ties50, ties_y2, yerr=None, xerr=xerr_ties, ms=10, capsize=5, fmt='o', label='Combined', color='slategray', alpha=0.5)
plt.errorbar(JPC05_keepers50, JPC05_y2, yerr=None, xerr=xerr_JPC05_keep, ms=10, capsize=5, fmt='o', color='cornflowerblue', alpha=0.5) 
plt.errorbar(VC23_keepers50, VC23_y2, yerr=None, xerr=xerr_VC23_keep, ms=10, capsize=5, fmt='o', color='mediumblue', alpha=0.5) 
plt.errorbar(VC11_keepers50, VC11_y2, yerr=None, xerr=xerr_VC11_keep, ms=10, capsize=5, fmt='o', color='blueviolet', alpha=0.5) 
plt.errorbar(JPC05_events50, JPC05_y, yerr=None, xerr=xerr_JPC05, ms=10, capsize=5, fmt='o', label='JPC05', color='cornflowerblue', alpha=0.5)
plt.errorbar(VC23_events50, VC23_y, yerr=None, xerr=xerr_VC23, ms=10, capsize=5, fmt='o', label='VC23', color='mediumblue', alpha=0.5)
plt.errorbar(VC11_events50, VC11_y, yerr=None, xerr=xerr_VC11, ms=10, capsize=5, fmt='o', label='VC11', color='blueviolet', alpha=0.5)
plt.legend()
plt.xlim(min(JPC05_allyears), max(JPC05_allyears))
plt.ylim(-0.6,3.6)
plt.grid(color='lightgray', linestyle=':', linewidth=1, axis='both', which='major')
plt.show()
plt.xlabel('Year (CE)')



#%% Plot the TieGraph with all Overlaps
import colorsys

# axis positions
y_pos      = {0: 3, 1: 2, 2: 1, 3: 0}
y_compile  = 0
row_labels = {3: 'VC23', 2: 'JPC05', 1: 'VC11', 0: 'Compilation'}

ovlp_mats = [globals()[name + '_overlap_matrix'] for name in record_names]

#### Make Color Palette
DISTINCT_PALETTE = [
    '#e6194b',   # vivid red
    '#3cb44b',   # vivid green
    '#4363d8',   # vivid blue
    '#f58231',   # vivid orange
    '#911eb4',   # vivid purple
    '#42d4f4',   # vivid cyan
    '#f032e6',   # vivid magenta
    '#bfef45',   # vivid lime
    '#469990',   # vivid teal
    '#9A6324',   # vivid brown
    '#800000',   # maroon
    '#000075',   # navy
    '#aaffc3',   # mint
    '#808000',   # olive
    '#dcbeff',   # lavender
    '#fffac8',   # cream yellow
    '#ffd8b1',   # apricot
    '#a9a9a9',   # grey
]

def get_distinct_colors(n):
    if n == 0:
        return []
    if n <= len(DISTINCT_PALETTE):
        return DISTINCT_PALETTE[:n]
    return ['#%02x%02x%02x' % tuple(int(c * 255) for c in
            colorsys.hsv_to_rgb(i / n, 0.85, 0.85))
            for i in range(n)]

#### Get Unique Colors 
components  = list(nx.connected_components(tie_graph)) #get all connections
multi_comps = [c for c in components if len(c) > 1] #connected nodes
solo_comps  = [c for c in components if len(c) == 1] #isolated nodes

n_groups  = len(multi_comps) #number of groups we need
group_clr = get_distinct_colors(n_groups) #color each group

node_color = {}
comp_pts   = []

# multi-node components
for k, comp in enumerate(multi_comps):
    clr = group_clr[k]
    for node in comp:
        node_color[node] = clr
    stacked = np.vstack([record_dists[ci][:, di] for ci, di in comp])
    cmean   = np.mean(stacked, axis=0)
    med     = float(np.median(cmean))
    lo      = float(med - np.percentile(cmean,  5))
    hi      = float(np.percentile(cmean, 95) - med)
    comp_pts.append(dict(x=med, lo=lo, hi=hi, color=clr))

# single-node components
for comp in solo_comps:
    node   = next(iter(comp))
    node_color[node] = 'black'
    ci, di = node
    samp   = record_dists[ci][:, di]
    med    = float(np.median(samp))
    lo     = float(med - np.percentile(samp,  5))
    hi     = float(np.percentile(samp, 95) - med)
    comp_pts.append(dict(x=med, lo=lo, hi=hi, color='black'))

#### Get All Overlap Lines
line_pairs = []

for ci in range(len(record_dists)):
    for cj in range(ci + 1, len(record_dists)):
        mat_ci = ovlp_mats[ci]
        mat_cj = ovlp_mats[cj]
        ovlp   = set()

        if mat_ci[cj] is not None:
            for di, olist in enumerate(mat_ci[cj]):
                for dj in olist:
                    ovlp.add((di, int(dj)))

        if mat_cj[ci] is not None:
            for dj, olist in enumerate(mat_cj[ci]):
                for di in olist:
                    ovlp.add((int(di), dj))

        for di, dj in ovlp:
            line_pairs.append(((ci, di), (cj, dj)))

#### Get Preserved Edges
final_line_pairs = []   # list of ((ci,di), (cj,dj), colour)

for node1, node2 in tie_graph.edges():
    clr = node_color.get(node1, 'black')   # both nodes share the same colour
    final_line_pairs.append((node1, node2, clr))

#### Get Error Bars
dep_err = {}
for ci in range(len(record_dists)):
    for di in range(np.size(record_dists[ci], axis=1)):
        samp = record_dists[ci][:, di]
        med  = float(core_medians[ci][di])
        dep_err[(ci, di)] = (float(med - np.percentile(samp,  5)),
                             float(np.percentile(samp, 95) - med))

#### Plot
fig, ax = plt.subplots(figsize=(24, 9), dpi=150)

for y in range(5):
    ax.axhline(y, color='lightgray', lw=0.5, zorder=0)

ax.axhline(0.5, color='gray', lw=1.0, ls='--', alpha=0.6, zorder=0)

#plot all overlaps gray
for (ci, di), (cj, dj) in line_pairs:
    ax.plot(
        [float(core_medians[ci][di]), float(core_medians[cj][dj])],
        [y_pos[ci],                   y_pos[cj]],
        color='black', lw=0.7, alpha=0.25, zorder=1
    )

#plot final connections in group colors
for (ci, di), (cj, dj), clr in final_line_pairs:
    ax.plot(
        [float(core_medians[ci][di]), float(core_medians[cj][dj])],
        [y_pos[ci],                   y_pos[cj]],
        color=clr, lw=2.5, alpha=0.85, zorder=2
    )

#plot error bars in group colors
for ci in range(len(record_dists)):
    y = y_pos[ci]
    for di in range(np.size(record_dists[ci], axis=1)):
        x   = float(core_medians[ci][di])
        clr = node_color.get((ci, di), 'black')
        lo, hi = dep_err[(ci, di)]
        ax.errorbar(x, y, xerr=[[lo], [hi]],
                    fmt='none', color=clr, capsize=4,
                    lw=1.2, alpha=0.55, zorder=3)

for pt in comp_pts:
    ax.errorbar(pt['x'], y_compile,
                xerr=[[pt['lo']], [pt['hi']]],
                fmt='none', color=pt['color'], capsize=4,
                lw=1.2, alpha=0.55, zorder=3)

#plot median deposit ages
for ci in range(len(record_dists)):
    y = y_pos[ci]
    for di in range(np.size(record_dists[ci], axis=1)):
        x   = float(core_medians[ci][di])
        clr = node_color.get((ci, di), 'black')
        ax.plot(x, y, 'o', color=clr, ms=10,
                markeredgewidth=0, zorder=4)

for pt in comp_pts:
    ax.plot(pt['x'], y_compile, 'o', color=pt['color'],
            ms=10, markeredgewidth=0, zorder=4)

#labels
ytick_vals = sorted(row_labels)
ax.set_yticks(ytick_vals)
ax.set_yticklabels([row_labels[k] for k in ytick_vals], fontsize=12)
ax.set_xlabel('Year (CE)', fontsize=12)
ax.set_title('95% Threshold', fontsize=13)
ax.set_xlim(min(JPC05_allyears), max(JPC05_allyears))
ax.set_ylim(-0.6, 3.6)
ax.grid(True, axis='x', alpha=0.3)

plt.tight_layout()



#%% Define Rate Parameter Function

def rate_parameter(allyrs, eventyrs, h):
    #initialize the rate parameter
    l = np.nan*(np.zeros(len(allyrs))) #l is lambda as in Mudelsee Eq 3.16
    
    for i in range(0,len(allyrs)): #loop through every year of the dataset
        l[i] = 0 
        for j in range(0, len(eventyrs)):
            y = (allyrs[i] - eventyrs[j]) / h
            l[i] = l[i] + (np.power((2*np.pi), -0.5) * np.exp((-y**2)/2))/h
    return(l*100)

#%% Load Frequency Data

#composite distribution (if not running compilation above)
comp_dist = np.genfromtxt(r"Compilation_Data/composite_dist.csv", delimiter=",", skip_header=0)
eventyrs = np.median(comp_dist, axis=0)

strmyrs = np.isin(JPC05_allyears, eventyrs.astype(int))
comp_stormyrs = np.savetxt(r'Compilation_Data/strmyrs.csv', strmyrs, delimiter=",")



#%% Bootstrap Composite

start_time = time.time()

#iterations 
iterations = 1000
#window size
h = 50
#initalize empty matrix
l_matrix_comp = np.zeros((iterations, len(JPC05_allyears))) #use AM4 because it's longest

#set age dist to draw from
age_dist = comp_dist

for i in range(0, iterations):
    
    #draw a random age from the probability distributions
    amax = len(age_dist) #maximum index
    amin = 0 # minimum index
    age_ind = np.random.randint(amin, high=amax, size=1) #get random index
    
    #draw random sample of event years from the random iteration
    eventyrs_temp = age_dist[age_ind,:]
    eventyrs_int = np.squeeze(eventyrs_temp.astype(int)) #convert to integer
    
    #create new set of event years with replacement, keeping the same total number of events and preserving the stratigraphic order
    rmax = np.size(age_dist, axis=1) #maximum event number index
    rmin = 0 #minimum event number index
    event_ind = np.random.randint(rmin, high=rmax, size=np.size(age_dist, axis=1))
    eventyrs_boot = eventyrs_int[event_ind]
    
    strmyrs_boot = np.isin(JPC05_allyears, eventyrs_boot) #boolean index of event years
    
    #rate parameter analyis on the new timeseries of event years
    l_temp = rate_parameter(JPC05_allyears, eventyrs_boot, h)
    
    #save lambda parameters
    l_matrix_comp[i,:] = l_temp
    
np.savetxt(r"Compilation_Data/l_matrix_comp.csv", l_matrix_comp, delimiter=',')

print('time to run bootstrapping is ', (time.time() - start_time)/60, ' minutes')  

#%% Load bootstrap data if not bootstrapping

l_matrix_comp = np.genfromtxt(r"Compilation_Data/l_matrix_comp.csv", delimiter=',', skip_header=0)

# individual rate parameter matrixes
JPC05_lmatrix = np.genfromtxt(r"JPC05_CoreData/l_matrix.csv", delimiter=",", skip_header=0)
VC23_lmatrix = np.genfromtxt(r"VC23_CoreData/l_matrix.csv", delimiter=",", skip_header=0)
VC11_lmatrix = np.genfromtxt(r"VC11_CoreData/l_matrix.csv", delimiter=",", skip_header=0)

# boolean event times
JPC05_storm_years = np.genfromtxt(r"JPC05_CoreData/storm_years.csv", delimiter=",", skip_header=0)
VC23_storm_years = np.genfromtxt(r"VC23_CoreData/storm_years.csv", delimiter=",", skip_header=0)
VC11_storm_years = np.genfromtxt(r"VC11_CoreData/storm_years.csv", delimiter=",", skip_header=0)

#%% Plot 100yr freq

######## Get statistics

#### JPC05
mean_rate_JPC05 = np.median(JPC05_lmatrix, axis=0)
low_rate2_JPC05 = np.percentile(JPC05_lmatrix, 5, axis=0)
low_rate1_JPC05 = np.percentile(JPC05_lmatrix, 16, axis=0)
high_rate2_JPC05 = np.percentile(JPC05_lmatrix, 95, axis=0)
high_rate1_JPC05 = np.percentile(JPC05_lmatrix, 84, axis=0)

#### VC23
mean_rate_VC23 = np.median(VC23_lmatrix, axis=0)
low_rate2_VC23 = np.percentile(VC23_lmatrix, 5, axis=0)
low_rate1_VC23 = np.percentile(VC23_lmatrix, 16, axis=0)
high_rate2_VC23 = np.percentile(VC23_lmatrix, 95, axis=0)
high_rate1_VC23 = np.percentile(VC23_lmatrix, 84, axis=0)

#### VC11
mean_rate_VC11 = np.median(VC11_lmatrix, axis=0)
low_rate2_VC11 = np.percentile(VC11_lmatrix, 5, axis=0)
low_rate1_VC11 = np.percentile(VC11_lmatrix, 16, axis=0)
high_rate2_VC11 = np.percentile(VC11_lmatrix, 95, axis=0)
high_rate1_VC11 = np.percentile(VC11_lmatrix, 84, axis=0)

#### Compilation
mean_rate_C = np.median(l_matrix_comp, axis=0)
low_rate2_C = np.percentile(l_matrix_comp, 5, axis=0)
low_rate1_C = np.percentile(l_matrix_comp, 16, axis=0)
high_rate2_C = np.percentile(l_matrix_comp, 95, axis=0)
high_rate1_C = np.percentile(l_matrix_comp, 84, axis=0)




#%% Final Figure

#### Plot
fig,ax = plt.subplots(2,1, dpi=150)
ax0 = ax[0]
ax1 = ax[1]


#individual records
ax0.grid(color='lightgray', linestyle=':', linewidth=0.7, axis='both', which='major',zorder=0)
ax0.plot(JPC05_allyears, mean_rate_JPC05, label='JPC05', color='cornflowerblue')
ax0.plot(VC23_allyears, mean_rate_VC23, label='VC23', color='mediumblue')
ax0.plot(VC11_allyears, mean_rate_VC11, label='VC11', color='blueviolet')
ax0.plot(JPC05_allyears, mean_rate_C, color='k', lw=1, linestyle='--', label='Compilation')
ax0.legend()
ax0.set_ylabel(r'Storm Frequency ($\lambda$)', fontname="Arial", fontweight='bold')
ax0.set_xlim(-100, 4000)
ax0.set_xlabel('Age (YBP)', fontname="Arial", fontweight='bold')

#compilation
ax1.grid(color='lightgray', linestyle=':', linewidth=0.7, axis='both', which='major')
ax1.axvspan(xmin=700, xmax=1000, color='tab:red', alpha=0.1, label='MCA',zorder=0)
ax1.axvspan(xmin=100, xmax=650, color='tab:blue', alpha=0.1, label='LIA', zorder=0)
ax1.plot(JPC05_allyears, mean_rate_C, color='k', lw=2, label='Compilation')
ax1.fill_between(JPC05_allyears, low_rate2_C, high_rate2_C, color = 'lightgray', edgecolor='darkgray', linewidth=0.5, label=r'1.65$\sigma$', zorder=1)
ax1.fill_between(JPC05_allyears, low_rate1_C, high_rate1_C, color='darkgray', edgecolor='gray', linewidth=0.5, label = r'1$\sigma$', zorder=1)
ax1.stem(JPC05_allyears, strmyrs, 'k', markerfmt =" ", basefmt=" ")
ax1.set_ylabel(r'Storm Frequency ($\lambda$)', fontname="Arial", fontweight='bold')
ax1.legend()
ax1.set_xlim(-100, 4000)
ax1.set_xlabel('Age (YBP)', fontname="Arial", fontweight='bold')



#%% Plot All Three Frequencies with Error Bars For Supplemental 

fig, ax = plt.subplots(3,1, dpi=150)
ax0 = ax[0]
ax1 = ax[1]
ax2 = ax[2]

#### JPC05
ax0.grid(color='lightgray', linestyle=':', linewidth=0.7, axis='both', which='major',zorder=0)
ax0.plot(JPC05_allyears, mean_rate_JPC05, color='k', lw=2, label='JPC05')
ax0.axvspan(xmin=700, xmax=1000, color='tab:red', alpha=0.1, label='MCA',zorder=0)
ax0.axvspan(xmin=100, xmax=650, color='tab:blue', alpha=0.1, label='LIA', zorder=0)
ax0.fill_between(JPC05_allyears, low_rate2_JPC05, high_rate2_JPC05, color = 'lightgray', edgecolor='darkgray', linewidth=0.5, label=r'1.65$\sigma$', zorder=1)
ax0.fill_between(JPC05_allyears, low_rate1_JPC05, high_rate1_JPC05, color='darkgray', edgecolor='gray', linewidth=0.5, label = r'1$\sigma$', zorder=1)
ax0.stem(JPC05_allyears, JPC05_storm_years, 'k', markerfmt =" ", basefmt=" ")
ax0.set_ylabel(r'Storm Frequency ($\lambda$)')
ax0.legend()
ax0.set_xlim(-300, 8300)
ax0.set_xlabel('Years Before Present (YBP)')


#### VC23
ax1.grid(color='lightgray', linestyle=':', linewidth=0.7, axis='both', which='major',zorder=0)
ax1.plot(VC23_allyears, mean_rate_VC23, color='k', lw=2, label='VC23')
ax1.axvspan(xmin=700, xmax=1000, color='tab:red', alpha=0.1, label='MCA',zorder=0)
ax1.axvspan(xmin=100, xmax=650, color='tab:blue', alpha=0.1, label='LIA', zorder=0)
ax1.fill_between(VC23_allyears, low_rate2_VC23, high_rate2_VC23, color = 'lightgray', edgecolor='darkgray', linewidth=0.5, label=r'1.65$\sigma$', zorder=1)
ax1.fill_between(VC23_allyears, low_rate1_VC23, high_rate1_VC23, color='darkgray', edgecolor='gray', linewidth=0.5, label = r'1$\sigma$', zorder=1)
ax1.stem(VC23_allyears, VC23_storm_years, 'k', markerfmt =" ", basefmt=" ")
ax1.set_ylabel(r'Storm Frequency ($\lambda$)')
ax1.legend()
ax1.set_xlim(-300, 8300)
ax1.set_xlabel('Years Before Present (YBP)')

#### VC11
ax2.grid(color='lightgray', linestyle=':', linewidth=0.7, axis='both', which='major',zorder=0)
ax2.plot(VC11_allyears, mean_rate_VC11, color='k', lw=2, label='VC11')
ax2.axvspan(xmin=700, xmax=1000, color='tab:red', alpha=0.1, label='MCA',zorder=0)
ax2.axvspan(xmin=100, xmax=650, color='tab:blue', alpha=0.1, label='LIA', zorder=0)
ax2.fill_between(VC11_allyears, low_rate2_VC11, high_rate2_VC11, color = 'lightgray', edgecolor='darkgray', linewidth=0.5, label=r'1.65$\sigma$', zorder=1)
ax2.fill_between(VC11_allyears, low_rate1_VC11, high_rate1_VC11, color='darkgray', edgecolor='gray', linewidth=0.5, label = r'1$\sigma$', zorder=1)
ax2.stem(VC11_allyears, VC11_storm_years, 'k', markerfmt =" ", basefmt=" ")
ax2.set_ylabel(r'Storm Frequency ($\lambda$)')
ax2.legend()
ax2.set_xlim(-300, 8300)
ax2.set_xlabel('Years Before Present (YBP)')


#%% Get Statistics

#### JPC05

#rate parameter mean
lmean_JPC05 = np.mean(mean_rate_JPC05)
lmean_16_JPC05 = np.percentile(mean_rate_JPC05, 16)
lmean_84_JPC05 = np.percentile(mean_rate_JPC05, 84)
lmean_std_JPC05 = np.std(mean_rate_JPC05[:4072])
lmean_var_JPC05 = np.mean(mean_rate_JPC05[:4072])/lmean_std_JPC05

#rate parameter max
ind = np.argmax(mean_rate_JPC05)
lmax_mean_JPC05 = mean_rate_JPC05[ind]
lmax_16_JPC05 = low_rate1_JPC05[ind]
lmax_84_JPC05 = high_rate1_JPC05[ind]

#rate parameter min
ind = np.argmin(mean_rate_JPC05)
lmin_mean_JPC05 = mean_rate_JPC05[ind]
lmin_16_JPC05 = low_rate1_JPC05[ind]
lmin_84_JPC05 = high_rate1_JPC05[ind]

#year of maximum rate parameter
lmax_ind_array = np.argmax(JPC05_lmatrix, axis=1)
year_max=np.zeros(len(lmax_ind_array))
for i in range(0,len(lmax_ind_array)):
    year_max[i] = JPC05_allyears[lmax_ind_array[i]]

yrmax_mean_JPC05 = np.mean(year_max)
yrmax_16_JPC05 = np.percentile(year_max, 16)
yrmax_84_JPC05 = np.percentile(year_max, 84)
yrmax_std_JPC05 = np.std(year_max)


#### VC23

#rate parameter mean
lmean_VC23 = np.mean(mean_rate_VC23)
lmean_16_VC23 = np.percentile(mean_rate_VC23, 16)
lmean_84_VC23 = np.percentile(mean_rate_VC23, 84)
lmean_std_VC23 = np.std(mean_rate_VC23[:4072])
lmean_var_VC23 = np.mean(mean_rate_VC23[:4072])/lmean_std_VC23

#rate parameter max
ind = np.argmax(mean_rate_VC23)
lmax_mean_VC23 = mean_rate_VC23[ind]
lmax_16_VC23 = low_rate1_VC23[ind]
lmax_84_VC23 = high_rate1_VC23[ind]

#rate parameter min
ind = np.argmin(mean_rate_VC23)
lmin_mean_VC23 = mean_rate_VC23[ind]
lmin_16_VC23 = low_rate1_VC23[ind]
lmin_84_VC23 = high_rate1_VC23[ind]


#year of maximum rate parameter
lmax_ind_array = np.argmax(VC23_lmatrix, axis=1)
year_max=np.zeros(len(lmax_ind_array))
for i in range(0,len(lmax_ind_array)):
    year_max[i] = VC23_allyears[lmax_ind_array[i]]

yrmax_mean_VC23 = np.mean(year_max)
yrmax_16_VC23 = np.percentile(year_max, 16)
yrmax_84_VC23 = np.percentile(year_max, 84)
yrmax_std_VC23 = np.std(year_max)


#### VC11
#rate parameter mean
lmean_VC11 = np.mean(mean_rate_VC11)
lmean_16_VC11 = np.percentile(mean_rate_VC11, 16)
lmean_84_VC11 = np.percentile(mean_rate_VC11, 84)
lmean_std_VC11 = np.std(mean_rate_VC11[:4072])
lmean_var_VC11 = np.mean(mean_rate_VC11[:4072])/lmean_std_VC11

#rate parameter mean before/after 2000 YBP
lmean_VC11_pre2000 = np.mean(mean_rate_VC11[:2071])
lmean_16_VC11_pre2000 = np.percentile(mean_rate_VC11[:2071], 16)
lmean_84_VC11_pre2000 = np.percentile(mean_rate_VC11[:2071], 84)

lmean_VC11_post2000 = np.mean(mean_rate_VC11[2071:])
lmean_16_VC11_post2000 = np.percentile(mean_rate_VC11[2071:], 16)
lmean_84_VC11_post2000 = np.percentile(mean_rate_VC11[2071:], 84)

#rate parameter max
ind = np.argmax(mean_rate_VC11)
lmax_mean_VC11 = mean_rate_VC11[ind]
lmax_16_VC11 = low_rate1_VC11[ind]
lmax_84_VC11 = high_rate1_VC11[ind]

#rate parameter min
ind = np.argmin(mean_rate_VC11)
lmin_mean_VC11 = mean_rate_VC11[ind]
lmin_16_VC11 = low_rate1_VC11[ind]
lmin_84_VC11 = high_rate1_VC11[ind]

#year of maximum rate parameter
lmax_ind_array = np.argmax(VC11_lmatrix, axis=1)
year_max=np.zeros(len(lmax_ind_array))
for i in range(0,len(lmax_ind_array)):
    year_max[i] = VC11_allyears[lmax_ind_array[i]]

yrmax_mean_VC11 = np.mean(year_max)
yrmax_16_VC11 = np.percentile(year_max, 16)
yrmax_84_VC11 = np.percentile(year_max, 84)
yrmax_std_VC11 = np.std(year_max)


#### Compilation
mean_rate_C = mean_rate_C[:4072]

#rate parameter mean
lmean_C = np.mean(mean_rate_C)
lmean_16_C = np.percentile(mean_rate_C, 16)
lmean_84_C = np.percentile(mean_rate_C, 84)
lmean_std_C = np.std(mean_rate_C)
lmean_var_C = lmean_C/lmean_std_C

#rate parameter max
ind = np.argmax(mean_rate_C)
lmax_mean_C = mean_rate_C[ind]
lmax_16_C = low_rate1_C[ind]
lmax_84_C = high_rate1_C[ind]

#rate parameter min
ind = np.argmin(mean_rate_C)
lmin_mean_C = mean_rate_C[ind]
lmin_16_C = low_rate1_C[ind]
lmin_84_C = high_rate1_C[ind]

#year of maximum rate parameter
lmax_ind_array = np.argmax(l_matrix_comp[:,:4072], axis=1)
year_max=np.zeros(len(lmax_ind_array))
for i in range(0,len(lmax_ind_array)):
    year_max[i] = VC11_allyears[:4072][lmax_ind_array[i]]

yrmax_mean_C = np.mean(year_max)
yrmax_16_C = np.percentile(year_max, 16)
yrmax_84_C = np.percentile(year_max, 84)
yrmax_std_C = np.std(year_max)


#%% Load Tsunami Data

#### Load Data
#Storms
age_dist = np.genfromtxt(r"Compilation_Data/composite_dist.csv", delimiter=",", skip_header=0)

#Witter Tsunamis
tsunamis = np.genfromtxt(r"Tsunami_Data/Witter_2019_Tsunami_Ages.csv", delimiter=",", skip_header=1)
tmean = tsunamis[:, 1]
tsigma = tsunamis[:, 2]
tmode = tsunamis[:, 3]
tlabels = np.genfromtxt(r"Tsunami_Data/Witter_2019_Tsunami_Ages.csv", delimiter=",", skip_header=1, dtype=str, usecols=0)

#Nelson Tsunamis 
age_dist_nelson = np.genfromtxt(r"Tsunami_Data/Nelson_2015_Fig11_tsunami_dist.csv", delimiter=",")
nelson_metadata = np.genfromtxt(r"Tsunami_Data/Nelson_2015_Fig11_metadata.csv", delimiter=",", skip_header=1, dtype=str)
nlabels = nelson_metadata[:, 1]

#%% Generate Tsunami Distributions 

#### Define Function
np.random.seed(42) #so distributions re reproducible

def skewed_dist(mean, mode, std, scale, size):
    skewness = (mean - mode) / std
    alpha = skewness * scale
    data = skewnorm.rvs(alpha, loc=mean, scale=std, size=size)
    return data

# make empty arrays
scale = 10
size = 5000
dsize = 100

#### Witter Distributions
t_dist = np.zeros((len(tsunamis), dsize))
t_xvals = np.zeros((len(tsunamis), dsize))
t_med = np.zeros(len(tsunamis))

for i in range(0, len(tsunamis)):
    dist = skewed_dist(tmean[i], tmode[i], tsigma[i], scale, size)
    dens = gaussian_kde(dist)
    xvals = np.linspace(min(dist), max(dist), dsize)
    t_dist[i, :] = dens(xvals)
    t_xvals[i, :] = xvals
    t_med[i] = np.median(dist)
    
#### Storm Distributions
s_dist = np.zeros((np.shape(age_dist)[1], dsize))
s_xvals = np.zeros((np.shape(age_dist)[1], dsize))
s_med = np.zeros(np.shape(age_dist)[1])

for i in range(0, np.shape(age_dist)[1]):
    ages = age_dist[:, i]
    ages = ages[~np.isnan(ages)]
    dens = gaussian_kde(ages)
    xvals = np.linspace(min(ages), max(ages), dsize)
    s_dist[i, :] = dens(xvals)
    s_xvals[i, :] = xvals
    s_med[i] = np.median(ages)
    
#### Nelson Distributions
n_dist = np.zeros((np.shape(age_dist_nelson)[1], dsize))
n_xvals = np.zeros((np.shape(age_dist_nelson)[1], dsize))
n_med = np.median(age_dist_nelson, axis=0)

for i in range(0, np.shape(age_dist_nelson)[1]):
    dens = gaussian_kde(age_dist_nelson[:, i])
    xvals = np.linspace(min(age_dist_nelson[:, i]), max(age_dist_nelson[:, i]), dsize)
    n_dist[i, :] = dens(xvals)
    n_xvals[i, :] = xvals
    
#%% Plot Storm and Tsunami Distributions

plt.figure(figsize=(12, 3.5), dpi=150)
plt.xlim(-75, 8000)
plt.grid(color='lightgray', linestyle=':', linewidth=0.7, axis='both', which='major')
plt.xlabel('Years Before Present (YBP)')
plt.ylabel('probability density')

for i in range(0, len(s_dist)):
    if i == 0:
        plt.fill_between(s_xvals[i, :], s_dist[i, :], color='k', alpha=0.1,
                         label="Storms")
    else:
        plt.fill_between(s_xvals[i, :], s_dist[i, :], color='k', alpha=0.1)

for i in range(0, len(t_dist)):
    if i == 0:
        plt.fill_between(t_xvals[i, :], t_dist[i, :], color='skyblue', alpha=0.4,
                         label="Witter Tsunamis")
    else:
        plt.fill_between(t_xvals[i, :], t_dist[i, :], color='skyblue', alpha=0.4)

for i in range(0, len(n_dist)):
    if i == 0:
        plt.fill_between(n_xvals[i, :], n_dist[i, :], color='mediumpurple', alpha=0.4,
                         label="Nelson Tsunamis")
    else:
        plt.fill_between(n_xvals[i, :], n_dist[i, :], color='mediumpurple', alpha=0.4)

plt.legend()
plt.xlim(-150,4100)
plt.ylim(-0.005,0.05)

#%% Find Storm Deposits Closest to Tsunami Deposits

# every tsunami deposit claims the storm deposit whose median age is nearest;
# two tsunamis are allowed to claim the same storm deposit
tsu_med = np.concatenate([t_med, n_med])
tsu_name = np.concatenate([["W-" + s.replace("Sand ", "") for s in tlabels],
                           ["N-" + s for s in nlabels]])

match = np.array([np.argmin(np.abs(s_med - m)) for m in tsu_med])
removed = np.unique(match)# storm deposits to drop
kept = np.setdiff1d(np.arange(len(s_med)), removed)

print(f"{len(removed)} of {len(s_med)} storm deposits matched to a tsunami deposit")
order = np.argsort(tsu_med)
for k in order:
    j = match[k]
    print(f"  {tsu_name[k]:<6}{tsu_med[k]:7.0f} YBP  ->  storm col {j:<4}"
          f"{s_med[j]:7.0f} YBP   (gap {abs(s_med[j]-tsu_med[k]):.0f} yr)")
    

#### Save As CSV

np.savetxt("Tsunami_Data/composite_dist_no_tsunamis.csv", age_dist[:, kept],
           delimiter=",", fmt="%.6E")

with open("storm_tsunami_matches.csv", "w") as f:
    f.write("storm_column,storm_median,status,matched_tsunami,tsunami_median,gap_yr\n")
    for j in range(len(s_med)):
        hits = np.where(match == j)[0]
        if len(hits):
            f.write(f"{j},{s_med[j]:.0f},removed,"
                    f"{'+'.join(tsu_name[h] for h in hits)},"
                    f"{'+'.join(f'{tsu_med[h]:.0f}' for h in hits)},"
                    f"{'+'.join(f'{abs(s_med[j]-tsu_med[h]):.0f}' for h in hits)}\n")
        else:
            f.write(f"{j},{s_med[j]:.0f},kept,,,\n")
            
#%% Plot Distributions Removed For Proximity to Tsunami Deposits
plt.figure(figsize=(10, 3.5), dpi=150)
plt.xlim(-100, 4100)
plt.grid(color='lightgray', linestyle=':', linewidth=0.7, axis='both', which='major')
plt.xlabel('Years Before Present (YBP)')
plt.ylabel('probability density')

for i in range(0, len(s_dist)):
    if i == 0:
        plt.fill_between(s_xvals[i, :], s_dist[i, :], color='k', alpha=0.1,
                         label="Skan Storms")
    else:
        plt.fill_between(s_xvals[i, :], s_dist[i, :], color='k', alpha=0.1)

for i in range(0, len(t_dist)):
    if i == 0:
        plt.fill_between(t_xvals[i, :], t_dist[i, :], color='skyblue', alpha=0.4,
                         label="Witter Tsunamis")
    else:
        plt.fill_between(t_xvals[i, :], t_dist[i, :], color='skyblue', alpha=0.4)

for i in range(0, len(n_dist)):
    if i == 0:
        plt.fill_between(n_xvals[i, :], n_dist[i, :], color='mediumpurple', alpha=0.4,
                         label="Nelson Tsunamis")
    else:
        plt.fill_between(n_xvals[i, :], n_dist[i, :], color='mediumpurple', alpha=0.4)

# stems on the median age of every storm deposit
STEM_HEIGHT = 0.01

plt.vlines(s_med[kept], 0, STEM_HEIGHT, color='k', linewidth=1.5,
           label="Events Kept", zorder=3)
plt.vlines(s_med[removed], 0, STEM_HEIGHT, color='tab:red', linewidth=1.5,
           label="Events Removed", zorder=4)


# deposit labels above each tsunami distribution
LABEL_DEPOSITS = False

if LABEL_DEPOSITS:
    for i in range(0, len(t_dist)):
        xpk = t_xvals[i, np.argmax(t_dist[i, :])]
        plt.text(xpk, max(t_dist[i, :]) * 1.04, tlabels[i].replace('Sand ', ''),
                 ha='center', va='bottom', fontsize=6, color='tab:blue')
    for i in range(0, len(n_dist)):
        xpk = n_xvals[i, np.argmax(n_dist[i, :])]
        plt.text(xpk, max(n_dist[i, :]) * 1.04, nlabels[i], ha='center',
                 va='bottom', fontsize=6, color='rebeccapurple')

plt.legend(ncol=2, fontsize=8)
plt.tight_layout()
plt.show()
plt.ylim(-0.005,0.05)


#%% Now Reboostrap Frequency Without Tsunamis

print('time to start bootstrapping without tsunamis is ', time.ctime())

start_time = time.time()


age_dist_noT = np.genfromtxt(r'Tsunami_Data/composite_dist_no_tsunamis.csv', delimiter=',', skip_header=0)

#### Initialize
#set number of iterations
iterations = 1000
#set window size
h = 50
#initialize empty matrix
l_matrix_noT = np.zeros((iterations, len(JPC05_allyears)))
event_matrix_bool_noT = np.nan * np.zeros((iterations,len(JPC05_allyears)))


#### Bootstraps
for ii in range(0,iterations):
    
    #draw a random event age from the Bacon probability distributions
    amax = len(age_dist_noT) #maximum index
    amin = 0 #minimum index
    age_ind = np.random.randint(amin, high=amax, size=1) #get random index
    
    #make alternative set of event years from the random age model iteration
    eventyrs_temp = age_dist_noT[age_ind,:]
    eventyrs_int = np.squeeze(eventyrs_temp.astype(int)) #convert to integer
    
    #create a new set of event years with replacement from the intermediate event years 
    #keeping the same total number of events
    #preserving stratigraphic order
    rmax = np.size(age_dist_noT,1) #maximum index
    rmin = 0 #minimum index
    event_ind = np.random.randint(rmin, high=rmax, size=np.size(age_dist_noT,1)) #rmax is exclusive of the final element so add 1
    eventyrs_boot = eventyrs_int[event_ind]
    
    strmyrs_boot = np.isin(JPC05_allyears, eventyrs_boot) #boolean index of event years
    
    #perform rate parameter analysis on the new timeseries of event years
    l_temp = rate_parameter(JPC05_allyears, eventyrs_boot, h)
    
    # plt.plot(allyrs, l_temp, color='k', alpha=0.1)
    
    l_matrix_noT[ii,:] = l_temp
    event_matrix_bool_noT[ii,:] = strmyrs_boot


print('time to run bootstrapping without tsunamis is ', (time.time() - start_time)/60, ' minutes') 

#### save output
np.savetxt(r"Tsunami_Data/l_matrix_no_tsunamis.csv", l_matrix_noT, delimiter=',')

#%% Get Stats of No Tsunami 

#### load bootstrap data here if not running now
l_matrix_noT = np.genfromtxt(r"Tsunami_Data/l_matrix_no_tsunamis.csv", delimiter=",", skip_header=0)


#### Get Stats   
mean_rate_ages_noT  = np.median(l_matrix_noT , axis=0)
low_rate2_ages_noT  = np.percentile(l_matrix_noT , 5, axis=0)
low_rate1_ages_noT  = np.percentile(l_matrix_noT , 16, axis=0)
high_rate2_ages_noT  = np.percentile(l_matrix_noT , 95, axis=0)
high_rate1_ages_noT  = np.percentile(l_matrix_noT , 84, axis=0)


#%% Plot Frequencies with and without Tsunamis

plt.figure(figsize=(10, 3.5), dpi=150)
plt.plot(JPC05_allyears[:4072], mean_rate_C, color='k', label = 'Bootstrapped Mean, All Events')
plt.fill_between(JPC05_allyears, low_rate2_C, high_rate2_C, color = 'k', alpha=0.1, label=r'2$\sigma$, All Events')
plt.fill_between(JPC05_allyears, low_rate1_C, high_rate1_C, color='k', alpha=0.3, label = r'1$\sigma$, All Events')
plt.plot(JPC05_allyears, mean_rate_ages_noT, color='navy', ls='--', label = 'Bootstrapped Mean, "Tsunamis" Removed')
plt.fill_between(JPC05_allyears, low_rate2_ages_noT, high_rate2_ages_noT, color = 'tab:blue', alpha=0.1, label=r'2$\sigma$, "Tsunamis" Removed')
plt.fill_between(JPC05_allyears, low_rate1_ages_noT, high_rate1_ages_noT, color='tab:blue', alpha=0.3, label = r'1$\sigma$, "Tsunamis" Removed')
plt.legend()
plt.xlabel('Years Before Present (YBP)')
plt.ylabel(r'Storm Frequency ($\lambda$)')
plt.xlim(-100,4100)
plt.grid(color='lightgray', linestyle=':', linewidth=0.7, axis='both', which='major')

