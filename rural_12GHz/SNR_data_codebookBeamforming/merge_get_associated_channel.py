# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 19:48:42 2023

@author: gangs
"""
import gzip
import numpy as np
#import glob
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

########################## merge SNR files and make one file ##########################
#files = glob.glob("SNR_map_*.gzip")
print("================= merging SNR files of each BS ======================")
merged = np.zeros((104, 8496))
for i in tqdm(range(104)):
    f = f'SNR_map_{i+1}.gzip'
    with gzip.open(f) as ff:
        data = np.load(ff)
    merged[i] = data

f = gzip.open(f'total_SNR.gzip', 'wb')
np.save(file = f, arr =merged)
f.close()


###################### make associated channel file #########################
print("===== createing channle files including all associated channel between UE and BS")
with gzip.open('total_SNR.gzip') as ff:
    data = np.load(ff)
associated_bs_indices =np.nanargmax(data, axis = 0)
n_RX = len(associated_bs_indices)


interation_keys = ['interactions']
for j in range(1,25):
    interation_keys.append(f'interactions.{j}')

df_dict = dict()
for i in tqdm(range(104)):
    df = pd.read_csv(f'../parsed_data_bs_to_ue/bs_{i+1}.csv', engine = 'python')
    df_dict[i+1] = df.drop(interation_keys, axis = 1)
   
df_ = df.drop(interation_keys, axis =1)
keys_ = list(df_.keys())
keys_.append('bs_idx')

i0 = associated_bs_indices[0]
df0 = pd.read_csv('../parsed_data_bs_to_ue/bs_%d.csv'%i0, engine = 'python')
df0 = df0.iloc[0]
n_keys = len(df0)

df_all = []
for j in tqdm(range(len(associated_bs_indices))):# over all Rx points
    bs_idx = associated_bs_indices[j] 
    df_j = df_dict[bs_idx+1] # bs index 0 corresponds to 1 in file
    df_j = df_j.iloc[j].to_list() # get channel for j'th point
    df_all.append(df_j)
df_all = np.array(df_all)
df_all = np.append( df_all, associated_bs_indices[:,None], axis = 1)

    
df_all = pd.DataFrame(df_all, columns=keys_)
#df_all = df_all[~np.isnan(df_all['avg_path_gain'])]
df_all.to_csv('associated_chan.csv')
