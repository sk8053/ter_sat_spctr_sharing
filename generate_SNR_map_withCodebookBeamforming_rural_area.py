# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 8:49:11 2023
Generate SNR values based on VQ codebook beamforming
Firstly set up antennas for each frequency 
SNR values of each BS are saved in SNR_data_codebookBeamforming
@author: seongjoon kang
"""
import numpy as np
import glob
import pandas as pd
from tqdm import tqdm
import sys
import os
sys.path.append(os.getcwd()+"/uav_interference_analysis")
from src.mmwchanmod.sim.get_channel_from_ray_tracing_data import get_channel_from_ray_tracing_data
from src.mmwchanmod.sim.array import URA, RotatedArray, multi_sect_array
from src.mmwchanmod.sim.antenna import Elem3GPP
from src.mmwchanmod.common.constants import PhyConst
from src.mmwchanmod.common.spherical import cart_to_sph
#from src.mmwchanmod.sim.drone_antenna_field import drone_antenna_gain
from src.mmwchanmod.sim.chanmod import dir_path_loss_multi_sect
import matplotlib.pyplot as plt
import gzip
import pickle
if 0:
    r_azm = np.random.uniform(low = -180, high = 180, size= (200, 10000))
    r_elev = np.random.uniform(low = -90, high = 90, size= (200, 10000))
    np.savetxt('random_azimuth_angles_new.txt', r_azm)
    np.savetxt('random_elev_angles_new.txt', r_elev)
    
n_bs = 104
rand_azm_factory = np.loadtxt('rural_12GHz/random_azm_angles.txt')
rand_elev_factory = np.loadtxt('rural_12GHz/random_elev_angles.txt')

in_out = 'outdoor'
for frequency in [12e9]: #12e9,  24e9
    if frequency == 6e9:##
        nant_ue = np.array([1,2])
        nant_gnb = np.array([2,2])
        nant_n_UE = nant_ue[0]*nant_ue[1]
        nant_n_gnb = nant_gnb[0]*nant_gnb[1]
        file_name1 = 'ue_codebook_6GHz.txt'
        file_name2 = 'bs_codebook_6GHz_rural.txt'
        
        dir_ = 'rural_scenario_Remcomm_6GHz/'
        
    
    elif frequency ==12e9: ##
        nant_ue = np.array([1,2])
        nant_gnb = np.array([8,8])
        nant_n_UE = nant_ue[0]*nant_ue[1]
        nant_n_gnb = nant_gnb[0]*nant_gnb[1]
        file_name1 = 'ue_codebook_12GHz.txt'
        file_name2 = 'bs_codebook_12GHz_rural.txt'
    
        dir_= 'rural_12GHz/'
        
    elif frequency == 18e9:
        nant_ue = np.array([1,3])
        nant_gnb = np.array([5,5])
        nant_n_UE = nant_ue[0]*nant_ue[1]
        nant_n_gnb = nant_gnb[0]*nant_gnb[1]
        file_name1 = 'ue_codebook_18GHz.txt'
        file_name2 = 'bs_codebook_18GHz_rural.txt'
        
        dir_= 'rural_scenario_Remcomm_18GHz/'
        
    elif frequency == 24e9: #
        nant_ue = np.array([1,3])
        nant_gnb = np.array([7,7])
        nant_n_UE = nant_ue[0]*nant_ue[1]
        nant_n_gnb = nant_gnb[0]*nant_gnb[1]
        file_name1 = 'ue_codebook_24GHz.txt'
        file_name2 = 'bs_codebook_24GHz_rural.txt'
        
        dir_= 'rural_scenario_Remcomm_24GHz/'
        
    else:
        ValueError(f"Unknown frequency{frequency} GHz")
        
    BW_dict ={6e9:100e6, 12e9:200e6, 18e9:300e6, 24e9:400e6}
    BW = BW_dict[frequency]
    tx_power = 33 # dB maximum TX power of BS
    KT = -174
    NF = 7
    
    thetabw_, phibw_ = 65, 65
    lam = PhyConst.light_speed / frequency
    
    element_UE = Elem3GPP(thetabw=thetabw_, phibw=phibw_)
    elem_bs = Elem3GPP(thetabw=thetabw_, phibw=phibw_)
    
    # create the Uniform Retangular Array for BS and UE
    #drone_antenna_gain = drone_antenna_gain()
    arr_ue_ = URA(elem=element_UE, nant=nant_ue, fc=frequency) # UE has 4*4
    arr_gnb_init = URA(elem=elem_bs, nant=nant_gnb,  fc= frequency) # gNB has 8*8
    # configure the antenna array of BSs having 3-sectors and down-tilted by -12
    arr_gnb = multi_sect_array(arr_gnb_init, sect_type='azimuth', nsect=3, theta0=-12)
    
    ue_cbook = np.loadtxt(f'{dir_}vq_codebook/'+file_name1, dtype = complex)
    bs_cbook = np.loadtxt(f'{dir_}vq_codebook/'+file_name2, dtype = complex)
    #rand_azms = np.random.uniform(-np.pi, np.pi, (22950,))
    #for index in range(50):
    for bs_ind in range(n_bs): #79, 104
        
        data = pd.read_csv(dir_+'parsed_data_bs_to_ue/bs_%d.csv'%(bs_ind+1), engine = 'python')
        
        SNR_map = []
        rand_azm_list, rand_elev_list = [] , []
        bs_sec_ind_list =[]
        
        for i in tqdm(range(len(data))):
            rand_azm = rand_azm_factory[bs_ind, i]
            rand_elev = rand_elev_factory[bs_ind, i]
            if np.array(np.isnan(data.iloc[i]['n_path'])):
                path_ = np.NaN

                SNR_map.append(-200)
                rand_azm_list.append(rand_azm)
                rand_elev_list.append(rand_elev)
                bs_sec_ind_list.append(1)
                
            else:

                # the array of UE is downl-tilted by -90,rand_azms[i] #
                arr_ue = RotatedArray(arr_ue_, theta0=rand_elev, phi0=rand_azm,  drone =False)
                
                path_ = int(data.iloc[i]['n_path'])
                chan = get_channel_from_ray_tracing_data(data.iloc[i])
            
                out = dir_path_loss_multi_sect(arr_gnb, [arr_ue], chan, isdrone= False,
                                            return_elem_gain= True)
            
                ind_ = out['sect_ind']
            
                bs_elem_gain = out['bs_elem_gain_dict'][ind_]
                ue_elem_gain = out['ue_elem_gain_dict'][ind_]
            
                bs_elem_gain_lin = 10**(0.05*bs_elem_gain)
                ue_elem_gain_lin = 10**(0.05*ue_elem_gain)
                path_loss = np.array(chan.pl)
                path_gain = 10**(-0.05*path_loss)
            
                g_l = path_gain*bs_elem_gain_lin*ue_elem_gain_lin
            
            
                bs_sv = out['bs_sv_dict'][ind_].copy()
                ue_sv = out['ue_sv_dict'][ind_].copy()
                
            
                bs_sv *= g_l[:,None]
                bs_sv, ue_sv = bs_sv.T, ue_sv.T
            
                H = bs_sv.dot(np.matrix.conj(ue_sv).T) # uplink channel
                
                gain_max = -1e5
                _SNR_map = np.zeros((nant_n_gnb, nant_n_UE))
                for b_i in range(nant_n_gnb):
                    for u_i in range(nant_n_UE):
                        bs_c = bs_cbook[b_i][:,None]
                        ue_c = ue_cbook[u_i][:,None]
                        gain = np.abs(np.conj(bs_c).T.dot(H).dot(ue_c))**2
                        snr = tx_power + 10*np.log10(gain) - KT - NF - 10*np.log10(BW)
                        
                        _SNR_map[b_i, u_i] = snr[0]
                        
                        
                SNR_map.append(np.max(_SNR_map.reshape(-1)))
                rand_azm_list.append(rand_azm)
                rand_elev_list.append(rand_elev)
                bs_sec_ind_list.append(ind_)
                
        print(np.sort(SNR_map)[::-1][:5])
        data_bs_i = dict(SNR = SNR_map, bs_sec= bs_sec_ind_list, rand_azm = rand_azm_list, rand_elev = rand_elev_list)
        
        with open(f'{dir_}SNR_data_codebookBeamforming/SNR_and_data_%d_{in_out}.pickle'%(bs_ind+1), 'wb') as f:
            pickle.dump(data_bs_i, f)
        
        
        
            
