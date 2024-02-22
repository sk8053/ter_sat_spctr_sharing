# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 12:26:17 2023

@author: seongjoon kang
"""
import numpy as np
from src.mmwchanmod.sim.get_channel_from_ray_tracing_data import get_channel_from_ray_tracing_data
from src.mmwchanmod.sim.array import URA, RotatedArray, multi_sect_array
from src.mmwchanmod.sim.antenna import Elem3GPP
from src.mmwchanmod.sim.chanmod import dir_path_loss_multi_sect
#import pickle
import copy


#from memory import Memory
#from power_optimizer import Power_Optimizer

class InterferenceCaculator(object):
    
    def __init__(self, frequency = 6e9, dir_= 'heraldsquare_mmwave_12GHz',n_total_bs= 104,
                 nant_gnb = np.array([8,8])):
        ## basic paramter settings for computing metrics such as SNR, SINR, and INR. 
        #self.Nf = Nf
        #self.BW = BW
        self.loss_list = []
        self.min_loss = 1e3
        self.dir_ = dir_

        #KT = -174 #[dBm/Hz] Johnson-Nyquest noise, noise power (dBm) in a register at room temperature
        # P_dbm (=KT) = 10log10(K_b T f/1mW): K_b: boltzmann's constant, T: temperature, f frequency
        # P_dbm (=KT) = -173.8 +10log10(f) at room temperature

        #self.noise_power = KT + Nf + 10*np.log10(BW) # this value is fixed once bandwdith and noise figures are determined
        self.frequency = frequency
        self.n_total_bs = n_total_bs
        #EK = -198.6 # [dBm/Hz] # pow2db(physconst('Boltzmann'))+30
        #G_T = 13
        #BW_sat = 30e6 # 30MHz ,38.821 Table 6.1.1.1-5, Maximum bandwidth for up and downlink
        thetabw_, phibw_ = 65, 65

        nant_ue = np.array([1, 2])
        nant_gnb = nant_gnb

        
        self.nant_n_ue = nant_ue[0]*nant_ue[1]
        self.nant_n_gnb = nant_gnb[0]*nant_gnb[1]
        #lam = PhyConst.light_speed / frequency
        
        # define antenna element for UE and BS
        element_ue = Elem3GPP(thetabw=thetabw_, phibw=phibw_)
        elem_bs = Elem3GPP(thetabw=thetabw_, phibw=phibw_)
        # define antenna array for UE 
        self.arr_ue_init = URA(elem=element_ue, nant=nant_ue, fc=frequency)
        # define antenna array for gNB
        arr_gnb = URA(elem=elem_bs, nant=nant_gnb,  fc= frequency) 
        # configure the antenna array of BSs having 3-sectors horizontally and down-tilted by -12
        self.arr_gnb = multi_sect_array(arr_gnb, sect_type='azimuth', nsect=3, theta0=-12)

        self.bs_id_to_ue_array = dict()

    def build_MIMO_channels(self,channel_parameters, associated_pair_dict,
                                   ue_rand_azm_elev_dict = None, f_c = 12e9-10e6,
                                    enable_calc_interference_channels = False):
        ### this function builds "uplink" MIMO channels for all the possible links between all the BSs and UEs
        
        # channel_parameters: dictionary including channel parameters in the form of pandas-dataframe
            # structure of this dictonary variables are the following
            # bs_index -> ue_index -> corresponding data frame
        # associated_pair_dict: a dictionary of all the associated pairs between BSs and UEs
            # bs_index-> associated ue_index
        # ue_rand_azm_elev_dict : angles of UEs when they do associate with BSs, we should keep these angles to reduce other randomness
        # enable_calc_interference_channels: compute interference channels between BSs and other interfering UEs

        # associated channel dictionary corresponding to each UE
        associated_H = dict() # ue_idx -> H
        # a list of interfering channel dictonary corresponding to each UE

        # build associated channels first
        self.bs_to_sect_ind_map = dict()

        for bs_idx in channel_parameters.keys():
            serv_ue_idx = associated_pair_dict[bs_idx]

            if ue_rand_azm_elev_dict == None:
                rand_azm = np.random.uniform(-180, 180, (1,))
                rand_elev = np.random.uniform(0, 90, (1,))
            else:
                # roate antenna array of UE in random direction
                (rand_azm, rand_elev) = ue_rand_azm_elev_dict[bs_idx][serv_ue_idx]

            arr_ue = RotatedArray(copy.deepcopy(self.arr_ue_init), theta0=rand_elev, phi0=rand_azm, drone=False)
            self.bs_id_to_ue_array[bs_idx] = arr_ue
            chan_param = channel_parameters[bs_idx][serv_ue_idx]
            if np.array(np.isnan(chan_param['n_path'])):  # if the link is outage
                H = np.zeros((self.nant_n_gnb, self.nant_n_ue))  # set up a virtual channel matrix
                path_loss = np.array([np.nan])
                bs_elem_gain = np.array([np.nan])
                ue_elem_gain = np.array([np.nan])
                ind_=0
            else:
                chan = get_channel_from_ray_tracing_data(chan_param)
                out = dir_path_loss_multi_sect(self.arr_gnb, [arr_ue], chan, isdrone=False, return_elem_gain=True)
                ind_ = out['sect_ind']

                bs_elem_gain = out['bs_elem_gain_dict'][ind_]
                ue_elem_gain = out['ue_elem_gain_dict'][ind_]

                bs_elem_gain_lin = 10 ** (0.05 * bs_elem_gain)
                ue_elem_gain_lin = 10 ** (0.05 * ue_elem_gain)
                path_loss = np.array(chan.pl)
                path_gain = 10 ** (-0.05 * path_loss)

                g = path_gain * bs_elem_gain_lin * ue_elem_gain_lin

                bs_sv = out['bs_sv_dict'][ind_]  # spatial signater in optimal sector of BS
                ue_sv = out['ue_sv_dict'][ind_]  # spatial signature in UE, ind_ doesn't affect

                bs_sv = bs_sv * g[:, None]
                bs_sv, ue_sv = bs_sv.T, ue_sv.T  # shape = (N_ant, N_path)

                #  channel matrix averaged over wideband
                # that is the sum of the all multipath components
                n_path = int(chan_param['n_path'])
                dly = [chan_param[f'delay_{i + 1}'] for i in range(n_path)]
                exp_phase = np.exp(-2 * np.pi * 1j * f_c * np.array(dly))
                #H = (exp_phase[None] * bs_sv).dot(np.matrix.conj(ue_sv).T)  # shape = (N_ant_bs,  N_ant_ue), uplink channel
                H = (exp_phase[None] * ue_sv).dot(np.matrix.conj(bs_sv).T) # downlink channel
            associated_H[bs_idx] = H
            self.bs_to_sect_ind_map[bs_idx]=ind_ # save the index of an associated sector of BS

        if enable_calc_interference_channels is True: # when we want to obtain interference channels between BSs and UEs
            interfering_H = dict()  # (ue_idx, bs_idx) -> all interfering channel matrices
            # build interfering channels
            for bs_idx in channel_parameters.keys():
                for j, ue_idx in enumerate(channel_parameters[bs_idx].keys()):
                    if associated_pair_dict[bs_idx] != ue_idx: # crate channel for only case that UEs are not associated with BSs
                        chan_param = channel_parameters[bs_idx][ue_idx]

                        if ue_rand_azm_elev_dict == None:
                            rand_azm = np.random.uniform(-180, 180, (1,))
                            rand_elev = np.random.uniform(0, 90, (1,))
                        else:
                            # roate antenna array of UE in random direction
                            #serv_ue_idx = associated_pair_dict[bs_idx]
                            (rand_azm, rand_elev) = ue_rand_azm_elev_dict[bs_idx][ue_idx]

                        arr_ue = RotatedArray(copy.deepcopy(self.arr_ue_init), theta0=rand_elev,phi0=rand_azm,  drone =False)

                        if np.array(np.isnan(chan_param['n_path'])):# if the link is outage
                            H = np.zeros((self.nant_n_gnb, self.nant_n_ue)) # set up a virtual channel matrix
                            path_loss = np.array([np.nan])
                            bs_elem_gain = np.array([np.nan])
                            ue_elem_gain = np.array([np.nan])
                        else:

                            chan = get_channel_from_ray_tracing_data(chan_param)

                            out = dir_path_loss_multi_sect(self.arr_gnb, [arr_ue], chan, isdrone= False,
                                                        return_elem_gain= True)

                            ind_ = self.bs_to_sect_ind_map[bs_idx] # find the sector index serving UE
                            bs_elem_gain = out['bs_elem_gain_dict'][ind_]
                            ue_elem_gain = out['ue_elem_gain_dict'][ind_]

                            bs_elem_gain_lin = 10**(0.05*bs_elem_gain)
                            ue_elem_gain_lin = 10**(0.05*ue_elem_gain)
                            path_loss = np.array(chan.pl)
                            path_gain = 10**(-0.05*path_loss)

                            g = path_gain*bs_elem_gain_lin*ue_elem_gain_lin

                            bs_sv = out['bs_sv_dict'][ind_] # spatial signater in optimal sector of BS
                            ue_sv = out['ue_sv_dict'][ind_] # spatial signature in UE, ind_ doesn't affect

                            bs_sv = bs_sv *g[:,None]
                            bs_sv, ue_sv = bs_sv.T, ue_sv.T # shape = (N_ant, N_path)

                            # 'uplink' channel matrix averaged over wideband
                            # that is the sum of the all multipath components
                            n_path = int(chan_param['n_path'])
                            dly = [chan_param[f'delay_{i + 1}'] for i in range(n_path)]
                            exp_phase = np.exp(-2 * np.pi * 1j * f_c * np.array(dly))
                            H = (exp_phase[None] * bs_sv).dot(np.matrix.conj(ue_sv).T)  # shape = (N_ant_bs,  N_ant_ue), uplink channel


                        H = np.matrix.conj(H).T
                        interfering_H[(bs_idx, ue_idx)] = H

            self.interfering_H = interfering_H

        self.associated_H = associated_H
        self.associated_pair_dict = associated_pair_dict  # we need map from ue_idx to bs_idx


    def decide_beamforming_vectors(self,associated_H = None,
                          indices_selected = None,
                          beamforming_scheme = 'SVD',
                          lambda_ = 1e3,
                          sat_itf_H = None):

        self.tx_beamforming_vect_set = dict()
        self.rx_beamforming_vect_set = dict()
        if associated_H == None:
            associated_H = self.associated_H


        delta_g_list = []

        for _idx in indices_selected: # this can be index of BS or UE
            # beamforming vector toward BS in uplink channel
            H_serv = associated_H[_idx]

            if beamforming_scheme == 'SVD':
                U, S, Vh = np.linalg.svd(H_serv)
                w_r = U[:, 0]  # take maximum eigen vector as decoder
                w_t = np.conj(Vh[0])
                w_t_svd = w_t

            elif beamforming_scheme == 'null_los' or beamforming_scheme == 'null_nlos':
                U, _, Vh = np.linalg.svd(H_serv)
                w_r = U[:, 0]  # take maximum singular vector as decoder
                w_r = w_r[:, None]
                w_t_svd = np.conj(Vh[0])
                bs_to_sat_itf = []

                for sect_and_sat_index in sat_itf_H[_idx].keys():
                    bs_to_sat_itf.append(sat_itf_H[_idx][sect_and_sat_index])

                bs_to_sat_itf = np.array(bs_to_sat_itf).T

                # normalize the interference channel matrix
                N_t, N_sat_N_sect = bs_to_sat_itf.shape # shape = (N_t, N_sat*N_sect)
                bs_to_sat_itf = np.sqrt(N_t * N_sat_N_sect) * bs_to_sat_itf / np.linalg.norm(bs_to_sat_itf, ord='fro')

                # normalize the terrestrial channel matrix
                N_t, N_r = H_serv.shape
                H_serv2 = np.sqrt(N_t * N_r) * H_serv / np.linalg.norm(H_serv, ord='fro')

                cov_terr = H_serv2.conj().T.dot(w_r).dot(w_r.conj().T).dot(H_serv2)
                cov_itf = bs_to_sat_itf.dot(bs_to_sat_itf.conj().T)

                Q = cov_terr - lambda_ * cov_itf

                eigen_values, v_nulls = np.linalg.eig(Q)
                I = np.argsort(eigen_values)[::-1]
                v_nulls = v_nulls[:, I]
                v_null = v_nulls[:, 0]
                w_t = v_null
            else:
                ValueError(f'unknown beamforming scheme {beamforming_scheme}')


            g = abs(w_r.conj().T.dot(H_serv).dot(w_t))**2
            g_old = abs(w_r.conj().T.dot(H_serv).dot(w_t_svd))**2
            g, g_old = np.squeeze(g), np.squeeze(g_old)

            _delta_g = float(g_old/g) # gain loss in linear scale

            delta_g_list.append(_delta_g)

            # for downlink case: bs_idx -> beamforming vector
            self.tx_beamforming_vect_set[_idx] = w_t
            self.rx_beamforming_vect_set[_idx] = w_r

        return delta_g_list


    def build_SAT_channel(self, channel_parameters, f_c = 12e9-10e6):
        ### this function builds "uplink" MIMO channels for all the possible links between all the BSs and satellites
        # channel_parameters: dictionary including channel parameters in the form of pandas-dataframe: bs_index -> channel parameters
        sat_H_list = dict()
        for _idx in channel_parameters.keys(): # _idx can be index of BS or satellites
            chan_param = channel_parameters[_idx]
            # roate antenna array of UE in random direction
            #arr_ue = RotatedArray(self.arr_ue_init, drone=False)
            arr_ue = self.bs_id_to_ue_array[_idx]
            if np.array(np.isnan(chan_param['n_path'])):  # if the link is outage

                H = np.zeros((self.nant_n_gnb,))  # set up a virtual channel matrix

                path_loss = np.array([np.nan])
                bs_elem_gain = np.array([np.nan])
                ue_elem_gain = np.array([np.nan])
                #for sector_ind in range(3):
                sector_ind = 0
                sat_H_list[(sector_ind, _idx)] = H  # sector index, satellite index or BS index
            else:
                chan = get_channel_from_ray_tracing_data(chan_param)

                # place a virtual UE  in the location of satellite
                # use channel data from sat to bs from ray-tracing resuls: sat -> BS
                # place a virtual UE in the position of satellite by switching angles: and consider transmission: BS->sat with invert = True
                out = dir_path_loss_multi_sect(self.arr_gnb, [arr_ue], chan, isdrone=False, disable_ue_elemgain =True, # here ue can be satellite
                                               return_elem_gain=True, invert =True) # invert = True, arrival angles and departure angles are switched

                #for sector_ind in range(3):
                sector_ind = self.bs_to_sect_ind_map[_idx]
                _elem_gain = out['bs_elem_gain_dict'][sector_ind] # consider only downlink transmission from BS
                _elem_gain_lin = 10 ** (0.05 * _elem_gain)

                path_loss = np.array(chan.pl)
                path_gain = 10 ** (-0.05 * path_loss)

                g = path_gain * _elem_gain_lin #* ue_elem_gain_lin
                _sv = out['bs_sv_dict'][sector_ind]  # spatial signature in optimal sector of BS
                _sv = np.conj(_sv)

                n_path = int(chan_param['n_path'])
                dly = [chan_param[f'delay_{i + 1}'] for i in range(n_path)]
                exp_phase = np.exp(-2 * np.pi * 1j * f_c * np.array(dly))
                _sv = (exp_phase[:, None] * _sv) * g[:, None]
                H = _sv.sum(0)   # channel frequency response
                sat_H_list[(sector_ind, _idx)] = H # (sector index, satellite index) or (sector index, BS index)

        return sat_H_list


'''
   def caclulate_metrics(self, associated_H = None, 
                      interfering_H = None,
                      tx_beamforming_vect_set = None,
                      rx_beamforming_vect_set = None,
                      associated_pair_dict = None,
                      is_waterfilling = False,
                      ind_selected:list()= None,
                     ):
    
    if associated_H == None or interfering_H == None:
        associated_H = self.associated_H
        interfering_H = self.interfering_H
        tx_beamforming_vect_set = self.tx_beamforming_vect_set
        rx_beamforming_vect_set = self.rx_beamforming_vect_set
        associated_pair_dict = self.associated_pair_dict
        tx_power_map = self.tx_power_map
        tx_power_amplitude_map = {_idx:10**(0.05*tx_power_map[_idx]) for _idx in tx_power_map.keys()}
        

    SNR_list, SINR_list, INR_list = [], [], []        
    # calculate interference power to other uplink transmision caused by this UE
    noise_power_lin = 10**(0.1*self.noise_power)
    for _idx in ind_selected:
        # extract serving channels and beamforming vectors
        H_serv = associated_H[_idx]
        w_t = tx_beamforming_vect_set[_idx]
        
        tx_power_amplitude_lin = tx_power_amplitude_map[_idx]#10**(0.05*tx_power_ue_map[ue_idx])

        # allocate Tx power across antennas
        #w_t = self.allocate_power_across_antennas(w_t.squeeze(), tx_power_amplitude_lin)
        w_t = w_t*tx_power_amplitude_lin
        # receive beamforming vector in the serving BS
        w_r = rx_beamforming_vect_set[_idx]

        rx_power = np.abs(w_r.conj().T.dot(H_serv).dot(w_t))**2

        rx_power = 10*np.log10(max(rx_power, 1e-22))
        SNR = rx_power - self.noise_power
        SNR_list.append(float(np.squeeze(SNR)))
        
        serving_idx = associated_pair_dict[_idx]
        itf_power = 0
        for _idx_itf in ind_selected:
            if _idx != _idx_itf:
                
                H_itf = interfering_H[(_idx_itf, serving_idx)] # interference channel from other UEs to the serving BS
                w_t_itf = tx_beamforming_vect_set[_idx_itf] # transmit beamforming vector from other UEs
                
                ## allocate power for interference channels
                tx_power_amplitude_lin = tx_power_amplitude_map[_idx_itf]
                w_t_itf = self.allocate_power_across_antennas(w_t_itf.squeeze(), tx_power_amplitude_lin)
                
                itf_power_i = np.abs(w_r.conj().T.dot(H_itf).dot(w_t_itf))**2
                itf_power += itf_power_i # interferenc powers are summed up in linear scale
                
        itf_plus_noise = itf_power + noise_power_lin
        
        SINR = rx_power - 10*np.log10(itf_plus_noise)
        itf_power = np.maximum(itf_power, 1e-20)
        INR = 10*np.log10(itf_power/noise_power_lin)
    
        SINR_list.append(float(np.squeeze(SINR)))
        INR_list.append(float(np.squeeze(INR)))

    return SNR_list, SINR_list, INR_list
    
    def allocate_power_across_antennas(self, w_t, 
                                tx_power_amplitude_ue_lin = 10**(0.05*23),# 23dBm is maximum Tx power of UE
                                water_filling_algorithm = False, # waterfilling algorithms
                                S = None): # singular values of channel matrix, H
     # power allocation across antennas given total Tx power
     if water_filling_algorithm is True:
         # this multiplication is needed because w_t is already normalized
         # w_t_i is not exponential term of phase
         power_list = self.water_filling_algorithm(S)*len(w_t)
         power_amp_per_ant = np.sqrt(power_list)
     else:
         # in this case, the power is eqully allocated across all antennas of a transmitter
         power_amp_per_ant = np.tile(1, len(w_t))*tx_power_amplitude_ue_lin #/np.linalg.norm(w_t) (= len(w_t))

     # 0<w_t_i <1 ,and |w_t|^2 = 1
     w_t = w_t*power_amp_per_ant

     return w_t    
 
        
    def build_MUMIMO_channels(self, channel_parameters,
                            associated_pair_dict,
                            ue_rand_azm_elev_dict=None,
                            three_gpp_pwctl=False,
                            decide_beamforming_bf=True,
                            beamforming_scheme='SVD',
                            init_access=False,
                            uplink=False,
                            f_c=12e9 - 10e6):
        ### this function builds "uplink" MIMO channels for all the possible links between all the BSs and UEs

        # channel_parameters: dictionary including channel parameters in the form of pandas-dataframe
        # structure of this dictonary variables are the following
        # bs_index -> ue_index -> corresponding data frame
        # associated_pair_dict: a dictionary of all the associated pairs between BSs and UEs
        # bs_index-> associated ue_index

        # three_gpp_pwctl: power control algorithm specificed in 3gpp, allocation of total Tx power

        # associated channel dictionary corresponding to each UE
        associated_H = dict()  # ue_idx ->
        bs_sect_ind_ue_ind = {bs_ind: dict() for bs_ind in range(self.n_total_bs)}
        bs_sect_ind_snr = {bs_ind: dict() for bs_ind in range(self.n_total_bs)}
        for bs_ind in range(self.n_total_bs):
            for ind_ in range(3):
                bs_sect_ind_ue_ind[bs_ind][ind_] = []
                bs_sect_ind_snr[bs_ind][ind_] = []

        # a list of interfering channel dictonary corresponding to each UE
        interfering_H = dict()  # (ue_idx, bs_idx) -> all interfering channel matrices
        associated_pair_dict_reversed = dict()

        # initial power of UEs are 23 dB, full power transmission
        for bs_idx in associated_pair_dict.keys():
            for ue_idx in associated_pair_dict[bs_idx]['ue_id']:
                ue_tx_power_map = {ue_idx: self.tx_power_ue}

        # build associated channels first
        self.bs_to_sect_ind_map = dict()

        for bs_idx in channel_parameters.keys():
            # serv_ue_id_per_bs, _ = associated_pair_dict[bs_idx]
            for serv_ue_idx in associated_pair_dict[bs_idx]['ue_id']:

                if ue_rand_azm_elev_dict == None:
                    rand_azm = np.random.uniform(-180, 180, (1,))
                    rand_elev = np.random.uniform(0, 90, (1,))
                else:
                    # roate antenna array of UE in random direction
                    (rand_azm, rand_elev) = ue_rand_azm_elev_dict[bs_idx][serv_ue_idx]

                arr_ue = RotatedArray(copy.deepcopy(self.arr_ue_init), theta0=rand_elev, phi0=rand_azm, drone=False)

                chan_param = channel_parameters[bs_idx][serv_ue_idx]
                if np.array(np.isnan(chan_param['n_path'])):  # if the link is outage
                    H = np.zeros((self.nant_n_gnb, self.nant_n_ue))  # set up a virtual channel matrix
                    path_loss = np.array([np.nan])
                    bs_elem_gain = np.array([np.nan])
                    ue_elem_gain = np.array([np.nan])
                    ind_ = 0
                    g = -1
                else:
                    chan = get_channel_from_ray_tracing_data(chan_param)
                    out = dir_path_loss_multi_sect(self.arr_gnb, [arr_ue], chan, isdrone=False, return_elem_gain=True)
                    ind_ = out['sect_ind']
                    bs_elem_gain = out['bs_elem_gain_dict'][ind_]
                    ue_elem_gain = out['ue_elem_gain_dict'][ind_]

                    bs_elem_gain_lin = 10 ** (0.05 * bs_elem_gain)
                    ue_elem_gain_lin = 10 ** (0.05 * ue_elem_gain)
                    path_loss = np.array(chan.pl)
                    path_gain = 10 ** (-0.05 * path_loss)

                    g = path_gain * bs_elem_gain_lin * ue_elem_gain_lin

                    bs_sv = out['bs_sv_dict'][ind_]  # spatial signature in optimal sector of BS
                    ue_sv = out['ue_sv_dict'][ind_]  # spatial signature in UE, ind_ doesn't affect

                    bs_sv = bs_sv * g[:, None]
                    bs_sv, ue_sv = bs_sv.T, ue_sv.T  # shape = (N_ant, N_path)

                    # 'uplink' channel matrix averaged over wideband
                    # that is the sum of the all multipath components
                    n_path = int(chan_param['n_path'])
                    dly = [chan_param[f'delay_{i + 1}'] for i in range(n_path)]
                    exp_phase = np.exp(-2 * np.pi * 1j * f_c * np.array(dly))
                    H = (exp_phase[None] * bs_sv).dot(
                        np.matrix.conj(ue_sv).T)  # shape = (N_ant_bs,  N_ant_ue), uplink channel

                # if uplink is True:
                #    associated_H[serv_ue_idx] = H
                #    associated_pair_dict_reversed[serv_ue_idx] = bs_idx
                # else:
                H = np.matrix.conj(H).T  # downlink channel
                associated_H[(bs_idx, serv_ue_idx)] = H
                _, S, _ = np.linalg.svd(H)
                s_max = np.max(S.reshape(-1))
                rx_gain = 20 * np.log10(s_max)
                snr = self.tx_power_gnb + rx_gain - self.noise_power

                bs_sect_ind_ue_ind[bs_idx][ind_].append(serv_ue_idx)

                bs_sect_ind_snr[bs_idx][ind_].append(snr)

                if three_gpp_pwctl is True:  # and np.array(np.isnan(chan_param['n_path'])) is False: #tx_power_ue_map
                    P_ex = (path_loss - bs_elem_gain - ue_elem_gain) * 0.6 + (-78)
                    P_ex = 10 * np.log10(np.sum(10 ** (0.1 * P_ex)))
                    ue_tx_power_map[serv_ue_idx] = min(P_ex, self.tx_power_ue)

                self.bs_to_sect_ind_map[bs_idx] = ind_  # save the index of an associated sector of BS

            if init_access is True:
                for _ind in range(3):
                    _best_snrs = np.sort(bs_sect_ind_snr[bs_idx][_ind])[::-1][:10]
                    bs_sect_ind_snr[bs_idx][_ind] = _best_snrs

                    I = np.argsort(bs_sect_ind_snr[bs_idx][_ind])[::-1]
                    I = I[:10]
                    __serv_ue_idx = bs_sect_ind_ue_ind[bs_idx][_ind]
                    _serv_ue_idx = np.array(__serv_ue_idx)[I]
                    bs_sect_ind_ue_ind[bs_idx][_ind] = list(_serv_ue_idx)

        if init_access is True:
            return bs_sect_ind_ue_ind, bs_sect_ind_snr

        else:
            # build interfering channels
            for bs_idx in channel_parameters.keys():
                # print(channel_parameters[bs_idx].keys())
                for j, ue_idx in enumerate(channel_parameters[bs_idx].keys()):
                    serving_ue_ids = associated_pair_dict[bs_idx]['ue_id']
                    if ue_idx not in serving_ue_ids:  # crate channel for only case that UEs are not associated with BSs
                        chan_param = channel_parameters[bs_idx][ue_idx]

                        if ue_rand_azm_elev_dict == None:
                            rand_azm = np.random.uniform(-180, 180, (1,))
                            rand_elev = np.random.uniform(0, 90, (1,))
                        else:
                            # roate antenna array of UE in random direction
                            # serv_ue_idx = associated_pair_dict[bs_idx]
                            (rand_azm, rand_elev) = ue_rand_azm_elev_dict[bs_idx][ue_idx]

                        arr_ue = RotatedArray(copy.deepcopy(self.arr_ue_init), theta0=rand_elev, phi0=rand_azm,
                                              drone=False)

                        if np.array(np.isnan(chan_param['n_path'])):  # if the link is outage
                            H = np.zeros((self.nant_n_gnb, self.nant_n_ue))  # set up a virtual channel matrix
                            path_loss = np.array([np.nan])
                            bs_elem_gain = np.array([np.nan])
                            ue_elem_gain = np.array([np.nan])
                        else:

                            chan = get_channel_from_ray_tracing_data(chan_param)

                            out = dir_path_loss_multi_sect(self.arr_gnb, [arr_ue], chan, isdrone=False,
                                                           return_elem_gain=True)

                            for ind_ in range(len(out['bs_sv_dict'])):
                                bs_elem_gain = out['bs_elem_gain_dict'][ind_]
                                ue_elem_gain = out['ue_elem_gain_dict'][ind_]

                                bs_elem_gain_lin = 10 ** (0.05 * bs_elem_gain)
                                ue_elem_gain_lin = 10 ** (0.05 * ue_elem_gain)
                                path_loss = np.array(chan.pl)
                                path_gain = 10 ** (-0.05 * path_loss)

                                g = path_gain * bs_elem_gain_lin * ue_elem_gain_lin

                                bs_sv = out['bs_sv_dict'][ind_]  # spatial signater in optimal sector of BS
                                ue_sv = out['ue_sv_dict'][ind_]  # spatial signature in UE, ind_ doesn't affect

                                bs_sv = bs_sv * g[:, None]
                                bs_sv, ue_sv = bs_sv.T, ue_sv.T  # shape = (N_ant, N_path)

                                # 'uplink' channel matrix averaged over wideband
                                # that is the sum of the all multipath components
                                n_path = int(chan_param['n_path'])
                                dly = [chan_param[f'delay_{i + 1}'] for i in range(n_path)]
                                exp_phase = np.exp(-2 * np.pi * 1j * f_c * np.array(dly))
                                H = (exp_phase[None] * bs_sv).dot(
                                    np.matrix.conj(ue_sv).T)  # shape = (N_ant_bs,  N_ant_ue), uplink channel

                                if uplink is False:  # if channel is downlink
                                    H = np.matrix.conj(H).T
                                    interfering_H[(bs_idx, ue_idx, ind_)] = H
                                else:
                                    interfering_H[(ue_idx, bs_idx, ind_)] = H

            self.associated_H = associated_H
            self.interfering_H = interfering_H
            if uplink is True:
                self.associated_pair_dict = associated_pair_dict_reversed  # we need map from ue_idx to bs_idx
                self.tx_power_map = ue_tx_power_map
            else:
                self.associated_pair_dict = associated_pair_dict  # we need map from ue_idx to bs_idx
                self.tx_power_map = {bs_idx: self.tx_power_gnb for bs_idx in channel_parameters.keys()}

    def decide_mumimo_beamforming_vectors(self, associated_H=None,
                                          indices_selected=None,
                                          interfering_H=None,
                                          tx_power_ue_map=None,
                                          beamforming_scheme='SVD',
                                          lambda_=1e3,
                                          # sat_elev_ang_observed:list() = None,
                                          uplink=True,
                                          sat_itf_H=None):
        self.tx_beamforming_vect_set = dict()
        self.rx_beamforming_vect_set = dict()
        if associated_H == None or interfering_H == None or tx_power_ue_map == None:
            associated_H = self.associated_H
            interfering_H = self.interfering_H
            tx_power_map = self.tx_power_map

        ue_cbook = self.ue_cbook
        if beamforming_scheme == 'codebook':
            # file_name2 = f'{self.dir}/bs_codebook_{int(self.frequency/1e9)}GHz.txt'
            bs_cbook = self.bs_cbook  # _dict
        elif beamforming_scheme == 'DFT':
            bs_cbook = self.dft_codebook  # _dict
        elif beamforming_scheme == 'VQ':
            bs_cbook = self.vq_codebook  # _dict

        tx_beamforming_vect_set, rx_beamforming_vect_set = dict(), dict()
        delta_g_list = []

        for _idx in indices_selected:  # this can be index of BS or UE
            # beamforming vector toward BS in uplink channel
            ue_idx_list, snr_list = self.associated_pair_dict[_idx]['ue_id'], \
            self.associated_pair_dict[_idx]['snr']
            # print(ue_idx_list, snr_list)
            if beamforming_scheme == 'SVD':
                w_t_dict = dict()
                for _ue_idx in ue_idx_list:
                    H_serv = associated_H[(_idx, _ue_idx)]
                    U, S, Vh = np.linalg.svd(H_serv)
                    w_r = U[:, 0]  # take maximum eigen vector as decoder
                    w_t = np.conj(Vh[0])
                    w_t_dict[_ue_idx] = w_t
            elif beamforming_scheme == 'null_los' or beamforming_scheme == 'null_nlos':
                w_t_dict = dict()
                w_r_dict = dict()

                bs_to_sat_itf = []
                # sat_itf_H : bs_index -> sat_index -> H
                for sat_idx in sat_itf_H[_idx].keys():
                    bs_to_sat_itf.append(sat_itf_H[_idx][sat_idx])

                bs_to_sat_itf = np.array(bs_to_sat_itf).T
                N_t, N_sat = bs_to_sat_itf.shape
                bs_to_sat_itf = np.sqrt(N_t * N_sat) * bs_to_sat_itf / np.linalg.norm(bs_to_sat_itf,
                                                                                      ord='fro')
                cov_itf_sat = bs_to_sat_itf.dot(bs_to_sat_itf.conj().T)

                for _ue_idx in ue_idx_list:
                    H_serv = associated_H[(_idx, _ue_idx)]
                    U, _, Vh = np.linalg.svd(H_serv)
                    w_r = U[:, 0]  # take maximum singular vector as decoder
                    w_r_dict[_ue_idx] = w_r

                    w_r = w_r[:, None]
                    w_t_old = np.conj(Vh[0])

                    N_t, N_r = H_serv.shape
                    H_serv2 = np.sqrt(N_t * N_r) * H_serv / np.linalg.norm(H_serv, ord='fro')

                    cov_itf = 0
                    for _ue_idx_itf, _snr in zip(ue_idx_list, snr_list):
                        if _ue_idx != _ue_idx_itf:
                            H_serv_itf = associated_H[(_idx, _ue_idx_itf)]
                            _cov_itf = H_serv_itf.conj().T.dot(w_r).dot(w_r.conj().T).dot(H_serv_itf)
                            _snr_lin = 10 ** (0.1 * _snr)
                            cov_itf += _snr_lin * _cov_itf  # need to multiply scalar values

                    H_serv_telda = np.conj(H_serv2).T.dot(w_r)
                    w_t = np.linalg.inv(lambda_ * cov_itf_sat + cov_itf + np.eye(cov_itf.shape[0])).dot(
                        H_serv_telda)
                    w_t = w_t / np.linalg.norm(w_t)
                    # Q = cov_terr - lambda_ * cov_itf

                    # eigen_values, v_nulls = np.linalg.eig(Q)
                    # I = np.argsort(eigen_values)[::-1]
                    # v_nulls = v_nulls[:, I]
                    # v_null = v_nulls[:, 0]
                    # w_t = v_null
                    w_t_dict[_ue_idx] = w_t

                    g = 10 * np.log10(abs(w_r.conj().T.dot(H_serv).dot(w_t)) ** 2)
                    g_old = 10 * np.log10(abs(w_r.conj().T.dot(H_serv).dot(w_t_old)) ** 2)
                    g, g_old = np.squeeze(g), np.squeeze(g_old)

                    _delta_g = float(g_old - g)
                    delta_g_list.append(10 ** (0.1 * _delta_g))
            # save beamforming vectors before power allocation
            # for downlink case: bs_idx -> beamforming vector
            # for uplink case: ue_idx -> beamforming vector

            tx_beamforming_vect_set[_idx] = w_t_dict
            rx_beamforming_vect_set[_idx] = w_r_dict

        self.tx_beamforming_vect_set = tx_beamforming_vect_set
        # print(self.tx_beamforming_vect_set.keys())
        self.rx_beamforming_vect_set = rx_beamforming_vect_set

        return delta_g_list

    def water_filling_algorithm(self, S):
        # water filling algorithms to allocate powers to individual antennas
        # return power values per antenna
        N0 = 10**(0.1*self.noise_power)
        tx_power_ue_lin = 10**(0.1*self.tx_power_ue)
        # in case that S includes zeros
        S = np.maximum(S,1e-22) # remove zero elements from S
        # we should decide the bounds, alpha, to fill water
        thresh_hold = 1e-6
        N = len(S) # total number of Tx antennas
        alpha_L = (N0/S).min() # intial lower bound of alpha
        alpha_U = (tx_power_ue_lin + (N0/S).sum())/N
        alpha_bar = (alpha_U + alpha_L)/2 # initial value of water level
        while abs(alpha_U- alpha_L) > thresh_hold:
            alpha_bar = (alpha_U + alpha_L)/2
            power_list = np.maximum(1/alpha_bar - N0/S,0)
            if tx_power_ue_lin>power_list.sum():
                alpha_U = alpha_bar
            else:
                alpha_L = alpha_bar
        # use alpha_bar as the bounds to fill water
        power_list= np.maximum(1/alpha_bar - N0/S,0)
        return power_list

#    @staticmethod
def min_dist_check(df):
    tx,ty, tz = df['tx_x'], df['tx_y'], df['tx_z']
    rx, ry, rz = df['rx_x'], df['rx_y'], df['rx_z']
    dist2d = np.sqrt((tx-rx)**2 + (ty-ry)**2)
    if dist2d>35: # minimum 2-d distance in rural scenario by 3GPP 38.901
        return True
    else:
        return False
'''
