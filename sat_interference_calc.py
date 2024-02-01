import numpy as np
import matplotlib.pyplot as plt

# import pandas as pd
plt.rcParams["font.family"] = "Times New Roman"
import pickle
import pandas as pd
from choose_sat import ChooseServingSat
from tqdm import tqdm
from interference_calculator import InterferenceCaculator#, #min_dist_check
import argparse
import random

random.seed(100)
parser = argparse.ArgumentParser(description='')

parser.add_argument('--uplink', dest='uplink', action='store_true', help='uplink or downlink')
parser.add_argument('--is_save', dest='is_save', action='store_true', help='save or not')
parser.add_argument('--total_observ_time', action='store', default=60, type=int, help=' total observation time')
parser.add_argument('--n_itf', action='store', default=21, type=int, help='total number of concurrent transmissions')
parser.add_argument('--n_iter', action='store', default=10, type=int, help='total number of iteration for randomization')
parser.add_argument('--bf', action='store', default='null_nlos', type=str, help='beamforming scheme')
parser.add_argument('--n_ant', action = 'store', default=8, type = int)
parser.add_argument('--lambda_', action = 'store', default=1e3, type = float)

min_elev = 25  # minimum elevation angle to observe satellites
print(f'minimum elevation angle is {min_elev}')
freq = 12e9
EK = -198.6  # [dBm/Hz] # pow2db(physconst('Boltzmann'))+30
G_T = 13
BW = 30e6  # 30MHz ,38.821 Table 6.1.1.1-5, Maximum bandwidth for up and downlink
f_c_list = freq + np.linspace(-BW/2, BW/2, 5)


BW_ter = 200e6
total_bs = 104  # total number of BSs
total_ue = 8496  # total number of UEs #13696
three_gpp_pwdctl = False
dir_ = f'rural_{int(freq / 1e9)}GHz'
dir_bs_to_ue = f'rural_{int(freq / 1e9)}GHz/parsed_data_bs_to_ue'
sat_track_file_name = f'data/satTrack_at_Colorado_minElem_60s_interval_{min_elev}.pickle'

args = parser.parse_args()

n_itf = args.n_itf
uplink = args.uplink
total_observ_time = args.total_observ_time  # [min]
n_iterations = args.n_iter  # number of iteration for more randomization
beamforming_scheme = args.bf
n_ant = args.n_ant
nant_gnb = np.array([n_ant,n_ant])
lambda_ = args.lambda_

print(f'carrier frequency is {freq}')

if uplink: # terrestrial uplink
    print('uplink transmission')
else:
    print('downlink transmission')

print(f'number of interfering UE or BS: {n_itf}')
print(f'total observation time: {total_observ_time}')
print(f'beamforming scheme: {beamforming_scheme} and URA:{nant_gnb}')
print(f'lambda: {lambda_}')


tx_power_ue = 23
tx_power_gnb = 33

if uplink is True:
    dir_sat_chan = 'parsed_data_sat_to_ue'
    tx_power = tx_power_ue
else:
    dir_sat_chan = 'parsed_data_sat_to_bs'
    tx_power = tx_power_gnb

interference_calculator = InterferenceCaculator(dir_=dir_,
                                                frequency=freq,
                                                tx_power_gnb=tx_power_gnb,
                                                tx_power_ue=tx_power_ue,
                                                BW=BW_ter,
                                                nant_gnb=nant_gnb)

####################### do association between BSs and UEs #################################
# merging all data across all BSs
SNR_all = np.zeros((total_bs, total_ue))  # shape = (total number of BSs, total number of UEs)
# use the same rotation angles as used for associations
rand_azm_total = dict()
rand_elev_total = dict()
for bs_idx in range(total_bs):  # for all BSs

    f = f'{dir_}/SNR_data_codebookBeamforming/SNR_and_data_%d_outdoor.pickle' % (bs_idx + 1)
    with open(f, 'rb') as handle:
        data = pickle.load(handle)
    SNR_all[bs_idx] = data['SNR'] # read SNR
    #random azimuth angle of all UE per BS after association
    rand_azm_total[bs_idx] = data['rand_azm'] # read azimuth angles
    rand_elev_total[bs_idx] = data['rand_elev'] # read elevation angles



# remove NAN values
SNR_all[np.isnan(SNR_all)] = -200
# 1% of probability to generate some attenuation
# such as attenuation effects from atmospheric conditions, rain, clouds, and scintillation
p = 1
obs_lon, obs_lat = -105.018174370992298, 40.139045580580948 # location of observer
choose_sat = ChooseServingSat(obs_lat, obs_lon, p, freq=freq, dir_=dir_sat_chan)
# upload geometrically tracked data for satellites
with open(sat_track_file_name, 'rb') as handle:
    tracked_data = pickle.load(handle)

rx_xyz = np.loadtxt('rural_12GHz/rural_ue_loc.txt') # locations of Rx
rxy = rx_xyz[:,:2]
tx_xyz = np.loadtxt('rural_12GHz/rural_bs_loc.txt') # locations of Tx
txy = tx_xyz[:,:2]
# BS -> UEs
bs_to_ues = {bs_idx:[] for bs_idx in range(len(txy))}
# perform clustering based on the locations of Rx
for i in range(len(txy)):
    txy_i = txy[i]
    dxy = np.linalg.norm(rxy - txy_i[None], axis = 1)
    I = np.argsort(dxy)[:84] # assume  84 UEs are close to one BS (8496/104)
    bs_to_ues[i] = I
# collect dataframe including one BS to all possible UEs
df_bs_ue_list = dict() # BS idx -> dataframe including all UEs
for bs_ind in tqdm(range(total_bs)):
    df = pd.read_csv(f'{dir_bs_to_ue}/bs_{bs_ind + 1}.csv', engine='python')
    df_bs_ue_list[bs_ind] = df

rand_azm_elev_dict = {bs_ind: dict() for bs_ind in range(total_bs)}
for bs_idx in range(total_bs):
    for ue_idx in range(len(rand_elev_total[bs_idx])):
        rand_azm = rand_azm_total[bs_idx][ue_idx]
        rand_elev = rand_elev_total[bs_idx][ue_idx]
        rand_azm_elev_dict[bs_idx][ue_idx] = (rand_azm, rand_elev)

itf_list = [] #inteference powers
# SNR, SINR, INR
snr_list_ter, sinr_list_ter, inr_list_ter = [], [], []
delta_g_list = []
sat_elev_observed_list = [] # elevation angles observed
for iter in tqdm(range(n_iterations)):  # repeat for randomization
    associated_ue_indices = [] # associated UEs for each iteration
    # if the associated UEs are selected,
    # we should choose the corresponding rotations between UEs and BSs
    # interfering BSs
    itf_bs_indices = []

    # for all BSs deployed
    for bs_idx in range(total_bs):
        # take all UEs associated to bs_idx
        ue_associated = bs_to_ues[bs_idx]

        SNR_UEs = SNR_all[bs_idx][ue_associated] # SNR values of UEs associated
        max_v = np.max(SNR_UEs)
        if max_v != -200:
            snr_s_selected = np.flip(np.sort(SNR_UEs))
            # choose 10 best UEs and serve only one UE among 10
            # or choose one best UE
            max_v_new = max_v #np.random.choice(snr_s_selected[:10], 1)
            itf_bs_indices.append(bs_idx) # interfering BS index
            ue_index_associated = np.where(SNR_all[bs_idx] == max_v_new)[0][0]
            associated_ue_indices.append(ue_index_associated)


    # associated channels can be easily found from bs_ind  and ue_ind
    # channel parameters between bs_ind and ue_ind as dataframe
    channel_parameters = {bs_ind: dict() for bs_ind in itf_bs_indices}
    for bs_ind in itf_bs_indices:
        for ue_ind in associated_ue_indices:
            chan_param_ = df_bs_ue_list[bs_ind].iloc[ue_ind]
            channel_parameters[bs_ind][ue_ind] = chan_param_
    # associated pair between UE and BS
    associated_pair_dict = {bs_idx: ue_idx for bs_idx, ue_idx in zip(itf_bs_indices, associated_ue_indices)}


    # decide the index of interfering UEs or BSs
    if uplink is True:
        itf_indices = associated_ue_indices
    else:
        itf_indices = itf_bs_indices

    for i in tqdm(range(total_observ_time)):
        _ind_selected = np.random.choice(itf_indices, n_itf, replace=False)  # BS or UE index selected
        # for each time, the UE or BS can observe several satellites
        # elevation angles, distances and satellite names corresponding to all the observed satellites
        elev_list = tracked_data[i]['elev_ang'].copy()
        dist_list = tracked_data[i]['dist'].copy()
        sat_ind_list = tracked_data[i]['sat_ind'].copy()
        sat_geo = np.array(tracked_data[i]['pos_geo'])[:, :2].copy()  # only take latitude and longitude
        n_serving_sat = len(elev_list)

        # calculate the channels from each BS to one best satellites
        # create channel parameter data structure: (sat_index, ue or bs index) -> dataframe of channel parameters
        channel_params_sat = {sat_idx: dict() for sat_idx in range(n_serving_sat)}
        channel_params_BS = {_ind: dict() for _ind in _ind_selected}
        sat_itf_H = dict()

        # set up channel parameters as dataframes
        for _ind in _ind_selected:  # UE or BS index set
            # Suppose each BS or UE would observe n_serving_sat satellites
            # return channel-parameter, satellite names, elevation angle list of n_serving_sat satellites
            df_list_sat, best_sat_names, df_LOS_list_sat = choose_sat.get_chan_df(elev_list, dist_list, sat_ind_list,
                                                                                  obs_ind=_ind, sat_geo_list=sat_geo)
            # shape of df_list_sat = (number of satellites, )
            # after choosing n-serving satellites, create data structure
            # (sat_index, ue or bs index) -> dataframe of channel parameters
            # sat_elev_list_per_ind[_ind] = best_elevs
            for sat_idx in range(n_serving_sat):
                channel_params_sat[sat_idx][_ind] = df_list_sat[sat_idx]
                if beamforming_scheme == 'null_nlos':  # interference nulling considering NLOS paths
                    channel_params_BS[_ind][sat_idx] = df_list_sat[sat_idx]
                elif beamforming_scheme == 'null_los':  # interference nulling considering only LOS paths
                    channel_params_BS[_ind][sat_idx] = df_LOS_list_sat[sat_idx]

        # take average over different carrier frequencies
        _delta_g_list = np.zeros((len(f_c_list), n_itf))
        _itf_list = np.zeros((len(f_c_list), n_serving_sat))

        for j, f in enumerate(f_c_list):
            interference_calculator.build_MIMO_channels(channel_parameters, associated_pair_dict,
                                                        ue_rand_azm_elev_dict=rand_azm_elev_dict,
                                                        # rotation angle when performing association between UEs and BSs
                                                        uplink=uplink,
                                                        beamforming_scheme=beamforming_scheme,
                                                        f_c=f)

            for _ind in _ind_selected:  # UE or BS index set
                if beamforming_scheme =='null_nlos' or beamforming_scheme == 'null_los':
                    H = interference_calculator.build_SAT_channel(channel_params_BS[_ind],
                                                              sat_to_bs=(not uplink),
                                                              bs_index=_ind,
                                                              f_c = f)  # one bs to several satellites
                    sat_itf_H[_ind] = H  # H is dictionary, sat index ->  H

            # every time, decide different beamforming vector to null out the side lobs
            delta_g = interference_calculator.decide_beamforming_vectors(
                                                                          indices_selected=_ind_selected,
                                                                          beamforming_scheme=beamforming_scheme,
                                                                          uplink=uplink,
                                                                          sat_itf_H=sat_itf_H,
                                                                          lambda_ = lambda_)

            _delta_g_list[j] = delta_g


            __itf_list = []
            for sat_idx in range(n_serving_sat):  # for every satellite
                # get channels from the chosen satellite to all UEs or BSs
                # return data shape: ue or bs index-> channel dict
                oneSat2_channel_list = interference_calculator.build_SAT_channel(channel_params_sat[sat_idx],
                                                                                 sat_to_bs=(not uplink),
                                                                                 f_c = f)
                itf = 0
                for _idx in oneSat2_channel_list.keys():  # _idx can be index of BS or UE
                    sat_h = oneSat2_channel_list[_idx]  # one channel between one BS or UE and the satellite
                    tx_beam_forming_vec = interference_calculator.tx_beamforming_vect_set[_idx]  # tx beamforming vector used in one BS
                    itf += np.abs(np.array(sat_h).conj().dot(tx_beam_forming_vec)) ** 2

                if np.isscalar(itf):
                    __itf_list.append(itf)
                else:
                    __itf_list.append(itf[0])

            _itf_list[j] = __itf_list

        # take average of the results over different carrier frequencies
        _delta_g_list = np.mean(_delta_g_list, axis = 0)
        _itf_list = np.mean(_itf_list, axis = 0)

        itf_list += list(_itf_list)
        delta_g_list += list(_delta_g_list)

        sat_elev_observed_list += elev_list

inr_list = 10*np.log10(itf_list) + G_T - EK - 10*np.log10(BW) + tx_power_gnb
delta_g_list = 10*np.log10(delta_g_list)
np.savetxt(f'data/delta_{beamforming_scheme}_{int(np.log10(lambda_))}.txt', delta_g_list)
np.savetxt(f'data/downlink_inr_{beamforming_scheme}_{int(np.log10(lambda_))}.txt',inr_list)
np.savetxt(f'data/downlink_elev_ang_{beamforming_scheme}_{int(np.log10(lambda_))}.txt', sat_elev_observed_list)
