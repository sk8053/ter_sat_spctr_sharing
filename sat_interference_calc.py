import numpy as np
import matplotlib.pyplot as plt

# import pandas as pd
plt.rcParams["font.family"] = "Times New Roman"
import pickle
import pandas as pd
from src.sat_iterference.choose_sat import ChooseServingSat
from tqdm import tqdm
from src.sat_iterference.interference_calculator import InterferenceCaculator#, #min_dist_check
import argparse
import random

random.seed(100)
parser = argparse.ArgumentParser(description='')
parser.add_argument('--total_observ_time', action='store', default=60, type=int, help=' total observation time')
parser.add_argument('--n_itf', action='store', default=21, type=int, help='total number of concurrent transmissions')
parser.add_argument('--n_iter', action='store', default=10, type=int, help='total number of iteration for randomization')
parser.add_argument('--bf', action='store', default='null_nlos', type=str, help='beamforming scheme')
parser.add_argument('--n_ant', action = 'store', default=8, type = int, help = 'number of Tx antennas')
parser.add_argument('--lambda_', action = 'store', default=1e2, type = float, help= 'regularization parameters to control interference nulling')

min_elev = 25  # minimum elevation angle to observe satellites

freq = 12e9 # carrier frequency in downlink transmission
EK = -198.6  # [dBm/Hz] # pow2db(physconst('Boltzmann'))+30
G_T = 13 # satellite gain-to-noise-temperature ratio
BW = 30e6  # 30MHz ,38.821 Table 6.1.1.1-5, Maximum bandwidth for up and downlink
f_c_list = freq + np.linspace(-BW/2, BW/2, 5) # possible subcarriers (it can be larger than 5)

BW_ter = 200e6  # bandwidth for terrestrial transmission
total_bs = 104  # total number of BSs
total_ue = 8496  # total number of UEs #13696
dir_ = f'rural_{int(freq / 1e9)}GHz' # directory having all the data files
dir_bs_to_ue = f'{dir_}/parsed_data_bs_to_ue' # data frames files between BSs and UEs
dir_sat_to_bs = f'{dir_}/parsed_data_sat_to_bs' # data frames files between satellites and BSs
# file name including all the tracking information during 60 seconds from 09/03, 2023, 08:10:20
sat_track_file_name = f'data/satTrack_at_Colorado_time_60m_min_elev_{min_elev}.pickle'

args = parser.parse_args()
n_itf = args.n_itf
total_observ_time = args.total_observ_time  # [min]
n_iterations = args.n_iter  # number of iteration for more randomization
beamforming_scheme = args.bf
n_ant = args.n_ant
nant_gnb = np.array([n_ant,n_ant])
lambda_ = args.lambda_
print('---------------- parameter settings -------------')
print(f'carrier frequency is {freq}')
print(f'minimum elevation angle is {min_elev}')
print('downlink transmission')
print(f'number of interfering UE or BS: {n_itf}')
print(f'total observation time: {total_observ_time}')
print(f'beamforming scheme: {beamforming_scheme} and URA:{nant_gnb}')
print(f'lambda: {lambda_}')
print(f'number of iteration for randomization: {n_iterations}')

tx_power_gnb = 33 #[dBm]
interference_calculator = InterferenceCaculator(dir_=dir_, frequency=freq, nant_gnb=nant_gnb)

####################### do association between BSs and UEs #################################
# merging all data across all BSs
SNR_all = np.zeros((total_bs, total_ue))  # shape = (total number of BSs, total number of UEs)

# use the same rotation angles as used for associations
# we need to keep these rotation angles for UE when building channels
rand_azm_elev_dict = {bs_ind: dict() for bs_ind in range(total_bs)}

# for all BSs, read SNR values when associating with all possible UEs
# we will use these SNR values per BS to choose associated UEs at random
for bs_idx in range(total_bs):
    f = f'{dir_}/SNR_data_codebookBeamforming/SNR_and_data_%d_outdoor.pickle' % (bs_idx + 1)
    with open(f, 'rb') as handle:
        data = pickle.load(handle)
    SNR_all[bs_idx] = data['SNR'] # read SNR
    for ue_idx in range(len(data['rand_azm'])):
        rand_azm_elev_dict[bs_idx][ue_idx] = (data['rand_azm'][ue_idx], data['rand_elev'][ue_idx])


# remove NAN values
SNR_all[np.isnan(SNR_all)] = -200

# 1% of probability to generate some attenuation
# such as attenuation effects from atmospheric conditions, rain, clouds, and scintillation
p = 1

# location of observer
obs_lon, obs_lat = -105.018174370992298, 40.139045580580948
choose_sat = ChooseServingSat(obs_lat, obs_lon, p, freq=freq, dir_=dir_sat_to_bs)
# upload geometrically tracked data for satellites using TLE files
with open(sat_track_file_name, 'rb') as handle:
    tracked_data = pickle.load(handle)

# create UE clusters per BS based on the location between them
rx_xyz = np.loadtxt('rural_12GHz/rural_ue_loc.txt') # locations of Rx
rxy = rx_xyz[:,:2]
tx_xyz = np.loadtxt('rural_12GHz/rural_bs_loc.txt') # locations of Tx
txy = tx_xyz[:,:2]

# ues_per_bs: a dictionary variable to save UE indices in clusters around a BS
ues_per_bs = {bs_idx:[] for bs_idx in range(len(txy))}

# perform clustering based on the distance between BSs and UEs
for i in range(len(txy)):
    txy_i = txy[i]
    dxy = np.linalg.norm(rxy - txy_i[None], axis = 1)
    I = np.argsort(dxy)[:82] # choose closest  82 UEs from one BS (8496/104)
    ues_per_bs[i] = I

# collect dataframe per BS for all possible UEs
df_bs_ue_list = dict() # BS idx -> dataframe including all UEs
print( '----------------------------------------------------------------------------')
print('============= Read channel parameters between the deployed BSs and UEs =========')
for bs_ind in tqdm(range(total_bs), desc='n_bs', ascii=True):
    df = pd.read_csv(f'{dir_bs_to_ue}/bs_{bs_ind + 1}.csv', engine='python')
    df_bs_ue_list[bs_ind] = df

itf_list = [] #inteference powers
delta_g_list = [] # gain loss
sat_elev_observed_list = [] # elevation angles observed

# repeat for randomization
# every iteration, we choose different active BSs causing interferences to satellites
for iter in tqdm(range(n_iterations), desc= 'n_iterations', ascii=True):
    associated_ue_indices = [] # associated UEs for each iteration
    # if the associated UEs are selected,
    # we should choose the corresponding rotations between UEs and BSs
    # interfering BSs, this is usually the all BSs
    all_bs_indices = []

    # for all BSs deployed
    for bs_idx in range(total_bs):
        # take all UEs associated to bs_idx
        ue_associated = ues_per_bs[bs_idx]

        SNR_UEs = SNR_all[bs_idx][ue_associated] # SNR values of UEs associated
        max_v = np.max(SNR_UEs)
        if max_v != -200: # if maximum SNR link is not outage
            all_bs_indices.append(bs_idx)  # interfering BS index
            # as the worst case, we assume that each BS serve UEs having maximum SNR
            snr_s_selected = [max_v] #np.flip(np.sort(SNR_UEs))[:1]
            # choose a SNR value at random from possible SNRs
            max_v_new = np.random.choice(snr_s_selected, 1)
            # choose UE corresponding to random SNR value chosen
            ue_index_associated = np.where(SNR_all[bs_idx] == max_v_new)[0][0]
            associated_ue_indices.append(ue_index_associated)


    # associated channels can be easily found from bs_ind  and ue_ind
    # channel parameters between bs_ind and ue_ind as dataframe
    channel_parameters = {bs_ind: dict() for bs_ind in all_bs_indices}
    for bs_ind in all_bs_indices:
        for ue_ind in associated_ue_indices:
            chan_param_ = df_bs_ue_list[bs_ind].iloc[ue_ind]
            channel_parameters[bs_ind][ue_ind] = chan_param_ # dataframe to build terrestrial channels
    # associated pair between UE and BS
    associated_pair_dict = {bs_idx: ue_idx for bs_idx, ue_idx in zip(all_bs_indices, associated_ue_indices)}

    for i in tqdm(range(total_observ_time), desc='total observation time', ascii=True):
        itf_bs_ind_active = np.random.choice(all_bs_indices, n_itf, replace=False)  # interfering active BSs indices selected
        # for each time, the BS can observe several satellites
        # elevation angles, distances and satellite names corresponding to all the observed satellites
        elev_list = tracked_data[i]['elev_ang'].copy()
        dist_list = tracked_data[i]['dist'].copy() # distances to the observed satellites from Earth station
        sat_ind_list = tracked_data[i]['sat_ind'].copy() # observed satellites
        sat_geo = np.array(tracked_data[i]['pos_geo'])[:, :2].copy()  # only take latitude and longitude
        n_serving_sat = len(elev_list)

        # calculate the channels from each BS to one best satellites
        # create channel parameter data structure: (sat_index, ue or bs index) -> dataframe of channel parameters
        channel_params_sat = {sat_idx: dict() for sat_idx in range(n_serving_sat)}
        channel_params_BS = {bs_ind: dict() for bs_ind in itf_bs_ind_active}


        # set up channel parameters as dataframes
        for bs_ind in itf_bs_ind_active:  # BS index set
            # Suppose each BS would observe n_serving_sat satellites
            # return channel-parameter, satellite names, elevation angle list of n_serving_sat satellites
            df_list_sat, best_sat_names, df_LOS_list_sat = choose_sat.get_chan_df(elev_list, dist_list, sat_ind_list,
                                                                                  obs_ind=bs_ind, sat_geo_list=sat_geo)
            # shape of df_list_sat = (number of satellites, )
            # after choosing n-serving satellites, create data structure
            # (sat_index, bs index) -> dataframe of channel parameters
            # sat_elev_list_per_ind[_ind] = best_elevs
            for sat_idx in range(n_serving_sat):
                channel_params_sat[sat_idx][bs_ind] = df_list_sat[sat_idx]
                if beamforming_scheme == 'null_nlos' or beamforming_scheme == 'SVD':  # interference nulling considering NLOS paths
                    channel_params_BS[bs_ind][sat_idx] = df_list_sat[sat_idx] # use dataframe considering all NLOS-path components
                elif beamforming_scheme == 'null_los':  # interference nulling considering only LOS paths
                    channel_params_BS[bs_ind][sat_idx] = df_LOS_list_sat[sat_idx] # use dataframe including only LOS-path component
                else:
                    ValueError(f'unknown beamforming scheme {beamforming_scheme}')

        # take average over different carrier frequencies
        _delta_g_list = np.zeros((len(f_c_list), n_itf))
        _itf_list = np.zeros((len(f_c_list), n_serving_sat))
        sat_itf_H = dict()
        for j, f in enumerate(f_c_list):
            interference_calculator.build_MIMO_channels(channel_parameters, associated_pair_dict,
                                                        ue_rand_azm_elev_dict=rand_azm_elev_dict, f_c=f)

            for bs_ind in itf_bs_ind_active:  # UE or BS index set
                H = interference_calculator.build_SAT_channel(channel_params_BS[bs_ind], f_c = f)  # one bs to several satellites
                sat_itf_H[bs_ind] = H  # H is dictionary, (sector_index, sat index) ->  interference channel


            # every time, decide different beamforming vector to null out the side lobs
            delta_g = interference_calculator.decide_beamforming_vectors(indices_selected= itf_bs_ind_active,
                                                                          beamforming_scheme=beamforming_scheme,
                                                                          sat_itf_H=sat_itf_H, # dictionary bs_idx->(sector index, sate_index) -> channels
                                                                          lambda_ = lambda_)
            _delta_g_list[j] = delta_g# gain loss in linear scale

            # change the keys of sat_itf_H for each satellite
            # old key: bs_index->(bs_sector_index, satellite_index)
            # new key: sat_index->(bs_sector_index, bs_index)
            sat_itf_H_update = {sat_ind:dict() for sat_ind in range(n_serving_sat)}
            for bs_ind in itf_bs_ind_active:  # UE or BS index set
                for key in sat_itf_H[bs_ind]:
                    sector_ind, sat_ind = key
                    sat_itf_H_update[sat_ind][(sector_ind, bs_ind)] = sat_itf_H[bs_ind][key]

            __itf_list = []
            for sat_ind in range(n_serving_sat):  # for every satellite
                # get channels from the chosen satellite to all UEs or BSs
                #oneSat2_channel_list = interference_calculator.build_SAT_channel(channel_params_sat[sat_idx], f_c = f)
                oneSat_channel_list = sat_itf_H_update[sat_ind]
                # oneSat_channel_list: (bs_sector_index, bs_index) -> channels

                itf = 0
                for bs_and_sect_idx in oneSat_channel_list:#oneSat2_channel_list.keys():  # bs_and_sect_idx is BS and sector index pair
                    _, bs_idx = bs_and_sect_idx
                    sat_h = oneSat_channel_list[bs_and_sect_idx]  # one channel between one BS or UE and the satellite
                    tx_beam_forming_vec = interference_calculator.tx_beamforming_vect_set[bs_idx]  # tx beamforming vector used in one BS
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

# calculate INR
inr_list = 10*np.log10(itf_list) + G_T - EK - 10*np.log10(BW) + tx_power_gnb
# gain loss in dB
delta_g_list = 10*np.log10(delta_g_list)
# save data
np.savetxt(f'data/delta_{beamforming_scheme}_{int(np.log10(lambda_))}.txt', delta_g_list)
np.savetxt(f'data/downlink_inr_{beamforming_scheme}_{int(np.log10(lambda_))}.txt',inr_list)
np.savetxt(f'data/downlink_elev_ang_{beamforming_scheme}_{int(np.log10(lambda_))}.txt', sat_elev_observed_list)
