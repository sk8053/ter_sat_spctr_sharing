import numpy as np
import random
import pandas as pd
import sys
import pathlib


ab_path = pathlib.Path().absolute().parent.__str__()
sys.path.append(ab_path+'/uav_interference_analysis/')


def get_channels(ue_loc, bs_loc,channel_model, network, three_gpp=False, frequency = 28e9, cell_type = 1,
                 aerial_fading = False, get_link_state = False, max_multi_n = 25):
    n_UE = len(ue_loc)
    if three_gpp is True:
        ue_loc2 = np.repeat(ue_loc[None, :], len(bs_loc), axis=0).reshape(-1, 3)
        bs_loc2 = np.repeat(bs_loc, len(ue_loc), axis=0)
        ue_loc2 = bs_loc2 + network._wrap_around(ue_loc2 - bs_loc2)

        channel_model.run_ns3(tx=bs_loc2, rx=ue_loc2, aerial_fading= aerial_fading, frequency=frequency)
        data = open(channel_model.path + 'ray_tracing.txt', 'r')
        three_gpp_channel_list, link_state_list = channel_model.getList(data, get_link_state = True)
        channel_list = np.array(three_gpp_channel_list).reshape(n_UE, -1)


    else:
        ue_loc2 = np.repeat(ue_loc, len(bs_loc), axis =0)
        bs_loc2 = np.repeat(bs_loc[None,:], len(ue_loc), axis = 0).reshape(-1,3)
        dist_vectors = ue_loc2 - bs_loc2
        dist_vectors = network._wrap_around(dist_vectors)
        cell_types = np.repeat([cell_type], len(dist_vectors))
        channels, link_state_list = channel_model.sample_path(dist_vectors, cell_types)
        channels = np.array(channels)
        channel_list = channels.reshape(n_UE, -1)

    if get_link_state is True:
        return channel_list, link_state_list
    else:
        return channel_list

def save_file(DATA, path: str):
    # function to save panda data
    random.seed(10)
    keys = DATA[0].keys()
    DATA_all = {key:[] for key in keys}
    for data in DATA:
        for key in keys:
            DATA_all[key].append(data[key])
    df = pd.DataFrame(DATA_all)
    #print(df['itf_no_ptx_list'])
    #power_optimizer = power_optimization(l_f = df['l_f'], ue_ids = df['ue_id'],
    #                                     itf_no_ptx_list= df['itf_no_ptx_list'],
    #                                     itf_id_list=df['itf_id_list'])
    #P_tx, SINR_list = power_optimizer.get_optimal_power_and_SINR(minstep=1e-7, debug=False)
    #df['tx_power_opt'] = 10*np.log10(P_tx)
    #df['SINR_opt'] = SINR_list

    df.to_csv(path, sep='\t', index=False)

def check_min_dist(loc):
    # check minimum distance between BSs
    I = np.argsort(loc[:,0])
    k =I[0]
    for i in I[1:]:
        dist = np.linalg.norm(loc[i] - loc[k])
        if dist <10:
            np.delete(loc, i, axis= 0)
        k = i
    return loc

def get_location(n:int = 50, MAX: int = 1000, isBS: bool = False, low:int =2, high:int = 5, h:int = 30 ):

    if n != 0:
        # set the locations of BSs
        loc_xy = np.random.uniform(low= 0, high =MAX, size =[n, 2])
        if isBS is True:
            height = np.random.uniform(low=low, high = high, size = (n,))
        else:
            height = np.repeat([h], n)
        loc = np.append(loc_xy, height[:,None], axis = 1)
        loc = check_min_dist(loc)
    else:
        loc = np.array([])

    return loc

def check_min_dist_UE_BS(bs_loc, g_UE_loc, n):

    n_gUE_dropped = n
    min_distance = 10
    # check minimum distance between ground UE and BS
    dist_vec = bs_loc - g_UE_loc[:,None]
    distance = np.linalg.norm(dist_vec[:,:2], axis = -1)

    for j in range(n_gUE_dropped):
        min = np.min(distance[j])
        #arg = np.argmin(distance[j])
        if min < min_distance:
            theta = np.random.uniform(low = - np.pi, high=np.pi, size = 1)
            #d_h = bs_loc[arg,2]-g_UE_loc[j,2]
            r_2d = 10 #np.sqrt(min_distance**2 - d_h**2)
            # change the location of a UE so that the distance is more than minimum
            g_UE_loc[j, 0] =  r_2d*np.cos(theta) + 0.05
            g_UE_loc[j, 1] =  r_2d*np.sin(theta) + 0.05

    return g_UE_loc

