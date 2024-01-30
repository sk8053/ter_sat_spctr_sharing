import random

import sys
import pathlib
ab_path = pathlib.Path().absolute().parent.__str__()

sys.path.append(ab_path+'/uav_interference_analysis/src/')

import numpy as np
from tqdm.auto import tqdm
from ue import  UE
from bs import BS
from network_channel import Channel_Info, Network
import pandas as pd
from ground_channel_generation import GroundChannel
from mmwchanmod.sim.drone_antenna_field import drone_antenna_gain
import argparse
from wrap_around import wrap_around
from power_optimizer import power_optimization
from utills import check_min_dist_UE_BS, get_location, save_file

parser = argparse.ArgumentParser(description='')
parser.add_argument('--height',action='store',default=30,type= int,\
    help='UAV_height')
parser.add_argument('--t_gpp', dest = 't_gpp', action = 'store_true')
parser.add_argument('--ptrl', dest = 'ptrl', action = 'store_true')

parser.add_argument('--n_UAV',action='store',default=100,type= int,\
    help='number of UAVs considered for association')
parser.add_argument('--n_gUE',action='store',default=100,type= int,\
    help='number of ground UEs considered for association')

parser.add_argument('--run_time',action='store',default=50,type= int,\
    help='simulation running time')
parser.add_argument('--frequency',action='store',default=28e9,type= float,\
    help='carrier frequency')
parser.add_argument('--alpha',action='store',default=0.5,type= float,\
    help='alpha for power control')
parser.add_argument('--ratio',action='store',default=0.5,type= float,\
    help='ratio of number')
parser.add_argument('--P0',action='store',default=-80,type= float,\
    help='reference power for power control')
parser.add_argument('--n', action ='store', default = 2, type = float,\
                    help = 'number of connections to each BS')

args = parser.parse_args()
UAV_height = args.height
# running time
run_time = args.run_time
alpha = args.alpha
P0 = args.P0
ratio = args.ratio
frequency = args.frequency
ptrl = args.ptrl
enable_3gpp_ch = args.t_gpp
max_n = args.n
n_UAV = args.n_UAV
n_gUE = args.n_gUE
uav_percentage = (n_UAV)/(n_gUE+ n_UAV)

if max_n>=1.0:
    max_n = int(max_n)

print ('number of connections to BS is ', max_n)
print ('frequency is ', frequency)
#print ('alpha is %s and P0 is %s'% (alpha, P0))

g_UE_Height = 1.6
print ('UAV height is ', UAV_height)
# set some parameters for network area
random.seed(100)
ISD = 200
ISD_ue_uav = 200
print ('ISD is ,', ISD)
lam = 4/(np.pi*ISD**2)
lam_ue_uav = 4/(np.pi*ISD_ue_uav**2)
XMAX, YMAX = 1000, 1000
area = XMAX * YMAX
print ('Average number of BS is ', int(lam*area))
min_distance = 10
# set network we observe
network = Network(X_MAX = XMAX, X_MIN = 0, Y_MAX = YMAX, Y_MIN = 0, Z_MAX = UAV_height)
# channel information

print ('Aerial channel for 3GPP is, ', enable_3gpp_ch)
channel_info = Channel_Info(three_gpp = enable_3gpp_ch)
channel_info.alpha = alpha
channel_info.P0 = P0

channel_info.ptrl = ptrl
channel_info.frequency = frequency
channel_info.long_term_bf = True
drone_antenna_gain = drone_antenna_gain()
#print ('long-term beamforming is ', channel_info.long_term_bf)
print ('power control of UAV and UE is , ', channel_info.ptrl)
if frequency == 2e9:
    channel_info.BW = 80e6
#N_g_UE = int(lam_ue_uav*area)

#channel_info.N_UAV = n_drop
#n_UAV_dropped, n_gUE_dropped =n_drop, n_drop # int(n_drop* n_UAV/(n_UAV+n_gUE)), int(n_drop* n_gUE/(n_UAV+n_gUE))
#print('number of UAV and gUE are %d and %d'% (n_UAV_dropped, n_gUE_dropped))
print ('uav percentage is %2.3f' %(int(uav_percentage*100)))
# every drop we set different number of BSs and their locations
# number of ground UE = number of BS, and their locations are closer
BS_DATA = []
random.seed(0)

for t in tqdm(range (run_time), desc= 'number of drops'):
    n_BS =  np.random.poisson(lam*area)
    #N_g_UE = np.random.poisson(lam_ue_uav*area)
    n_total = n_BS *16
    n_UAV_dropped = int(uav_percentage * n_total)
    n_gUE_dropped = n_total - n_UAV_dropped
    channel_info.N_UAV = n_UAV_dropped
    # set the locations of BSs
    bs_loc = get_location(n = n_BS, MAX = 1000, isBS = True, low = 2, high = 5)
    # generate all UAV locations we will utilize to drop over simulation time
    UAV_loc = get_location(n = n_UAV_dropped, MAX = 1000, isBS = False, h = UAV_height)

    if n_UAV_dropped !=0:
        UAV_loc2 = np.repeat(UAV_loc, len(bs_loc), axis =0)
        bs_loc2 = np.repeat(bs_loc[None,:], len(UAV_loc), axis = 0).reshape(-1,3)
        dist_vectors = UAV_loc2 - bs_loc2
        dist_vectors =wrap_around(dist_vectors, network)
        cell_types = np.repeat([1], len(dist_vectors))
        uav_channels, uav_link_states = channel_info.channel.sample_path(dist_vectors, cell_types)
        uav_channels, uav_link_states = np.array(uav_channels), np.array(uav_link_states)
        uav_channel_list, uav_link_states = uav_channels.reshape(n_UAV_dropped, -1), uav_link_states.reshape(n_UAV_dropped, -1)

    if enable_3gpp_ch is True:
        UAV_loc2 = np.repeat(UAV_loc[None, :], n_BS, axis=0).reshape(-1, 3)
        bs_loc2 = np.repeat(bs_loc, n_UAV_dropped, axis=0)
        UAV_loc2 = bs_loc2 + wrap_around(UAV_loc2 - bs_loc2, network)
        uav_channels = GroundChannel(tx=bs_loc2, rx=UAV_loc2, aerial_fading=True, frequency=frequency)
        data = open(uav_channels.path + 'ray_tracing.txt', 'r')
        uav_channel_list = uav_channels.getList(data)
        uav_channel_list = np.array(uav_channel_list).reshape(n_UAV_dropped, -1)

    # set the locations of ground UEs
    g_UE_loc = get_location(n = n_gUE_dropped, MAX = 1000, isBS = False, h = g_UE_Height)
    g_UE_loc = check_min_dist_UE_BS(bs_loc, g_UE_loc, n_gUE_dropped)
    # check minimum distance between ground UE and BS
    # generate ground channels for all UEs
    # repeat g_UE locations by number of BSs
    g_UE_loc2 = np.repeat(g_UE_loc[None,:],n_BS, axis = 0).reshape(-1,3)
    # repeat BS locations for all g_UEs
    bs_loc2 = np.repeat(bs_loc, n_gUE_dropped, axis=0)
    # apply wrap_around between BS and g_UE
    g_UE_loc2 = bs_loc2 + wrap_around(g_UE_loc2 - bs_loc2, network)
    g_channels = GroundChannel(tx=bs_loc2, rx = g_UE_loc2, aerial_fading = False, frequency = frequency)
    data = open(g_channels.path + 'ray_tracing.txt', 'r')
    g_channel_list =g_channels.getList(data)
    g_channel_list = np.array(g_channel_list).reshape(n_gUE_dropped,-1)

    BS_set = []
    # firstly drop BSs
    for a in range(n_BS):
        bs = BS(network = network, bs_type=1, bs_id=a, channel_info = channel_info, loc =bs_loc[a][:,None]) # 1 is terrestrial, 0 is aerial
        BS_set.append(bs)
    # secondly drop UAVs
    uav_list = []
    g_ue_list = []

    for k in np.arange(n_UAV_dropped):

        uav = UE(channel_info=channel_info, network=network, UE_id=k, ground_UE=False, loc=UAV_loc[k], bs_info=BS_set,
                 drone_antenna_gain = drone_antenna_gain, enable_3gpp_ch = enable_3gpp_ch)

        uav.set_channel(uav_channel_list[k], uav_link_states[k])
        uav.association()
        uav_list.append(uav)
    # drop g UEs
    for k in np.arange(n_gUE_dropped):
        g_UE = UE(channel_info=channel_info, network=network, bs_info=BS_set,  UE_id=k + n_UAV_dropped,
                   ground_UE=True, loc=g_UE_loc[k])
        g_UE.set_channel(chan_list=g_channel_list[k])
        g_UE.association()  # drop ground UE
        g_ue_list.append(g_UE)

    total_ue_list = np.append(uav_list, g_ue_list)
    # decide the association and allocate resources
    total_connected_ue_id_list = []
    for bs in BS_set:
        if bs.connected is True:
            connection_complete, connected_ue_id = bs.decide_connection_multi_UE(max_n=max_n)
            if connection_complete is True:
                total_connected_ue_id_list = np.append(total_connected_ue_id_list, connected_ue_id)
                #resource_indices = np.arange(len(connected_ue_id))
                #np.random.shuffle(resource_indices)
                #for j, id in zip(resource_indices, connected_ue_id):
                #    total_ue_list[id].set_resource_index(j)

    UL_DATA = np.array([])

    connected_BS_list = np.array([])
    for bs in BS_set:
        if bs.connected is True:
            data = bs.get_interference(total_UE_list= total_ue_list, total_connected_ue_id_list = total_connected_ue_id_list)
            UL_DATA = np.append(UL_DATA, data)
            connected_BS_list = np.append(connected_BS_list, bs)

    #DL_DATA = np.array([])

    #connected_uav_ids = total_connected_ue_id_list[total_connected_ue_id_list<n_UAV_dropped]
    #connected_uavs= total_ue_list[np.array(connected_uav_ids, dtype = int)]

    #for ue_id in total_connected_ue_id_list:
    #    ue = total_ue_list[int(ue_id)]
    #    if ue.get_ue_type() == 'g_ue' and n_UAV_dropped !=0:
    #        itf_ue_uav = ue.get_interference_from_uav(connected_uavs = connected_uavs)
    #    else:
    #        itf_ue_uav = -200

    #    data = ue.get_interference(connected_BS = connected_BS_list)
    #    data['itf_ue_uav'] = itf_ue_uav
    #    DL_DATA = np.append(DL_DATA, data)

    # collect all data and save from only UAVs
    '''
    if enable_3gpp_ch is True:
        path_ul = 'data_with_3gpp_channel/data_%dm_height/uplink_interf_and_data_%dG_%d.txt' % (
            UAV_height,int(frequency/1e9), t)
        path_dl = 'data_with_3gpp_channel/data_%dm_height/downlink_interf_and_data_%dG_%d.txt' % (
            UAV_height,int(frequency/1e9), t)
    else:
    '''
    '''
    if frequency == 2e9:
        if ptrl is True:
            path_ul = ab_path + '/uav_interference_analysis/test_data/sub6_%d_stream_ptrl/UAV_%d/uplink_itf_ns=%d_h=%d_%dG_%d.txt' % \
                      (max_n, uav_p, max_n, UAV_height, int(frequency / 1e9), t)
            path_dl = ab_path + '/uav_interference_analysis/test_data/sub6_%d_stream_ptrl/UAV_%d/downlink_itf_ns=%d_h=%d_%dG_%d.txt' % \
                      (max_n, uav_p, max_n, UAV_height, int(frequency / 1e9), t)
        else:
            path_ul = ab_path + '/uav_interference_analysis/test_data/sub6_%d_stream/UAV_%d/uplink_itf_ns=%d_h=%d_%dG_%d.txt' % \
                      (max_n, uav_p, max_n, UAV_height, int(frequency / 1e9), t)
            path_dl = ab_path + '/uav_interference_analysis/test_data/sub6_%d_stream/UAV_%d/downlink_itf_ns=%d_h=%d_%dG_%d.txt' % \
                      (max_n, uav_p, max_n, UAV_height, int(frequency / 1e9), t)
    else:
        if ptrl is True:
            path_ul = ab_path + '/uav_interference_analysis/test_data/%d_stream_ptrl/UAV_%d/uplink_itf_ns=%d_h=%d_%dG_%d.txt' % \
                      (max_n, uav_p, max_n, UAV_height, int(frequency / 1e9), t)
            path_dl = ab_path + '/uav_interference_analysis/test_data/%d_stream_ptrl/UAV_%d/downlink_itf_ns=%d_h=%d_%dG_%d.txt' % \
                      (max_n, uav_p, max_n, UAV_height, int(frequency / 1e9), t)
        else:
            path_ul = ab_path + '/uav_interference_analysis/test_data/%d_stream/UAV_%d/uplink_itf_ns=%d_h=%d_%dG_%d.txt' % \
                      (max_n, uav_p, max_n, UAV_height, int(frequency / 1e9), t)
            path_dl = ab_path + '/uav_interference_analysis/test_data/%d_stream/UAV_%d/downlink_itf_ns=%d_h=%d_%dG_%d.txt' % \
                      (max_n, uav_p, max_n, UAV_height, int(frequency / 1e9), t)
    '''
    path_ul = ab_path + '/uav_interference_analysis/test_data/%d_stream_ptrl/UAV_%d/uplink_itf_ns=%d_h=%d_%dG_%d.txt' % \
              (max_n, n_UAV, max_n, UAV_height, int(frequency / 1e9), t)
    #path_dl = ab_path + '/uav_interference_analysis/test_data/%d_stream_ptrl/UAV_%d/downlink_itf_ns=%d_h=%d_%dG_%d.txt' % \
    #          (max_n, n_UAV, max_n, UAV_height, int(frequency / 1e9), t)
    save_file(UL_DATA, path_ul)
    #save_file(DL_DATA, path_dl)
