import random

import sys
import pathlib
ab_path = pathlib.Path().absolute().parent.__str__()
#print (ab_path)
sys.path.append(ab_path+'/uav_interference_analysis/src/')
sys.path.append('src/')

import numpy as np
from tqdm.auto import tqdm
from device.ue import  UE
from device.bs import BS
from helper.network_channel import Channel_Info, Network
#import pandas as pd
#from ground_channel_generation import GroundChannel
from mmwchanmod.sim.drone_antenna_field import drone_antenna_gain
import argparse

#from power_optimizer import power_optimization
from helper.utills import check_min_dist_UE_BS, get_location, save_file, get_channels
random.seed(10)
parser = argparse.ArgumentParser(description='')

parser.add_argument('--height',action='store',default=60,type= int,\
    help='uav_height')
parser.add_argument('--n_drop',action='store',default=10,type= int, help='number of drops per cell')

parser.add_argument('--t_gpp', dest = 't_gpp', action = 'store_true')

parser.add_argument('--ptrl', dest = 'ptrl', action = 'store_true')

parser.add_argument('--n_UAV',action='store',default=5,type= int,\
    help='number of UAVs considered for association')

parser.add_argument('--n_gUE',action='store',default=5,type= int,\
    help='number of ground UEs considered for association')

parser.add_argument('--run_time',action='store',default=30,type= int,\
    help='simulation running time')

parser.add_argument('--frequency',action='store',default=28e9,type= float,\
    help='carrier frequency')
parser.add_argument('--alpha',action='store',default=0.8,type= float,\
    help='alpha for power control')

parser.add_argument('--P0',action='store',default=-82,type= float,\
    help='reference power for power control')

parser.add_argument('--ISD_d',action='store',default=200,type= int,\
    help='inter-site distance between rooptop BSs')

parser.add_argument('--n', action ='store', default = 2, type = int,\
                    help = 'number of connections to each BS')

args = parser.parse_args()
uav_height = args.height
# running time
run_time = args.run_time
alpha = args.alpha
P0 = args.P0

frequency = args.frequency
ptrl = args.ptrl
enable_3gpp_ch = args.t_gpp

max_n = args.n
ISD_d = args.ISD_d

n_UAV = args.n_UAV
n_gUE = args.n_gUE
uav_percentage = (n_UAV)/(n_gUE+ n_UAV)

print ('uav percentage is %2.3f' %(int(uav_percentage*100)))
print ('number of connections to BS is ', max_n)
print ('alpha is %s and P0 is %s'% (alpha, P0))

n_drop = args.n_drop

g_ue_height = 1.6
print ('UAV height is ', uav_height)
# set some parameters for network area
random.seed(100)
ISD_s = 200

uav_access_bs_t  = True
enable_dl_itf = False # enabling downlink interference computation

if uav_access_bs_t is True:
    print ('----------------- open access  ---------------')
else:
    print('----------------- close access  ---------------')

print ('ISD between standard BSs is ,', ISD_s)
print ('ISD between dedicated BSs is ,', ISD_d)

lam_s = 4/(np.pi*ISD_s**2)
if ISD_d !=0:
    lam_d = 4/(np.pi*ISD_d**2)
else:
    lam_d = 0 

XMAX, YMAX = 1000, 1000
area = XMAX * YMAX
#print ('Average number of standard BSs is ', int(lam_s*area))
#print ('Average number of dedicated BSs is ', int(lam_d*area))
min_distance = 10
# set network we observe
network = Network(X_MAX = XMAX, X_MIN = 0, Y_MAX = YMAX, Y_MIN = 0, Z_MAX = uav_height)
# channel information

print ('3 Gpp Aerial channel is, ', enable_3gpp_ch)
channel_info = Channel_Info(city = 'uav_moscow')
channel_info.alpha = alpha
channel_info.P0 = P0
channel_info.ptrl = ptrl
#channel_info.long_term_bf = True
channel_info.frequency = frequency

drone_antenna_gain = drone_antenna_gain()
print ('power control is', ptrl)

if enable_3gpp_ch is True:
    uav_channel_model = channel_info.ground_channel
    channel_info.BW = 80e6
else:
    uav_channel_model = channel_info.aerial_channel

print ('frequency is ', frequency)
# every drop we set different number of BSs and their locations
# number of ground UE = number of BS, and their locations are closer
print ('number of ue per cell is, ', n_drop)
for t in tqdm(range (run_time), desc= 'number of drops'):
    n_bs_s = np.random.poisson(lam_s*area) # number of standard BSs
    n_bs_d = np.random.poisson(lam_d *area) # number of dedicated BSs

    if uav_access_bs_t is True:
        n_total = n_bs_s *n_drop
        n_uav_dropped = int(uav_percentage * n_total)
        n_gue_droppped = n_total - n_uav_dropped
        if n_uav_dropped == 0:
            n_bs_d =0
    else:
        n_uav_dropped = n_drop*n_bs_d
        n_gue_droppped = n_drop * n_bs_s
   # print (n_uav_dropped)
    channel_info.N_UAV = n_uav_dropped
    # locations of standard BSs
    bs_loc_s = get_location(n = n_bs_s, MAX = 1000, isBS = True, low = 10, high = 10.5)

    if n_uav_dropped !=0:
      # generate all UAV locations we will utilize to drop over simulation time
      uav_loc = get_location(n = n_uav_dropped, MAX = 1000, isBS = False, h = uav_height)
      # channel between UAV and standard BSs
      uav_channel_list_s = get_channels(uav_loc, bs_loc_s,channel_info.aerial_channel, network,
                                                     frequency = frequency, cell_type = 1,
                                                    three_gpp = False )
    else:
        print ('purely ground UEs are deployed')

    # set the locations of ground UEs
    g_ue_loc = get_location(n = n_gue_droppped, MAX = 1000, isBS = False, h = g_ue_height)
    g_ue_loc = check_min_dist_UE_BS(bs_loc_s, g_ue_loc, n_gue_droppped)
    g_channel_list_s = get_channels(g_ue_loc, bs_loc_s, channel_info.ground_channel, network,
                                  frequency = frequency, three_gpp = True, aerial_fading = False)

    # if dedicated BSs are added
    if n_bs_d !=0:
        bs_loc_d = get_location(n=n_bs_d, MAX=1000, isBS=True, low=10, high=30)
        bs_loc = np.append(bs_loc_s, bs_loc_d, axis = 0)
        # channel between UAV and dedicated BSs
        uav_channel_list_d = get_channels(uav_loc, bs_loc_d, channel_info.aerial_channel, network,
                                                            frequency=frequency, cell_type=0,
                                                            three_gpp = False, aerial_fading = True)
        uav_channel_list = np.append(uav_channel_list_s, uav_channel_list_d, axis = 1)

        g_channel_list_d = get_channels(g_ue_loc, bs_loc_d, channel_info.ground_channel, network,
                                        frequency=frequency, three_gpp=True)
        g_channel_list = np.append(g_channel_list_s, g_channel_list_d, axis=1)
    # if only standard BSs are deployed
    else:
        if n_uav_dropped !=0:
            uav_channel_list = uav_channel_list_s
        bs_loc = bs_loc_s
        g_channel_list = g_channel_list_s
    bs_set = []
    # firstly drop terrestrial BSs
    for a in range(n_bs_s):
        # deploy terrestrial and aerial BSs
        bs_t = BS(network = network, bs_type=1, bs_id= a, channel_info = channel_info, loc =bs_loc_s[a][:,None]) # 1 is terrestrial, 0 is aerial
        bs_set.append(bs_t)
    # secondly drop aerial BSs
    for a in range(n_bs_d):
        bs_a = BS(network=network, bs_type=0, bs_id= a + n_bs_s, channel_info=channel_info, loc=bs_loc_d[a][:, None])  # 1 is terrestrial, 0 is aerial
        bs_set.append(bs_a)

    # thirdly drop UAVs
    uav_list = []
    for k in np.arange(n_uav_dropped):
        uav = UE(channel_info=channel_info, network=network, UE_id=k, ground_UE=False, loc=uav_loc[k], bs_info=bs_set,
                 drone_antenna_gain = drone_antenna_gain)
        uav.set_channel(chan_list=uav_channel_list[k])
        uav.association(uav_access_bs_t= uav_access_bs_t)
        uav_list.append(uav)
    # lastly ground UEs
    g_ue_list = []
    for k in np.arange(n_gue_droppped):
        g_UE = UE(channel_info=channel_info, network=network, bs_info=bs_set,  UE_id=k + n_uav_dropped,
                   ground_UE=True, loc=g_ue_loc[k])
        g_UE.set_channel(chan_list=g_channel_list[k])
        g_UE.association()  # drop ground UE
        g_ue_list.append(g_UE)

    total_ue_list = np.append(uav_list, g_ue_list)
    # decide the association and allocate resources
    total_connected_ue_id_list = []
    connected_bs_list = np.array([])
    for bs in bs_set:
        if bs.connected is True:
            connection_complete, connected_ue_ids = bs.decide_connection_multi_UE(max_n=max_n, uav_access_bs_t=uav_access_bs_t)
            if connection_complete is True:
                # compute the channel only for connected UEs for saving time
                for c_ue in total_ue_list[connected_ue_ids]:
                    c_ue.compute_channels()
                
                total_connected_ue_id_list = np.append(total_connected_ue_id_list, connected_ue_ids)
                connected_bs_list = np.append(connected_bs_list, bs)

    UL_DATA = np.array([])
    for bs in connected_bs_list:
        data = bs.get_interference(total_UE_list= total_ue_list, total_connected_ue_id_list = total_connected_ue_id_list)
        UL_DATA = np.append(UL_DATA, data)

    # collect all data and save from only UAVs

    if ptrl is True:
        if uav_access_bs_t is True:
            path_ul = ab_path + '/uav_interference_analysis/test_data/%d_stream_ptrl/uplink_itf_UAV=%d_ISD_d_=%d_ns=%d_h=%d_%dG_%d.txt' % \
                (max_n, n_UAV, ISD_d, max_n, uav_height, int(frequency / 1e9), t)
        else:
            path_ul = ab_path + '/uav_interference_analysis/test_data_closed/%d_stream_ptrl/uplink_itf_UAV=%d_ISD_d_=%d_ns=%d_h=%d_%dG_%d.txt' % \
                      (max_n, n_UAV, ISD_d, max_n, uav_height, int(frequency / 1e9), t)
    else:
        if uav_access_bs_t is True:
            path_ul = ab_path + '/uav_interference_analysis/test_data/%d_stream_ptrl/full_power_uplink_itf_UAV=%d_ISD_d_=%d_ns=%d_h=%d_%dG_%d.txt' % \
                    (max_n, n_UAV, ISD_d, max_n, uav_height, int(frequency / 1e9), t)
        else:
            path_ul = ab_path + '/uav_interference_analysis/test_data_closed/%d_stream_ptrl/full_power_uplink_itf_UAV=%d_ISD_d_=%d_ns=%d_h=%d_%dG_%d.txt' % \
                      (max_n, n_UAV, ISD_d, max_n, uav_height, int(frequency / 1e9), t)

    save_file(UL_DATA, path_ul)

    if enable_dl_itf is True:
        DL_DATA = np.array([])
        connected_uav_ids = total_connected_ue_id_list[total_connected_ue_id_list<n_uav_dropped]
        connected_uavs= total_ue_list[np.array(connected_uav_ids, dtype = int)]
        for ue_id in total_connected_ue_id_list:
            ue = total_ue_list[int(ue_id)]
            if ue.get_ue_type() == 'g_ue' and n_uav_dropped !=0:
                itf_ue_uav = ue.get_interference_from_uav(connected_uavs = connected_uavs)
            else:
                itf_ue_uav = -200

            data = ue.get_interference(connected_BS = connected_bs_list)
            data['itf_ue_uav'] = itf_ue_uav
            DL_DATA = np.append(DL_DATA, data)

        path_dl = ab_path + '/uav_interference_analysis/test_data/%d_stream_ptrl/downlink_itf_UAV=%d_ISD_d_=%d_ns=%d_h=%d_%dG_%d.txt' % \
                      (max_n, n_UAV,  ISD_d,  max_n, uav_height,  int(frequency / 1e9), t)

        save_file(DL_DATA, path_dl)

