import numpy as np
from mmwchanmod.sim.chanmod import MPChan
import os
import pathlib
ab_path = pathlib.Path().absolute().parent.__str__()

class GroundChannel():
    def __init__(self, ):
        self.path = pathlib.Path().absolute().home().__str__() + '/ns3-mmwave/'
        pass

    def run_ns3(self, tx=np.zeros([1,1]), rx = np.zeros([1,1]), aerial_fading:bool = False, frequency:int = 28e9):
        tx, rx = tx.reshape(-1, 3), rx.reshape(-1, 3)
        os.chdir (self.path)
        f = open('location.txt', 'w')
        for i in range(len(tx)):
            f.write("tx\t" + str(tx[i][0]) + "\t" + str(tx[i][1]) + "\t" + str(tx[i][2]) + '\n')
            f.write("rx\t" + str(rx[i][0]) + "\t" + str(rx[i][1]) + "\t" + str(rx[i][2]) + '\n')
        f.close()
        os.system('rm ray_tracing.txt')
        #os.system('./waf --run scratch/three-gpp-channel-example > /dev/null')
        if aerial_fading is True:
            aerial_fading_ = 'true'
            scenario = 'UMi-StreetCanyon_Aerial'
        else:
            aerial_fading_ = 'false'
            scenario = 'UMi-StreetCanyon'
        os.system('./waf --run \" scratch/three-gpp-channel-example -aerial_fading=' + aerial_fading_
                  +' -frequency=' + str(frequency) + '\"' + '> /dev/null')
        os.system('rm location.txt')
        os.chdir(ab_path + '/uav_interference_analysis/')

    def getList(self, a, get_link_state = False):
        data = dict()
        data_list = list()
        chan_list = list()
        chan = MPChan()
        tx_loc = list()
        rx_loc = list()
        link_state_list = list()
        while True:
            line = a.readline()
            line = line.split()
            if not line: break

            data_s = list(map(float, line[1:]))
            data_s = np.array(data_s)
            data[line[0]] = data_s

            if line[0] == 'delay':
                chan.dly = np.array(data_s) / 1e9
            if line[0] == 'pathloss':
                chan.pl = data_s
            if line[0] == 'aod':
                data_s = data_s % 360
                data_s = data_s - 360 * (data_s > 180)
                chan.ang[:, MPChan.aod_phi_ind] = data_s
            if line[0] == 'aoa':
                data_s = data_s % 360
                data_s = data_s - 360 *(data_s> 180)
                chan.ang[:, MPChan.aoa_phi_ind] = data_s
            if line[0] == 'zod':
                chan.ang = np.zeros((len(data_s), MPChan.nangle), dtype=np.float32)
                chan.ang[:, MPChan.aod_theta_ind] = data_s
            if line[0] == 'zoa':
                chan.ang[:, MPChan.aoa_theta_ind] = data_s
            if line[0] == 'link_state':
                chan.link_state = data_s[0]
                link_state_list.append(int(chan.link_state))
            if line[0] == 'aoa':
                data_list.append(data)
                data = dict()
                chan_list.append(chan)
                chan = MPChan()

            if line[0] == 'TX':
                tx_loc.append(data[line[0]])
            if line[0] == 'RX':
                rx_loc.append(data[line[0]])
        if get_link_state is True:
            return chan_list, link_state_list
        else:
            return chan_list
