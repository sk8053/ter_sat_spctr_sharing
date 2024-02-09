import numpy as np
import scipy.stats
import scipy.constants
import sys;
sys.path.append('..')
from mmwchanmod.sim.antenna import Elem3GPP
from mmwchanmod.sim.array import URA, RotatedArray
from mmwchanmod.sim.chanmod import dir_path_loss_multi_sect
from mmwchanmod.common.constants import LinkState
from helper.utills import get_channels

class UE(object):
    """
    Class for implementing UAV
    """
    def __init__(self, bs_info=None, network = None, channel_info = None, UE_id:int = 0,
                 UAV_Height_max: int = 120, UAV_Height_min:int =10, ground_UE:bool = False
                 , loc:list = [],  drone_antenna_gain = None): #, enable_3gpp_ch:bool = False):

        self.UAV_Height_MAX = UAV_Height_max # UAV height max
        self.UAV_Height_MIN = UAV_Height_min # UAV height min
        self.network = network
        self.channel_info = channel_info
        thetabw_, phibw_ = 65, 65 # half power of beamwidth
        # velocity setting
        self.ground_UE = ground_UE
        # UAV ID setting
        self.ue_id = UE_id
        frequency = channel_info.frequency

        self.lambda_ = scipy.constants.speed_of_light / frequency
        self.wrx = 0
        self.rx_sv = dict()
        self.ue_elem_gain = dict()
        self.bs_elem_gain = dict()
        self.bs_locations = np.empty(shape=[0, 3])

        # obtain the information on all BSs
        self.cell_types = np.array([])
        self.arr_gnb_list = []

        #self.pl_gain = np.zeros(len(bs_info))
        self.received_power = np.zeros(len(bs_info))
        # UE, BS beamforming vectors
        self.ue_w_bf, self.ue_w_bf_dl, self.bs_w_bf = list(), list(),list()
        # UE and BS spatial signatures
        self.ue_sv, self.bs_sv = dict(), dict()
        # enable 3gpp channel
        #self.enable_3gpp_ch = enable_3gpp_ch
        # associated section index of serving BS
        self.serving_bs_sect_ind = 0
        # map from BS id to associated section index
        self.bs_sect_ind_dict = dict()
        # all the BSs
        self.bs_set = bs_info
        # power control parameters for uplink transmission
        self.alpha = channel_info.alpha
        self.P0 = channel_info.P0
        # transmission power of UE, default value is maximum, 23dBm
        self.tx_power = channel_info.UE_TX_power
        # carrier frequency
        self.frequency = frequency
        # collect some information from all the BSs
        for bs in bs_info:
            self.bs_locations = np.vstack((self.bs_locations, bs.get_location()))  # collect the location information
            self.cell_types = np.append(self.cell_types, bs.bs_type)  # get BS types, terrestrial or aerial
            self.arr_gnb_list.append(bs.get_antenna_array())  # get antenna arrays for all BSs
        # number of sector of BSs
        self.sect_bs_n = bs_info[0].get_number_of_sectors()
        ## intial connection with a random BS
        self.serving_bs_id = -1
        # location of UE
        self.loc = loc
        # index of resource
        self.resource_id = -1
        # set initial location randomly
        # UAV antenna element array setting
        element_ue = Elem3GPP(thetabw=thetabw_, phibw=phibw_)
        if self.frequency ==2e9: # and self.ground_UE is True:
            arr_ue = URA(elem=element_ue, nant=np.array([1, 1]), fc=frequency, drone_antenna_gain=drone_antenna_gain)
        elif self.frequency ==140e9:
            arr_ue = URA(elem=element_ue, nant=np.array([8, 8]), fc=frequency, drone_antenna_gain=drone_antenna_gain)
        else:
            arr_ue = URA(elem=element_ue, nant=np.array([4, 4]), fc=frequency, drone_antenna_gain=drone_antenna_gain)

       # else:
        #    raise ValueError("The antenna configuration is not set for the frequency,", self.frequency)
        # configure antenna arrays for UAVs and UEs
        rand_azm = np.random.uniform(low = -180, high = 180, size=(1,))[0]
        if self.ground_UE is True:
            self.arr_ue = RotatedArray(arr_ue, theta0=0, phi0= rand_azm , drone = False)
            self.arr_ue_list = [self.arr_ue]
        else:
            self.arr_ue = RotatedArray(arr_ue, theta0=-90,phi0=rand_azm,  drone = True)
            self.arr_ue_list =[self.arr_ue]

    # some get-functions
    def get_Tx_power(self):
        return self.tx_power
    def get_id(self):
        return self.ue_id
    def get_array(self):
        return self.arr_ue_list
    def get_bf(self):
        return self.ue_w_bf
    def get_bf_dl(self):
        return self.ue_w_bf_dl
    def get_ue_type(self):
        serv_ue_loc = np.squeeze(self.get_current_location())
        ue_height = serv_ue_loc[2]
        if ue_height > 20:
            return 'uav'
        else:
            return 'g_ue'
    def get_bs_sect_n(self):
        return self.serving_bs_sect_ind
    def get_resource_index(self):
        return self.resource_id

    def set_resource_index(self, resource_id):
        self.resource_id = resource_id


    def set_channel(self, chan_list=None, link_state = None):
        # set ground user channels that obtained from ns3
        # we assume all ground users are static
        '''
        if np.ndim(chan_list) ==0:
            ue_loc = np.repeat(self.get_current_location().reshape(-1,3), len(self.bs_locations), axis = 0)
            self.g_channel = GroundChannel(tx=self.bs_locations, rx=ue_loc)
            data = open(self.g_channel.path+'ray_tracing.txt', 'r')
            self.g_chan_list = self.g_channel.getList(data)
        else:
            self.g_chan_list = list(chan_list)
        '''
        self.channels = chan_list
        #for chan in chan_list:
        #    if chan.link_state !=0 and len(chan.pl)>10:
        #        print (chan.dly*1e9)
        #        chan.pl = chan.pl[:10]
        #        chan.ang = chan.ang[:10]

        self.link_state = link_state

    def get_random_locations(self):
        """
        obtain the random location over the given network area
        """
        X_MAX, Y_MAX = self.network.X_MAX, self.network.Y_MAX
        X_MIN, Y_MIN = self.network.X_MIN, self.network.Y_MIN
        xx = np.random.uniform(low=X_MIN, high=X_MAX, size=1)
        yy = np.random.uniform(low=Y_MIN, high=Y_MAX, size=1)
        trajectory_2D = np.column_stack((xx, yy))
        uav_height = np.array([self.UAV_Height_MAX])
        loc = np.append(trajectory_2D, uav_height[:, None], axis=1)
        return loc

    def set_locations(self,loc= np.array([])):
        # set location of UAV and UE from outside
        if self.ground_UE is True:
            self.UAV_Height_MAX = 1.6

        if len (loc) == 0: # if location is empy, random locations are chosen
            self.loc = self.get_random_locations()
        else:
            self.loc = np.array(loc).reshape(-1,3)

    def get_long_term_beamforming_vectors(self, ue_sv:list, bs_sv:list, n_rand:int = 10, uplink:bool = True, get_w_bs= False
                                          , path_gain:list = None):
        '''
        compute the long-term beamfomring vector based on channel information at BS and UE sides

        Parameters
        ----------
        ue_sv: spatial signature at UE side
        bs_sv: spatial signature at BS side
        n_rand: number of iteration for small-scale fading randomization
        uplink: determine if channel is uplink or downlink
        Returns
        -------
        w_ue: list
        beamforming vector at UE side
        w_bs: list
        beamforming vector at BS side

        '''
        n_r, n_t = ue_sv.shape[1], bs_sv.shape[1]
        #print(pl.shape, ue_sv.shape)
        path_gain_lin = 10**(-0.05*(path_gain - min(path_gain)))

        ue_sv, bs_sv = ue_sv.T, bs_sv.T
        bs_sv = path_gain_lin[None] * bs_sv
        Cov_H_bs, Cov_H_ue = np.zeros((n_rand, n_t, n_t), dtype=complex), np.zeros((n_rand, n_r, n_r), dtype=complex)

        for i in range(n_rand):

            #ue_sv2 = ue_sv #* np.exp(-1j * theta_random[:, None])
            if uplink is True:
                H = bs_sv.dot(np.matrix.conj(ue_sv).T)  # H(f) shape = 64*16
                theta_random = np.random.uniform(low=-np.pi, high=np.pi, size=(H.shape[0],))
                H = H*np.exp(-1j*theta_random[:,None])
                Cov_H_ue[i] =np.matrix.conj(H).T.dot(H)
            else: # downlink channel case
                H = ue_sv.dot(np.matrix.conj(bs_sv).T)  # H(f) shape = 16*64
                theta_random = np.random.uniform(low=-np.pi, high=np.pi, size=(H.shape[0],))
                H = H * np.exp(-1j*theta_random[:,None])
                Cov_H_ue[i] = H.dot(np.matrix.conj(H).T)
            #if get_w_bs is True:
            #    Cov_H_bs[i] = np.matrix.conj(H).T.dot(H)


        Cov_H_ue = np.mean(Cov_H_ue, axis=0)  # 16*16 Qrx
        eig_value_ue, eig_vector_ue = np.linalg.eig(Cov_H_ue)
        w_ue = eig_vector_ue[:, np.argmax(eig_value_ue)]  # 16*1

        #if get_w_bs is True:
        #    Cov_H_bs = np.mean(Cov_H_bs, axis=0)  # 64*64 Qtx
        #    eig_value_bs, eig_vector_bs = np.linalg.eig(Cov_H_bs)
        #    w_bs = eig_vector_bs[:, np.argmax(eig_value_bs)]  # 64*1
        #    return w_ue, w_bs
        #else:
        return w_ue

    def association (self, uav_access_bs_t:bool = True):
        '''
        1) compute channels between UEs (or UAVs) and BSs
        2) save all the data related to channel information
        3) perform association based on the received power from BSs
        4) choose optimal one BS and its sector as serving BS and sector

        Parameters
        ----------
        uav_access_bs_t: bool
        enable UAVs to access to terrestrial BSs
        Returns
        -------
        None
        '''
        if len(self.loc) == 0:
            self.loc = self.get_random_locations()

        # first, get channels for all BSs
        loc = self.loc.reshape(1,3)
        dist_vectors = loc - self.bs_locations
        #dist_vectors = self.wrap_around(dist_vectors)

        dist3D = np.linalg.norm(dist_vectors, axis = 1)
        bs_sect_ind_dict = dict()
        # then compute path loss, channel matrices, and beamforming vectors for all links
        for bs_id, channel in enumerate(self.channels):

            data = dir_path_loss_multi_sect(self.arr_gnb_list[bs_id], self.arr_ue_list, channel,
                                            isdrone=not self.ground_UE,
                                            return_elem_gain= True)

            fspl = 20 * np.log10(dist3D[bs_id]) + 20 * np.log10(self.frequency) - 147.55
            channel.pl[channel.pl < fspl] = np.random.uniform(low = fspl, high = fspl + 100)

            if len(channel.pl) ==0: # outage case
                channel.pl = np.array([250])
                #self.pl_gain[bs_id] = np.zeros((1,))  # path gain = path loss + bf gain + elem gain
                self.bs_elem_gain[bs_id] = {i:np.array([-250]) for i in range(self.sect_bs_n)}
                self.ue_elem_gain[bs_id] = {i:np.array([-250]) for i in range(self.sect_bs_n)}
                self.received_power[bs_id] = -np.inf
                bs_sect_ind = 0
            else:
                # for every UE, make maps from BS ID to channel parameters
                self.bs_elem_gain[bs_id] = data['ue_elem_gain_dict']
                self.ue_elem_gain[bs_id] = data['bs_elem_gain_dict']

                bs_sect_ind = data['sect_ind']
                received_power_db = self.channel_info.BS_TX_power - channel.pl  + self.bs_elem_gain[bs_id][bs_sect_ind] \
                                    + self.ue_elem_gain[bs_id][bs_sect_ind]
                received_power_lin = 10**(0.1*received_power_db)
                self.received_power[bs_id] = 10*np.log10(received_power_lin.sum())

            # map from bs id to sector id for one UE
            bs_sect_ind_dict[bs_id] = bs_sect_ind

        # Do association
        # find serving BSs having maximum received power
        if self.ground_UE is True: # UEs can only be connected with standard BSs
            max_power =  max(self.received_power[self.cell_types==1])
        # UAV can connect with either standard BSs and rooftop BSs
        elif self.ground_UE is False and len(self.cell_types) !=0:
            if uav_access_bs_t is True:
                max_power = max(self.received_power) # [self.cell_types == 0]
            else: # in this case, UAV can only access to dedicated BSs.
                max_power = max(self.received_power[self.cell_types == 0])
        else:
            raise RuntimeError('UAVs are trying to connects to dedicated BSs, but there is no dedicated BS ')


        serv_bs_id = np.where(self.received_power == max_power)[0][0]


        if self.channels[serv_bs_id].link_state != LinkState.no_link:

            self.bs_set[serv_bs_id].connect_ue(self.ue_id, self.received_power[serv_bs_id], self.get_ue_type())
            self.serving_bs_sect_ind = bs_sect_ind_dict[serv_bs_id]
            self.serving_bs_id = serv_bs_id
            # convey index of serving sector of BS to serving BS
            self.bs_set[serv_bs_id].set_serv_bs_sect(ue_id = self.ue_id, n = self.serving_bs_sect_ind)

            # power control
            # compute Tx power of UE according to simple power control algorithm according to the formula given to 3GPP
            if self.get_ue_type() == 'g_ue': # ground UEs performing power-control always
                P_ex = (self.channels[serv_bs_id].pl - self.bs_elem_gain[serv_bs_id][self.serving_bs_sect_ind]
                                - self.ue_elem_gain[serv_bs_id][self.serving_bs_sect_ind]) * self.alpha + self.P0
                P_ex = 10*np.log10(np.sum(10**(0.01*P_ex)))
                self.tx_power = min(P_ex, self.channel_info.UE_TX_power)
            else: # UAV can do power -control or not
                if self.channel_info.ptrl is True:
                    P_ex = (self.channels[serv_bs_id].pl - self.bs_elem_gain[serv_bs_id][self.serving_bs_sect_ind]
                                - self.ue_elem_gain[serv_bs_id][self.serving_bs_sect_ind]) * self.alpha + self.P0
                    P_ex = 10*np.log10(np.sum(10**(0.01*P_ex)))
                    self.tx_power = min(P_ex, self.channel_info.UE_TX_power)
                else:
                    self.tx_power = 23

    def compute_channels(self, codebook_bf = False):
        '''
        compute channel between UE and BS
         '''
        if len(self.loc) == 0:
            self.loc = self.get_random_locations()
        pl_list = []
        # then compute path loss, channel matrices, and beamforming vectors for all links
        for bs_id, channel in enumerate(self.channels):
            data = dir_path_loss_multi_sect(self.arr_gnb_list[bs_id], self.arr_ue_list, channel,
                                            isdrone=not self.ground_UE,
                                            return_elem_gain=False)
            if channel.link_state == LinkState.no_link: # outage case
                if self.frequency == 2e9: # and self.ground_UE is True:
                    N_ue = 1
                    N_bs = 8
                elif self.frequency ==140e9:
                    N_ue = 64
                    N_bs = 256
                else:
                    N_ue = 16
                    N_bs = 64
                self.ue_sv[bs_id] = {i:np.zeros((1, N_ue)) for i in range(self.sect_bs_n)}  # rx_sv without element gain
                self.bs_sv[bs_id] = {i:np.zeros((1, N_bs)) for i in range(self.sect_bs_n)}  # tx_sv without element gain
                data['ue_sv_dict'] = self.ue_sv[bs_id]
                data['bs_sv_dict'] = self.bs_sv[bs_id]
            else:
                # for every UE, make maps from BS ID to channel parameters
                self.ue_sv[bs_id] = data['ue_sv_dict'] # rx_sv without element gain
                self.bs_sv[bs_id] = data['bs_sv_dict'] # tx_sv without element gain
            # BSs have to have the channel information
            data['ue_elem_gain_dict'] = self.ue_elem_gain[bs_id]
            data['bs_elem_gain_dict'] = self.bs_elem_gain[bs_id]
            self.bs_set[bs_id].set_channels(ue_id=self.ue_id, link_state=channel.link_state, pl=channel.pl, data=data)
            pl_list.append(channel.pl)

        if codebook_bf is False:
            # calculate the long-term beamforming vectors
            path_gain = pl_list[self.serving_bs_id] - self.bs_elem_gain[self.serving_bs_id][self.serving_bs_sect_ind] \
                        - self.ue_elem_gain[self.serving_bs_id][self.serving_bs_sect_ind]
            self.ue_w_bf = self.get_long_term_beamforming_vectors \
                        (ue_sv = self.ue_sv[self.serving_bs_id][self.serving_bs_sect_ind],
                         bs_sv = self.bs_sv[self.serving_bs_id][self.serving_bs_sect_ind], uplink= True
                         , n_rand = 10, path_gain = path_gain)
            self.ue_w_bf_dl = self.get_long_term_beamforming_vectors \
                        (ue_sv = self.ue_sv[self.serving_bs_id][self.serving_bs_sect_ind],
                         bs_sv = self.bs_sv[self.serving_bs_id][self.serving_bs_sect_ind], uplink= False
                         ,n_rand = 10, path_gain = path_gain)
        else: # do codebook-based beamforming
            if self.get_ue_type() == 'g_ue':
                codebook = np.loadtxt('ue_codebook.txt', dtype = complex)
            else:
                codebook = np.loadtxt('uav_codebook.txt', dtype= complex)
            bs_sv = self.bs_sv[self.serving_bs_id][self.serving_bs_sect_ind].T
            ue_sv = self.ue_sv[self.serving_bs_id][self.serving_bs_sect_ind].T

            H = bs_sv.dot(np.matrix.conj(ue_sv).T) # uplink channel
            H_dl = ue_sv.dot(np.matrix.conj(bs_sv).T)  # H(f) downlink channel
            _, _, vh = np.linalg.svd(H)
            u,_,_ = np.linalg.svd(H_dl)
            s_u, s_d = -200, -200
            for  code in codebook:
                
                s_i_ul = vh.dot(code)
                p_ul = np.linalg.norm(s_i_ul)
                if s_u < p_ul:
                    self.ue_w_bf = code  # only for uplink
                    s_u = p_ul

                s_i_dl = code.dot(u)
                p_dl = np.linalg.norm(s_i_dl)
                if s_d < p_dl:
                    self.ue_w_bf_dl = code  # only for uplink
                    s_d = p_dl

        self.compute_snrs()

    def compute_snrs(self):
        '''
        Compute SNR between UE and all the BSs including its serving BS

        '''
        serv_bs_id = self.serving_bs_id
        for bs_id in range(len(self.channels)):
            for bs_sect_ind in range(self.sect_bs_n):
                if self.channels[bs_id].pl[0] != 250:  # if not outage

                    ue_sv_s = self.ue_sv[bs_id][bs_sect_ind]  # .T,
                    ue_w_bf = self.ue_w_bf

                    bf_gain_UE_side_lobe = 10 * np.log10(np.abs(ue_sv_s.dot(ue_w_bf)) ** 2 + 1e-20)

                    pl_gain_no_BS_after_assocition = self.channels[bs_id].pl - bf_gain_UE_side_lobe - \
                                                     self.ue_elem_gain[bs_id][bs_sect_ind] \
                                                     - self.bs_elem_gain[bs_id][bs_sect_ind]
                    pl_lin_no_BS = 10 ** (-0.1 * (pl_gain_no_BS_after_assocition))
                    pl_gain_no_BS_dB = -10 * np.log10(pl_lin_no_BS.sum())
                else:  # outage case
                    pl_gain_no_BS_dB = 250
                # if bs is not serving BS
                if bs_id != serv_bs_id:
                    serv_ue = False
                    self.bs_set[bs_id].set_snr_by_sidelobe(ue_tx_power=self.tx_power,
                                                           ue_id=self.ue_id,
                                                           pl_gain_no_BS_after_association=pl_gain_no_BS_dB,
                                                           sect_n=bs_sect_ind, serv_ue=serv_ue)

                # if bs is serving BS and its sector is serving sector
                elif bs_id == serv_bs_id and bs_sect_ind == self.serving_bs_sect_ind:
                    serv_ue = True
                    self.bs_set[bs_id].set_snr_by_sidelobe(ue_tx_power=self.tx_power,
                                                           ue_id=self.ue_id,
                                                           pl_gain_no_BS_after_association=pl_gain_no_BS_dB,
                                                           sect_n=bs_sect_ind, serv_ue=serv_ue)

    def get_interference(self, connected_BS = None):
        '''
        interference for downlink case: BS -> UEs
        which is the sum of received powers at UE side from other BSs except serving BSs

        Parameters
        ----------
        connected_BS: list
        the set of associated BSs

        Returns
        -------
        data: a dictionary
        this includes all the information: SINR, SNR, total interference
        intra-cell interference, inter-cell interference, ue id, serving bs id
        beamforming gain, ue type, link state
        '''
        inter_itf, intra_itf = 0,0
        serv_bs_id = self.serving_bs_id # serving BS ID

        serv_tx_rx_bf = 0
        serv_sec_id = self.serving_bs_sect_ind
        # do not consider outage case
        if self.channels[serv_bs_id].link_state != LinkState.no_link:
            # take the beamforming vector toward serving BS at UE side
            # consider all the associated BSs
            for bs in connected_BS:
                bs_id = bs.get_id()
                if self.channels[bs_id].link_state != LinkState.no_link: # check outage case
                    # take all the connected UEs and UAVs to a BS
                    connected_ue_ids = bs.get_connected_ue_list()
                    for c_ue_id in connected_ue_ids:
                        # take beamforming vector toward connected UEs at BS
                        w_bs = bs.get_mmse_precoders(c_ue_id)
                        # obtain the index of sector of serving BS
                        serv_sec_id_at_bs = bs.get_serv_bs_sect(c_ue_id)
                        # get the channel between UAV and the BS that causes interference
                        # we specify the channel including serving sector index of BSs
                        bs_sv = self.bs_sv[bs_id][serv_sec_id_at_bs].T # BS channel between UE and a sector of BS
                        ue_sv = self.ue_sv[bs_id][serv_sec_id_at_bs].T # UE channel between UE and a sector of BS
                        # obtain multi-path channel from BS -> UE
                        H = ue_sv.dot(np.matrix.conjugate(bs_sv.T))
                        n_t, n_r = H.shape
                        # normalize channel
                        H *=np.sqrt(n_t*n_r)/np.linalg.norm(H, 'fro')

                        # compute the beamforming gain
                        # since w_ue is derived from long-term-beamforming scheme
                        # do not take the conjugate to w_ue
                        tx_rx_bf = np.abs(np.conj(self.ue_w_bf_dl).dot(H).dot(w_bs))**2
                        pl_lin = 10**(-0.1*self.channels[bs_id].pl)
                        bs_tx_power = self.channel_info.BS_TX_power
                        # compute received power by taking element gain between the UE and a sector of BS
                        rx_p =  bs_tx_power*tx_rx_bf*pl_lin* \
                                10**(0.1*self.bs_elem_gain[bs_id][serv_sec_id_at_bs]) \
                                * 10**(0.1*self.ue_elem_gain[bs_id][serv_sec_id_at_bs])
                        # if it's not serving BS, received power is interference
                        # inter-cell interference
                        if bs_id != serv_bs_id:
                            itf = rx_p
                            # accumulate all interference values in linear-scale
                            inter_itf += itf.sum()
                        # intra-cell interference
                        elif bs_id == serv_bs_id and c_ue_id != self.ue_id:
                            intra_itf += rx_p.sum()
                        # otherwise it's the signal from serving BS
                        else:
                            l_f = 10*np.log10(rx_p.sum())
                            serv_tx_rx_bf = 10*np.log10(tx_rx_bf)


        # then add interference-power to noise power in linear scale
        KT_lin, NF_lin = 10 ** (0.1 * self.channel_info.KT), 10 ** (0.1 * self.channel_info.NF)
        total_itf = intra_itf + inter_itf
        Noise_Interference = KT_lin * NF_lin * self.channel_info.BW + total_itf

        # obtain SINR in dB-scale
        SINR =  self.channel_info.BS_TX_power + l_f - 10 * np.log10(Noise_Interference)
        SNR = self.channel_info.BS_TX_power + l_f - self.channel_info.KT - self.channel_info.NF - 10*np.log10(self.channel_info.BW)
        # covert interference value from linear scale to dB scale
        # offset value is added in case interference is zero
        total_itf_db = 10*np.log10(total_itf+ 1e-20)

        serv_ue_loc = np.squeeze(self.get_current_location())
        ue_height = serv_ue_loc[2]
        if ue_height > 20:
            ue_type = 'uav'
        else:
            ue_type = 'g_ue'
        link_state = self.channels[serv_bs_id].link_state
        return {'SINR':SINR,
                 'SNR':SNR,
                 'total_itf':total_itf_db,
                 'intra_itf':10*np.log10(intra_itf+1e-20),
                 'inter_itf':10*np.log10(inter_itf+1e-20),
                 'link_state':link_state,
                 'ue_id':self.ue_id,
                 'serving_BS_ID':self.serving_bs_id,
                  'bf_gain':serv_tx_rx_bf,
                   'ue_type':ue_type,
                   'l_f':l_f}


    def get_current_location(self):
        return self.loc

    def wrap_around(self, dist_vec):
        """
        wrap around implementation over the given network area
        """
        dist_x, dist_y, dist_z = dist_vec.T
        ind_x = np.where(self.network.X_MAX - np.abs(dist_x) <= np.abs(dist_x))
        ind_y = np.where(self.network.Y_MAX - np.abs(dist_y) <= np.abs(dist_y))
        dist_x[ind_x] = (-1) * np.sign(dist_x[ind_x]) * (self.network.X_MAX - np.abs(dist_x[ind_x]))
        dist_y[ind_y] = (-1) * np.sign(dist_y[ind_y]) * (self.network.Y_MAX - np.abs(dist_y[ind_y]))
        return np.column_stack((dist_x, dist_y, dist_z))


    def get_channels_to_uav(self, uavs):
        """
        Obtain the channel elements  between the UE and UAVs

        Parameters
        ----------
        uavs: uav objects connected with BSs

        Returns
        -------
        ue_sv: dictionary variable
        spatial signature at UE side

        bs_sv: dictionary variable
        spatial signature at BS side

        ue_elem: dictionary variable
        element gains between UE and UAVs at UE side

        bs_elem: dictionary variable
        element gains between UE and UAVs at UAVs side

        channel_dict: dictionary variable
        channel elements between UE snd UAVs

        """
        uav_locations = []

        for uav in uavs:
            uav_locations.append(uav.get_current_location())
        uav_loc = np.array(uav_locations).reshape(-1,3)
        ue_loc = self.loc.reshape(1, 3)  # location of UE

        channels, link_states = get_channels(ue_loc = uav_loc, bs_loc = ue_loc,
                                             channel_model =self.channel_info.aerial_channel,
                                             network = self.network,
                                        frequency=self.frequency, cell_type=1)

        dist_vectors = uav_loc- ue_loc # distance vector between a UE and UAVs

        #dist_vectors = self.wrap_around(dist_vectors)

        #cell_types = np.repeat([1], len(dist_vectors))

        #channels, link_states = self.channel_info.channel.sample_path(dist_vectors, cell_types)
        dist3D = np.linalg.norm(dist_vectors, axis=1)
        # then compute path loss, channel matrices, and beamforming vectors for all links
        H_dict = dict()
        ue_elem, uav_elem = dict(), dict()
        channel_dict = dict()
        channels = np.squeeze(channels)
        for i, (uav, channel) in enumerate(zip(uavs, channels)):
            uav_array = uav.get_array()
            uav_id = uav.get_id()

            fspl = 20 * np.log10(dist3D[i]) + 20 * np.log10(self.frequency) - 147.55
            channel.pl[channel.pl < fspl] = np.random.uniform(low = fspl, high = fspl +100)
            ## obtain channel matrices between the UE and UAVs
            # firstly, get channel from UE -> UAV
            data1 = dir_path_loss_multi_sect(self.arr_ue_list,uav_array, channel,
                                            isdrone= True, return_elem_gain = True)
            uav_elem[uav_id] = data1['ue_elem_gain_dict'][0]
            ue_elem[uav_id] = data1['bs_elem_gain_dict'][0]

            data2 = dir_path_loss_multi_sect(self.arr_ue_list, uav_array, channel,
                                             isdrone=True, return_elem_gain= False)

            uav_sv= data2['ue_sv_dict'][0].T  # rx_sv without element gain
            ue_sv= data2['bs_sv_dict'][0].T  # tx_sv without element gain

            # but we want to get channel from UAV -> UE
            H = ue_sv.dot(np.matrix.conjugate(uav_sv.T))
            n_t, n_r = H.shape
            # normalize channel
            N = np.linalg.norm(H, 'fro')
            if N !=0:
                H *= np.sqrt(n_t * n_r) /N
            channel_dict[uav_id] = channel
            H_dict[uav_id] = H

        return H_dict, ue_elem, uav_elem, channel_dict

    def get_interference_from_uav(self, connected_uavs = None):
        """
        Obtain the interference between the UE and UAVs

        Parameters
        ----------
        connected_uavs: uav objects connected with BSs
        Returns
        -------
        itf_dB: interference between UE and UAVs in dB scale

        """
        H_dict, ue_elem, uav_elem, channels = self.get_channels_to_uav(connected_uavs)
        # do not consider outage case
        #serv_bs_id = self.serving_bs_id
        # take the beamforming vector toward serving BS at UE side
        w_ue = self.ue_w_bf_dl
        # consider all the associated BSs
        itf = 0
        for uav in connected_uavs:
            uav_id = uav.get_id()
            if channels[uav_id].link_state != LinkState.no_link: # check outage case

                # take beamforming vector at UAV side
                w_uav = uav.get_bf()
                # get spatial signature at UAV and UE

                H = H_dict[uav_id]

                # compute the beamforming gain
                tx_rx_bf = np.abs(np.conj(w_ue).dot(H).dot(w_uav))**2
                pl_lin = 10**(-0.1*channels[uav_id].pl)
                bs_tx_power = self.channel_info.BS_TX_power

                # compute received power by taking element gain between the UE and UAVs
                rx_p =  bs_tx_power*tx_rx_bf*pl_lin* \
                        10**(0.1*uav_elem[uav_id]) \
                        * 10**(0.1* ue_elem[uav_id])
                itf += rx_p.sum()
        itf_dB = 10 * np.log10(itf + 1e-20)

        return itf_dB
