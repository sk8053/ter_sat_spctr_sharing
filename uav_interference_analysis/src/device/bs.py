import numpy as np
import sys;
sys.path.append('..')
from mmwchanmod.sim.antenna import Elem3GPP
from mmwchanmod.sim.array import URA, multi_sect_array

class BS(object):
    """
    implement base station
    """
    def __del__(self):
        pass
    def __init__(self, bs_type: int = 1, bs_id: int = 0, network = None, channel_info = None, loc: list =[], n_sect:int= 3):
        # 1 is terrestrial, 0 is aerial
        # half power beamwidth for horizontal and vertical axis
        thetabw, phibw = 65, 65
        # number of sectors
        self.n_sect = n_sect
        # range of BS heights
        self.Tr_BS_Height_MIN = 5
        self.Tr_BS_Height_MAX = 8
        self.Ar_BS_Height_MIN = 10
        self.Ar_BS_Height_MAX = 30
        # indicate if BS is terrestrial or aerial BS
        self.bs_type = bs_type
        # study area
        self.network = network
        # channel information including tx power and chanel models and etc.
        self.channel_info = channel_info
        # set carrier frequency
        self.frequency = channel_info.frequency
        # Identification number of BSs
        self.bs_id = bs_id
        # indicate if connection is established or not
        self.connected = False
        # obtain total number of UAVs
        self.N_UAV = channel_info.N_UAV
        # total number of connected UEs
        self.total_ue_n = 0
        # set all antenna arrays
        # firstly set antenna elements for terrestrial and aerial BSs
        elem_gnb_t = Elem3GPP(thetabw=thetabw, phibw=phibw)
        elem_gnb_a = Elem3GPP(thetabw=thetabw, phibw=phibw)
        # change the number of antenna elements depending on frequencies.
        if self.frequency == 2e9:
            arr_gnb_t = URA(elem=elem_gnb_t, nant=np.array([8, 1]), fc=self.frequency)
            arr_gnb_a = URA(elem=elem_gnb_a, nant=np.array([8, 1]), fc=self.frequency)
            self.N_a = 8
        elif self.frequency ==140e9:
            arr_gnb_t = URA(elem=elem_gnb_t, nant=np.array([16, 16]), fc=self.frequency)
            arr_gnb_a = URA(elem=elem_gnb_a, nant=np.array([16, 16]), fc=self.frequency)
            self.N_a = 64
        else:
            arr_gnb_t = URA(elem=elem_gnb_t, nant=np.array([8, 8]), fc=self.frequency)
            arr_gnb_a = URA(elem=elem_gnb_a, nant=np.array([8, 8]), fc=self.frequency)
            self.N_a = 64
        # sectorize the antenna arrays for BSs,
        # BSs are equipped with multiple sectors
        if bs_type == 1: # 1 is terrestrial BS, 0 is aerial BS
            arr_gnb_list_t = multi_sect_array(arr_gnb_t, sect_type='azimuth', nsect=self.n_sect, theta0=-12)
            self.arr_gnb_list = arr_gnb_list_t
        else:
            arr_gnb_list_a = multi_sect_array(arr_gnb_a, sect_type='azimuth', nsect=self.n_sect, theta0=45)
            self.arr_gnb_list = arr_gnb_list_a

        self.bs_sv, self.ue_sv = dict(), dict()
        self.link_state = dict()

        self.elem_gain_UE = dict()
        self.elem_gain_BS = dict()

        self.connected_g_ue_id_list = np.array([], dtype= int)
        self.connected_uav_id_list = np.array([], dtype = int)
        self.connected_rx_power_list = np.array([])
        self.H = dict()
        #self.H_first = dict()
        self.SNR_u = dict()
        self.SNR_u_from_itf = dict()


        self.set_locations(loc = loc)
        # mmse-precoder for downlink transmission
        self.w_mmse_precoder_list = dict()
        self.pl = dict()
        # connected UAVs and UEs
        self.connected_entities = None
        self.serving_sect_ind = dict() # index of serving sector at BS

    def set_locations(self,loc = np.array([])):
        # set locations for BSs
        if len(loc) == 0:
            X_MAX, Y_MAX, Z_MAX = self.network.X_MAX, self.network.Y_MAX, self.network.Z_MAX
            X_MIN, Y_MIN, Z_MIN = self.network.X_MIN, self.network.Y_MIN, self.network.Z_MIN
            xx = np.random.uniform(low =X_MIN, high = X_MAX, size =(1,))
            yy = np.random.uniform(low =Y_MIN, high = Y_MAX, size =(1,))
            init_loc = np.column_stack((xx, yy))

            if self.bs_type == 1:  # if bs is terrestrial
                BS_Height_MAX, BS_Height_MIN = self.Tr_BS_Height_MAX, self.Tr_BS_Height_MIN
            else:
                BS_Height_MAX, BS_Height_MIN = self.Ar_BS_Height_MAX, self.Ar_BS_Height_MIN

            init_loc_z =np.random.uniform(low = BS_Height_MIN, high = BS_Height_MAX, size =(1,))
            self.locations = np.append(init_loc, init_loc_z[:, None], axis=1)
        else:
            self.locations = np.array(loc).reshape(-1,3)


    def disconnect_to_uav(self):
        self.connected = False

    def get_location(self):
        return np.squeeze(self.locations)
    def get_id(self):
        return self.bs_id
    def get_antenna_array(self):
        return self.arr_gnb_list

    def set_snr_by_sidelobe(self, ue_tx_power, ue_id,  pl_gain_no_BS_after_association, sect_n, serv_ue = False):
        # set beamforming vector for uplink case
        snr =self.channel_info.UE_TX_power - pl_gain_no_BS_after_association - self.channel_info.KT \
                                - self.channel_info.NF - 10 * np.log10(self.channel_info.BW/100000)

        if serv_ue is True:
            self.SNR_u[(ue_id, sect_n)] = 10 ** (0.1 * snr)
        else:
            self.SNR_u_from_itf[(ue_id, sect_n)] = 10 ** (0.1 * snr)

    def set_channels(self,ue_id = 0,  link_state = None, pl = None, data = None):
        """
        set channels between UEs and this BS

        Parameters
        ----------
        ue_id: int
        UE identification number

        link_state: int
        link state: LOS = 1, NLOS = 2, outage = 0

        pl: list
        pathloss values between UEs and this BS

        data: dictionary
        this includes all the information related to channel

        """
        for i in range(self.n_sect):
            self.elem_gain_UE[(ue_id, i)] = data['ue_elem_gain_dict'][i] # element gain at UE side
            self.elem_gain_BS[(ue_id, i)] = data['bs_elem_gain_dict'][i]
            self.bs_sv[(ue_id, i)] = data['bs_sv_dict'][i].T
            self.ue_sv[(ue_id, i)] = data['ue_sv_dict'][i].T

            elem_gain_UE_lin = 10**(0.05*self.elem_gain_UE[(ue_id, i)])
            elem_gain_BS_lin = 10**(0.05*self.elem_gain_BS[(ue_id, i)])
            #pl_lin = 10**(-0.05*(pl-min(pl)))*elem_gain_UE_lin*elem_gain_BS_lin
            g_l = 10**(-0.05*pl)*elem_gain_UE_lin*elem_gain_BS_lin

            # channel from UE to BS
            # H = sum{ g_l * e_rx * e_tx^H}
            self.bs_sv[(ue_id, i)] *= g_l[None]
            self.H[(ue_id, i)] = self.bs_sv[(ue_id,i)].dot(np.matrix.conj(self.ue_sv[(ue_id, i)]).T)

            #N_r, N_t = self.H[(ue_id, i)].shape
            #H_fro_norm = np.linalg.norm(self.H[(ue_id,i)], 'fro')
            #if H_fro_norm !=0:
            #    self.H[(ue_id,i)] *= np.sqrt(N_t * N_r) / H_fro_norm
        self.link_state[ue_id] = link_state
        # pure pathloss
        self.pl[ue_id] = np.array(pl, dtype = float)



    def connect_ue (self, ue_id, Rx_power, ue_type = 'g_ue'):
        # this function save connection information from UEs
        # but connection-completion is not decided here
        self.connected = True
        if ue_type == 'g_ue':
            self.connected_g_ue_id_list = np.append(self.connected_g_ue_id_list, ue_id)
        else:
            self.connected_uav_id_list = np.append(self.connected_uav_id_list, ue_id)
        self.connected_rx_power_list = np.append(self.connected_rx_power_list, Rx_power)

    
    def decide_connection_multi_UE(self,uav_access_bs_t:bool= True, max_n:int=2):
        """
        Decide and constrain the connections based on maximum number, max_n

        Parameters
        ----------
        max_n: int
        number of UEs which are served simultaneously by BS

        Returns
        -------
        connection_complete: bool
        indicates whether connection is complete or not
        connections become complete when desidred UAVs and UEs are associated

        connection_list: list
        id list of connected UAVs or UEs
        """
        connection_list = np.append(self.connected_uav_id_list,self.connected_g_ue_id_list)
        self.total_ue_n = len(connection_list)
        if uav_access_bs_t is True:
            if len(connection_list)>= max_n:
                np.random.shuffle(connection_list)
                connection_list = connection_list[:max_n]
                connection_complete = True
            else:
                connection_list = np.array([])
                connection_complete = False
            self.connected_entities = connection_list
        else:
            if self.bs_type == 1 and len(self.connected_g_ue_id_list) >= max_n:
                np.random.shuffle(self.connected_g_ue_id_list)
                connection_list = self.connected_g_ue_id_list[:max_n]
                connection_complete = True
            elif self.bs_type == 0 and len(self.connected_uav_id_list) >= max_n:
                np.random.shuffle(self.connected_uav_id_list)
                connection_list = self.connected_uav_id_list[:max_n]
                connection_complete = True
            else:
                connection_list = np.array([])
                connection_complete = False

            self.connected_entities = connection_list

        return connection_complete, self.connected_entities


    def get_beamforming_vectors_from_codebook(self,connected_UE_list:list):
        """
        find beamforming vector from codebook at BS side

        Parameters
        ----------
        connected_UE_list: list of ue objects
        list of connected ue and uav objects

        Returns
        ----------
        w_codebook_bf_list: list
        list of mmse beamforing vectors corresponding to each connected UE or UAV for uplink case
        """
        codebook = np.loadtxt('src/device/bs_codebook.txt', dtype = complex)
        w_codebook_bf_list = dict()
        for ue in connected_UE_list:
            u_id = ue.get_id()
            bs_sect_n = ue.get_bs_sect_n()
            w_UE = ue.get_bf()
            HW = self.H[(u_id, bs_sect_n)].dot(w_UE)
            s = -200
            code_op, i_op = None, None

            for i, code in enumerate(codebook):

                p = np.abs(np.conj(code).T.dot(HW))**2

                if s<p:
                    code_op = code/np.linalg.norm(code)
                    s = p
                    i_op = i
            w_codebook_bf_list[u_id] = code_op
            # remove the selected beamforming vector from codebook
            #codebook = np.delete(codebook,i_op, axis = 0)
        return w_codebook_bf_list

    def get_MMSE_beamform_vectors(self,connected_UE_list, snr_sum_rest_UEs):
        """
        compute the MMSE beamforming vectors at BSs
        to enable spatial division multiplexing

        Parameters
        ----------
        connected_UE_list: list of ue objects
        list of connected ue and uav objects

        snr_sum_rest_UEs: float
        this is sum of snr from other interfering UEs or UAVs
        this value is used for computing MMSE beamforming vector

        Returns
        ----------
        w_mmse_list: list
        list of mmse beamforing vectors corresponding to each connected UE or UAV for uplink case
        w_mmse_list_downlink: list
        list of mmse beamforing vectors corresponding to each connected UE or UAV for downlink case
        """
        sum_cov, sum_cov_dl = 0,0

        N_a = self.N_a
        U =  len(self.connected_entities)
        w_mmse_list = dict()
        w_mmse_list_downlink = dict()
        for ue2 in connected_UE_list:
            u_ = ue2.get_id()
            w_UE2 = ue2.get_bf()
            w_UE2_dl = ue2.get_bf_dl()
            bs_sect_n = ue2.get_bs_sect_n()
            HW = self.H[(u_, bs_sect_n)].dot(w_UE2)
            HW_dl = self.H[(u_, bs_sect_n)].dot(w_UE2_dl)
            snr_ratio = self.SNR_u[(u_, bs_sect_n)] #/self.SNR_u[(u_, bs_sect_n)]
            sum_cov += snr_ratio * (np.outer(HW, np.matrix.conjugate(HW)))
            sum_cov_dl += snr_ratio * ( np.outer(HW_dl,np.matrix.conjugate(HW_dl)))

        for i, ue in enumerate(connected_UE_list):
            u = ue.get_id()
            bs_sect_n = ue.get_bs_sect_n()
            A_downlik = sum_cov_dl + N_a*U*(1 + snr_sum_rest_UEs[bs_sect_n]) * np.eye(len(sum_cov_dl))
            A = sum_cov + (1 + snr_sum_rest_UEs[bs_sect_n]) * np.eye(len(sum_cov))

            w_UE = ue.get_bf()
            w_UE_dl = ue.get_bf_dl()
            w_BS = np.linalg.inv(A).dot(self.H[(u, bs_sect_n)]).dot(w_UE)
            w_BS_downlink = np.linalg.inv(A_downlik).dot(self.H[(u, bs_sect_n)]).dot(w_UE_dl)
            if np.linalg.norm (w_BS) == 0:
                w_BS = w_BS / (np.linalg.norm(w_BS)+1e-20)
                w_BS_downlink = w_BS_downlink / (np.linalg.norm(w_BS_downlink) + 1e-20)
            else:
                w_BS = w_BS / np.linalg.norm(w_BS)  # norm is computed even if values are complex numbers
                w_BS_downlink  = w_BS_downlink/np.linalg.norm(w_BS_downlink)
                w_mmse_list[u] = w_BS
                w_mmse_list_downlink[u] = w_BS_downlink #np.matrix.conj(w_BS_downlink)
        return w_mmse_list, w_mmse_list_downlink

    def get_interference(self,total_UE_list:list =None, total_connected_ue_id_list:list = None, codebook_bf = False):
        """
        Compute interference MU-MIMO uplink case by employing MMSE receiver at BS side
        1) decide the MMSE beamforming or codebook-based beamforming vector for the given connections
        2) compute the beamforming gains and pathloss gains
        3) calculate inter- and intra- cell interference

        Parameters
        ----------
        total_UE_list: list
        list of UEs and UAVs dropped in the network area

        total_connected_ue_id_list: list
        list of UEs and UAVs only associated with BSs

        Returns
        ----------
        data: dictionary variable
        this dictionary includes all the information we want
        such as SNR, SINR, inter-cell and intral cell interferences
        """
        # decide connected UAVs and UEs in this BS
        connected_id_list = self.connected_entities
        connected_id_list = np.array(connected_id_list, dtype = int)
        # decide the UAVs and UEs in the other cells
        rest_id_list = list(set(total_connected_ue_id_list) - set(connected_id_list))
        rest_id_list = np.array(rest_id_list, dtype=int)
        # get objects of UE and UAVs correspondingly
        rest_UE_list = total_UE_list[rest_id_list]
        connected_UE_list = total_UE_list[connected_id_list]


        snr_sum_rest_UEs = np.zeros(self.n_sect)
        for bs_sect_n in range(self.n_sect):
            for r_ue in rest_UE_list:
                r_ue_id = r_ue.get_id()
                snr_sum_rest_UEs[bs_sect_n] += self.SNR_u_from_itf[(r_ue_id, bs_sect_n)]

        if len(connected_id_list) !=0:
            # let's create a data structure
            data =[{'SINR':None,'SNR':None, 'intra_itf':1e-20, 'inter_itf':1e-20,'bs_id':self.bs_id,'bs_type':None,
                    'ue_id':None,'ue_type':None, 'tx_power':None, 'l_f':None
                    , 'itf_gUE':1e-20, 'itf_UAV':1e-20, 'link_state':None,'n_los':0, 'n_nlos':0,
                    'los_itf':1e-20, 'nlos_itf':1e-20 , 'itf_no_ptx_list':[], 'itf_id_list':[],
                    'bf_gain':None, 'n_t':0}
                   for i in range (len(connected_id_list))]

            if codebook_bf is False:
                # get mmse list for uplink and mmse precoder list for downlink
                w_bs_bf_list,self.w_mmse_precoder_list = self.get_MMSE_beamform_vectors(connected_UE_list, snr_sum_rest_UEs)
            else:
                w_bs_bf_list = self.get_beamforming_vectors_from_codebook(connected_UE_list)

            # the list of large-scale fading including pathloss
            l_f_db_list = np.zeros(len(connected_UE_list))
            for i, ue in enumerate(connected_UE_list):
                u = ue.get_id()
                bs_sect_ind = ue.get_bs_sect_n()
                w_UE = ue.get_bf()
                w_BS =  w_bs_bf_list[u]

                #pl_lin = 10**(-0.1*self.pl[u])
                #rx_elem_gain_lin_u = 10 ** (0.1 * self.elem_gain_BS[(u, bs_sect_ind)])
                #tx_elem_gain_lin_u = 10 ** (0.1 * self.elem_gain_UE[(u, bs_sect_ind)])

                if self.frequency == 2e9: # and ue.ground_UE is True:
                    g = np.conj(w_BS).T.dot(self.H[(u, bs_sect_ind)])
                    l_f = abs(g*np.conj(g))[0]
                    #l_f =  tx_rx_bf*(pl_lin* rx_elem_gain_lin_u * tx_elem_gain_lin_u).sum()
                else:
                    g = np.conj(w_BS).T.dot(self.H[(u, bs_sect_ind)]).dot(w_UE)
                    l_f = abs(g * np.conj(g))
                    #l_f = tx_rx_bf * (pl_lin * rx_elem_gain_lin_u * tx_elem_gain_lin_u).sum()


                l_f_db = 10*np.log10(l_f+ 1e-20)
                l_f_db_list[i] = l_f_db

                data[i]['ue_id'] = int(u)
                data[i]['link_state'] = self.link_state[u]
                data[i]['l_f'] = l_f_db

                # compute beamforming gain
                N_r, N_t = self.H[(u, bs_sect_ind)].shape
                H_fro_norm = np.linalg.norm(self.H[(u, bs_sect_ind)], 'fro')
                if H_fro_norm !=0:
                    K = np.sqrt(N_t * N_r) / H_fro_norm
                else:
                    K = 0
                bf_gain = l_f*(K**2)
                data[i]['bf_gain'] = 10*np.log10(bf_gain)

                #print (data[i]['bf_gain'])
                data[i]['n_t'] = self.total_ue_n
                serv_ue_loc = np.squeeze(ue.get_current_location())
                #data[i]['ue_x'] = serv_ue_loc[0]
                #data[i]['ue_y'] = serv_ue_loc[1]
                #data[i]['ue_z'] = serv_ue_loc[2]
                #data[i]['bs_x'] = self.locations[0][0]
                #data[i]['bs_y'] = self.locations[0][1]
                #data[i]['bs_z'] = self.locations[0][2]
                data[i]['tx_power'] = ue.get_Tx_power()
               # data[i]['ue_elem'] = 10*np.log10(np.max(tx_elem_gain_lin_u)+ 1e-20)
               # data[i]['bs_elem'] = 10*np.log10(np.max(rx_elem_gain_lin_u) + 1e-20)
                if serv_ue_loc[2] > 20:
                    data[i]['ue_type'] = 'uav'
                else:
                    data[i]['ue_type'] = 'g_ue'

                if self.bs_type ==1:
                    data[i]['bs_type'] = 'bs_s'
                else:
                    data[i]['bs_type'] = 'bs_d'
            # compute intra-cell interference
            for i, ue in enumerate(connected_UE_list):

                connected_UE_list2 = np.delete(connected_UE_list, i)

                u = ue.get_id()
                w_BS = w_bs_bf_list[u]
                bs_sect_ind = ue.get_bs_sect_n()
                #resource_id_connected = ue.get_resource_index()

                itf =0
                itf_gUE = 0
                itf_UAV = 0
                n_los, n_nlos  =0, 0
                los_itf, nlos_itf = 0, 0

                for ue2 in connected_UE_list2:
                    u2 = ue2.get_id()
                    #resource_id_itf = ue2.get_resource_index()
                    if self.link_state[u2] != 0.0 and u!=u2: #and resource_id_connected == resource_id_itf:
                        w_UE = ue2.get_bf()
                        #pl_lin = 10 ** (-0.1 * self.pl[u2])
                        #rx_elem_gain_lin_u = 10 ** (0.1 * self.elem_gain_BS[(u2, bs_sect_ind)])
                        #tx_elem_gain_lin_u = 10 ** (0.1 * self.elem_gain_UE[(u2, bs_sect_ind)])
                        Ptx = ue2.get_Tx_power()
                        P_t_lin = 10**(0.1*Ptx)

                        if self.frequency == 2e9: #and ue2.ground_UE is True:
                            g = np.conj(w_BS).T.dot(self.H[(u2,bs_sect_ind)])
                            #itf0 = (P_t_lin * abs(g*np.conj(g))*pl_lin*rx_elem_gain_lin_u * tx_elem_gain_lin_u ).sum() #*rx_elem_gain_lin_u * tx_elem_gain_lin_u
                            itf0 = P_t_lin*abs(g)**2
                            itf += itf0
                        else:
                            g = np.conj(w_BS).T.dot(self.H[(u2, bs_sect_ind)]).dot(w_UE)
                            #itf0 = (P_t_lin * pl_lin * abs(g * np.conj(g)) * rx_elem_gain_lin_u * tx_elem_gain_lin_u).sum()
                            itf0 = P_t_lin*abs(g)**2
                            itf += itf0

                        if ue2.get_ue_type() == 'g_ue':
                            itf_gUE += itf0
                        else:
                            itf_UAV += itf0
                        data[i]['itf_no_ptx_list'].append(10*np.log10(itf0/P_t_lin + 1e-20))
                        data[i]['itf_id_list'].append(u2)
                        if ue2.get_ue_type() == 'uav':
                            if self.link_state[u2] == 2.0:
                                n_nlos +=1
                                nlos_itf +=itf0
                            elif self.link_state[u2] == 1.0:
                                n_los  +=1
                                los_itf += itf0
                    

                data[i]['itf_gUE'] = itf_gUE
                data[i]['itf_UAV'] = itf_UAV
                data[i]['intra_itf'] = np.asscalar(10*np.log10(itf + 1e-25))
                data[i]['n_los'] = n_los
                data[i]['n_nlos'] = n_nlos
                data[i]['los_itf'] = los_itf
                data[i]['nlos_itf'] = nlos_itf

            if data[i]['intra_itf'] == 1e-20: # in cases that there is no intra-interference
                data[i]['intra_itf'] = -200
            # compute inter-cell interference
            for i, ue in enumerate(connected_UE_list):
                u = ue.get_id()
                w_BS = w_bs_bf_list[u]
                bs_sect_ind = ue.get_bs_sect_n()

                itf, itf_UAV, itf_gUE =0, 0, 0
                n_los, n_nlos = 0, 0
                los_itf, nlos_itf = 0, 0

                #resource_id_connected = ue.get_resource_index()

                for r_ue in rest_UE_list:
                    u2 = r_ue.get_id()
                    #resource_id_itf = r_ue.get_resource_index()
                    if self.link_state[u2] != 0.0: #and resource_id_connected == resource_id_itf:
                        Ptx = r_ue.get_Tx_power()
                        P_t_lin = 10**(0.1*Ptx)

                        w_UE  = r_ue.get_bf()

                        pl = self.pl[u2]
                        #pl_lin = 10 ** (-0.1 * pl)

                        #rx_elem_gain_lin_u = 10 ** (0.1 * self.elem_gain_BS[(u2,bs_sect_ind)])
                        #tx_elem_gain_lin_u = 10 ** (0.1 * self.elem_gain_UE[(u2, bs_sect_ind)])

                        if self.frequency == 2e9: # and r_ue.ground_UE is True:
                            g = np.conj(w_BS).T.dot(self.H[(u2, bs_sect_ind)])#.dot(w_UE)
                            #itf0 = (P_t_lin *abs(g*np.conj(g)) *pl_lin *rx_elem_gain_lin_u * tx_elem_gain_lin_u).sum() #rx_elem_gain_lin_u * *rx_elem_gain_lin_u * tx_elem_gain_lin_u
                            itf0 = P_t_lin*abs(g)**2
                            itf += itf0
                        else:
                            g = np.conj(w_BS).T.dot(self.H[(u2, bs_sect_n)]).dot(w_UE)
                            #itf0 = (P_t_lin * pl_lin * abs(g * np.conj(g)) * rx_elem_gain_lin_u * tx_elem_gain_lin_u).sum()  # rx_elem_gain_lin_u *
                            itf0 = P_t_lin*abs(g)**2
                            itf += itf0

                        data[i]['itf_no_ptx_list'].append(10*np.log10(itf0/P_t_lin + 1e-20))
                        data[i]['itf_id_list'].append(u2)
                        if r_ue.get_ue_type() == 'g_ue':
                            itf_gUE += itf0
                        else:
                            itf_UAV += itf0
                        if r_ue.get_ue_type() =='uav':
                            if self.link_state[u2] ==2.0:
                                n_nlos +=1
                                nlos_itf += itf0
                            elif self.link_state[u2] == 1.0:
                                n_los +=1
                                los_itf += itf0


                data[i]['itf_gUE'] = 10*np.log10(itf_gUE + data[i]['itf_gUE'] + 1e-20)
                data[i]['itf_UAV'] = 10*np.log10(itf_UAV + data[i]['itf_UAV'] + 1e-20)
                data[i]['n_los'] += n_los
                data[i]['n_nlos'] += n_nlos
                data[i]['los_itf'] = 10*np.log10(los_itf + data[i]['los_itf']+1e-20)
                data[i]['nlos_itf'] = 10 * np.log10(nlos_itf + data[i]['nlos_itf']+1e-20)


                KT_lin, NF_lin = 10 ** (0.1 * self.channel_info.KT), 10 ** (0.1 * self.channel_info.NF)
                interference = 10**(0.1*data[i]['intra_itf']) + itf
                Noise_Interference = KT_lin * NF_lin * self.channel_info.BW + interference

                # obtain SINR in dB-scale
                ue_tx_power = ue.get_Tx_power()
                SINR = ue_tx_power + l_f_db_list[i] - 10 * np.log10(Noise_Interference)

                SNR = ue_tx_power + l_f_db_list[i] - self.channel_info.KT - \
                      self.channel_info.NF - 10*np.log10(self.channel_info.BW)
                data[i]['SNR'] = SNR
                data[i]['SINR'] = SINR
                data[i]['inter_itf'] = 10*np.log10(itf+1e-20)
            return data

    # get-set functions
    def get_mmse_precoders(self, ue_id):
        return self.w_mmse_precoder_list[ue_id]
    def get_connected_ue_list(self):
        return np.array(self.connected_entities, dtype = int)
    def set_serv_bs_sect(self, ue_id, n=0):
        self.serving_sect_ind[ue_id] = n
    def get_serv_bs_sect(self, ue_id):
        return self.serving_sect_ind[ue_id]
    def get_number_of_sectors(self):
        return self.n_sect

