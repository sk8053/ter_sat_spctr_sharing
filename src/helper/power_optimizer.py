from pyswarm import pso
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

class power_optimization():
    '''
    class to optimize uplink transmit power
    '''

    def __init__(self, l_f=None, ue_ids=None, itf_no_ptx_list=None, itf_id_list=None, n_s = None):
        KT = -174
        NF = 6
        BW = 400e6
        # compute noise power in linear scale
        self.noise_power = 10 ** (0.1 * KT) * 10 ** (0.1 * NF) * BW
        # large scale gain
        self.l_f = 10 ** (0.1 * l_f)  # from linear scale to dB scale
        # ue id list
        self.ue_ids = ue_ids
        # all the interference of each UE from other UEs without transmit power
        self.itf_no_tx_dict = dict()
        self.n_s = n_s
        L = len(l_f)
        # read the interference without transmit power
        for j in range(L):
            itf_no_tx_j = np.array(itf_no_ptx_list[j], dtype=float)
            ue_id = self.ue_ids[j]
            self.itf_no_tx_dict[ue_id] = 10 ** (0.1 * itf_no_tx_j)
        # interference ue ids for each ue
        self.itf_id_dict = dict()
        # map from itf_id -> index of power
        self.tx_ind_dict = dict()
        for j in range(L):
            itf_ids = np.array(itf_id_list[j], dtype=int)
            ue_id = self.ue_ids[j]
            self.tx_ind_dict[ue_id] = j
            self.itf_id_dict[ue_id] = itf_ids


    def target_function(self, P_tx):
        # target function to minimze
        SINR_sum = 0
        for j, ue_id in enumerate(self.ue_ids):
            I = 0.0
            for k, itf_id in enumerate(self.itf_id_dict[ue_id]):
                I += self.itf_no_tx_dict[ue_id][k] * P_tx[self.tx_ind_dict[itf_id]]
            rate = (self.n_s/10) * np.log2(1 + self.l_f[j] * P_tx[j] / (I + self.noise_power))
            SINR_sum += np.log10(rate)

        return SINR_sum

    def con(self, P_tx):
        # constraint to optimization
        x1 = P_tx[0]
        x2 = P_tx[1]
        return

    def get_optimal_power_and_SINR(self, minstep=1e-3, debug = False):
        # minimize the target function
        # use particle-swarm optimizer

        # define upper and lower bound of power
        L = len(self.l_f)
        lb = np.ones(L) * 10 ** (0.1 * 1)
        ub = np.ones(L) * 10 ** (0.1 * 23)

        P_tx, fopt = pso(self.target_function, lb, ub, maxiter=100,
                         swarmsize=100, minstep=minstep,
                         minfunc=minstep, omega=0.5,
                         phip=0.5, phig=0.5, debug=debug)  # , f_ieqcons=con)
        SINR_list = []

        for j, ue_id in enumerate(self.ue_ids):
            I = 0.0
            for k, itf_id in enumerate(self.itf_id_dict[ue_id]):
                I += self.itf_no_tx_dict[ue_id][k] * P_tx[self.tx_ind_dict[itf_id]]
            SINR_ = 10 * np.log10(self.l_f[j] * P_tx[j] / (I + self.noise_power))
            SINR_list.append(SINR_)
        return P_tx, SINR_list