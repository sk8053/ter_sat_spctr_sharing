"""
chanmod.py:  Methods for modeling multi-path channels
"""

import numpy as np
#import numexpr as ne
import copy
from src.mmwchanmod.common.constants import LinkState

class MPChan(object):
    """
    Class for storing list of rays
    """
    nangle = 4
    aoa_phi_ind = 0
    aoa_theta_ind = 1
    aod_phi_ind = 2
    aod_theta_ind = 3
    ang_name = ['AoA_Phi', 'AoA_theta', 'AoD_phi', 'AoD_theta']

    large_pl = 250

    def __init__(self):
        """
        Constructor

        Creates an empty channel
        """
        # Parameters for each ray
        self.pl  = np.zeros(0, dtype=np.float32)
        self.dly = np.zeros(0, dtype=np.float32)
        self.ang = np.zeros((0,MPChan.nangle), dtype=np.float32)
        self.link_state = LinkState.no_link

    def comp_omni_path_loss(self):
        """
        Computes the omni-directional channel gain
        Returns
        -------
        pl_omni:  float
            Omni-directional path loss
        """
        if self.link_state == LinkState.no_link:
            pl_omni = np.inf
        else:
            pl_min = np.min(self.pl)
            pl_lin = 10**(-0.1*(self.pl-pl_min))
            pl_omni = pl_min-10*np.log10(np.sum(pl_lin) )

        return pl_omni


    def rms_dly(self):
        """
        Computes the RMS delay spread
        Returns
        -------
        dly_rms:  float
            RMS delay spread (std dev weighted by paths)
        """
        if self.link_state == LinkState.no_link:
            dly_rms = 0
        else:
            # Compute weights
            pl_min = np.min(self.pl) #
            w = 10**(-0.1*(self.pl-pl_min))
            w = w / np.sum(w)

            # Compute weighted RMS
            dly_mean = w.dot(self.dly)
            dly_rms = np.sqrt( w.dot((self.dly-dly_mean)**2) )

        return dly_rms

def dir_path_loss(tx_arr, rx_arr, chan, return_elem_gain=True,\
                       return_bf_gain=True):
    """
    Computes the directional path loss between RX and TX arrays
    Parameters
    ----------
    tx_arr, rx_arr : ArrayBase object
        TX and RX arrays
    chan : MPChan object
        Multi-path channel object
    return_elem_gain : boolean, default=True
        Returns the TX and RX element gains
    return_bf_gain : boolean, default=True
        Returns the TX and RX beamforming gains
    Returns
    -------
    pl_eff:  float
        Effective path loss with BF gain
    tx_elem_gain, rx_elem_gain:  (n,) arrays
        TX and RX element gains on each path in the channel
    tx_bf_gain, rx_nf_gain:  (n,) arrays
        TX and RX BF gains on each path in the channel

    """

    if chan.link_state == LinkState.no_link:
        pl_eff = MPChan.large_pl
        tx_elem_gain = np.array(0)
        rx_elem_gain = np.array(0)
        tx_bf = np.array(0)
        rx_bf = np.array(0)

    else:

        # Get the angles of the path
        # Note that we have to convert from inclination to elevation angle
        aod_theta = 90 - chan.ang[:,MPChan.aod_theta_ind]
        aod_phi = chan.ang[:,MPChan.aod_phi_ind]
        aoa_theta = 90 - chan.ang[:,MPChan.aoa_theta_ind]
        aoa_phi = chan.ang[:,MPChan.aoa_phi_ind]

        tx_sv, tx_elem_gain = tx_arr.sv(aod_phi, aod_theta, return_elem_gain=True)
        rx_sv, rx_elem_gain = rx_arr.sv(aoa_phi, aoa_theta, return_elem_gain=True)


        # Compute path loss with element gains
        pl_elem = chan.pl - tx_elem_gain - rx_elem_gain

        # Select the path with the lowest path loss
        im = np.argmin(pl_elem)

        # Beamform in that direction
        wtx = np.conj(tx_sv[im,:])
        wtx = wtx / np.sqrt(np.sum(np.abs(wtx)**2))
        wrx = np.conj(rx_sv[im,:])
        wrx = wrx / np.sqrt(np.sum(np.abs(wrx)**2))

        # Compute the gain with both the element and BF gain
        # Note that we add the factor 10*np.log10(nanttx) to
        # account the division of power across the TX antennas
        tx_bf = 20*np.log10(np.abs(tx_sv.dot(wtx)))
        rx_bf = 20*np.log10(np.abs(rx_sv.dot(wrx)))
        pl_bf = chan.pl - tx_bf - rx_bf

        # Subtract the TX and RX element gains
        tx_bf -= tx_elem_gain
        rx_bf -= rx_elem_gain

        # Compute effective path loss
        pl_min = np.min(pl_bf)
        pl_lin = 10**(-0.1*(pl_bf-pl_min))
        pl_eff = pl_min-10*np.log10(np.sum(pl_lin) )

    # Get outputs
    if not (return_bf_gain or return_elem_gain):
        return pl_eff
    else:
        out =[pl_eff]
        if return_elem_gain:
            out.append(tx_elem_gain)
            out.append(rx_elem_gain)
        if return_bf_gain:
            out.append(tx_bf)
            out.append(rx_bf)
        return out

def dir_path_loss_multi_sect(bs_arr_list:list, ue_arr_list:list, chan:MPChan,
                             isdrone:bool = False,  return_elem_gain = True, disable_ue_elemgain = False, 
                             disable_bs_elemgain = False, invert = False):
    """
    Computes the directional path loss between list of RX and TX arrays.
    This is typically used when the TX or RX have multiple sectors
    Parameters
    ----------
    bs_arr_list, ue_arr_list : list of ArrayBase objects
        BS and UE arrays
    chan : MPChan object
        Multi-path channel object
    return_arr_ind : boolean, default=True
        Returns the index of the chosen array
    return_elem_gain : boolean, default=True
        Returns the TX and RX element gains
    return_bf_gain : boolean, default=True
        Returns the TX and RX beamforming gains
    Returns
    -------
    pl_eff:  float
        Effective path loss with BF gain
    ind_tx, ind_rx: int
        Index of the selected TX and RX arrays
    tx_elem_gain, rx_elem_gain:  (n,) arrays
        TX and RX element gains on each path in the channel
    tx_bf_gain, rx_nf_gain:  (n,) arrays
        TX and RX BF gains on each path in the channel

    """

    if chan.link_state == LinkState.no_link:
        #sect_ind_ue = 0
        sect_ind_bs = 0
        zero_array_bs = np.zeros(shape =(1, len(bs_arr_list)))
        zero_array_ue = np.zeros(shape=(1, len(ue_arr_list)))

        zero_array = np.array([0])
        #w_bs, w_ue = zero_array, zero_array
        n_sect = len(bs_arr_list)
        bs_elem_gain_dict, ue_elem_gain_dict = {i:zero_array for i in range(n_sect)}, {i:zero_array for i in range(n_sect)}
        bs_sv_dict, ue_sv_dict = {i:zero_array_bs for i in range(n_sect)}, {i:zero_array_ue for i in range(n_sect)}
    else:
        #im = 0
        #sect_ind_ue = 0
        sect_ind_bs = 0
        # Get the angles of the path
        # Note that we have to convert from inclination to elevation angle
        if invert is True:
            aod_theta = 90 - chan.ang[:, MPChan.aoa_theta_ind]
            aod_phi = chan.ang[:, MPChan.aoa_phi_ind]
            aoa_theta = 90 - chan.ang[:, MPChan.aod_theta_ind]
            aoa_phi = chan.ang[:, MPChan.aod_phi_ind]
        else:
            aod_theta = 90 - chan.ang[:,MPChan.aod_theta_ind]
            aod_phi = chan.ang[:,MPChan.aod_phi_ind]
            aoa_theta = 90 - chan.ang[:,MPChan.aoa_theta_ind]
            aoa_phi = chan.ang[:,MPChan.aoa_phi_ind]

        # Loop over the array combinations to find the best array
        # with the lowest path loss
        pl_min = MPChan.large_pl
        bs_elem_gain_dict, ue_elem_gain_dict = dict(), dict()
        bs_sv_dict, ue_sv_dict = dict(), dict()

        for i_ue, ue_arr in enumerate(ue_arr_list):
            for i_bs, bs_arr in enumerate(bs_arr_list):
                # this is up-link case: arrival angles at BSs
                #if return_elem_gain is True:
                bs_elem_gaini = bs_arr.sv(aod_phi, aod_theta,\
                                                return_elem_gain=True, drone = False)
                # departure angles at UEs
                ue_elem_gaini = ue_arr.sv(aoa_phi, aoa_theta
                                                ,return_elem_gain=True, drone = isdrone)
               
                if disable_ue_elemgain is True:
                    pl_elemi = chan.pl - bs_elem_gaini
                elif disable_bs_elemgain is True:
                    pl_elemi = chan.pl - ue_elem_gaini
                else:
                    # Compute path loss with element gains
                    pl_elemi = chan.pl - bs_elem_gaini - ue_elem_gaini

                bs_elem_gain_dict[i_bs] = bs_elem_gaini
                ue_elem_gain_dict[i_bs] = ue_elem_gaini
                # Select the path with the lowest path loss
                pl_mini = np.min(pl_elemi)
                if pl_mini < pl_min:
                    pl_min = pl_mini
                    im = np.argmin(pl_elemi)
                    sect_ind_ue = i_ue
                    sect_ind_bs = i_bs

                #else:
                bs_svi = bs_arr.sv(aod_phi, aod_theta, \
                                          return_elem_gain=False, drone=False)
                
                # departure angles at UEs
                ue_svi = ue_arr.sv(aoa_phi, aoa_theta
                                          , return_elem_gain=False, drone=isdrone)  #
                #ue_svi = ue_svi[:2,:]
                #bs_svi = bs_svi[:2,:]
                bs_sv_dict[i_bs] = bs_svi#/ bs_elem_gain_lin_i[:, None]
                ue_sv_dict[i_bs]  = ue_svi#/ ue_elem_gain_lin_i[:, None]


   # if return_elem_gain is True: # if it returns element gains only
   #     out = {'bs_elem_gain_dict':bs_elem_gain_dict,
   #            'ue_elem_gain_dict':ue_elem_gain_dict,
   #            'sect_ind': sect_ind_bs
   #           }
   # else: # if it returns spatial channels only
   #   out ={
   #       'bs_sv_dict':bs_sv_dict,
   #       'ue_sv_dict':ue_sv_dict,
   #       }
      
    out = {'bs_elem_gain_dict':bs_elem_gain_dict,
           'ue_elem_gain_dict':ue_elem_gain_dict,
           'sect_ind': sect_ind_bs,
           'bs_sv_dict':bs_sv_dict,
           'ue_sv_dict':ue_sv_dict,
          }
    return out





''' 
       
        bs_sv = bs_sv_dict[sect_ind_bs]
        ue_sv = ue_sv_dict[sect_ind_bs]
        
        n_rand = 10
        if long_term_bf is True:
            bs_sv2, ue_sv2 = copy.deepcopy(bs_sv), copy.deepcopy(ue_sv)
            #Cov_H_tx, Cov_H_rx, H_list = [],[],[]
            n_r, n_t, n_path = ue_sv.shape[1], bs_sv.shape[1], ue_sv.shape[0]

            Cov_H_bs, Cov_H_ue = np.zeros((n_rand, n_t, n_t), dtype = complex), np.zeros((n_rand, n_r, n_r), dtype = complex)
            H_ue_list, H_bs_list = [], []
            H_list = np.zeros((n_rand, n_r, n_t), dtype = complex)

            H_frob = np.zeros((n_rand,))
            #n_bs, n_ue = bs_sv.shape[1], ue_sv.shape[1]
            for i in range (n_rand):
                theta_random = np.random.uniform(low=-np.pi, high=np.pi, size=(ue_sv.shape[0],))
                ue_sv2 = ue_sv*np.exp(-1j*theta_random[:,None])
                if uplink is True:
                    H = ue_sv2.T.dot(np.conj(bs_sv2))# H(f)
                else:
                    H = np.conj(ue_sv2).T.dot(bs_sv2)  # H(f)

                H_rx = ue_sv2
                H_tx = bs_sv2

                H_ue_list.append(H_rx) # channel matrix at RX side
                H_bs_list.append(H_tx) # channel matrix at TX side

                #H *=np.sqrt(n_tx*n_rx)/np.linalg.norm(H,'fro') # normalize H

                H_frob[i] = np.linalg.norm(H,ord= 'fro')**2
                H_list[i] = H
                Cov_H_bs[i] = np.matrix.conj(H).T.dot(H)
                Cov_H_ue[i] = H.dot(np.matrix.conj(H).T)


            #H_norm_factor = np.mean(H_frob)/(n_tx*n_rx)
            #G_omni = np.mean(H_frob) / (n_bs * n_ue)

            Cov_H_ue = np.mean(Cov_H_ue, axis=0)  # 16*16 Qrx
            Cov_H_bs = np.mean(Cov_H_bs, axis=0)  # 64*64 Qtx

            eig_value_ue,eig_vector_ue = np.linalg.eig(Cov_H_ue)
            w_ue= eig_vector_ue[:,np.argmax(eig_value_ue)] # 16*1
            w_ue = w_ue / np.linalg.norm(w_ue)

            eig_value_bs, eig_vector_bs = np.linalg.eig(Cov_H_bs)
            w_bs = eig_vector_bs[:,np.argmax(eig_value_bs)]  # 64*1
            w_bs = w_bs / np.linalg.norm(w_bs)

        else:
            # Beamform in the dominant direction
            w_bs = np.conj(bs_sv[im,:])
            w_bs = w_bs / np.sqrt(np.sum(np.abs(w_bs)**2))
            w_ue = np.conj(ue_sv[im,:])
            w_ue = w_ue / np.sqrt(np.sum(np.abs(w_ue)**2))

             
   # compute long-term beamforming gain over small scale fading         
            bf_gain_list = np.zeros((n_rand,), dtype= complex)
            ue_bf = np.zeros((n_rand,n_path))
            bs_bf = np.zeros((n_rand,n_path))
            #s_list= []
            for H, H_ue, H_bs in zip(H_list, H_ue_list, H_bs_list):
                g = np.matrix.conj(w_ue).dot(H).dot(w_bs)
                bf_gain_list[i] = g*np.conj(g)

                #H /= np.sqrt(H_norm_factor)
                #bf gain at Tx
                g_bs = H_bs.dot(np.conj(w_bs))
                bs_bf_ = g_bs*np.conj(g_bs)
                #bf gain at Rx
                g_ue = H_ue.dot(np.conj(w_ue))
                ue_bf_ = g_ue*np.conj(g_ue)

                ue_bf[i]= abs(ue_bf_) # abs removes zero imaginary part, 0j
                bs_bf[i] = abs(bs_bf_)
            
            beamforming gains at Tx and Rx
            bs_bf = np.mean(bs_bf, axis = 0)
            ue_bf = np.mean(ue_bf, axis =0)

            bs_bf = 10*np.log10(bs_bf)
            ue_bf = 10*np.log10(ue_bf)
            bf_gain = np.mean(bf_gain_list)
            bs_ue_bf = 10*np.log10(abs(bf_gain)) - 10*np.log10(G_omni)
    
            if frequency == 28e9 or frequency == 140e9:
                pl_bf = chan.pl - bs_ue_bf - bs_elem_gain - ue_elem_gain

            else:
                pl_bf = chan.pl - bs_bf - bs_elem_gain - ue_elem_gain
            
'''
