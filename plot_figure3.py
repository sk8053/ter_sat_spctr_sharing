## vector-quantization codebook ###
### generate BS or UE codebook based on K-mean clustering algorithm #####
### the reference paper is  Pengfei Xia and G. B. Giannakis, "Design and analysis of transmit-beamforming based on limited-rate feedback,"  ####
#### this file generates codebook for any antenna configurations #######
## author: seongjoon kang
## email: sk8053@nyu.edu
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.getcwd() + "/uav_interference_analysis")
from src.mmwchanmod.sim.array import URA
from src.mmwchanmod.sim.antenna import Elem3GPP
from src.mmwchanmod.common.constants import PhyConst
from src.mmwchanmod.common.spherical import cart_to_sph
from src.mmwchanmod.sim.drone_antenna_field import drone_antenna_gain
from tqdm import tqdm
from scipy import linalg
#import matplotlib.path as mplPath
import cvxpy as cp
import seaborn as sb
import pickle
plt.rcParams["font.family"] = "Times New Roman"

frequency = 12e9 # 6e9, 9e9
ue_codebook = False # False -> bs codebook
# UE antenna arrays for each carrier frequency
ue_ant_dict ={
    6e9:np.array([1,2]),
    12e9:np.array([1,2]),
    18e9:np.array([1,3]),
    24e9:np.array([1,3]),
    }
# BS antenna arrays for each carrier frequency
bs_ant_dict ={
    6e9:np.array([8,8]),
    12e9:np.array([8,8]),
    18e9:np.array([8,8]),
    24e9:np.array([8,8]),
    }

if ue_codebook is True:
    ant_array = ue_ant_dict[frequency] # get ue antenna array from the corresponding frequency
else:
    ant_array = bs_ant_dict[frequency] # get bs antenna array from the corresponding frequency
n_ants = ant_array[0]*ant_array[1] # total number antenna


thetabw_, phibw_ = 65, 65
lam = PhyConst.light_speed / frequency
element_bs = Elem3GPP(thetabw=thetabw_, phibw=phibw_)
_arrays = URA(elem=element_bs, nant=ant_array, fc=frequency)
N_t = n_ants
print(f'max gain = {10*np.log10(n_ants)}')
gain_MAX = 10*np.log10(n_ants)-3
print(gain_MAX)


theta2_map = {theta:dict() for theta in np.arange(10,80)}
phi_bins = np.arange(-50, 50, step=10)  # starting bin should not be edge angle
theta_bins = np.arange(-65, 0, step=10)  # starting bin should not be edge angle

##### the below lines are made for testing and plotting ####
def plot_beam_space_gain(w, theta_t, phi_t, ind, theta_itfs = None, phi_itfs = None):
    n_bins_phi = 120
    n_bins_theta = 180
    phi_s = np.linspace(-60, 60, n_bins_phi)
    theta_s = np.linspace(90, -90, n_bins_theta)
    g_map = np.zeros((n_bins_theta, n_bins_phi))
    g_max = -100
    for i in range(n_bins_theta):
        theta = theta_s[i]
        for j in range(n_bins_phi):
            phi = phi_s[j]
            sv_ij = _arrays.sv(phi, theta, return_elem_gain=False)
            g_map[i, j] = 10 * np.log10(abs(sv_ij.conj().dot(w)) ** 2)
            if theta>20 and g_map[i, j]>g_max:
                g_max = g_map[i, j]

    pcm = plt.imshow(g_map, cmap='jet', vmin=-50, vmax=19, aspect='auto')
    cb = plt.colorbar(pcm, pad=0.01, location='right', cmap='jet')  # pcm, ax=[ax1, ax2, ax3, ax4],
    cb.set_label(label='Beamforming gain (dB)', size=14)
    cb.ax.tick_params(labelsize=12)

    plt.scatter(60 + phi_t, 90 - theta_t, c='k', s=120)

    sv = _arrays.sv(phi_t, theta_t, return_elem_gain=False)
    g = 10 * np.log10(abs(sv.conj().dot(w)) ** 2)
    plt.text(60 + phi_t - 8, 90 - theta_t + 10, f'{np.round(g[0], 2)} dB', color='w', fontsize=13, weight='bold')

    plt.grid(linestyle=':')
    plt.xticks(np.arange(n_bins_phi + 1, step=10), np.arange(-60, 61, step=10), fontsize=13)
    plt.yticks(np.arange(n_bins_theta + 1, step=10), np.arange(90, -91, step=-10), fontsize=13)
    plt.xlabel(r'Azimuth angle ($\phi$ [$^\circ$])', fontsize=16)
    plt.ylabel(fr'Elevation angle ($\theta$ [$^\circ$])', fontsize=16)

    if len(theta_itfs) >0:
        for theta, phi in zip(theta_itfs, phi_itfs):
            plt.scatter(60 + phi, 90 - theta, c='r', s=120, marker='X')

    plt.subplots_adjust(top=0.99,bottom=0.116,left=0.1,right=1.0,hspace=0.2,wspace=0.2)
    plt.savefig(f'figures/beam_gain.png', dpi=500)
    plt.show()
    return np.round(g_max,2)

if 1:
    #theta_c = 20
    phi_t = -2
    theta_t = -10

    with open('theta2.pickle', 'rb') as f:
        theta2_map = pickle.load(f)

    #theta2_map = {70:19 , 60:18, 50:18, 40:26, 30:36, 20:47, 10:59}
    spacing = 10

    phi_ts = np.arange(-60, 60+spacing, step =spacing)
    theta_ts = np.arange(-5, -65-spacing, step =-spacing)

    theta_itfs = np.arange(20, 132, step=1.1) #12+20
    phi_itfs = np.arange(-50, 55, step=10.5)

    phi_ts = np.array([-30])
    theta_ts = np.array([-30])

    theta_itfs = np.array([20,  30, 30, 40,  50, 20, 60,40, 60, 80])  # 12+20
    phi_itfs = np.array([-20, -30, 30, 40, 50, -10, 0, 10, -50, 0])

    codebook = []
    ind = 1
    for theta_t in np.flip(theta_ts):
        for phi_t in phi_ts:

            SV = []
            for i in range(len(phi_itfs)):
                #for j in range(len(theta_itfs)):
                if phi_itfs[i] != phi_t:
                    sv_ij = _arrays.sv(phi_itfs[i], theta_itfs[i], return_elem_gain=False)
                    if len(SV)== 0:
                        SV = sv_ij
                    else:
                        SV = np.append(SV, sv_ij, axis=0)

            sv = _arrays.sv(phi_t, theta_t, return_elem_gain=False)
            print(sv.shape, SV.shape)
            lambda_ = 10
            SV2 = SV.T
            cov_itf = SV2.dot(SV2.conj().T)
            cov_ter = np.outer(sv, sv.conj())
            #cov_itf = SV2.conj().T.dot(SV2)
            Q = cov_ter - lambda_*cov_itf
            eigen_values, v_nulls = np.linalg.eig(Q)

            I = np.argsort(eigen_values)[::-1]
            v_nulls = v_nulls[:, I]
            v_null = v_nulls[:, 0]
            w_t = v_null

            w = w_t
            w = w / np.linalg.norm(w)

            g = 10 * np.log10(abs(sv.conj().dot(w)) ** 2)

            g_max = plot_beam_space_gain(w, theta_t, phi_t, ind, theta_itfs, phi_itfs)
            #print(f'phi_t:{phi_t}, theta_t:{theta_t}, and g:{g[0]}, and g_max:{g_max}')
            print(f'phi_t:{phi_t}, theta_t:{theta_t}, and g:{g[0]}')
            ind +=1
            codebook.append(w)

    codebook = np.array(codebook)

