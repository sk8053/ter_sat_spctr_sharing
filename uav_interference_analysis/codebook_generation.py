import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append("C://Users/gangs/Downloads/uav_interference_analysis/src")
from src.mmwchanmod.sim.array import URA
from src.mmwchanmod.sim.antenna import Elem3GPP
from src.mmwchanmod.common.constants import PhyConst
from src.mmwchanmod.common.spherical import cart_to_sph
from src.mmwchanmod.sim.drone_antenna_field import drone_antenna_gain
from tqdm import tqdm
from scipy import linalg
#import matplotlib.path as mplPath


def get_uniform_on_sphere(size:int =64, low_phi = -180, high_phi = 180):
    xyz = np.random.normal(0,1,size=(size*5,3))
    xyz = xyz / np.linalg.norm(xyz,axis=1)[:,None]
    _, phi, theta = cart_to_sph(xyz)

    xyz= xyz[(phi<=high_phi) & (phi>=low_phi)]
    return xyz[:size]

frequency = 28e9
thetabw_, phibw_ = 65, 65
lam = PhyConst.light_speed / frequency
element_bs = Elem3GPP(thetabw=thetabw_, phibw=phibw_)

#drone_antenna_gain = drone_antenna_gain()
#element_ue = Elem3GPP(thetabw=thetabw_, phibw=phibw_)
#arr_ue = URA(elem=element_ue, nant=np.array([4, 4]), fc=frequency)
#arr_uav = URA(elem=element_ue, nant=np.array([4, 4]), fc=frequency, drone_antenna_gain=drone_antenna_gain)
codebook_size = 16
n_ant = codebook_size

arr_bs = URA(elem=element_bs, nant=np.array([int(np.sqrt(n_ant)), int(np.sqrt(n_ant))]), fc=frequency)

test_size = 20000
xyz_bs = get_uniform_on_sphere(size = test_size, low_phi=-120, high_phi=120)
_, phi_bs, theta_bs = cart_to_sph(xyz_bs)
elem_gain_bs = arr_bs.sv(phi_bs, 90-theta_bs, return_elem_gain=True)
sv_bs_test = arr_bs.sv(phi_bs, 90-theta_bs, return_elem_gain=False)
sv_bs_test = sv_bs_test #*10**(0.05*elem_gain_bs[:,None])


#initiallize the codebook of BS
init_xyz= get_uniform_on_sphere(size = codebook_size, low_phi=-120, high_phi= 120)
_, phi_init, theta_init = cart_to_sph(init_xyz)
bs_codebook = arr_bs.sv(phi_init, 90-theta_init, return_elem_gain=False)
#np.random.normal(loc = 0, scale=1.0, size = (codebook_size,64)) \
              #+ 1j *np.random.normal(loc = 0, scale=1.0, size = (codebook_size,64))
bs_codebook = bs_codebook/np.linalg.norm(bs_codebook,axis=1)[:,None]


conv_delta = 100
# run K-means clustering algorithm

while conv_delta !=0:
    classified_sv = {i: [] for i in range(codebook_size)}
    # firstly classify spatial signature vectors as clusters
    for j in range(test_size):
        sv_bs_i = sv_bs_test[j]
        # for each center point, or each direction of spatial signature
        ro = bs_codebook.dot(np.conj(sv_bs_i))
        ro  = np.abs(ro)**2

        # classify the spatial signature, sv_bs_i
        code_index = np.argmax(ro)
        classified_sv[code_index].append(sv_bs_i)

    # update the codebook based on the classified spatial signatures
    new_codebook = np.zeros_like(bs_codebook)
    sum_cluster_size = []
    for k in range (codebook_size):
        #take codewords in cluster k
        codewords_k = np.array(classified_sv[k])

        #if len(codewords_k) ==0:
            #print('empty cluster')
         #   codewords_k = np.random.normal(0,1,size= (1,64)) + 1j*np.random.normal(0,1,size= (1,64))
         #   codewords_k = codewords_k/np.linalg.norm(codewords_k)

        codewords_k = codewords_k.reshape(-1, n_ant)
        # then sum over all outer products of them
        sum = np.zeros((n_ant,n_ant), dtype= complex)
        sum_cluster_size.append(codewords_k.shape[0])
        for m in range (codewords_k.shape[0]):
            sum += np.outer(codewords_k[m], np.matrix.conjugate(codewords_k[m]))

        sum_eig_val, sum_eig_vec = np.linalg.eig(sum)

        # update codebook vectors as maximum eigenvectors
        new_codebook[k] = sum_eig_vec[:,np.argmax(np.real(sum_eig_val))]


    delta = new_codebook - bs_codebook
    bs_codebook = new_codebook

    conv_delta = np.linalg.norm(delta,'fro')
    print (conv_delta, np.std(sum_cluster_size))


print (bs_codebook.shape)

np.savetxt( 'ue_codebook.txt', bs_codebook,fmt='%.8e')
