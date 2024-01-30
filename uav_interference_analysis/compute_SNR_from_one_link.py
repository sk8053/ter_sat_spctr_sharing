# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 15:22:27 2022

@author: seongjoonkang
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append("C://Users/gangs/Downloads/uav_interference_analysis/src")
from src.mmwchanmod.sim.array import URA, RotatedArray, multi_sect_array
from src.mmwchanmod.sim.antenna import Elem3GPP
from src.mmwchanmod.common.constants import PhyConst
from src.mmwchanmod.common.spherical import cart_to_sph
from src.mmwchanmod.sim.drone_antenna_field import drone_antenna_gain
from mmwchanmod.sim.chanmod import dir_path_loss_multi_sect

frequency = 28e9
thetabw_, phibw_ = 65, 65
lam = PhyConst.light_speed / frequency

element_uav = Elem3GPP(thetabw=thetabw_, phibw=phibw_)
elem_bs = Elem3GPP(thetabw=thetabw_, phibw=phibw_)

# create the Uniform Retangular Array for BS and UAV
arr_uav = URA(elem=element_uav, nant=np.array([4, 4]), fc=frequency) # UAV has 4*4
arr_gnb = URA(elem=elem_bs, nant=np.array([8, 8]),  fc= frequency) # gNB has 8*8

rand_azm = np.random.uniform(low = -180, high = 180, size=(1,))[0]

# the array of UAV is downl-tilted by -90
arr_uav = RotatedArray(arr_uav, theta0=-90,phi0=rand_azm,  drone = True)
# configure the antenna array of BSs having 3-sectors and down-tilted by -12
arr_gnb = multi_sect_array(arr_gnb, sect_type='azimuth', nsect=3, theta0=-12)


dir_path_loss_multi_sect(arr_gnb, arr_uav, channel, isdrone= True,
                                return_elem_gain= True)