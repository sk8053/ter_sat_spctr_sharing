import scipy.constants
import sys;
sys.path.append('..')
from mmwchanmod.datasets.download import load_model
from mmwchanmod.sim.ground_channel_generation import GroundChannel
import tensorflow.keras.backend as K
import numpy as np

class Channel_Info(object):
    def __init__(self, city:str = 'uav_boston'):
        self.frequency = 28e9
        self.lambda_ = scipy.constants.speed_of_light / self.frequency
        self.NF = 6
        self.KT = -174
        self.UE_TX_power = 23
        self.BS_TX_power = 30
        self.BW = 400e6
        self.alpha = 0.9
        self.P0 = -50
        self.N_UAV = 60
        self.long_term_bf = True
        self.ptrl = True
        K.clear_session
        self.aerial_channel = load_model(city, src='remote')
        self.aerial_channel.load_link_model()
        self.aerial_channel.load_path_model()
        self.ground_channel = GroundChannel()

class Network(object):
    def __init__(self, X_MAX:int = 1000, X_MIN:int = 0, Y_MAX:int = 1000,
                 Y_MIN:int = -100,Z_MAX:int=35, Z_MIN:int = 0):
        #fig = plt.figure()
        #ax = plt.axes(projection='3d')
        self.X_MAX, self.Y_MAX, self.Z_MAX = X_MAX, Y_MAX, Z_MAX
        self.X_MIN, self.Y_MIN, self.Z_MIN = X_MIN, Y_MIN, Z_MIN

    @staticmethod
    def _wrap_around(dist_vec, X_MAX = 1000, Y_MAX = 1000):
        dist_x, dist_y, dist_z = dist_vec.T
        ind_x = np.where(X_MAX - np.abs(dist_x) <= np.abs(dist_x))
        ind_y = np.where(Y_MAX - np.abs(dist_y) <= np.abs(dist_y))
        dist_x[ind_x] = (-1) * np.sign(dist_x[ind_x]) * (X_MAX - np.abs(dist_x[ind_x]))
        dist_y[ind_y] = (-1) * np.sign(dist_y[ind_y]) * (Y_MAX - np.abs(dist_y[ind_y]))
        return np.column_stack((dist_x, dist_y, dist_z))

