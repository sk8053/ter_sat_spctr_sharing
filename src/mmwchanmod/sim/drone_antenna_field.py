import  numpy as np
import sys
import pathlib

ab_path = pathlib.Path().absolute().__str__() + '/uav_interference_analysis/src/mmwchanmod/sim/'
sys.path.append(ab_path)
class drone_antenna_gain():
    def __init__(self):
        #p = np.loadtxt(ab_path+'azi_ele_angles.txt')
        #v = np.loadtxt(ab_path+'values.txt')
        #az, ele = p.T
        df = np.loadtxt(ab_path + 'drone_antenna_pattern_real_.csv')
        ele, az = df[:,0]-90, df[:,1]-180
        v = df[:,2]

        az, ele = np.array(az.T, dtype = int), np.array(ele.T, dtype = int)
        self.gain = dict()
        for i, (a, e) in enumerate(zip(az, ele)):
            self.gain[(str(a), str(e))] =v[i]

