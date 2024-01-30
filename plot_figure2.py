
import sys
import os
sys.path.append(os.getcwd()+"/uav_interference_analysis")
from src.mmwchanmod.sim.antenna import Elem3GPP
from src.mmwchanmod.sim.array import URA
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
import numpy as np

def response(phi, theta):
    phibw, thetabw = 65, 65
    Am = 30
    gain_max = 8
    slav = 30

    # Rotate the angles relative to element boresight.
    # Note the conversion from inclination to elevation angles

    phi1 = phi
    theta1 = theta

    # Put the

    # Put the phi from -180 to 180
    phi1 = phi1 % 360
    phi1 = phi1 - 360 * (phi1 > 180)

    # print (min(phi), max(phi))
    # plot_pattern(self.response,100,0,0,100,'rect_phi')

    if thetabw > 0:
        Av = -np.minimum(12 * ((theta1) / thetabw) ** 2, slav)
    else:
        Av = 0
    if phibw > 0:
        Ah = -np.minimum(12 * (phi1 / phibw) ** 2, Am)
    else:
        Ah = 0
    gain = gain_max - np.minimum(-Av - Ah, Am)

    return gain

R_E = 6371e3
h = 600e3
theta = np.linspace(0,np.pi/2, 100)
theta_deg = np.rad2deg(theta)
f = 12e9
thetabw, phibw = 65, 65
elem_ = Elem3GPP(thetabw=thetabw, phibw=phibw)
ant_array = np.array([8,8])
_arrays = URA(elem=elem_, nant=ant_array, fc=f)
sv = _arrays.sv(np.array([0]), np.array([-10]), return_elem_gain=False)
w = sv.conj().T
w = w/np.linalg.norm(w)
g_list =[]
theta_range = np.rad2deg(theta)
for t in theta_range:
    g = elem_.response(phi =0,theta = t)
    g_list.append(g)

d_theta = np.sqrt(R_E**2*np.sin(theta)**2 + h**2 + 2*h*R_E) - R_E*np.sin(theta)
fspl = 20*np.log10(d_theta) + 20*np.log10(f) - 147.55

phi = np.repeat([0], len(theta_deg))
#plt.plot(theta_range, g_list)
#plt.plot(theta_deg, fspl-response(phi, theta_deg+12))
#plt.plot(theta_deg, fspl, 'r')
#theta_deg = np.arange(-90,90, 1)
#sv_list = _arrays.sv(np.repeat([0],len(theta_deg)), theta_deg, return_elem_gain=False)
#bf_g = 10*np.log10(abs(sv_list.dot(w))**2)
#bf_g = np.squeeze(bf_g)
#plt.plot(theta_deg, bf_g)
#phi = np.repeat([0], len(theta_deg))
#elem_gain = response(phi, theta_deg+12)
#plt.plot(theta_deg, fspl-elem_gain-bf_g)
#plt.grid()

plt.plot(theta_deg, fspl-response(phi, theta_deg+12), 'k', label = 'FSPL + antenna gain', lw = 2.5)
plt.plot(theta_deg, fspl, 'r:', label = 'FSPL only', lw = 2.5)
plt.grid()
plt.xticks(np.arange(90+10, step =10), fontsize = 13)
plt.yticks(np.arange(165, 195, step = 5),fontsize =13)
plt.ylabel('FSPL w/ and w/o antenna gain [dB]', fontsize = 16)
plt.xlabel(r'Elevation angle [$^\circ$]', fontsize = 16)
plt.legend(fontsize = 15)

plt.subplots_adjust(top=1.0,
bottom=0.115,
left=0.105,
right=0.995,
hspace=0.2,
wspace=0.2)
plt.savefig('figures/pathloss_elem_gain.png', dpi = 500)
plt.show()




