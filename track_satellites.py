#import datetime
import numpy as np
from astropy.coordinates import EarthLocation
from astropy import time
from pycraf import satellite
import datetime
from datetime import timezone, timedelta
from sgp4.api import jday
#import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from sgp4.api import SGP4_ERRORS
import csv
from skyfield.api import load # wgs84,
#from skyfield.api import EarthSatellite
ts = load.timescale()
from util import get_speed_of_satellite, eci2lla
import astropy.constants as constants
import pymap3d as pm

R_E = constants.R_earth.value
f_c = 12e9
min_elev = 25
file_name = 'satTrack_at_Colorado_time_60m_min_elev'
cur_time = datetime.datetime(2023, 9,3, 8, 10, 20) # 09/03, 2023, 08:10:20
utc_time  = cur_time.astimezone(timezone.utc) # convert current time to utc time
print(f'current time: {cur_time} and utc time: {utc_time}')

# satellite heights are set up in starlink tle file
# read satellite constellation TLE file and save it as a dictionary variable
with open('starlink.tle') as f:
    tles = f.readlines()
    tle_dict = dict()
    for j in range(int(len(tles)/3)):
        key = tles[j*3].replace(" ", "")
        tle_dict[key[:-1]] = tles[j*3]+tles[j*3+1]+ tles[j*3+2]

# the geodetic location of an observer
lon, lat = -105.018174370992298, 40.139045580580948
alt = 1.6

location = EarthLocation(lat = lat, lon = lon, height = alt) # longitude, latitude, height
# obtain the location of an observer in ECEF frame from the geodetic location, (latitude, longitude)
ob_x_ecef, ob_y_ecef, ob_z_ecef = pm.geodetic2ecef(lat, lon, alt)
print(f'location of an observer in ECEF frame:{ob_x_ecef, ob_y_ecef, ob_z_ecef}')

# create an empty dictionary data for saving
data_to_save = {j:dict() for j in range(60)}
for j in range(60):
    for feature_key in ['utc_time', 'sat_ind', 'elev_ang', 'azm_ang', 'dist','pos','pos_geo', 'vel', 'doppler_shift']:
        data_to_save[j][feature_key] =[]

ff = open(f'data/{file_name}_{min_elev}.csv','wt',encoding='utf-8', newline="")
file_writer = csv.writer(ff)
file_writer.writerow(['utc_time', 'n_sat', 'sat_ind', 'elev_ang', 'azm_ang', 'dist','pos','pos_geo', 'vel', 'doppler_shift'])

for j in range(60): # from 0 minutes to 60 minutes, one hour
    #dt = datetime.datetime(utc_time.year, utc_time.month, utc_time.day, utc_time.hour, utc_time.minute, utc_time.second)  #
    utc_time = utc_time + timedelta(seconds=60)
    #obstime = time.Time(dt)
    print(f'time index:{j}, observed utc time {time.Time(utc_time)}')
    for key in tqdm(tle_dict.keys()):
        tle_string = tle_dict[key]
        satname, sat = satellite.get_sat(tle_string)
        # create a SatelliteObserver instance
        sat_obs = satellite.SatelliteObserver(location)
        az, el, dist = sat_obs.azel_from_sat(tle_string, time.Time(utc_time)) # [deg], [deg], [km]

        if el.value >=min_elev:
            data_to_save[j]['utc_time'].append(time.Time(utc_time))
            data_to_save[j]['sat_ind'].append(key)
            data_to_save[j]['elev_ang'].append(el.value) # [deg]
            data_to_save[j]['azm_ang'].append(az.value) #  [deg]
            data_to_save[j]['dist'].append(dist.value*1e3) #[m]
            jd, fr = jday(utc_time.year, utc_time.month, utc_time.day, utc_time.hour, utc_time.minute, utc_time.second)  # utc_time.minute
            err_code, position, velocity = sat.sgp4(jd, fr) # position and velocity are in the frame of ECI
            # position: True Equator Mean Equinox position (km), in other words, ECI frame
            # velocity: True Equator Mean Equinox velocity (km/s), in other words, ECI frame
            if err_code >0:
                ValueError(f'time:{j}, sat:{key},{SGP4_ERRORS[err_code]}')

            position_m = [x*1e3 for x in list(position)]
            data_to_save[j]['pos'].append(position_m) #[m]
            dist_to_E = np.linalg.norm(position_m)
            #scale_k = R_E/dist_to_E
            #lon, lat, alt = pm.eci2geodetic(x=position_m[0]*scale_k,y = position_m[1]*scale_k, z =position_m[2]*scale_k, t = utc_time)
            lon, lat, alt = pm.eci2geodetic(x=position_m[0] , y=position_m[1], z=position_m[2] , t=utc_time)
            #lon, lat, alt = eci2lla(x=position_m[0] , y=position_m[1], z=position_m[2] , dt=utc_time)

            data_to_save[j]['pos_geo'].append([lon, lat, alt])

            velocity_m = [x*1e3 for x in list(velocity)]
            data_to_save[j]['vel'].append(list(velocity_m)) #[m]

            sat_xyz = np.array(list(position))*1e3 # km -> m
            v = np.array(list(velocity))*1e3 # km -> m

            R_h = np.linalg.norm(sat_xyz)
            sat_h = R_h - R_E # actual height of satellite
            sat_velocity = get_speed_of_satellite(sat_h)

            ob_x, ob_y, ob_z = pm.ecef2eci(ob_x_ecef, ob_y_ecef, ob_z_ecef, time.Time(utc_time)) # the location of an observer in ECI frame
            ob_xyz = np.array([ob_x, ob_y, ob_z])

            sat2obs = ob_xyz - sat_xyz
            # angle between sat-to-observer vector and velocity vector of satellite
            theta = np.arccos(sat2obs.dot(v)/(np.linalg.norm(sat2obs)*np.linalg.norm(v)))
            # doppler shift from 3GPP 38.811 5.3.4.3
            doppler_shift = (f_c/constants.c.value)*sat_velocity*np.cos(theta)
            data_to_save[j]['doppler_shift'].append(doppler_shift) # [Hz]

    # time, number of satellite, satellite indices, elevation angle, azimuth angle, distance, position, velocity
    file_writer.writerow([time.Time(utc_time), len(data_to_save[j]['sat_ind']),data_to_save[j]['sat_ind'], data_to_save[j]['elev_ang'], data_to_save[j]['azm_ang'],
                          data_to_save[j]['dist'], data_to_save[j]['pos'],data_to_save[j]['pos_geo'], data_to_save[j]['vel'], data_to_save[j]['doppler_shift']])


ff.close()

with open(f'data/{file_name}_{min_elev}.pickle', 'wb') as handle:
    pickle.dump(data_to_save, handle)

