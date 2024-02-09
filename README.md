# Terrestrial-Satellite Spectrum Sharing in the Upper Mid-Band with Interference Nulling
The growing demand for broader bandwidth in cellular networks has turned the upper mid-band (7-24 GHz) into a focal point for expansion. 
However, the integration of terrestrial cellular and incumbent satellite services, particularly in the 12 GHz band, poses significant interference challenges. 
This paper investigates the interference dynamics in terrestrial-satellite coexistence scenarios and introduces a novel beamforming approach that leverages available ephemeris data for dynamic interference mitigation. 
By establishing spatial radiation nulls directed towards visible satellites, our technique ensures the protection of satellite uplink communications without markedly compromising terrestrial downlink quality. 
Through a practical case study, we demonstrate that our approach maintains the satellite uplink signal-to-noise ratio (SNR) degradation under 1 dB and incurs a median SNR penalty of only 0.1 dB for the terrestrial downlink. 
Our findings offer a promising pathway for efficient spectrum sharing in the upper mid-band, fostering a concurrent enhancement in both terrestrial and satellite network capacity.

* Seongjoon Kang, Giovanni Geraci, Marco Mezzavilla, Sundeep Rangan
https://arxiv.org/abs/2311.12965

This paper will be published in [ICC 2024](https://icc2024.ieee-icc.org/)

## Satellite Tracking
In order to track the location of satellites in the given time, run the file, **‘track_satellites.py’**.  <br /> 
```
python3 track_satellites.py
```
After finshing running it, all the tracking information is saved in directory, **data/** <br />
For example, when observing satellite during 60 minutes with 25 &deg; minimum elevation angle, the **track_satellites.py** generates the following pickle file in **data/** <br />
```
satTrack_at_Colorado_time_60m_min_elev_25.pickle
```
If you want to change the time, change the variable, **cur_time** in the code. Default time is set as 08:10:20 09/03, 2023.<br />
Also, for different satellite constellations, you can download a different TLE file from [Celestrack](https://celestrak.org/). <br />
Now **Starlink** TLE file, **starlink.tle** downloaded in the September 2023, is in use. <br />

## Calculation of Interferences to Satellites
Interference nulling scheme suggested in our paper can be evaluated by running **sat_interference_calc.py**.<br />
There are several options to change simulation parameters. <br />
```
python3 sat_interference_calc.py --lambda_ 1 --bf SVD --n_iter 10 --total_observ_time 60
```
- **lambda_**: regularization parameters to control interference nulling. This value corresponds to &lambda; in the paper. <br />
  In the paper, we used &lambda; = 1, 10. <br />
- **bf**: beamforming scheme. There are three options: null_los, null_nlos, and SVD. <br />
      null_los: beamforming by creating nulls on LOS paths only.<br />
      null_nlos: beamforming by creating nulls on all the paths. <br />
      SVD: Singular Value Decomposition based beamforming, which doesn't consider interference nulling. <br />
- **n_iter**: number of iterations to choose different active BSs at random, which simultaneously transmit data to the best UE.
- **total_observ_time**: total observation time (minutes) of satellites by Earth station in the given region. The maximum value is 60 minutes.
- **n_itf**: number of BSs transmitting data concurrently. We set it as 21 in the paper, 20% of total BSs deployed. 

 All the simulation results are saved in the directory **data/**

## Ray-tracing Data
The ray-tracing data to model channels is located in the directory **rural_12GHz/**. <br />
- In **rural_12GHz/parsed_data_sat_to_bs**: each csv file in this directory includes all the channel paramters to all the BSs deployed. <br />
  The file names show the locations of satellites <br />
   For example, **sat_inclination_10_phi_120.csv** means that the satellite is located at 10 &deg; inclination angle and 120 &deg; horizontal angle. <br />



