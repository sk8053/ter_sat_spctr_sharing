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

# Satellite Tracking
In order to track the location of satellites in the given time, run the file, **‘track_satellites.py’**.  <br /> 
If you want to change the time, change the variable, **cur_time** in the code. Default time is set as 08:10:20 09/03, 2023.<br />
Also, for different satellite constellations, you can download a different TLE file from [Celestrack](https://celestrak.org/). <br />
Now **Starlink** TLE file **starlink.tle** downloaded in the September 2023 is in use. <br />
