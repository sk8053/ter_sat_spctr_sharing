
from astropy.coordinates import GCRS, ITRS, EarthLocation, CartesianRepresentation
from astropy import units as u
from astropy.time import Time
import numpy as np
#import scipy.constants  as p
import astropy.constants as constants

def get_speed_of_satellite(sat_height= 600e3):
    # get constant speed of satellite for the given height
    # ex.
    # -	At 600 km : V =7.5622 km.s-1
    # -	At 1500 km : V =7.1172 km.s-1
    # -	At 10000 km : V =4.9301 km.s-1
    # reference: https://rjallain.medium.com/calculating-the-speed-to-get-to-low-earth-orbit-and-other-calculations-c4df88f4cd2e
    # gravitational constant
    G = constants.G.value # unit = N m^2/kg^2
    #Earth mass
    M_earth = constants.M_earth.value # the mass of Earth
    R_earth = constants.R_earth.value # the radius of Earth
    speed = np.sqrt(G*M_earth/(R_earth+sat_height)) # speed value as a scalar
    # angular velocity
    #w = speed/(R_earth + sat_height)
    # satellite orbital period
    #T = (2*np.pi)/w

    return speed

def get_observation_time(alpha=30, sat_height = 600e3):
    # get maximum observation time at a location of an observer on Earth
    # given minimum elevation angle of a satellite
    R_earth  = constants.R_earth.value # the radius of Earth
    a = np.deg2rad(alpha)
    v = get_speed_of_satellite(sat_height)
    time = 2*((R_earth+sat_height)/v)*(np.pi/2 - a - np.arcsin((R_earth/(R_earth+sat_height))*np.cos(a)))
    return time
def get_doppler_freq_shift(f_c,  sat_height, u_t):
    """
    this functions follows the 3gpp 38.811, 5.3.4.3
    :param f_c: carrier frequency
    :param sat_height: the height of satellite
    :param u_t: angle at the origin of Earth between satellite and UE at time t [degree]
              u_t is changing over time t
    :return: doppler shift frequency
    """
    c = constants.c.value
    u_t = np.deg2rad(u_t)
    R_earth = constants.R_earth.value
    sat_velocity = get_speed_of_satellite(sat_height)

    gammar = (R_earth+sat_height)/R_earth

    f_d = (f_c/c)*sat_velocity*(np.sin(u_t)/np.sqrt(1+gammar**2 - 2*gammar*np.cos(u_t)))

    return f_d

# https://groups.google.com/g/astropy-dev/c/AIdMCZykFtw/m/LX0ZwdESBQAJ?pli=1
def eci2lla(x,y,z,dt):
    """
    Convert Earth-Centered Inertial (ECI) cartesian coordinates to latitude, longitude, and altitude, using astropy.

    Inputs :
    x = ECI X-coordinate (m)
    y = ECI Y-coordinate (m)
    z = ECI Z-coordinate (m)
    dt = UTC time (datetime object)
    Outputs :
    lon = longitude (radians)

    lat = geodetic latitude (radians)
    alt = height above WGS84 ellipsoid (m)
    """
    # convert datetime object to astropy time object
    tt=Time(dt,format='datetime')

    # Read the coordinates in the Geocentric Celestial Reference System
    gcrs = GCRS(CartesianRepresentation(x=x*u.m, y=y*u.m,z=z*u.m), obstime=tt)

    # Convert it to an Earth-fixed frame
    itrs = gcrs.transform_to(ITRS(obstime=tt))

    el = EarthLocation.from_geocentric(itrs.x, itrs.y, itrs.z)

    # conversion to geodetic
    lon, lat, alt = el.to_geodetic()

    return lon.value, lat.value, alt.value #, (itrs.x.value, itrs.y.value, itrs.z.value)