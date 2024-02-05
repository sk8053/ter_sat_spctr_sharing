import numpy as np
import pandas as pd
import glob
import itur

class ChooseServingSat(object):

    def __init__(self, obs_lat, obs_lon, p, freq = 18e9, dir_:str= 'parsed_data_sat_to_bs'):
        self.obs_lat =obs_lat
        self.obs_lon = obs_lon
        self.p = p #values exceeded during p % of the average
        self.freq = freq  # carrier frequency
        # read csv files for one observer
        self.elev_to_df = {e:dict() for e in [10,20,30,40,50,60,70,80,90]}
        for elev in [10,20,30,40,50,60,70,80,90]:
            incl_ang = 90 - elev
            if incl_ang != 0:
                for phi in [0,60,120,180,240, 300]:
                    file_name = f'rural_{int(freq/1e9)}GHz/{dir_}/sat_inclination_{incl_ang}_phi_{phi}.csv'
                    # each elevation angle can map to several azimuth angles
                    self.elev_to_df[elev][phi] = pd.read_csv(file_name)
            else:
                file_name = f'rural_{int(freq/1e9)}GHz/{dir_}/sat_inclination_{incl_ang}_phi_0.csv'
                # each elevation angle can map to several azimuth angles
                self.elev_to_df[elev][0] = pd.read_csv(file_name)

        #sample geodetic locations directed to the azimuth angles
        self.azimuth_to_LatLon =np.array( [[ 40.071024883264556,-104.692390260897568], #phi = 0
                                           [ 40.384950998480129,-104.914360373175015], # phi = 60
                                            [ 40.398901262320805,-105.398720678060158], # phi = 120
                                           [40.066413452242159, -105.624391573035723], # phi = 180
                                            [39.760154612246957, -105.381364551232764 ], # phi = 240
                                           [ 39.745444936776295, -104.859572530632775]]) #phi = 300

    def get_chan_df(self, elev_list:list(),  dist_list:list(),sat_ind_list:list(), obs_ind:int = 0, n_serving_sat:int = None, sat_geo_list:list() = []):

        """
        :param elev_list: elevation angles corresponding to each satellite
        :param dist_list: distance between an observer and satellites
        :param sat_ind_list: the list of the names of satellites
        :param obs_ind: the list of indices of satellite
        :param n_serving_sat: number of satellites serving IoT devices
        :return: the channel parameter set of the best satellites
        """

        elev_list = np.array(elev_list)
        # round off elevation angles so that they can be mapped to remocomm data
        elev_list = np.round(elev_list/10).astype(int)*10
        pl_keys = ['path_loss_%d'%(s+1) for s in range(25)]
        fr_keys=['link state', 'n_path', 'path_loss_1', 'delay_1', 'aoa_1', 'aod_1', 'zoa_1', 'zod_1']
        #fr_keys+=['delay_2','path_loss_2','aoa_2', 'aod_2', 'zoa_2', 'zod_2']
        df_list = [] # save channel parameters of satellites
        df_LOS_list =[]
        # collect all the channel parameters corresponding to all the satellites
        for k, (elev, dist) in enumerate(zip(elev_list, dist_list)):

            if len(sat_geo_list) == 0:
                # select azimuth angle randomly
                phi = np.random.choice(np.arange(360,step = 60),1)[0]
            else:
                sat_geo_k = sat_geo_list[k]

                delta = np.linalg.norm(self.azimuth_to_LatLon - sat_geo_k, axis = 1)
                min_k = delta.argmin()
                phi = min_k *60

            if elev==90:
                phi = 0
            df = self.elev_to_df[elev][phi]
            # choose the index of an observer
            df_el = df.iloc[obs_ind].copy()

            if elev==0 and phi==0:
                other_pl = 0
            else:
                # scale pathloss value by the given distance
                other_pl = self.get_other_pathloss(self.obs_lat, self.obs_lon, elev, self.freq, self.p)
            df_el[pl_keys] += 20 * np.log10(dist / df_el['distance']) + other_pl.value

            df_list.append(df_el)
            #elev_phi.append([elev, phi])
            df_first = df_el[fr_keys].copy()
            #df_first['aoa_1'] = phi
            #df_first['zoa_1'] = 90-elev
            #df_first['aod_1'] = df_first['aoa_1'] -180
            #df_first['zod_1'] = 180-df_first['zoa_1']
            df_first['link state'] = 1
            df_first['n_path'] = 1
            df_LOS_list.append(df_first)

        #df_first_array = np.array(df_first_list)
        #I = np.argsort(first_pl_list) # sort first path loss in ascending order
        I = range(len(df_list))
        if n_serving_sat == None:
            I_best = I
        else:
            # choose n-best satellites
            #I_best = I[:n_bestsat]
            I_best = np.random.choice(I, (n_serving_sat,), replace=False)

        best_df_array = [df_list[i] for i in I_best]
        df_LOS_list =[df_LOS_list[i] for i in I_best]
        best_sat_names = np.array(sat_ind_list)[I_best]
        #best_elevs = elev_list[I_best]
        return best_df_array, best_sat_names , df_LOS_list

    def get_other_pathloss(self, lat, lon,el, freq = 18e9, p =1):

        # Location of the receiver ground stations
        #lat = 41.39
        #lon = -71.05
        # Link parameters
        el = el # Elevation angle [deg]
        f = int(freq/1e9) * itur.u.GHz  # Frequency [GHz]
        D = 0.2 * itur.u.m  # Receiver antenna diameter  [m]
        p = p # We compute values exceeded during p % of the average
        # year
        # Compute atmospheric parameters
        hs = itur.topographic_altitude(lat, lon)

        # T = itur.surface_mean_temperature(lat, lon)
        # P = itur.standard_pressure(hs)
        # rho_p = itur.surface_water_vapour_density(lat, lon, p, hs)
        # rho_sa = itur.models.itu835.water_vapour_density(lat, hs)
        # T_sa = itur.models.itu835.temperature(lat, hs)
        # V = itur.models.itu836.total_water_vapour_content(lat, lon, p, hs)
        # Compute rain and cloud-related parameters
        # R_prob = itur.models.itu618.rain_attenuation_probability(lat, lon, el, hs)
        # R_pct_prob = itur.models.itu837.rainfall_probability(lat, lon)
        # R001 = itur.models.itu837.rainfall_rate(lat, lon, p)
        # h_0 = itur.models.itu839.isoterm_0(lat, lon)
        # h_rain = itur.models.itu839.rain_height(lat, lon)
        # L_red = itur.models.itu840.columnar_content_reduced_liquid(lat, lon, p)

        A_w = itur.models.itu676.zenit_water_vapour_attenuation(lat, lon, p, f, h=hs)
        # Compute attenuation values
        # A_g = itur.gaseous_attenuation_slant_path(f, el, rho_p, P, T)
        A_r = itur.rain_attenuation(lat, lon, f, el, hs=hs, p=p)
        A_c = itur.cloud_attenuation(lat, lon, el, f, p)
        A_s = itur.scintillation_attenuation(lat, lon, f, el, p, D)
        A_t = itur.atmospheric_attenuation_slant_path(lat, lon, f, el, p, D)

        return A_w + A_r + A_c + A_s + A_t
