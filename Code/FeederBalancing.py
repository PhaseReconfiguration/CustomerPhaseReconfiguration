import math
import numpy as np
import os
import pandapower as pp
from pandapower.plotting.plotly import simple_plotly
from pandapower.plotting.plotly import pf_res_plotly
import pandapower.topology as top
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time
import itertools
from tqdm import tqdm #loading bar
import pickle 

class FeederBalancing:
    def __init__(self, input_path):
        # Conventions:
        # - Use kWh for timeseries (easier to understand even it pandapower asks for MWh at the end)
        np.random.seed(19)
        self.input_path = input_path

        self.avalilable_phases = ['A', 'B', 'C']
        self.len_timeseries = 4*24*365 #to change depending on the availbale data
        self.dict_phasecode_to_number = {'Monophasé (sans neutre)':1, 'Monophasé':1, 'Triphasé':2, 'Tétraphasé':3}
        self.pv_scaling_factor = 712 #2850/4=712 -> energy produced by a system of 1 kWhp in an year. 
        self.pv_installation_sizes = [self.pv_scaling_factor * 4, self.pv_scaling_factor * 16] #Ref: https://www.yesenergysolutions.co.uk/advice/how-much-energy-solar-panels-produce-home
        self.ev_scaling_factor = 350
        self.ev_installation_sizes = [self.ev_scaling_factor * 3.5, self.ev_scaling_factor * 18] #Ref: TODO
        self.hp_scaling_factor = 300
        self.hp_installation_sizes = [self.hp_scaling_factor * 2.5, self.hp_scaling_factor * 16] #Ref: TODO

        self.consumption_file = pd.read_excel(os.path.join(self.input_path, "RESA", "anonymized_consumption_file.xlsx"))
        self.list_load_timeseries = pd.read_csv(os.path.join(input_path, 'RESA', 'anonymized_Load_SM_timeseries.csv'), sep=',', index_col=0).reset_index()
        self.load_timeseries_default = pd.read_csv(os.path.join(self.input_path, 'Timeseries', '1-LV-rural2--1-sw', 'LoadProfile.csv'), sep=';').reset_index()
        self.list_pv_timeseries = pd.read_csv(os.path.join(input_path, 'RESA', 'anonymized_PV_SM_timeseries.csv'), sep=',', index_col=0).reset_index()
        self.pv_timeseries_default = pd.read_csv(os.path.join(self.input_path, 'Timeseries', '1-LV-rural2--1-sw', 'RESProfile.csv'), sep=';').reset_index()
        self.evhp_timeseries = pd.read_excel(os.path.join(self.input_path, 'Timeseries', 'HPEV timeseries.xlsx')).reset_index()[:self.len_timeseries]
        self.ev_timeseries_high = pd.read_csv(os.path.join(self.input_path, 'Timeseries', 'EV_High.csv'),  index_col=0)[:self.len_timeseries]['High']
        self.hp_timeseries_high = pd.read_csv(os.path.join(self.input_path, 'Timeseries', 'HP_High.csv'),  index_col=0)[:self.len_timeseries]['High']

        self.net, self.choosable_buses, self.distances = self.import_network(self.input_path)
        self.number_customers = len(self.net.asymmetric_load)
        self.feeders = [270,61]
        self.assign_feeders(self.feeders) #Depends on the network (The first bus(es) after the substation low-voltage bus )
        self.temp_P = None
        self.feeder_colors = np.random.choice(list(mcolors.CSS4_COLORS.keys()), len(self.feeders))
        
        self.timesteps = list(range(self.len_timeseries))
        self.number_timesteps = len(self.timesteps)
        self.load_timeseries(plot_timeseries=False)
        self.assign_ts_to_phase()
        self.feeder_eans = {
                f: self.net.asymmetric_load.loc[self.net.asymmetric_load['feeder'] == f, 'ean'].tolist()
                for f in range(len(self.feeders)) }
        self.considered_feeder = None
        self.feeder_index_eans = {
                f: self.net.asymmetric_load.loc[self.net.asymmetric_load['feeder'] == f, 'ean'].index.tolist()
                for f in range(len(self.feeders)) }
        self.considered_index_eans = None

        self.aggragates_init = [self.aggregate_feeder_load(self.P, self.feeder_eans[f]) for f in range(len(self.feeders))]
        self.generate_Psi()
        self.generate_Bs()
        
        self.unbalance_loss = []
        self.associated_loss = []
        self.changes_loss = []
        self.loss_distance = []

    def import_network(self, input_path):
        net = pp.from_pickle(os.path.join(input_path, 'anonymized_net.p'))
        choosable_buses = list(net.load.bus)

        self.NetInfo(net)
        simple_plotly(net, bus_color=net.bus['color'])

        distances = 1 / np.array(net.asymmetric_load['distance'])
        
        return net, choosable_buses, distances
    def NetInfo(self, net):
        #Print various statistics about the network.
        print(f'There are {len(net.ext_grid)} substation(s)')
        print(f'There are {len(net.trafo)} transformer(s). Rated power: {net.trafo.sn_mva.values} kVAR')
        print(f'There are {len(net.bus)} nodes')
        print(f"There are {len(net.line)} lines")
        print(f"There are {np.sum(net.bus['info']=='customer')} customers")

    def get_phasecode_from_number(self, number_phases):
        value = {i for i in self.dict_phasecode_to_number if self.dict_phasecode_to_number[i]==number_phases}
        return value
    def get_number_from_phasecode(self, phasecode):
        return self.dict_phasecode_to_number[phasecode]

    def get_score(self, h_surface, number_phases):
        #PV: https://pdf.sciencedirectassets.com/280851/1-s2.0-S2211467X22X00072/1-s2.0-S2211467X23001281/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEJ3%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIQCEnMLu0y6ENVyHyFXISbvxs%2BeVM%2BtJ4eVvSQHVHB2I%2BQIgOdtj3nYwO4mdCwqRqJCo5yh79c1SIQyWITtKvLDhjEAqsgUIRhAFGgwwNTkwMDM1NDY4NjUiDNwMccNyvzGK1FTB8CqPBXh1jHYvDjcWjAkjrBprgxY%2F5IXIb71jaq46KjuFnum9HcL%2Fxxj6xvh%2BBhvBiKaAWuJ%2BhOgKJblPAqb7t%2BT9O%2FRbJH5RUWFHL1pepYZgPIpS%2B7vaY8CtYgYkwC04gNrACswZUU0Tn%2FIR3ZVDiOD2nVo5F%2BKw1hzGKljqvdY1kZKtuSRd5MZRW8nDqfDK%2BkPzp%2F3zUsHMY2AHnyDtK4RQcA7u24MiUS9neAe1QXMTS%2FjqQ0ZzFSD28zMY6Pe0PelnlJkNVrz%2BjmEHsw8hfrPoCZN1zjWQmcKXrB0HHOYUtCFWebfPHwjpEnsMLgQ%2BeKZUGxwVNu7qhq4YJHwfzJhlGQavJ6pkXyrj8aAwI1959OusgejJ55BOJYU7285d6muckAgBzW%2FD9fxQkAjaI1Np713Yf5cqQ3iPwxJwt14jGh868RjQLoKul61a9a%2BMG4Fudb0BahvNhRYgORRrNKaVkprXBOSuOG9AVtwk8%2BB%2Bi1eLmR5frXjZig9lLzpWf9sCMMnZdTHFAcEOrhWh1Tha4wGICuXuBbvxQlmQsdF6DqHzK2K9bQsH0dSAEk6DOOoA6dEHOEeZo8FWqgbX9z80DyNqFxB3xyS8x1BwTNTByaKAiTeZDb65fjnua7hlFDDviTvFgKchSBuFaOy37Q4aA%2Fm1S2M%2FesjNe3O79Tf4tstuVbZIlmMkB%2Bma4D%2Fqv8BcEAzo1rhUtUgkP3iW%2F43WQmMvCv%2BmN3TQi%2FLOa3l0tFl14LfZOBv5Pfe%2B9Kxu803nYOkP2tOu7YOSN%2FOo1DQMTLBoZI02YOYC7US4jFjeiiA8TXdkPn4W2Wual%2B%2FcUfAa4g6%2FH0v%2Bc1CB0BmRaIy02tTis7VfNxDKBPhn8Z2DmUcwrKOcugY6sQFUt8TxKH5tsIb6vk6IEyek0EwqjIrlrTAqPubj7n4oWYna4JhceuTndqEygtfUPBNL3O6j8qlpZaMCfT4jqSN1e5ELpkIlsgF29Vgf%2FBsFS2%2BcbiPvTk6lCYxG92NQqheYErJ5bv14WmC5MKdVhGA1hMnx86TTUlKEDpPbUqeFL3WSWwY3JbYlAW0zajjRNf3DOggIBpOWxgk0kVpb%2Bo%2B8RjRQZznKo9aLV2NH8H83wvo%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20241127T133941Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY6NZOVK4C%2F20241127%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=e17752cbf36b4c3b6e636634e6604889bebf2c8db4d942705f9d6ad7f6d3c6f6&hash=c4025a6d6e934b7461dea9fc5af2a340b50d2e2f589342c412508ffaf3416501&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S2211467X23001281&tid=spdf-fe3a5c3a-e8c3-485f-bd50-aea73e4c6115&sid=53e02f3a4ed4054da778a3b4a62cf4c2c620gxrqb&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=18075e045a5c59555e55&rr=8e928696084cb9a1&cc=be
        #EV: https://escholarship.org/uc/item/2b05w8pk
        score = h_surface/80 + number_phases/1.75 #Means values
        return score

    def phases_from_score(self, h_surface, phases_load, tech_installation_sizes):
        number_phases_load = len(phases_load)
        score = self.get_score(h_surface, number_phases_load)
        if(score < 3 or number_phases_load < len(self.avalilable_phases)): #TODO: The <3 might need some additional consideration.
            annual_consumption = tech_installation_sizes[0]
            phases = phases_load[:1]
        else:
            annual_consumption = tech_installation_sizes[1]
            phases = self.avalilable_phases
        
        return phases, annual_consumption
    
    def get_phase_splitting_values(self, number_phases):
        if number_phases==1:
            return [1]
        else:
            values = np.random.normal(1/number_phases, 0.05, number_phases)
            values = values / values.sum()
            return values

    def shift_timeseries(self, shifting_value):
        if(shifting_value!=0):
            period = np.random.randint(-shifting_value, shifting_value)
        else:
            period = 0
        return period
    def load_timeseries(self, plot_timeseries=True):
        assigned_load_timeseries = pd.DataFrame() #Better a Pandas DF: rows-timesteps, columns-EANs
        
        def assign_find_closest_load_ts(customers_wo_ts, assigned_load_timeseries, list_load_timeseries, avalilable_ts_for_ean):
            timeseries_default1 = self.load_timeseries_default['H0-A_pload'][:self.len_timeseries]
            timeseries_default2 = self.load_timeseries_default['H0-B_pload'][:self.len_timeseries]
            timeseries_default = [timeseries_default1, timeseries_default2]
            for c in customers_wo_ts:
                ean = c.ean
                annual_consumption = c['ann_cons']
                phases = c['phase_load']
                phasecode = self.get_phasecode_from_number(len(phases))

                closest_ean = None
                closest_diff = np.inf
                # Find the closest matching timeseries based on consumption and phasecode
                for _,other_c in self.consumption_file[(self.consumption_file['RACCORD_CLIENT']==phasecode) & (self.consumption_file['EAN'].isin(avalilable_ts_for_ean)) ].iterrows():
                    other_ean = str(other_c['EAN'])
                    ts_consumption = other_c['Total Actif calcule']
                    ts_consumption = ts_consumption if ts_consumption is not None else other_c['P_CONTRACTUELLE']

                    # Calculate difference in consumption and check phase compatibility
                    consumption_diff = abs(annual_consumption - ts_consumption)
                    if consumption_diff < closest_diff:
                        closest_ean = other_ean
                        closest_diff = consumption_diff

                multipliers = self.get_phase_splitting_values(len(phases))
                factor = np.random.random(1)*0.2+0.8
                for i,p in enumerate(phases):
                    if(closest_ean):
                        ts = list_load_timeseries[f"{closest_ean}_{p}"]
                    else:
                        ts = timeseries_default[i%len(timeseries_default)] #Use a standard one
                    tmp_load_timeseries = self.normalize_time_series(ts, annual_consumption * multipliers[i])
                    assigned_load_timeseries[f'{ean}_{p}'] = tmp_load_timeseries * factor

            return assigned_load_timeseries
        
        avalilable_ts_for_ean = [i[:-2] for i in self.list_load_timeseries.columns.values[1:]]
        avalilable_ts_for_ean = set(avalilable_ts_for_ean)

        customers_wo_ts = []
        for _,c in self.net.asymmetric_load.iterrows():
            ean = c['ean']
            if(ean in avalilable_ts_for_ean):
                phases = c['phase_load']
                annual_consumption = c['ann_cons']
                multipliers = self.get_phase_splitting_values(len(phases))
                for i,p in enumerate(phases):
                    if(f"{ean}_{p}" in self.list_load_timeseries.columns):
                        ts = self.list_load_timeseries[f"{ean}_{p}"]
                    elif(p=='B' or p=='C'): #If we enter here it means that the customer is 3-phase but the time series for B and C are not available (most likely tons of missing values). Therefore use the A-one
                        ts = self.list_load_timeseries[f"{ean}_A"]
                    tmp_load_timeseries = self.normalize_time_series(ts, annual_consumption * multipliers[i])[:self.len_timeseries]
                    assigned_load_timeseries[f"{ean}_{p}"] = tmp_load_timeseries
            else:
                customers_wo_ts.append(c)
        assigned_load_timeseries =  assign_find_closest_load_ts(customers_wo_ts, assigned_load_timeseries, self.list_load_timeseries, avalilable_ts_for_ean)


        def assign_find_closest_pv_ts(customers_wo_ts, assigned_timeseries):
            timeseries_default1 = self.pv_timeseries_default['PV1'][:self.len_timeseries]
            timeseries_default2 = self.pv_timeseries_default['PV3'][:self.len_timeseries]
            timeseries_default3 = self.pv_timeseries_default['PV7'][:self.len_timeseries]
            timeseries_default = [timeseries_default1, timeseries_default2, timeseries_default3]

            pv_pen_rate = 0.85 #[0,1]
            current_pv_pen_rate =  len(customers_wo_ts) / len(self.net.asymmetric_load)
            delta_pv_pen_rate = pv_pen_rate # pv_pen_rate - current_pv_pen_rate #TODO: fix to a different amount
            choosable_buses_missing_pv = [i.bus for i in customers_wo_ts]
            self.net = self.PVinstallation(self.net, choosable_buses_missing_pv, delta_pv_pen_rate)
            for j,c in enumerate(customers_wo_ts):
                ean = c['ean']
                c = self.net.asymmetric_load[self.net.asymmetric_load['ean'] == ean]
                phases = c['phase_pv'].values[0]
                if(phases is not None):
                    annual_prod = c['ann_pv_prod'].values[0]

                    multipliers = self.get_phase_splitting_values(len(phases))
                    period = self.shift_timeseries(4*2)
                    factor = np.random.random(1)*0.2+0.8
                    for i,p in enumerate(phases):
                        tmp_load_timeseries = self.scale_time_series(timeseries_default[(j+i)%len(timeseries_default)], annual_prod * multipliers[i], self.pv_scaling_factor)
                        assigned_timeseries[f'{ean}_{p}'] = -tmp_load_timeseries.shift(periods=period, fill_value=0) * factor
            return assigned_timeseries

        assigned_pv_timeseries = pd.DataFrame()
        customers_wo_ts = []
        avalilable_ts_for_ean = set(self.list_pv_timeseries.columns.values[1:])
        for j,c in self.net.asymmetric_load.iterrows():
            ean = c['ean']
            if(ean in avalilable_ts_for_ean):
                annual_prod = np.sum(self.list_pv_timeseries[ean])
                phases_load = c['phase_load']

                # Decide the number of phases based on the annual consumption
                number_phase = 1 if annual_prod <= self.pv_installation_sizes[0] else 3 #It can generate conflicts with the load phase
                number_phase = min(number_phase, len(phases_load)) #so avoid that the load is 1-phase but the PV 3-phase
                
                phases = phases_load if number_phase < len(self.avalilable_phases) else  self.avalilable_phases[:number_phase]
                
                self.net.asymmetric_load.at[j, 'phase_pv'] = ''.join(phases)
                self.net.asymmetric_load.at[j, 'ann_pv_prod'] = int(annual_prod)

                for i,p in enumerate(phases):
                    multipliers = self.get_phase_splitting_values(len(phases))
                    tmp_pv_timeseries = self.normalize_time_series(self.list_pv_timeseries[ean], annual_prod * multipliers[i])[:self.len_timeseries]
                    assigned_pv_timeseries[f"{ean}_{p}"] = -tmp_pv_timeseries
            else:
                customers_wo_ts.append(c)

        assigned_pv_timeseries =  assign_find_closest_pv_ts(customers_wo_ts, assigned_pv_timeseries)

        def assign_timeseries_EVHP(column, pen_rate):
            if(column=='ev'):
                tech_installation_scaling_factor = self.ev_scaling_factor
                tech_installation_sizes = self.ev_installation_sizes
                ts_to_consider = "5000 kWh"
            else:
                tech_installation_scaling_factor = self.hp_scaling_factor
                tech_installation_sizes = self.hp_installation_sizes
                ts_to_consider = "Total PAC"

            assigned_timeseries = pd.DataFrame()
            for i ,c in self.net.asymmetric_load.iterrows():
                install_random = np.random.rand()
                if(install_random>pen_rate):
                    continue
                ean = c['ean']
                phases_load = c['phase_load']
                h_surface = c['hh_surf']

                phases, annual_consumption = self.phases_from_score(h_surface, phases_load, tech_installation_sizes)
                
                self.net.asymmetric_load.at[i, f'ann_{column}_cons'] = annual_consumption
                self.net.asymmetric_load.at[i, f'phase_{column}'] = ''.join(phases)

                multipliers = self.get_phase_splitting_values(len(phases))
                period = self.shift_timeseries(4*5)
                factor = np.random.random(1)*0.2+0.8
                for j,p in enumerate(phases):
                    if(annual_consumption>tech_installation_sizes[0]):
                        ts = self.ev_timeseries_high if column=='ev' else self.hp_timeseries_high
                        tmp_load_timeseries = self.scale_time_series(ts, annual_consumption * multipliers[j], tech_installation_scaling_factor)
                    else:
                        tmp_load_timeseries = self.normalize_time_series(self.evhp_timeseries[ts_to_consider], annual_consumption * multipliers[j])
                    assigned_timeseries[f"{ean}_{p}"] = tmp_load_timeseries.shift(periods=period, fill_value=0) * factor
            return assigned_timeseries
        
        ev_pen_rate = 0.7
        assigned_ev_timeseries = assign_timeseries_EVHP('ev', ev_pen_rate)
        hp_pen_rate = 0.6
        assigned_hp_timeseries = assign_timeseries_EVHP('hp', hp_pen_rate)

        self.assigned_load_timeseries = assigned_load_timeseries
        self.assigned_pv_timeseries = assigned_pv_timeseries
        self.assigned_ev_timeseries = assigned_ev_timeseries
        self.assigned_hp_timeseries= assigned_hp_timeseries
        print(f"Customers with:\nPV: {self.net.asymmetric_load['phase_pv'].notnull().sum()},\nEV: {self.net.asymmetric_load['phase_ev'].notnull().sum()},\nHP: {self.net.asymmetric_load['phase_hp'].notnull().sum()}")


    def assign_ts_to_phase(self):
        def get_least_used_phase(P, ean, phases_load, inverse=False):
            subset_P = P[ [f"{ean}_{p}" for p in phases_load] ]
            phase_usages = subset_P.sum(axis=0)

            ind = np.argmax(phase_usages) if inverse else np.argmin(phase_usages)
            phase = phases_load[ind]
            return phase

        def update_least_used_phase(P, phases_load, phases_tech, ean, annual_consumption, tech_installation_sizes, assigned_timeseries, inverse=False):
            if(phases_tech is not None): #Tech not installed
                is_annual_consumption_large = annual_consumption > tech_installation_sizes[0]
                if(is_annual_consumption_large): #Consumption is large: split the tech among the phases almost equally (already done in the previous step)
                    for p in phases_tech:
                        P[f'{ean}_{p}'] += assigned_timeseries[f"{ean}_{p}"]
                else: #Consumption is not large: assign the tech to the least used phase
                    least_used_phase = get_least_used_phase(P, ean, phases_load, inverse=inverse)
                    column_least_used = f'{ean}_{least_used_phase}'
                    P[column_least_used] += assigned_timeseries[f"{ean}_{phases_tech[0]}"]

        temp_columns_P = [f"{ean}_{p}" for ean in self.net.asymmetric_load['ean'] for p in self.avalilable_phases]
        tmp_values_P = np.zeros( [self.len_timeseries, len(temp_columns_P)] )
        P = pd.DataFrame(tmp_values_P, columns=temp_columns_P)
        for i,c in self.net.asymmetric_load.iterrows():
            phases = c['phase_load']
            ean = c['ean']
            for p in phases:
                column = f'{ean}_{p}'
                P[column] += self.assigned_load_timeseries[column]
            if(len(phases) == 1): #Customers connected to a single phase
                column = f'{ean}_{phases[0]}'
                if(c['phase_ev'] is not None):
                    P[column] += self.assigned_ev_timeseries[column]
                if(c['phase_hp'] is not None):
                    P[column] += self.assigned_hp_timeseries[column]
                if(c['phase_pv'] is not None):
                    P[column] += self.assigned_pv_timeseries[column]
            else: #Customers with multiple phases
                phases_pv = c['phase_pv']
                update_least_used_phase(P, phases, phases_pv, ean, c['ann_pv_prod'], self.pv_installation_sizes, self.assigned_pv_timeseries, inverse=True)
                phases_ev = c['phase_ev']
                update_least_used_phase(P, phases, phases_ev, ean, c['ann_ev_cons'], self.ev_installation_sizes, self.assigned_ev_timeseries)
                phases_hp = c['phase_hp']
                update_least_used_phase(P, phases, phases_hp, ean, c['ann_hp_cons'], self.hp_installation_sizes, self.assigned_hp_timeseries)
        
        self.P = P

    def CalculatePossibleCombinations(self, choosable_buses, percentuage):
        #Calculate the possible combinations given choosable buses and a percentuage of changes.
        choosable_elements = len(choosable_buses)
        n_combinations = math.comb(choosable_elements, int(choosable_elements * percentuage)) # comb(n,m) = n! / [(n-m)!m!], n>m
        return n_combinations

    def chose_buses(self, choosable_buses, penetration_rate, probabilities):
        # Function to choose buses for PV scenarios
        elements_to_select = round(len(choosable_buses)*penetration_rate)
        ##Without probabilities
        # ids = np.random.choice(choosable_buses, elements_to_select, replace=False)
        ##With probabilities
        p = probabilities / np.sum(probabilities) #Make the sum of p equal to 1 (Required by numpy)
        ids = np.random.choice(choosable_buses, elements_to_select, replace=False, p=p)
        return ids

    def PVinstallation(self, net, choosable_buses, penetration):
        prob = net.asymmetric_load[net.asymmetric_load.bus.isin(choosable_buses)]['probPV']
        ids = self.chose_buses(choosable_buses, penetration, prob)

        for i in ids:
            h_surface = net.asymmetric_load.loc[net.asymmetric_load['bus']==i, 'hh_surf'].values[0]
            phases_load = net.asymmetric_load.loc[net.asymmetric_load['bus']==i, 'phase_load'].values[0]

            phases, annual_prod = self.phases_from_score(h_surface, phases_load, self.pv_installation_sizes)
            ind = net.asymmetric_load[net.asymmetric_load['bus']==i].index[0]
            net.asymmetric_load.at[ind, 'ann_pv_prod'] = annual_prod
            net.asymmetric_load.at[ind, 'phase_pv'] = ''.join(phases)
        return net
    def EVinstallation(self, net, choosable_buses, penetration):
        ids = self.Chose_buses(self, choosable_buses, penetration, net.asymmetric_load['probEV'])

        for i in ids:
            #Add EV to customer's bus. No need to create a new element, just add 'E' to 'tech' columns
            net.asymmetric_load.loc[net.asymmetric_load['bus'] == i, 'tech'] += 'E'
        return net
    def HPinstallation(self, net, choosable_buses, penetration):
        ids =self. Chose_buses(choosable_buses, penetration, net.asymmetric_load['probHP'])

        for i in ids:
            #Add HP to customer's bus. No need to create a new element, just add 'H' to 'tech' columns
            net.asymmetric_load.loc[net.asymmetric_load['bus'] == i, 'tech'] += 'H'
        return net

    def normalize_time_series(self, timeseries, total_consumption):
        #Use this function in order to set the sum over time of a given timeseries equal to total_consumption
        t = timeseries / np.sum(timeseries) * total_consumption
        return pd.Series(t)
    def scale_time_series(self, timeseries, scaling_factor, tech_installation_scaling_factor):
        #Use this function in order to set the max a given timeseries equal to scaling_factor (Used to PV peak for example)
        t = timeseries * scaling_factor / tech_installation_scaling_factor
        return pd.Series(t)

    def get_meaningful_days_timesteps(self, meaningful_days):
        timeseries_steps = []
        multiplier = 4 * 24 #Number timesteps in a day considering 15 minutes resolutions (1h = 4 * 15minutes)
        for day in meaningful_days:
            if(day<1 or day>365):
                raise("Error in timestep parsing. Required a day in [1,365], received {day}")
            timeseries_steps.extend(list(range((day-1) * multiplier, day * multiplier)))
        
        return sorted(timeseries_steps)

    def plot_P(self,number=-1):
        for _,c in self.net.asymmetric_load[:number].iterrows():
            ean = c['ean']
            for p in self.avalilable_phases:
                plt.plot(self.P.iloc[self.timesteps][f'{ean}_{p}'].values)
            
            print(f"Client ean: {ean}. \n Load: #phases: {c['phase_load']}, consumption: {c['ann_cons']}. \n PV: #phases: {c['phase_pv']}, consumption: {c['ann_pv_prod']}. \n EV: #phases: {c['phase_ev']}, consumption: {c['ann_ev_cons']}. \n HP: #phases: {c['phase_hp']}, consumption: {c['ann_hp_cons']}.")
            plt.show()

    def generate_Psi(self):
        Psi = []

        for i in range(len(self.avalilable_phases)):
            perm = itertools.permutations(self.avalilable_phases, i+1) 
            for j in perm: 
                Psi.append(''.join(j))
        self.Psi = Psi

    def get_index_from_phase_confg(self, phase):
        for i,p in enumerate(self.Psi):
            if(p == phase):
                return i
        print(f"Phase {phase} not in {self.Psi}")

    def generate_Bs(self):
        B_init = np.zeros( (len(self.net.asymmetric_load), len(self.Psi)) )
        B_init_nobinary = []

        B_feas = np.zeros( (len(self.net.asymmetric_load), len(self.Psi)) )
        indexes_per_n_phases = {}
        indexes_per_n_phases_customers = []
        for i in range(len(self.avalilable_phases)):
            indexes = [k for k,j in enumerate(self.Psi) if len(j)==i+1] #return all the configuration indexes with a given number of phases
            indexes_per_n_phases[i+1] = indexes

        for i,(_,c) in enumerate(self.net.asymmetric_load.iterrows()):
            phase = c['phase_load']
            index_confg = self.get_index_from_phase_confg(phase)
            B_init[i,index_confg] = 1
            B_init_nobinary.append(index_confg)
            for j in indexes_per_n_phases[len(phase)]:
                B_feas[i,j] = 1
            indexes_per_n_phases_customers.append( indexes_per_n_phases[len(phase)] )

        self.B_init = B_init
        self.B_init_nobinary = B_init_nobinary
        self.B_init_opposite = 1 - B_init
        self.B_feas = B_feas
        self.B_feas_nobinary = indexes_per_n_phases
        self.B_feas_nobinary_per_customer = indexes_per_n_phases_customers

    def get_feeder(self, net, bus, prev_buses = [], prev_lines = []):
        #Find this info plotting the network: _ = simple_plotly(net, aspectratio=(10,8))
        trafo_bus_id = self.net.trafo['lv_bus'].values[0] #may depend on the network
        trafo_line_ids = [] #Depends on the network
        prev_buses.append(bus)

        elems = pp.toolbox.get_connected_elements_dict(net, bus)
        for b in elems['bus']:
            if(b!=trafo_bus_id and b not in prev_buses):
                self.get_feeder(net, b, prev_buses, prev_lines)
        for l in elems['line']:
            if(l not in trafo_line_ids and l not in prev_lines):
                prev_lines.append(l)
    def assign_feeders(self, starting_buses):
        for i,b in enumerate(starting_buses):
            buses = []
            lines = []
            self.get_feeder(self.net, b, buses, lines)

            self.net.bus.loc[self.net.bus.index.isin(buses), 'feeder'] = i
            self.net.line.loc[self.net.line.index.isin(lines), 'feeder'] = i
            self.net.asymmetric_load.loc[self.net.asymmetric_load['bus'].isin(buses), 'feeder'] = i
        self.customers_index_per_feeder = {i:self.net.asymmetric_load.loc[self.net.asymmetric_load['feeder'] == i].index.values for i in range(len(self.feeders))}

    def change_P(self, B):
        P_new = self.P.copy(deep=True)
        for i,changed in enumerate(np.sum(B * self.B_init_opposite, axis=1)):
            if(changed==1):
                c = self.net.asymmetric_load.iloc[i] #it may give issues if the indexes are not the same as expected
                ean = c['ean']
                old_conf = c['phase_load']
                new_conf = self.Psi[np.argmax(B[i])]
                for j,p in enumerate(new_conf):
                    P_new[f'{ean}_{new_conf[j]}'] = self.P[f'{ean}_{old_conf[j]}'] #Swap columns

                # Clear phases that are not in use anymore
                old_phases = set(old_conf)  # Collect all old phases
                used_phases = set(new_conf)  # Collect all new phases
                unused_phases = old_phases - used_phases  # Find old phases not reused
                for old_phase in unused_phases:
                    P_new[f'{ean}_{old_phase}'] = 0

        self.temp_P = P_new
        return P_new
    def aggregate_feeder_load_old(self, P, t,feeder_eans):
        A = [sum(P.loc[t, f'{ean}_{p}'] for ean in feeder_eans) for p in self.avalilable_phases]
        return A
    def aggregate_feeder_load(self, P, feeder_eans):
        # Remove the t parameter since we'll process all timesteps
        A = []
        # For each phase
        for p in self.avalilable_phases:
            # Create column names for this phase
            phase_cols = [f'{ean}_{p}' for ean in feeder_eans]
            # Sum all EANs for this phase across all timesteps
            phase_sum = P[phase_cols].sum(axis=1)
            A.append(phase_sum)
        
        # Stack the phases side by side to get a Tx3 array
        return np.column_stack(A)
    
    def get_B_from_genetic(self, solution):
        B = np.zeros( (self.number_customers, len(self.Psi)) )
        for i,s in enumerate(solution):
            B[i,s] = 1
        return B.copy()
    def get_genetic_from_B(self, B):
        solution = []
        for i in range(B.shape[0]):
            solution.append(np.argmax(B[i]))  # Find the column index of the maximum value (1 in this case)
        return solution
    def check_constraint_feasible_configuration(self, B):
        for i in range(self.number_customers):
            if not np.all(B[i] <= self.B_feas[i]):
                return -1

    def objective_function(self, B, complete=True):
        num_feeders = len(self.feeders)
        P = self.change_P(B)
        loss_unbalance = 0
        loss_aggregate = 0
        
        # Process each feeder
        for f in range(num_feeders):
            # Get Tx3 array of aggregated loads for all timesteps
            A = self.aggregate_feeder_load(P, self.feeder_eans[f])  # Shape: (T, 3)
            
            # Calculate mean across phases for all timesteps
            mu = np.mean(A, axis=1)  # Shape: (T,)
            
            # Calculate unbalance loss for all timesteps
            # Reshape mu to (T,1) for broadcasting
            loss = np.square(np.abs(A - mu[:, np.newaxis])).sum(axis=1)  # Shape: (T,)
            loss_unbalance += np.sum(loss)
            
            # Calculate aggregate loss
            # self.aggregates_init[f] should be shape (T, 3)
            loss_A = np.maximum(0, A - self.aggragates_init[f])  # Shape: (T, 3)
            loss_aggregate += np.sum(loss_A)
        
        # These calculations remain unchanged as they don't involve the time dimension
        loss_changes = np.sum(B * self.B_init_opposite)
        loss_distance = np.sum(self.distances * np.sum(B * self.B_init_opposite, axis=1))
        
        # Calculate final loss
        loss = (loss_unbalance * self.scale_unbalance - 
                loss_aggregate * self.scale_aggregate + 
                (loss_changes * self.scale_changes + 
                loss_distance * self.scale_distances) * complete)
        
        # Store loss components
        self.unbalance_loss.append(loss_unbalance)
        self.associated_loss.append(loss_aggregate)
        self.changes_loss.append(loss_changes)
        self.loss_distance.append(loss_distance)
        
        return loss



    def load_time_series_at_timestep(self, P, current_net, time_step):
        power_factor = 0.98
        for i,l in current_net.asymmetric_load.iterrows():
            for p in self.avalilable_phases:
                load = P[f"{l['ean']}_{p}"].iloc[time_step] / 1000 #to convert in MW
                current_net.asymmetric_load.loc[i, f'p_{p.lower()}_mw'] = load
                current_net.asymmetric_load.loc[i, f'q_{p.lower()}_mvar'] = load * power_factor

    def run_simulations(self, B, output_path):
        lbar = tqdm(total=len(self.timesteps))
        ti = time.time()
        issues_to_consider = ['voltage', 'load_line', 'loss_line', 'load_trafo', 'loss_trafo']

        times=[]
        debug_time_executions = {'generate': [], 'ts': [], 'pf': [], 'res': []}
        P = self.change_P(B)

        results = []
        for f in range(len(self.feeders)):
            res = []
            for timestep in range(len(self.timesteps)):
                r = {}
                for issues in issues_to_consider:
                    r[issues] = [[],[],[]]
                res.append(r)
            results.append(res)
        
        net = pp.pandapowerNet(self.net.copy())

        for t, timestep in enumerate(self.timesteps):
            tii = time.time()
            self.load_time_series_at_timestep(P, net, timestep)
            debug_time_executions['ts'].append(time.time()-tii)
            
            #Run PF
            tii = time.time()
            pp.runpp_3ph(net) #asymmetric (multi phases)
            debug_time_executions['pf'].append(time.time()-tii)
            
            tii = time.time()
            #Voltage. Drop not useful elements
            for f in range(len(self.feeders)):
                buses = net.bus['feeder']==f
                lines = net.line['feeder']==f
                for i,p in enumerate(self.avalilable_phases):
                    voltages =      net.res_bus_3ph    [f'vm_{p.lower()}_pu'][buses].values
                    loading_lines = net.res_line_3ph   [f'loading_{p.lower()}_percent'][lines].values
                    loss_lines =    net.res_line_3ph   [f'p_{p.lower()}_l_mw'][lines].values
                    loading_trafo = net.res_trafo_3ph  [f'loading_{p.lower()}_percent'].values
                    loss_trafo =    net.res_trafo_3ph  [f'p_{p.lower()}_l_mw'].values

                    results[f][t][issues_to_consider[0]][i].append(voltages)
                    results[f][t][issues_to_consider[1]][i].append(loading_lines)
                    results[f][t][issues_to_consider[2]][i].append(loss_lines)
                    results[f][t][issues_to_consider[3]][i].append(loading_trafo)
                    results[f][t][issues_to_consider[4]][i].append(loss_trafo)
            debug_time_executions['res'].append(time.time()-tii)
            lbar.update(1)

        timestep = round((time.time()-ti)*100)/100
        print(f'Elapsed time: {timestep} s ({(timestep/60):.1f} m). Average: {(timestep/len(self.timesteps)):.5f} s/pf')

        lbar.close()
        return debug_time_executions, results