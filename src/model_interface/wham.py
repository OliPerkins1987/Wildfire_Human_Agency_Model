# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 10:41:39 2021

@author: Oli
"""


import agentpy as ap
import numpy as np
import pandas as pd
from copy import deepcopy
from dask.distributed import Client
from collections import defaultdict


from Core_functionality.AFTs.agent_class import AFT
from Core_functionality.AFTs.arable_afts import Swidden, SOSH, MOSH, Intense_arable
from Core_functionality.AFTs.livestock_afts import Pastoralist_r, Pastoralist_p,Ext_LF_r, Int_LF_r, Ext_LF_p, Int_LF_p
from Core_functionality.AFTs.forestry_afts import Hunter_gatherer_f, Agroforestry, Logger, Managed_forestry, Abandoned_forestry  
from Core_functionality.AFTs.nonex_afts import Hunter_gatherer_n, Recreationalist, SLM, Conservationist
from Core_functionality.AFTs.land_system_class import land_system
from Core_functionality.AFTs.land_systems import Cropland, Pasture, Rangeland, Forestry, Urban, Unoccupied, Nonex

from Core_functionality.top_down_processes.arson import arson
from Core_functionality.top_down_processes.background_ignitions import background_rate
from Core_functionality.top_down_processes.fire_constraints import fuel_ct, dominant_afr_ct

from Core_functionality.Trees.Transfer_tree import define_tree_links, update_pars, predict_from_tree_fast
from Core_functionality.prediction_tools.regression_families import regression_link, regression_transformation

from output_analysis.utility import get_afr_vals
from output_analysis.ncdf_write_func import write_nc


###################################################################

### Core model class

###################################################################


class WHAM(ap.Model):

    def setup(self):
        
        ### Multi-processor
        if self.p.bootstrap == True:
            
            self.client = Client(n_workers=self.p.n_cores)
        
        # Spatio-temporal boundaries
        self.xlen     = self.p.xlen
        self.ylen     = self.p.ylen
        self.timestep = self.p.start_run

        # Create grid
        self.grid = ap.Grid(self, (self.xlen, self.ylen), track_empty=False)
        self.Area = np.array(self.p.Maps['Area']).reshape(self.p.ylen, self.p.xlen)
        
        # Create land systems
        self.ls     = ap.AgentList(self, 
                       [y[0] for y in [ap.AgentList(self, 1, x) for x in self.p.LS]])
        
        # Create AFTs
        self.agents = ap.AgentList(self, 
                       [y[0] for y in [ap.AgentList(self, 1, x) for x in self.p.AFTs]])
        
        if self.p.Policy == True:
        
            # Create Observers
            self.Observers = dict(zip([x for x in self.p.Observers.keys()], 
                                  [ap.AgentList(self, 1, y) for y in self.p.Observers.values()]))
        

        ########################################
        ### Agent class setup
        ########################################
        
        ### LS
        self.ls.setup()
        self.ls.get_pars(self.p.AFT_pars)
        self.ls.get_boot_vals(self.p.AFT_pars)
        
        ### AFTs
        self.agents.setup()
        self.agents.get_dist_pars(self.p.AFT_pars)
        
        ### Fire use
        
        if self.p.AFT_fire == True:
        
            self.agents.get_fire_pars()
        
        ### Fire observers
        
        if self.p.Policy == True:
    
            ### Observers
            for observer in self.Observers.keys():
            
                self.Observers[observer].setup()
        
            if 'background_rate' in self.Observers.keys():
            
                self.Observers['background_rate'].get_fire_pars()
        
            if 'arson' in self.Observers.keys():
            
                self.Observers['arson'].get_fire_pars()
        
        
        ### Nitrogen use
        
        if self.p.AFT_Nfer == True:
            
            self.agents.get_Nfer_pars()
        
        
        ### Results containers
        self.results = {}
        
        for i in self.p.reporters:
            
            self.results[i] = []
        
    
    def go(self):
        
        while self.timestep <= self.p.end_run:
    
            self.step()
            print(self.timestep)
            self.timestep += 1
        
        if self.p.bootstrap == True:
            
            self.client.close()
        
            
    ########################################################################
    
    ### AFT distribution functions
    
    ########################################################################
    
    def allocate_X_axis(self):
        
        ### Gather X-axis vals from land systems       
        ls_scores    = dict(zip([type(x).__name__ for x in self.ls], 
                        [x.Dist_vals for x in self.ls]))
        
        #################################################
        ### Perform calculation to get X_axis
        #################################################
        
        ### 1) Competition between Unoccupied and forestry
        unoc_habitat            =  (ls_scores['Cropland'] + ls_scores['Pasture'] + ls_scores['Rangeland'] + ls_scores['Urban']) <= self.p.theta
        ls_scores['Unoccupied'] =  ls_scores['Unoccupied'] * unoc_habitat
        ls_scores['Forestry']   =  np.select([ls_scores['Unoccupied'] > 0.5, ls_scores['Unoccupied'] <= 0.5], 
                                    [ls_scores['Forestry'] * 0.5, ls_scores['Forestry']], default = 0)
            
        ### Calc forestry
        ls_scores['Forestry'] =  ls_scores['Forestry'] * (self.p.Maps['Forest'].data[self.timestep, :, :]).reshape(self.xlen * self.ylen)
        ls_scores['Forestry'] =  np.select([ls_scores['Forestry']>0], [ls_scores['Forestry']], default = 0)
        
        ### 2) Competition between Unoccupied and nonex
        ### Get remaining vegetation fraction
        Open_vegetation                =  self.p.Maps['Mask'] - ls_scores['Cropland'] - ls_scores['Pasture'] - ls_scores['Rangeland'] - ls_scores['Forestry'] - ls_scores['Urban']
        Open_vegetation                =  np.array([x if x >=0 else 0 for x in Open_vegetation])
        
        ### calc nonex & unoccupied
        ls_scores['Unoccupied']        =  Open_vegetation * ls_scores['Unoccupied']
        ls_scores['Nonex']             =  Open_vegetation - ls_scores['Unoccupied']
                
        ### 3) re-scale against Mask: coastal pixels
        ls_frame                       = pd.DataFrame(ls_scores)
        ls_frame['tot']                = self.p.Maps['Mask'] / ls_frame.sum(axis = 1) 
        ls_frame.iloc[:, 0:-1  ]       = ls_frame.iloc[:,0:-1].multiply(ls_frame.tot, axis="index")                            
        ls_frame                       = ls_frame.iloc[:, 0:-1].to_dict('series')
       
        ### reshape and stash
        self.X_axis                    =  dict(zip([x for x in ls_frame.keys()], 
                                            [np.array(x).reshape(self.ylen, self.xlen) for x in ls_frame.values()]))
               
    def allocate_AFT(self):
        
        ''' combines AFT competitiveness scores to allocate cell space'''""
        
        self.AFT_scores = {}
        
        ### Gather competitiveness scores from AFTs
        land_systems = [y for y in pd.Series([x for x in self.agents.ls]).unique()]
        aft_scores   = {}
        aft_habitats = {}
    
        if type(land_systems) == str:
            land_systems = [land_systems] #catch the case where only 1 ls type
            
        for l in land_systems:
            
            ### get predictions
            aft_scores[l] = dict(zip([type(x).__name__ for x in self.agents if x.ls == l], 
                                     [x.Dist_vals for x in self.agents if x.ls == l]))
            
            ### get aft habitats
            aft_habitats[l] = dict(zip([type(x).__name__ for x in self.agents if x.ls == l], 
                                     [x.Habitat_vals for x in self.agents if x.ls == l]))

            ### apply giving in habitats
            aft_scores[l] = self.giving_in(aft_scores[l], aft_habitats[l])

            ### calculate total by land system by cell
            tot_y         = np.add.reduce([x for x in aft_scores[l].values()])
            
            ### divide by total & reshape to world map
            for k in aft_scores[l].keys():
                
                aft_scores[l][k]   = np.array(aft_scores[l][k]/tot_y).reshape(self.ylen, self.xlen)
                aft_scores[l][k]   = aft_scores[l][k] * self.X_axis[l]
                self.AFT_scores[k] = aft_scores[l][k]
        
        self.calc_AFR(aft_scores)
       
        
    def giving_in(self, aft_dict, habitat_dict):
        
        ''' removes AFT distribution outside of boolean habitat
        conditional on there being another AFT to occupy the space'''
        
        habitat_dist = {}
        
        for z in [x for x in aft_dict.keys()]:
        
            habitat_dist[z] = [aft_dict[z] * habitat_dict[z]]
        
        habitat_sum = np.nansum(np.array([x for x in habitat_dist.values()]), axis = 0)
                
        #######################################
        ### apply habitat
        #######################################
        
        for z in [x for x in aft_dict.keys()]:
            
            aft_dict[z] = np.select([habitat_sum > 0], habitat_dist[z], default = aft_dict[z])
            
        return(aft_dict)
        
            
        
    def calc_AFR(self, aft_dist_by_ls):
        
        '''' voting mechanism by AFT to define AFR coverage '''
        
        self.LFS = {}
        
        for l in aft_dist_by_ls.keys():
            
            afr_temp = dict(zip([x for x in aft_dist_by_ls[l].keys()], 
                                [a.afr for a in self.agents if type(a).__name__ in aft_dist_by_ls[l].keys()]))
        
            afr_dict = defaultdict(list)
            
            [afr_dict[value].append(key) for key, value in afr_temp.items()]
        
            for key, value in afr_dict.items():
                
                for r in range(len(afr_dict[key])):
                    
                    afr_dict[key][r] = aft_dist_by_ls[l][value[r]]
                    
                afr_dict[key] = np.add.reduce(afr_dict[key])
        
            self.LFS[l] = dict(afr_dict)
                
        ### stash afr scores
        self.AFR        = get_afr_vals(self.LFS)
    
    
    ###################################################################################
    
    ### Fire use functions
    
    ###################################################################################
    
    def calc_BA(self, group_lc):
        
        ''' gathers deliberate fire and multiplies by AFT coverage'''
        
        
        ### gather and some fire types across AFTs
        
        self.Managed_fire = {}
        self.Managed_igs  = {}
        
        for i in self.p.Fire_types.keys():
            
            self.Managed_fire[i] = {}
            self.Managed_igs[i]  = {}
            
            for a in self.agents:
                    
                if i in a.Fire_vals.keys():
                    
                    self.Managed_fire[i][a] = np.array(a.Fire_vals[i]).reshape(self.p.ylen, self.p.xlen)
                    self.Managed_fire[i][a] = self.Managed_fire[i][a] * self.AFT_scores[type(a).__name__]
                    self.Managed_igs[i][a]  = self.Managed_fire[i][a] / (a.Fire_use[i]['size'] / 100) ##size(ha) -> size(km2)
            
            
            self.Managed_fire[i] = np.nansum([x for x in self.Managed_fire[i].values()], 
                                                 axis = 0)
            
            self.Managed_igs[i]  = np.nansum([x for x in self.Managed_igs[i].values()], 
                                                 axis = 0)
            
            
        #################################
        ### Calculate deforestation fire
        #################################
        
        if 'deforestation' in self.Observers.keys():
        
            self.Observers['deforestation'][0].clear_vegetation()
            self.Managed_fire['defor'] = self.Observers['deforestation'][0].VC_vals
            self.Managed_igs['defor']  = self.Observers['deforestation'][0].VC_igs    
                
        
        #######################################
        ### Divide outputs by seasonality map
        #######################################
        
        for i in self.p.Fire_types.keys():
        
            if self.p.Seasonality == True:
            
                self.Managed_fire[i] = self.p.Fire_seasonality[i].data * self.Managed_fire[i]
                self.Managed_igs[i]  = self.p.Fire_seasonality[i].data * self.Managed_igs[i]
        
          
        #################################
        ### apply constraints
        #################################
        
        if self.p.apply_fire_constraints == True:
            
            self.fire_constraints()


        ### Total up managed fire
        self.Managed_fire['Total']  = np.nansum([x for x in self.Managed_fire.values() if type(x) != np.float64], 
                                                 axis = 0)
        

################################################################
### Constraints on fire not captured by DAFI / AFT calculations
################################################################
        
        
    def fire_constraints(self):
        
        '''top down constraints on fire'''
        
        for c in self.Observers.values():
            
            if 'ct' in type(c[0]).__name__:
                
                c.constrain()
        
    
    #######################################################################
    
    ### Unmanaged fire
    
    #######################################################################
    
    
    def calc_background_ignitions(self):
        
        ''' Accidental ignitions'''
        
        ### Get background rate
        self.Background_ignitions = np.array(self.Observers['background_rate'].Fire_vals[0]).reshape(self.ylen, self.xlen)


    def calc_arson(self):

        ''' Arson '''        

        ### Get arson
        self.Arson                = np.array(self.Observers['arson'].Fire_vals[0]).reshape(self.ylen, self.xlen)
        
        
    def calc_escaped_fires(self):
        
        if self.p['escaped_fire'] == True:
        
            ####################################################
            ### 1) calculate base escaped rates by fire type
            ####################################################
        
        
            self.Escaped_fire = {}
            base = self.model.p.AFT_pars['Fire_escape']['Overall']
        
        
            for i in self.Managed_igs.keys():
                
                base_rate            = base.loc[base['Intention'] == i,'Base_escape_rate'].iloc[0]
                self.Escaped_fire[i] = self.Managed_igs[i] * base_rate         
    
    
            ####################################################
            ### 2) caculate impact of fire control
            ####################################################
            
            ### fire control distribution
            self.Observers['fire_control_measures'].control()
        
            ##############################
            ### impact of fire control
            ##############################
            
            control_impact = self.model.p.AFT_pars['Fire_escape']['Overall']
            control_weights= {}
            
            for a in self.Observers['fire_control_measures']:
                
                for f in a.Control_vals.keys():
                    
                    # filt data for fire
                    ctl_filt = control_impact.loc[control_impact['Intention'] == f,:]
                    
                    # combine impact of controlled & uncontrolled
                    controlled = (1-a.Control_vals[f]) *  ctl_filt.loc[ctl_filt['Controlled'] == True, 'Escape_weight'].iloc[0]
                    no_control = (a.Control_vals[f]) *  ctl_filt.loc[ ctl_filt['Controlled'] == False, 'Escape_weight'].iloc[0]
                    
                    # sum and stash
                    ctl_w              = controlled+no_control
                    control_weights[f] = np.array(ctl_w).reshape(self.p.ylen, self.p.xlen)
                    
            ####################################################
            ### 3) Combine
            ####################################################
            
            for f in self.Escaped_fire.keys():
                
                self.Escaped_fire[f] = self.Escaped_fire[f] * control_weights[f]
                
    
    ###############################################
    
    ### Suppression of unmanaged ignitions
    ### Currently calculated offline
    
    ###############################################
    
    def suppress_fire(self):
    
        pass
    
    
    ###############################################
    
    ### Nitrogen
    
    ###############################################
    
    def calc_Nfer(self):
        
        Nfer_scores   = {}
        AFT_Nuser     = [x for x in self.agents if x.Nfer_use != {}]
        
        for a in AFT_Nuser:
            
            a.Nfer_vals = (np.array(a.Nfer_vals).reshape(self.ylen, self.xlen) * 
                                self.AFT_scores[type(a).__name__])
                        
        for l in ['Cropland', 'Pasture']:
            
            ### get predictions
            Nfer_scores[l] = dict(zip([type(x).__name__ for x in AFT_Nuser if x.ls == l], 
                                     [x.Nfer_vals for x in AFT_Nuser if x.ls == l]))
                        
            
            ### calculate total by land system by cell
            Nfer_scores[l] = np.nansum([x for x in Nfer_scores[l].values()], axis = 0)
        
        ### add linear correction?
        self.Nitrogen_fertiliser = Nfer_scores
    
    
    
    #####################################################################################
    
    ### scheduler
    
    #####################################################################################
    
    def step(self):
        
        ### ls distribution
        self.ls.get_vals()
        self.allocate_X_axis()

        ### aft distribution
        self.agents.compete()
        self.allocate_AFT()
        
        ####################################################
        ### Fire use
        ####################################################
        
        if self.p.AFT_fire == True:
            
            self.agents.fire_use()
            self.calc_BA(group_lc = False)
        
            #################################################
            ### Background & arson ignitions
            #################################################
        
            if 'background_rate' in self.Observers.keys():
        
                self.Observers['background_rate'].ignite()
                self.calc_background_ignitions()
                
            if 'arson' in self.Observers.keys():
        
                self.Observers['arson'].ignite()
                self.calc_arson()
                       
            ### Suppression
            self.agents.fire_suppression()
            self.suppress_fire()
            self.calc_escaped_fires()
        
        ####################################################
        ### Nitrogen
        ####################################################
        
        if self.p.AFT_Nfer == True:
        
            self.agents.nfer_use()
            self.calc_Nfer()
        
        ### update
        self.update()
    
    
    def update(self):
                
        self.record()        ### store data in model object
        self.write_out()     ### write data to disk
        
    
    ####################################################################
    
    ### Reporters
    
    ####################################################################
    
    def record(self):
        
        ''' stash reported variables in RAM '''
        
        for i in self.p.reporters:
        
            self.results[i].append(deepcopy(self[i]))        
            
    
    def write_out(self):
        
        '''choose data to write'''
                
        if self.p.write_annual == True:
            
            for i in self.results.keys():
    
                if type(self.results[i][self.timestep])  == dict:
    
                    for file in self.results[i][self.timestep].keys():
            
                        ### compile data
                        path = self.p.write_fp + '\\' + i + '_' + file + '_' + str(self.timestep) + '.nc'
                        data = self.results[i][self.timestep][file]
        
                        ### write out
                        write_nc(fn = path, annual = True,
                                 vals = data, mod = self)
            
            
                elif type(self.results[i][self.timestep]) == np.ndarray:
        
                    ### compile data
                    path = self.p.write_fp + '\\' + i + '_' + str(self.timestep) + '.nc'
                    data = self.results[i][self.timestep]
        
                    ### write out
                    write_nc(fn = path, annual = True,
                      vals = data, mod = self)
    
    
    
    ####################################################################
    
    ### End conditions
    
    ####################################################################
    
    def end(self):
        
        pass
    
          

