# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 18:13:36 2021

@author: baoj529
"""

import numpy as np
import random
from timeit import default_timer as timer

def convertobs2list(observation, all_available, ind_row, ind_col, dim1, dim2):

    # convert to list_unit, list_inlet and list_outlet
    matrix_conn = np.reshape(observation,(dim1, dim2))
    matrix_conn = matrix_conn[ind_row, :]
    matrix_conn = matrix_conn[:, ind_col]
    ind_tmp = np.where(matrix_conn == 0.5)
    matrix_conn[ind_tmp] = 0

    inlet_conn_repeat = np.sum(matrix_conn,axis=1)
    outlet_conn_repeat = np.sum(matrix_conn,axis=0)

    if np.any(inlet_conn_repeat>1) or np.any(outlet_conn_repeat>1):
        list_unit = []	
        list_inlet = []
        list_outlet = []
    else:
        ind_1, ind_2 = np.where(matrix_conn == 1)
        user_inputs_list_inlet_all = all_available.list_inlet_all[ind_row, :]
        user_inputs_list_outlet_all = all_available.list_outlet_all[ind_col, :]
        list_inlet = user_inputs_list_inlet_all[ind_1, :]
        list_outlet = user_inputs_list_outlet_all[ind_2, :]
        list_unit = np.unique(np.concatenate((list_inlet[:,0], list_outlet[:,0])))

    return list_unit, list_inlet, list_outlet

def pre_screen(list_unit, list_inlet, list_outlet, constraints_consume, constraints_satisfied):

    # start from initial score of 500
    minimum_score = -1000
    pres_score = 500
    delta_scoreA = 400  #penalty option 1
    delta_scoreB = 200   #penalty option 2
    delta_scoreC = 100   #penalty option 3
    delta_scoreD = 50   #penalty option 4
    last_score = pres_score

    ts = timer()

    # physics constraint 1: repeated connections
    if len(list_unit) == 0:
        pres_score = minimum_score
        # print('constraint 1 fails')
        constraints_satisfied[0, 0] = -1
        return pres_score, constraints_consume, constraints_satisfied

    constraints_consume[0, 0] += timer()-ts
    ts = timer()
    if pres_score == last_score:
        constraints_satisfied[0, 0] = 1
    else:
        last_score = pres_score

    # physics constraint 2: N(inlet) = N(outlet)
    #   (always satisfied)
    if len(list_inlet) != len(list_outlet):
        pres_score = pres_score-delta_scoreB
        # print('constraint 2 fails')

        # cut off list_inlet or list_outlet to make them equal-length
        len_min = min(len(list_inlet), len(list_outlet))
        ind = np.arange(len_min)
        list_inlet = list_inlet[ind]
        list_outlet = list_outlet[ind]

    constraints_consume[0, 1] += timer()-ts
    ts = timer()
    if pres_score == last_score:
        constraints_satisfied[0, 1] = 1
    else:
        last_score = pres_score

    # physics constraint 3: unit cannot connect to itself
    comp = (list_inlet[:, 0] == list_outlet[:, 0])
    if np.any(comp == True):
        pres_score = minimum_score
        # print('constraint 3 fails')
        constraints_satisfied[0, 2] = -1
        return pres_score, constraints_consume, constraints_satisfied

    constraints_consume[0, 2] += timer()-ts
    ts = timer()
    if pres_score == last_score:
        constraints_satisfied[0, 2] = 1
    else:
        last_score = pres_score

    # physics constraint 4: several units essential
    ind = np.where((list_unit == 'mixer_0') | (list_unit == 'flash_0') \
        | (list_unit == 'StReactor_1') | (list_unit == 'product_0'))
    if len(ind[0]) != 4:
        pres_score = pres_score-delta_scoreC*(4-len(ind[0]))
        # print('constraint 4 fails')

    constraints_consume[0, 3] += timer()-ts
    ts = timer()
    if pres_score == last_score:
        constraints_satisfied[0, 3] = 1
    else:
        last_score = pres_score

    # physics constraint 5: inlet cannot directly connect outlet 
    #   (repeated with other constraints)
    ind = np.where(list_outlet[:, 0] == 'mixer_0')
    if np.any(ind[0]):
        if list_inlet[ind[0], 0] in ['flash_0', 'exhaust_1', 'exhaust_2', 'product_0']:
            pres_score = minimum_score
            # print('constraint 5 fails')
            constraints_satisfied[0, 4] = -1
            return pres_score, constraints_consume, constraints_satisfied

    constraints_consume[0, 4] += timer()-ts
    ts = timer()
    if pres_score == last_score:
        constraints_satisfied[0, 4] = 1
    else:
        last_score = pres_score

    # physics constraint 6: exhaust can only connect to splitter.inlet_2 or flash 
	# product_0.inlet can only connect to flash_0.liq_outlet
    ind = np.where(list_inlet[:, 0] == 'exhaust_1')
    if np.any(ind[0]):
        if list_outlet[ind[0], 0] not in ['flash_0', 'flash_1', 'flash_2']:
            if list_outlet[ind[0], 0] not in ['splitter_1', 'splitter_2']:
                pres_score = minimum_score
                # print('constraint 6.1 fails')
                constraints_satisfied[0, 5] = -1
                return pres_score, constraints_consume, constraints_satisfied
            elif list_outlet[ind[0], 1] != 'outlet_2':
                pres_score = minimum_score
                # print('constraint 6.2 fails')
                constraints_satisfied[0, 5] = -1
                return pres_score, constraints_consume, constraints_satisfied

    ind = np.where(list_inlet[:, 0] == 'exhaust_2')
    if np.any(ind[0]):
        if list_outlet[ind[0], 0] not in ['flash_0', 'flash_1', 'flash_2']:
            if list_outlet[ind[0], 0] not in ['splitter_1', 'splitter_2']:
                pres_score = minimum_score
                # print('constraint 6.3 fails')
                constraints_satisfied[0, 5] = -1
                return pres_score, constraints_consume, constraints_satisfied
            elif list_outlet[ind[0], 1] != 'outlet_2':
                pres_score = minimum_score
                # print('constraint 6.4 fails')
                constraints_satisfied[0, 5] = -1
                return pres_score, constraints_consume, constraints_satisfied

    # # this is for methanol example only	
    # ind = np.where(list_inlet[:, 0] == 'product_0')
    # if np.any(ind[0]):
    #     if list_outlet[ind[0], 0] != 'flash_0' or list_outlet[ind[0], 1] != 'liq_outlet':
    #         pres_score = pres_score-delta_scoreA
    #         # print('constraint 6.5 fails')

    constraints_consume[0, 5] += timer()-ts
    ts = timer()
    if pres_score == last_score:
        constraints_satisfied[0, 5] = 1
    else:
        last_score = pres_score

    # physics constraint 7: liq_outlet cannot connect to compressor or expander
    ind = np.where(list_outlet[:, 1] == 'liq_outlet')
    if np.any(ind[0]):
        for ele in ind[0]: # there may be multiple flashes
            if list_inlet[ele, 0] in ['compressor_1', 'compressor_2', 'expander_1', 'expander_2']:
                pres_score = minimum_score
                # print('constraint 7 fails')
                constraints_satisfied[0, 6] = -1
                return pres_score, constraints_consume, constraints_satisfied
    
    constraints_consume[0, 6] += timer()-ts
    ts = timer()
    if pres_score == last_score:
        constraints_satisfied[0, 6] = 1
    else:
        last_score = pres_score

    # physics constraint 8: cannot connect to same kind of units
    for i in range(len(list_inlet)):
        if list_outlet[i, 0] in ['compressor_1', 'compressor_2', 'expander_1', 'expander_2'] \
            and list_inlet[i, 0] in ['compressor_1', 'compressor_2', 'expander_1', 'expander_2']:
            pres_score = minimum_score
            # print('constraint 8.1 fails')
            constraints_satisfied[0, 7] = -1
            return pres_score, constraints_consume, constraints_satisfied

        elif list_outlet[i, 0] in ['heater_1', 'heater_2', 'cooler_1', 'cooler_2'] \
            and list_inlet[i, 0] in ['heater_1', 'heater_2', 'cooler_1', 'cooler_2']:
            pres_score = minimum_score
            # print('constraint 8.2 fails')
            constraints_satisfied[0, 7] = -1
            return pres_score, constraints_consume, constraints_satisfied

        elif list_outlet[i, 0] in ['StReactor_1', 'StReactor_2'] \
            and list_inlet[i, 0] in ['StReactor_1', 'StReactor_2']:
            pres_score = minimum_score
            # print('constraint 8.3 fails')
            constraints_satisfied[0, 7] = -1
            return pres_score, constraints_consume, constraints_satisfied

    constraints_consume[0, 7] += timer()-ts
    ts = timer()
    if pres_score == last_score:
        constraints_satisfied[0, 7] = 1
    else:
        last_score = pres_score
    
    # # physics constraint 8.5: heater/cooler cannot be in the upstream of compressor/expander
    # for str_tmp in ['heater_1', 'heater_2', 'cooler_1', 'cooler_2']:
    #     ind = np.where(list_outlet[:, 0] == str_tmp)
    #     if np.any(ind[0]):
    #         if list_inlet[ind[0], 0] in ['compressor_1', 'compressor_2', 'expander_1', 'expander_2']:
    #             pres_score = pres_score-delta_scoreC
    #             print('constraint 8.5 fails')

    # physics constraint 9: splitter/flash cannot connect to mixer completely
    ind = np.where(list_inlet[:, 0] == 'mixer_1')
    if len(ind[0]) == 2:
        unique = np.unique(list_outlet[ind[0], 0])
        if len(unique) <2: # it means mixer connect to the same unit: either flash or splitter
            pres_score = minimum_score
            # print('constraint 9.1 fails')
            constraints_satisfied[0, 8] = -1
            return pres_score, constraints_consume, constraints_satisfied

    ind = np.where(list_inlet[:, 0] == 'mixer_2')
    if len(ind[0]) == 2:
        unique = np.unique(list_outlet[ind[0], 0])
        if len(unique) <2: # it means mixer connect to the same unit: either flash or splitter
            pres_score = minimum_score
            # print('constraint 9.2 fails')
            constraints_satisfied[0, 8] = -1
            return pres_score, constraints_consume, constraints_satisfied

    constraints_consume[0, 8] += timer()-ts
    ts = timer()
    if pres_score == last_score:
        constraints_satisfied[0, 8] = 1
    else:
        last_score = pres_score

    # physics constraint 10: for each unit, all inlets and outlets must be selected
    for str_tmp in list_unit:
        if str_tmp not in ['mixer_0', 'exhaust_1', 'exhaust_2', 'product_0']:
            ind_1 = np.where(list_inlet[:, 0] == str_tmp)
            ind_2 = np.where(list_outlet[:, 0] == str_tmp)
            if str_tmp in ['mixer_1', 'mixer_2']:
                if len(ind_1[0])<2 or len(ind_2[0])<1:
                    pres_score = pres_score-delta_scoreD
                    # print('contraint 10.1 fails')
            elif str_tmp in ['flash_0', 'flash_1', 'flash_2', 'splitter_1', 'splitter_2']:
                if len(ind_1[0])<1 or len(ind_2[0])<2:
                    pres_score = pres_score-delta_scoreD
                    # print('contraint 10.2 fails')
            else: #all the rest units have one inlet and one outlet
                if len(ind_1[0])<1 or len(ind_2[0])<1:
                    pres_score = pres_score-delta_scoreD
                    # print('contraint 10.3 fails')

    constraints_consume[0, 9] += timer()-ts
    ts = timer()
    if pres_score == last_score:
        constraints_satisfied[0, 9] = 1
    else:
        last_score = pres_score

    # physics constraint 11: all units must be in the same cycle
    connect2feed, unit_1_downstream, unit_2_downstream = \
        all_units_downstream(list_inlet[:, 0], list_outlet[:, 0], \
            'mixer_0', 'StReactor_1', 'StReactor_2')
    # print('constraint 11 debug-connect2feed: ', connect2feed)
    # print('constraint 11 debug-unit_1_downstream: ', unit_1_downstream)
    # print('constraint 11 debug-unit_2_downstream: ', unit_2_downstream)

    if 0 in connect2feed:
        # print('constraint 11 fails')
        pres_score = pres_score-delta_scoreB

    constraints_consume[0, 10] += timer()-ts
    ts = timer()
    if pres_score == last_score:
        constraints_satisfied[0, 10] = 1
    else:
        last_score = pres_score

    # physics constraint 12: it must be in the sequence: feed -> reactor -> (flash -> product) and exhaust
    if 'flash_0' in list_unit:
        if 'flash_0' not in unit_1_downstream and 'flash_0' not in unit_2_downstream:
            # print('contraint 12.1 fails')
            pres_score = pres_score-delta_scoreC

    if 'exhaust_1' in list_unit:
        if 'exhaust_1' not in unit_1_downstream and 'exhaust_1' not in unit_2_downstream:
            # print('contraint 12.2 fails')
            pres_score = pres_score-delta_scoreC
    
    if 'exhaust_2' in list_unit:
        if 'exhaust_2' not in unit_1_downstream and 'exhaust_2' not in unit_2_downstream:
            # print('contraint 12.3 fails')
            pres_score = pres_score-delta_scoreC

    constraints_consume[0, 11] += timer()-ts
    if pres_score == last_score:
        constraints_satisfied[0, 11] = 1
    else:
        last_score = pres_score

    return pres_score, constraints_consume, constraints_satisfied

def all_units_downstream(units_in, units_out, unit_0, unit_1, unit_2):  
    # identify if the outlet units are connect to unit_0
    connected = np.zeros(len(units_out), dtype=int) # numpy array: int
    unit_1_downstream = [] # list: string
    unit_2_downstream = [] # list: string
    unit_1_record = False
    unit_2_record = False
    
    # mark every children, grand children, ... connected to unit_0
    ind_1 = np.where(units_out == unit_0)
    
    if np.any(connected[ind_1[0]]==0):
        connected[ind_1[0]] = 1
        for i1 in ind_1[0]:
            ind_2 = np.where(units_out == units_in[i1])
            if unit_1 == units_in[i1]:
                unit_1_record = True
            if unit_2 == units_in[i1]:
                unit_2_record = True
            if unit_1_record:
                unit_1_downstream.append(units_in[i1])
            if unit_2_record:
                unit_2_downstream.append(units_in[i1])
            if np.any(connected[ind_2[0]]==0):
                connected[ind_2[0]] = 1
                for i2 in ind_2[0]:
                    ind_3 = np.where(units_out == units_in[i2])
                    if unit_1 == units_in[i2]:
                        unit_1_record = True
                    if unit_2 == units_in[i2]:
                        unit_2_record = True
                    if unit_1_record:
                        unit_1_downstream.append(units_in[i2])
                    if unit_2_record:
                        unit_2_downstream.append(units_in[i2])
                    if np.any(connected[ind_3[0]]==0):
                        connected[ind_3[0]] = 1
                        for i3 in ind_3[0]:
                            ind_4 = np.where(units_out == units_in[i3])
                            if unit_1 == units_in[i3]:
                                unit_1_record = True
                            if unit_2 == units_in[i3]:
                                unit_2_record = True
                            if unit_1_record:
                                unit_1_downstream.append(units_in[i3])
                            if unit_2_record:
                                unit_2_downstream.append(units_in[i3])
                            if np.any(connected[ind_4[0]]==0):
                                connected[ind_4[0]] = 1
                                for i4 in ind_4[0]:
                                    ind_5 = np.where(units_out == units_in[i4])
                                    if unit_1 == units_in[i4]:
                                        unit_1_record = True
                                    if unit_2 == units_in[i4]:
                                        unit_2_record = True
                                    if unit_1_record:
                                        unit_1_downstream.append(units_in[i4])
                                    if unit_2_record:
                                        unit_2_downstream.append(units_in[i4])
                                    
                                    if np.any(connected[ind_5[0]]==0):
                                        connected[ind_5[0]] = 1
                                        for i5 in ind_5[0]:
                                            ind_6 = np.where(units_out == units_in[i5])
                                            if unit_1 == units_in[i5]:
                                                unit_1_record = True
                                            if unit_2 == units_in[i5]:
                                                unit_2_record = True
                                            if unit_1_record:
                                                unit_1_downstream.append(units_in[i5])
                                            if unit_2_record:
                                                unit_2_downstream.append(units_in[i5])
                                            if np.any(connected[ind_6[0]]==0):
                                                connected[ind_6[0]] = 1
                                                for i6 in ind_6[0]:
                                                    ind_7 = np.where(units_out == units_in[i6])
                                                    if unit_1 == units_in[i6]:
                                                        unit_1_record = True
                                                    if unit_2 == units_in[i6]:
                                                        unit_2_record = True
                                                    if unit_1_record:
                                                        unit_1_downstream.append(units_in[i6])
                                                    if unit_2_record:
                                                        unit_2_downstream.append(units_in[i6])
                                                    if np.any(connected[ind_7[0]]==0):
                                                        connected[ind_7[0]] = 1
                                                        for i7 in ind_7[0]:
                                                            ind_8 = np.where(units_out == units_in[i7])
                                                            if unit_1 == units_in[i7]:
                                                                unit_1_record = True
                                                            if unit_2 == units_in[i7]:
                                                                unit_2_record = True
                                                            if unit_1_record:
                                                                unit_1_downstream.append(units_in[i7])
                                                            if unit_2_record:
                                                                unit_2_downstream.append(units_in[i7])
                                                            if np.any(connected[ind_8[0]]==0):
                                                                connected[ind_8[0]] = 1
                                                                for i8 in ind_8[0]:
                                                                    ind_9 = np.where(units_out == units_in[i8])
                                                                    if unit_1 == units_in[i8]:
                                                                        unit_1_record = True
                                                                    if unit_2 == units_in[i8]:
                                                                        unit_2_record = True
                                                                    if unit_1_record:
                                                                        unit_1_downstream.append(units_in[i8])
                                                                    if unit_2_record:
                                                                        unit_2_downstream.append(units_in[i8])
                                                                    if np.any(connected[ind_9[0]]==0):
                                                                        connected[ind_9[0]] = 1
                                                                        for i9 in ind_9[0]:
                                                                            ind_10 = np.where(units_out == units_in[i9])
                                                                            if unit_1 == units_in[i9]:
                                                                                unit_1_record = True
                                                                            if unit_2 == units_in[i9]:
                                                                                unit_2_record = True
                                                                            if unit_1_record:
                                                                                unit_1_downstream.append(units_in[i9])
                                                                            if unit_2_record:
                                                                                unit_2_downstream.append(units_in[i9])
                                                                            if np.any(connected[ind_10[0]]==0):
                                                                                connected[ind_10[0]] = 1
                                                                                for i10 in ind_10[0]:
                                                                                    ind_11 = np.where(units_out == units_in[i10])
                                                                                    if unit_1 == units_in[i10]:
                                                                                        unit_1_record = True
                                                                                    if unit_2 == units_in[i10]:
                                                                                        unit_2_record = True
                                                                                    if unit_1_record:
                                                                                        unit_1_downstream.append(units_in[i10])
                                                                                    if unit_2_record:
                                                                                        unit_2_downstream.append(units_in[i10])
                                                                                    if np.any(connected[ind_11[0]]==0):
                                                                                        connected[ind_11[0]] = 1
                                                                                        for i11 in ind_11[0]:
                                                                                            ind_12 = np.where(units_out == units_in[i11])
                                                                                            if unit_1 == units_in[i11]:
                                                                                                unit_1_record = True
                                                                                            if unit_2 == units_in[i11]:
                                                                                                unit_2_record = True
                                                                                            if unit_1_record:
                                                                                                unit_1_downstream.append(units_in[i11])
                                                                                            if unit_2_record:
                                                                                                unit_2_downstream.append(units_in[i11])
                                                                                            if np.any(connected[ind_12[0]]==0):
                                                                                                connected[ind_12[0]] = 1
    
    # return the identification
    return connected, unit_1_downstream, unit_2_downstream

#%%
class fs_gen():
    def __init__(self, n_rows, n_cols, n_features):
        self.num_elements =     n_cols # n_cols
        self.MAX_Iteration =    n_rows # n_rows
        self.n_features =       n_features # n_cols*n_rows
        self.action_space =     np.arange(self.num_elements-1)# remove the last column: step counter
        self.n_actions  =       len(self.action_space)

    def reset(self,episode):
        s = np.zeros(self.n_features,dtype=float)
        picked_true_fs=random.randint(0,1)
        self.step_counter = 0
        return s, picked_true_fs

    def update_env(self,S,old_obs_,i_step):
        new_obs_ = np.copy(old_obs_)
        
        # step counter always filled
        new_obs_[i_step*self.num_elements+self.num_elements-1] = 1.0 

        unavailable = False
        noaction = False
        if new_obs_[i_step*self.num_elements+S] == 0.5:
            new_obs_[i_step*self.num_elements+S] = 1.0
            # new_obs_[i_step*self.num_elements+S] = new_obs_[i_step*self.num_elements+S]*2
            if S == self.n_actions-1:
                noaction = True
        else: # if step on an unavailable spot
            unavailable = True
            new_obs_[S] = 1.0
        
        return new_obs_, unavailable, noaction

    def step(self,action,old_obs,episode,i_step,all_available,constraints_consume,\
        reward,extra_reward,ind_row,ind_col,N_noaction,N_unavailable,system_complexity,\
            constraints_satisfied):

        # penalty options
        minimum_score = -1000
        pres_score = 500
        if system_complexity == 4:
            delta_score_noaction = 400
        elif system_complexity == 3:
            delta_score_noaction = 320
        elif system_complexity == 2:
            delta_score_noaction = 240
        elif system_complexity == 1:
            delta_score_noaction = 160
        else:
            delta_score_noaction = 0

        # update observation
        S=action
        new_obs, unavailable, noaction = self.update_env(S,old_obs,i_step)

        # determine reward accordingly
        if unavailable == True: 
            N_unavailable += 1
            # 1. if step on unavailable spot
            R = minimum_score
            extra_R = -800.0
        elif noaction == True:
            # 2. if step on no-action spot
            # N_noaction += 1
            if i_step in ind_row:
                # 2.1 the i_step row is partially filled
                N_noaction += 1
                R = reward
                extra_R = -delta_score_noaction # may assign penalty later
            else:
                # 2.2 the i_step row is empty
                R = reward
                extra_R = 0.0 # may assign reward later, should have reward
        else:
            # 3. if step on available spot
            list_unit, list_inlet, list_outlet = convertobs2list(new_obs, all_available, ind_row, ind_col, len(all_available.list_inlet_all), len(all_available.list_outlet_all)+2)
            R, constraints_consume, constraints_satisfied = pre_screen(list_unit, list_inlet, list_outlet, constraints_consume, constraints_satisfied)
            extra_R = 0.0

        # evaluate reward
        if R >= pres_score:
            pass_pre_screen = True
        else:
            pass_pre_screen = False

        episode_done = False
        if R <= minimum_score or i_step > self.MAX_Iteration-2:
            episode_done = True

        return new_obs,R,extra_R,episode_done,pass_pre_screen,constraints_consume,N_noaction,N_unavailable,constraints_satisfied