import numpy as np
import os
import MET_IDAES_v16 as IDAES
sys.path.append(os.path.abspath("../"))
from RL_ENV import pre_screen, convertobs2list
import matplotlib.pyplot as plt

def post_processing(user_inputs, all_available, P):

    ind_case = P['index']
    unit_pool = P['unit_pool']
    dir = P['dir']
    run_idaes = P['run_idaes']
    run_idaes_target = P['run_idaes_target']
    run_idaes_save = P['run_idaes_save']
    visualize_idaes = P['visualize_idaes']
    total_iterations = P['total_iterations']
    N_save = []

    temp1 = ['.'.join(strings) for strings in all_available.list_inlet_all]
    temp2 = ['.'.join(strings) for strings in user_inputs.list_inlet_all]
    ind_row = [temp1.index(i) for i in temp2] # available rows/inlets
    temp1 = ['.'.join(strings) for strings in all_available.list_outlet_all]
    temp2 = ['.'.join(strings) for strings in user_inputs.list_outlet_all]
    ind_col = [temp1.index(i) for i in temp2] # available columns/outlets

#%% load RL results

    print('\nPostProcessing of index ', ind_case, ' for ', unit_pool, ' unit-pool')
    
    filename = dir+'training_time_'+str(ind_case)+'.csv'
    if os.path.exists(filename):
        training_time = np.loadtxt(filename, dtype=float, delimiter=',')
        print('time consuming: ', training_time)
        [t1, t2, t3] = training_time
    else:
        print('simulation not finished')
        t1 = 0; t2 = 0; t3 = 0

    filename = dir+'obs_runIdaes_flowsheet_'+str(ind_case)+'.csv'
    if not os.path.exists(filename):
        print('no pass-pre-screen flowsheets saved')
    else:
        filename = dir+'obs_runIdaes_flowsheet_'+str(ind_case)+'.csv'
        obs_idaes = np.loadtxt(filename, dtype=int, delimiter=',')
        
        filename = dir+'obs_runIdaes_reward_'+str(ind_case)+'.csv'
        reward_idaes = np.loadtxt(filename, dtype=int, delimiter=',')
        
        filename = dir+'obs_runIdaes_extra_reward_'+str(ind_case)+'.csv'
        extra_reward_idaes = np.loadtxt(filename, dtype=int, delimiter=',')
        
        filename = dir+'obs_runIdaes_costing_'+str(ind_case)+'.csv'
        cost_idaes = np.loadtxt(filename, dtype=float, delimiter=',')
        
        filename = dir+'obs_runIdaes_status_'+str(ind_case)+'.csv'
        status_idaes = np.loadtxt(filename, dtype=float, delimiter=',')
        
        tt_reward_idaes = reward_idaes+extra_reward_idaes
        index_iteration = status_idaes[:, -1].astype(int)
        flowrate_idaes = status_idaes[:, 1]

        # extract all feasible observations
        if reward_idaes.size == 1:
            tmp = obs_idaes
            obs_idaes = np.reshape(tmp, (tmp.size, reward_idaes.size))
            tmp = reward_idaes
            reward_idaes = np.reshape(tmp, (tmp.size, ))
            tmp = extra_reward_idaes
            extra_reward_idaes = np.reshape(tmp, (tmp.size, ))
            tmp = cost_idaes
            cost_idaes = np.reshape(tmp, (tmp.size, reward_idaes.size))
            tmp = status_idaes
            status_idaes = np.reshape(tmp, (tmp.size, reward_idaes.size))

        ind_feasi = np.where((reward_idaes>=5000) & (extra_reward_idaes>=0))
        if len(ind_feasi[0])>0:
            obs_feasi = obs_idaes[ind_feasi]
            reward_feasi = reward_idaes[ind_feasi]
            extra_reward_feasi = extra_reward_idaes[ind_feasi]
            cost_feasi = cost_idaes[ind_feasi]
            status_feasi = status_idaes[ind_feasi]
            
            N_idaes = len(reward_idaes)
            N_feasi = len(reward_feasi)
            max_iteration = status_idaes[-1, -1]
            max_perc = max_iteration/total_iterations*100 #[%]
            index_iteration_feasi = index_iteration[ind_feasi].astype(int)

            #%% visualize RL results
            # plot number of inlets, number of outlet
            inlet_num = []
            outlet_num = []
            for ii in range(N_feasi):
                n_inlets = len(all_available.list_inlet_all)
                n_outlets = len(all_available.list_outlet_all)
                temp_matrix = np.reshape(obs_feasi[ii,:], (n_inlets, n_outlets))
    
                inlet_count = 0
                outlet_count = 0
                for k in range(n_inlets):
                    if np.sum(temp_matrix[k,:]) == 1:
                        inlet_count += 1
                for k in range(n_outlets):  
                    if np.sum(temp_matrix[:,k]) == 1:
                        outlet_count += 1 
                inlet_num.append(inlet_count)
                outlet_num.append(outlet_count)
                
            plt.figure()
            plt.plot(np.arange(len(inlet_num)), inlet_num)
            plt.ylabel('# of connections')
            plt.title('index_'+str(ind_case))
            plt.savefig(dir+'N_connections_'+str(ind_case)+'.png')

            # plot scatter points of cost_mol vs flow rate
            x = status_feasi[:, 0]/(365*24*3600)
            y = status_feasi[:, 1]*1.0
            text = list(range(1, len(x)+1))
            
            fig, ax = plt.subplots()
            ax.scatter(x, y)
            for k, txt in enumerate(text):
                ax.annotate(txt, (x[k], y[k]))
            plt.ylabel('methanol flow rate [mol/s]')
            plt.xlabel('cost per mol [$/mol]')
            plt.title('index_'+str(ind_case))
            plt.savefig(dir+'status_'+str(ind_case)+'.png')
            
            plt.figure()
            plt.plot(np.arange(len(x)), x)
            plt.ylabel('cost per mol  [$/mol]')
            
            plt.figure()
            plt.plot(np.arange(len(y)), y)
            plt.ylabel('methanol flow rate [mol/s]')
            
            print('optimal flowsheets:')
            # print(status_feasi)
            # print(cost_feasi)
            print('# of cases to IDAES: ', N_idaes)
            print('# of feasible cases: ', N_feasi)
            if len(inlet_num)>0:
                print('# of inlets: ', max(inlet_num))
                print('# of outlets:', max(outlet_num))
                N_max_conn = max(inlet_num)
            else:
                N_max_conn = 0
            
            #%%
            N_idaes_hist = np.arange(len(index_iteration))
            N_feasi_hist = np.arange(len(index_iteration_feasi))
            index_iteration_feasi = np.insert(index_iteration_feasi, 0, index_iteration_feasi[0]-1)
            index_iteration_feasi = np.insert(index_iteration_feasi, 0, 0)
            N_feasi_hist = np.insert(N_feasi_hist, 0, 0)
            N_feasi_hist = np.insert(N_feasi_hist, 0, 0)
            #
            fig, ax1 = plt.subplots(figsize=(8, 6), dpi=80)
            ax1.set_xlabel('# of episode [-]', fontsize = 20)
            ax1.set_ylabel('# of pass-pre-screen flowsheets [-]', color = 'blue', fontsize = 20)
            plot_1 = ax1.plot(index_iteration, N_idaes_hist, 'b-', label = 'pass-pre-screen flowsheet')
            ax1.tick_params(axis = 'y', labelcolor = 'blue')
            ax2 = ax1.twinx()
            ax2.set_ylabel('# of feasible flowsheets [-]', color = 'red', fontsize = 20)
            plot_2 = ax2.plot(index_iteration_feasi, N_feasi_hist, 'r-', label = 'feasible flowsheet')
            ax2.tick_params(axis = 'y', labelcolor = 'red')
            lns = plot_1+plot_2
            labels = [l.get_label() for l in lns]
            plt.legend(lns, labels, loc = 0, fontsize = 20)
            plt.yticks(fontsize=16)
            plt.xticks(fontsize=16)
            str_tmp = 'Methanol with '+str(unit_pool)+' units in pool'
            plt.title(str_tmp, fontsize = 20)
            plt.show()   
            #
            x = status_feasi[:, 0]/(365*24*3600)
            y = status_feasi[:, 1]*1.0
            y = y[x<0.08]
            x = x[x<0.08]
            text = list(range(1, len(x)+1))
            
            fig, ax = plt.subplots(figsize=(8, 6), dpi=80)
            ax.scatter(x, y)
            # for k, txt in enumerate(text):
            #     ax.annotate(txt, (x[k], y[k]))
            plt.ylabel('Methanol flow rate [mol/s]', fontsize = 20)
            plt.xlabel('Cost per mol [$/mol]', fontsize = 20)
            # plt.xlabel('Benzene purity [%]', fontsize = 14)
            # plt.ylabel('Benzene flow rate [mol/s]', fontsize = 14)
            plt.yticks(fontsize=16)
            plt.xticks(fontsize=16)
            plt.title(str_tmp, fontsize = 20)
            plt.show()
            
            run_idaes_index = [np.argmin(x), np.argmax(y)]
            run_idaes_obs = obs_feasi[run_idaes_index, :]
            #%%
        else:
            N_idaes = reward_idaes.size
            N_feasi = 0
            if N_idaes == 1:
                max_iteration = status_idaes[-1]
            else:
                max_iteration = status_idaes[-1, -1]
            max_perc = max_iteration/total_iterations*100 #[%]
            N_max_conn = 0
        
        N_save = [ind_case, unit_pool, N_idaes, N_feasi, max_perc, N_max_conn, t1, t2, t3]

#%% evaluate feasible flowsheets
        if run_idaes:
            score_re = []
            extra_score_re = []
            status_re = []
            costs_re = []
            ind_feasi_re = []
            
            # if run_idaes_target == 'feasible':
            #     N_target = N_feasi
            #     obs_target = obs_feasi
            # else:
            #     N_target = N_idaes
            #     obs_target = obs_idaes
            N_target = len(run_idaes_index)
            obs_target = run_idaes_obs
            
            for ii in range(N_target):
            # for ii in [0]:
                print('\n------------------------------------------------------------\n')
                print('Evaluate observation ', ii+1, ' of index ', ind_case)
                obs = obs_target[ii,:]
                
                list_unit, list_inlet, list_outlet = convertobs2list(obs, all_available, ind_row, ind_col, len(all_available.list_inlet_all), len(all_available.list_outlet_all))
                flowsheet_name = 'index_'+str(ind_case)+'_observation_'+str(ii+1)
                
                constraints_consume = np.zeros((1, 12))
                constraints_satisfied = np.zeros((1, 14))
                pres_score, constraints_consume, constraints_satisfied = pre_screen(list_unit, list_inlet, list_outlet, constraints_consume, constraints_satisfied)
                print('Pass the Pre-screen: with score of ', pres_score)
                
                list_inlet = ['.'.join(strings) for strings in list_inlet]
                list_outlet = ['.'.join(strings) for strings in list_outlet]
                list_unit = list(list_unit)
                score, extra_score, status, costs = IDAES.run_optimization(flowsheet_name, list_unit, list_inlet, list_outlet, visualize_flowsheet = visualize_idaes)
                score_re.append(score)
                extra_score_re.append(extra_score)
                status_re.append(status)
                costs_re.append(costs)
                
                if score >= 5000:
                    ind_feasi_re.append(ii)
                    print('\nRecord ', len(ind_feasi_re), ' feasible flowsheets')
            
            # save re-evaluation results
            if run_idaes_save:
                tmp_array = np.array(score_re)
                fmt = ",".join(["%d"])
                np.savetxt(dir+'score_re'+str(ind_case)+'.csv',tmp_array,fmt=fmt,comments='')
                tmp_array = np.array(extra_score_re)
                fmt = ",".join(["%d"])
                np.savetxt(dir+'extra_score_re'+str(ind_case)+'.csv',tmp_array,fmt=fmt,comments='')
                tmp_array = np.array(status_re)
                tmp_array = np.array(tmp_array[:,1:], dtype=float)
                fmt = ",".join(["%f"]*len(tmp_array[0]))
                np.savetxt(dir+'status_re'+str(ind_case)+'.csv',tmp_array,fmt=fmt,comments='')
                tmp_array = np.array(costs_re)
                fmt = ",".join(["%f"]*len(tmp_array[0]))
                np.savetxt(dir+'costs_re'+str(ind_case)+'.csv',tmp_array,fmt=fmt,comments='')

            # extract all feasible observations
            print('Re-evaluation - # of feasible cases: ', len(ind_feasi_re))

        # load saved re-evaluated results
        filename = dir+'score_re'+str(ind_case)+'.csv'
        if os.path.exists(filename):
            filename = dir+'score_re'+str(ind_case)+'.csv'
            score_re = np.loadtxt(filename, dtype=int, delimiter=',')
            
            filename = dir+'extra_score_re'+str(ind_case)+'.csv'
            extra_score_re = np.loadtxt(filename, dtype=int, delimiter=',')
            
            filename = dir+'status_re'+str(ind_case)+'.csv'
            status_re = np.loadtxt(filename, dtype=float, delimiter=',')
            
            filename = dir+'costs_re'+str(ind_case)+'.csv'
            costs_re = np.loadtxt(filename, dtype=float, delimiter=',')

            # extract all feasible observations
            ind_feasi_re = np.where(score_re>=5000)
            obs_feasi_re = obs_idaes[ind_feasi_re]
            reward_feasi_re = score_re[ind_feasi_re]
            extra_reward_feasi_re = extra_score_re[ind_feasi_re]
            cost_feasi_re = costs_re[ind_feasi_re]
            status_feasi_re = status_re[ind_feasi_re]
            N_feasi_re = len(reward_feasi_re)

    #%% visualize RL results
            # plot scatter points of status
            x = status_feasi_re[:, 0]*1.0
            y = status_feasi_re[:, 1]
            text = list(range(1, len(x)+1))
            
            fig, ax = plt.subplots()
            ax.scatter(x, y)
            for k, txt in enumerate(text):
                ax.annotate(txt, (x[k], y[k]))
            plt.ylabel('methanol flow rate [mol/s]')
            plt.xlabel('cost per mol [$/mol]')
            plt.title('index_'+str(ind_case))
            plt.savefig(dir+'status_'+str(ind_case)+'_re.png')
            
            print('optimal flowsheets (re-evaluation):')
            # print(status_feasi)
            # print(cost_feasi)
            print('# of cases to IDAES: ', N_idaes)
            print('# of feasible cases: ', N_feasi_re)
            if len(inlet_num)>0:
                print('# of inlets: ', max(inlet_num))
                print('# of outlets:', max(outlet_num))
    
    return N_save
                
#%%
if __name__ == "__main__":

    # User inputs and model setups
    class user_inputs_18:
        def __init__(self):
            self.list_unit_all = np.array(['mixer_0', 'mixer_1', 'flash_0', \
                'exhaust_1', 'exhaust_2', 'product_0', 'StReactor_1', 'StReactor_2', \
                    'compressor_1', 'compressor_2', 'heater_1', 'heater_2', \
                        'cooler_1', 'expander_1', 'splitter_1', 'splitter_2'], dtype=str) #16
            self.list_inlet_all = np.array([x.split(".") for x in [\
                'mixer_1.inlet_1', 'mixer_1.inlet_2', 'flash_0.inlet', \
                    'exhaust_1.inlet', 'exhaust_2.inlet', 'product_0.inlet', 'StReactor_1.inlet', 'StReactor_2.inlet', \
                        'compressor_1.inlet', 'compressor_2.inlet', 'heater_1.inlet', 'heater_2.inlet', \
                            'cooler_1.inlet', 'expander_1.inlet', 'splitter_1.inlet','splitter_2.inlet']], dtype=str) #16
            self.list_outlet_all = np.array([x.split(".") for x in [\
                'mixer_0.outlet', 'mixer_1.outlet', 'flash_0.vap_outlet', 'flash_0.liq_outlet', \
                    'StReactor_1.outlet', 'StReactor_2.outlet', 'compressor_1.outlet', \
                        'compressor_2.outlet', 'heater_1.outlet', 'heater_2.outlet', 'cooler_1.outlet', \
                        'expander_1.outlet', 'splitter_1.outlet_1', 'splitter_1.outlet_2', 'splitter_2.outlet_1', \
                            'splitter_2.outlet_2']], dtype=str) #16

    # all available units # 22 units
    class all_available:
        def __init__(self):
            self.list_unit_all = np.array(['mixer_0', 'flash_0', 'exhaust_1', 'exhaust_2', \
                'product_0', 'mixer_1', 'mixer_2', 'heater_1', \
                    'heater_2', 'StReactor_1', 'StReactor_2', 'flash_1', \
                        'splitter_1', 'splitter_2', 'compressor_1', 'compressor_2', \
                            'cooler_1', 'cooler_2', 'expander_1', 'expander_2'], dtype=str) #20
            self.list_inlet_all = np.array([x.split(".") for x in ['flash_0.inlet', 'exhaust_1.inlet', 'exhaust_2.inlet', 'product_0.inlet', \
                'mixer_1.inlet_1', 'mixer_1.inlet_2', 'mixer_2.inlet_1', 'mixer_2.inlet_2', \
                    'heater_1.inlet', 'heater_2.inlet', 'StReactor_1.inlet', 'StReactor_2.inlet', \
                        'flash_1.inlet', 'splitter_1.inlet', 'splitter_2.inlet', 'compressor_1.inlet', \
                            'compressor_2.inlet', 'cooler_1.inlet', 'cooler_2.inlet', 'expander_1.inlet', \
                                'expander_2.inlet']], dtype=str) # 21
            self.list_outlet_all = np.array([x.split(".") for x in ['mixer_0.outlet', 'mixer_1.outlet', 'mixer_2.outlet', 'flash_0.vap_outlet', \
                'flash_0.liq_outlet', 'heater_1.outlet', 'heater_2.outlet', 'StReactor_1.outlet', \
                    'StReactor_2.outlet', 'flash_1.liq_outlet', 'flash_1.vap_outlet', 'splitter_1.outlet_1', \
                        'splitter_1.outlet_2', 'splitter_2.outlet_1', 'splitter_2.outlet_2', 'compressor_1.outlet', \
                            'compressor_2.outlet', 'cooler_1.outlet', 'cooler_2.outlet', 'expander_1.outlet', \
                                'expander_2.outlet']], dtype=str) #21

    # prameters for the post-processing
    P = {'index': 1, 
        'unit_pool': 1,
        'dir': '../result/',
        'run_idaes': True,
        'visualize_idaes': True,
        'run_idaes_target': 'feasible',
        'run_idaes_save': False,
        'total_iterations': 1000000
    }
    P['index'] = 18
    P['unit_pool'] = 18
    P['dir'] = './result/'
    user_inputs = user_inputs_18()

    # Calling processing for pass-pre-screen flowsheets
    N_save = post_processing(user_inputs, all_available(), P)

