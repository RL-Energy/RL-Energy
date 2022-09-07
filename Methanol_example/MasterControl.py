#%% Loading required packages.
from RL_CORE import RL_call
import numpy as np

#%%
if __name__ == "__main__":

    # User inputs and model setups
    class user_inputs_14:
        def __init__(self):
            self.list_unit_all = np.array(['mixer_0', 'mixer_1', 'flash_0', 'exhaust_1', \
                'exhaust_2', 'product_0', 'StReactor_1', 'compressor_1', \
                    'heater_1', 'cooler_1', 'expander_1', 'splitter_1'], dtype=str) #12
            self.list_inlet_all = np.array([x.split(".") for x in [\
                'mixer_1.inlet_1', 'mixer_1.inlet_2', 'flash_0.inlet', 'exhaust_1.inlet', \
                    'exhaust_2.inlet', 'product_0.inlet', 'StReactor_1.inlet', 'compressor_1.inlet', \
                        'heater_1.inlet', 'cooler_1.inlet', 'expander_1.inlet', 'splitter_1.inlet']], dtype=str) #12
            self.list_outlet_all = np.array([x.split(".") for x in [\
                'mixer_0.outlet', 'mixer_1.outlet', 'flash_0.vap_outlet', 'flash_0.liq_outlet', \
                    'StReactor_1.outlet', 'compressor_1.outlet', 'heater_1.outlet', 'cooler_1.outlet', \
                        'expander_1.outlet', 'splitter_1.outlet_1', 'splitter_1.outlet_2']], dtype=str) #11

    class user_inputs_15:
        def __init__(self):
            self.list_unit_all = np.array(['mixer_0', 'mixer_1', 'flash_0', 'exhaust_1', \
                'exhaust_2', 'product_0' 'StReactor_1', 'compressor_1', \
                    'compressor_2', 'heater_1', 'cooler_1', 'expander_1', \
                        'splitter_1'], dtype=str) #13
            self.list_inlet_all = np.array([x.split(".") for x in [\
                'mixer_1.inlet_1', 'mixer_1.inlet_2', 'flash_0.inlet', 'exhaust_1.inlet', \
                    'exhaust_2.inlet', 'product_0.inlet', 'StReactor_1.inlet', 'compressor_1.inlet', \
                        'compressor_2.inlet', 'heater_1.inlet', 'cooler_1.inlet', 'expander_1.inlet', \
                            'splitter_1.inlet']], dtype=str) #13
            self.list_outlet_all = np.array([x.split(".") for x in [\
                'mixer_0.outlet', 'mixer_1.outlet', 'flash_0.vap_outlet', 'flash_0.liq_outlet', \
                    'StReactor_1.outlet', 'compressor_1.outlet', 'compressor_2.outlet', 'heater_1.outlet', \
                        'cooler_1.outlet', 'expander_1.outlet', 'splitter_1.outlet_1', 'splitter_1.outlet_2']], dtype=str) #12
    
    class user_inputs_16:
        def __init__(self):
            self.list_unit_all = np.array(['mixer_0', 'mixer_1', 'flash_0', 'exhaust_1', \
                'exhaust_2', 'product_0' 'StReactor_1','compressor_1', \
                    'compressor_2', 'heater_1', 'heater_2', 'cooler_1', \
                        'expander_1', 'splitter_1'], dtype=str) #14
            self.list_inlet_all = np.array([x.split(".") for x in [\
                'mixer_1.inlet_1', 'mixer_1.inlet_2', 'flash_0.inlet', 'exhaust_1.inlet', \
                    'exhaust_2.inlet', 'product_0.inlet', 'StReactor_1.inlet', 'compressor_1.inlet', \
                        'compressor_2.inlet', 'heater_1.inlet', 'heater_2.inlet', 'cooler_1.inlet', \
                            'expander_1.inlet', 'splitter_1.inlet']], dtype=str) #14
            self.list_outlet_all = np.array([x.split(".") for x in [\
                'mixer_0.outlet', 'mixer_1.outlet', 'flash_0.vap_outlet', 'flash_0.liq_outlet', \
                    'StReactor_1.outlet', 'compressor_1.outlet', 'compressor_2.outlet', 'heater_1.outlet', \
                        'heater_2.outlet', 'cooler_1.outlet', 'expander_1.outlet', 'splitter_1.outlet_1', \
                            'splitter_1.outlet_2']], dtype=str) #13

    class user_inputs_17:
        def __init__(self):
            self.list_unit_all = np.array(['mixer_0', 'mixer_1', 'flash_0', 'exhaust_1', \
                'exhaust_2', 'product_0' 'StReactor_1','compressor_1', \
                    'compressor_2', 'heater_1', 'heater_2', 'cooler_1', \
                        'expander_1', 'splitter_1', 'splitter_2'], dtype=str) #15
            self.list_inlet_all = np.array([x.split(".") for x in [\
                'mixer_1.inlet_1', 'mixer_1.inlet_2', 'flash_0.inlet', 'exhaust_1.inlet', \
                    'exhaust_2.inlet', 'product_0.inlet', 'StReactor_1.inlet', 'compressor_1.inlet', \
                        'compressor_2.inlet', 'heater_1.inlet', 'heater_2.inlet', 'cooler_1.inlet', \
                            'expander_1.inlet', 'splitter_1.inlet', 'splitter_2.inlet']], dtype=str) #15
            self.list_outlet_all = np.array([x.split(".") for x in [\
                'mixer_0.outlet', 'mixer_1.outlet', 'flash_0.vap_outlet', 'flash_0.liq_outlet', \
                    'StReactor_1.outlet', 'compressor_1.outlet', 'compressor_2.outlet', 'heater_1.outlet', \
                        'heater_2.outlet', 'cooler_1.outlet', 'expander_1.outlet', 'splitter_1.outlet_1', \
                            'splitter_1.outlet_2', 'splitter_2.outlet_1', 'splitter_2.outlet_2']], dtype=str) #15
    
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

    class user_inputs_19:
        def __init__(self):
            self.list_unit_all = np.array(['mixer_0', 'mixer_1', 'mixer_2', 'flash_0', \
                'exhaust_1', 'exhaust_2', 'product_0', 'StReactor_1', 'StReactor_2', \
                    'compressor_1', 'compressor_2', 'heater_1', 'heater_2', \
                        'cooler_1', 'expander_1', 'splitter_1', 'splitter_2'], dtype=str) #17
            self.list_inlet_all = np.array([x.split(".") for x in [\
                'mixer_1.inlet_1', 'mixer_1.inlet_2', 'mixer_2.inlet_1', 'mixer_2.inlet_2', 'flash_0.inlet', \
                    'exhaust_1.inlet', 'exhaust_2.inlet', 'product_0.inlet', 'StReactor_1.inlet', 'StReactor_2.inlet', \
                        'compressor_1.inlet', 'compressor_2.inlet', 'heater_1.inlet', 'heater_2.inlet', \
                            'cooler_1.inlet', 'expander_1.inlet', 'splitter_1.inlet','splitter_2.inlet']], dtype=str) #18
            self.list_outlet_all = np.array([x.split(".") for x in [\
                'mixer_0.outlet', 'mixer_1.outlet', 'mixer_2.outlet', 'flash_0.vap_outlet', 'flash_0.liq_outlet', \
                    'StReactor_1.outlet', 'StReactor_2.outlet', 'compressor_1.outlet', \
                        'compressor_2.outlet', 'heater_1.outlet', 'heater_2.outlet', 'cooler_1.outlet', \
                        'expander_1.outlet', 'splitter_1.outlet_1', 'splitter_1.outlet_2', 'splitter_2.outlet_1', \
                            'splitter_2.outlet_2']], dtype=str) #17

    class user_inputs_20:
        def __init__(self):
            self.list_unit_all = np.array(['mixer_0', 'mixer_1', 'mixer_2', 'flash_0', 'flash_1', \
                'exhaust_1', 'exhaust_2', 'product_0', 'StReactor_1', 'StReactor_2', \
                    'compressor_1', 'compressor_2', 'heater_1', 'heater_2', \
                        'cooler_1', 'expander_1', 'splitter_1', 'splitter_2'], dtype=str) #19
            self.list_inlet_all = np.array([x.split(".") for x in [\
                'mixer_1.inlet_1', 'mixer_1.inlet_2', 'mixer_2.inlet_1', 'mixer_2.inlet_2', 'flash_0.inlet', 'flash_1.inlet', \
                    'exhaust_1.inlet', 'exhaust_2.inlet', 'product_0.inlet', 'StReactor_1.inlet', 'StReactor_2.inlet', \
                        'compressor_1.inlet', 'compressor_2.inlet', 'heater_1.inlet', 'heater_2.inlet', \
                            'cooler_1.inlet', 'expander_1.inlet', 'splitter_1.inlet', 'splitter_2.inlet']], dtype=str) #19
            self.list_outlet_all = np.array([x.split(".") for x in [\
                'mixer_0.outlet', 'mixer_1.outlet', 'mixer_2.outlet', 'flash_0.vap_outlet', 'flash_0.liq_outlet', 'flash_1.vap_outlet', 'flash_1.liq_outlet', \
                    'StReactor_1.outlet', 'StReactor_2.outlet', 'compressor_1.outlet', \
                        'compressor_2.outlet', 'heater_1.outlet', 'heater_2.outlet', 'cooler_1.outlet', \
                        'expander_1.outlet', 'splitter_1.outlet_1', 'splitter_1.outlet_2', 'splitter_2.outlet_1', \
                            'splitter_2.outlet_2']], dtype=str) #20

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

    # prameters for the RL-GNN-IDAES integrated framework.
    P = {'model_restore':False,'model_save':True,
        'model_index_restore':1, 'model_index':1, 'visualize': False,
        'threshold_learn':20,'GNN_enable':False,'learning_rate':0.01,'reward_decay':0.5, 'N_hidden': 200,
        'replace_target_iter':5,'memory_size':20000,'batch_size':3200,
        'Episode_max_mode':'dynamic','Episode_max':1e6,'Additional_step':1e2, # Episode_max_mode: dynamic or static
        'e_greedy_max':0.9, 'e_greedy_min':0.0, 'e_greedy_increment':1e-6, 'increment_degradation': [1, 2, 5, 10],
        'CNN_enable': False, 'IDAES_enable': True, 'complexity': 4} # system complexity indicated by 1, 2, 3, 4, larger number indicates more complex design

    initial_index = 19 #20
    units = 19 #20
    exec("user_inputs = user_inputs_"+str(units)+"()")

    # 1st cycle
    P['model_index'] = initial_index
    P['e_greedy_min'] = 0.0
    P['e_greedy_max'] = 0.9
    RL_call(user_inputs,all_available(), P) # Calling RL framework to train the model

    # 2nd cycle and more
    P['model_restore'] = True
    P['model_index_restore'] = initial_index
    P['model_index'] = initial_index+20
    P['e_greedy_min'] = 0.9
    P['e_greedy_max'] = 0.9
    RL_call(user_inputs,all_available(), P) # Calling RL framework to train the model