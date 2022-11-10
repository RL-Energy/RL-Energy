import numpy as np
import os
import MET_IDAES_v16 as IDAES
sys.path.append(os.path.abspath("../"))
from RL_ENV import pre_screen, convertobs2list
import matplotlib.pyplot as plt

#%%
if __name__ == "__main__":
    
    # User inputs
    individual_observation = [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    flowsheet_name = 'flowsheet 0'
    visualize_idaes = True

    # User inputs and model setups
    class user_inputs:
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

    # processing user-selected observation
    user_inputs = user_inputs()
    all_available = all_available()
    temp1 = ['.'.join(strings) for strings in all_available.list_inlet_all]
    temp2 = ['.'.join(strings) for strings in user_inputs.list_inlet_all]
    ind_row = [temp1.index(i) for i in temp2] # available rows/inlets
    temp1 = ['.'.join(strings) for strings in all_available.list_outlet_all]
    temp2 = ['.'.join(strings) for strings in user_inputs.list_outlet_all]
    ind_col = [temp1.index(i) for i in temp2] # available columns/outlets
    list_unit, list_inlet, list_outlet = convertobs2list(individual_observation, all_available, ind_row, ind_col, len(all_available.list_inlet_all), len(all_available.list_outlet_all))
    
    list_inlet = ['.'.join(strings) for strings in list_inlet]
    list_outlet = ['.'.join(strings) for strings in list_outlet]
    list_unit = list(list_unit)
    score, extra_score, status, costs = IDAES.run_optimization(flowsheet_name, list_unit, list_inlet, list_outlet, visualize_flowsheet = visualize_idaes)
