import ipywidgets as widgets 
from ipywidgets import AppLayout, Box, Button, GridspecLayout, Layout ,HBox, Label
from IPython.display import display
import numpy as np

def select_units():

    unit_list = [
        'Heater',
        'Cooler',
        'Heat Exchanger',
        'Reactor',
        'Mixer',
        'Flash',
        'Splitter',
        'Compressor',
        'Expander',
        'Turbine',
		'Exhaust',
		'Product'
        ]

    u1 = widgets.Dropdown(
        options=['0', '1', '2', '3'],
        value='1',
        description=unit_list[0],
        disabled=False,
        style = {'description_width': 'initial'})

    u2 = widgets.Dropdown(
        options=['0', '1', '2', '3'],
        value='0',
        description=unit_list[1],
        disabled=False,
        style = {'description_width': 'initial'})

    u3 = widgets.Dropdown(
        options=['0', '1', '2', '3'],
        value='0',
        description=unit_list[2],
        disabled=False,
        style = {'description_width': 'initial'})

    u4 = widgets.Dropdown(
        options=['0','1', '2', '3'],
        value='1',
        description=unit_list[3],
        disabled=False,
        style = {'description_width': 'initial'})

    u5 = widgets.Dropdown(
        options=['0','1', '2', '3'],
        value='2',
        description=unit_list[4],
        disabled=False,
        style = {'description_width': 'initial'})

    u6 = widgets.Dropdown(
        options=['0','1', '2', '3'],
        value='2',
        description=unit_list[5],
        disabled=False,
        style = {'description_width': 'initial'})

    u7 = widgets.Dropdown(
        options=['0', '1', '2', '3'],
        value='1',
        description=unit_list[6],
        disabled=False,
        style = {'description_width': 'initial'})

    u8 = widgets.Dropdown(
        options=['0', '1', '2', '3'],
        value='1',
        description=unit_list[7],
        disabled=False,
        style = {'description_width': 'initial'})

    u9 = widgets.Dropdown(
        options=['0', '1', '2', '3'],
        value='0',
        description=unit_list[8],
        disabled=False,
        style = {'description_width': 'initial'})

    u10 = widgets.Dropdown(
        options=['0', '1', '2', '3'],
        value='0',
        description=unit_list[9],
        disabled=False,
        style = {'description_width': 'initial'})
	
    u11 = widgets.Dropdown(
       options=['0', '1', '2', '3'],
       value='2',
       description=unit_list[10],
       disabled=False,
       style = {'description_width': 'initial'})
	   
    u12 = widgets.Dropdown(
       options=['1'],
       value='1',
       description=unit_list[11],
       disabled=False,
       style = {'description_width': 'initial'})


    def build_unit_lists(u1,u2,u3,u4,u5,u6,u7,u8,u9,u10,u11,u12):
        num_heaters = int(u1)
        num_coolers = int(u2)
        num_heatex = int(u3)
        num_reactors = int(u4)
        num_mixers = int(u5)
        num_flash = int(u6)
        num_splitters = int(u7)
        num_compressors = int(u8)
        num_expanders = int(u9)
        num_turbines = int(u10)
        num_exhaust = int(u11)
        num_product = int(u12)
        unit_nums = [num_heaters, num_coolers, num_heatex, num_reactors, num_mixers, num_flash, num_splitters, 
                     num_compressors, num_expanders, num_turbines,num_exhaust,num_product]


        list_unit_all = []
        if num_heaters > 0:
            for i in range(num_heaters):
                k = 'heater_' + str(i+1)
                list_unit_all.append(k)


        if num_coolers > 0:
            for i in range(num_coolers):
                k = 'cooler_' + str(i+1)
                list_unit_all.append(k)

        if num_heatex > 0:
            for i in range(num_heatex):
                k = 'heatex_' + str(i+1)
                list_unit_all.append(k)

        if num_reactors > 0:
            for i in range(num_reactors):
                k = 'StReactor_' + str(i+1)
                list_unit_all.append(k)

        if num_mixers > 0:
            for i in range(num_mixers):
                k = 'mixer_' + str(i)
                list_unit_all.append(k)

        if num_flash > 0:
            for i in range(num_flash):
                k = 'flash_' + str(i)
                list_unit_all.append(k)					
        if num_splitters > 0:
            for i in range(num_splitters):
                k = 'splitter_' + str(i+1)
                list_unit_all.append(k)
        if num_compressors > 0:
            for i in range(num_compressors):
                k = 'compressor_' + str(i+1)
                list_unit_all.append(k)
        if num_expanders > 0:
            for i in range(num_expanders):
                k = 'expander_' + str(i+1)
                list_unit_all.append(k)
				
        if num_turbines > 0:
            for i in range(num_turbines):
                k = 'turbines_' + str(i+1)
                list_unit_all.append(k)
				
        if num_exhaust > 0:
            for i in range(num_exhaust):
                k = 'exhaust_' + str(i+1)
                list_unit_all.append(k)
				
        if num_product > 0:
            for i in range(num_product):
                k = 'product_' + str(i)
                list_unit_all.append(k)



        list_inlet_all = []
        list_outlet_all = []

    
        
        for i in range(len(list_unit_all)):
            if (list_unit_all[i]=='mixer_1')or (list_unit_all[i]=='mixer_2'):
                list_inlet_all.append(list_unit_all[i] + '.inlet_1')
                list_inlet_all.append(list_unit_all[i] + '.inlet_2')
            
            else:
                list_inlet_all.append(list_unit_all[i] + '.inlet')

        for i in range(len(list_unit_all)):
            if (list_unit_all[i]=='splitter_1') or (list_unit_all[i]=='splitter_2') or (list_unit_all[i]=='splitter_3'):
                list_outlet_all.append(list_unit_all[i] + '.outlet_1')
                list_outlet_all.append(list_unit_all[i] + '.outlet_2')
            elif (list_unit_all[i]=='flash_0') :
                list_outlet_all.append(list_unit_all[i] + '.vap_outlet')
                list_outlet_all.append(list_unit_all[i] + '.liq_outlet')
            elif (list_unit_all[i]=='flash_1') :
                list_outlet_all.append(list_unit_all[i] + '.vap_outlet')
                list_outlet_all.append(list_unit_all[i] + '.liq_outlet')
            elif (list_unit_all[i]=='flash_2') :
                list_outlet_all.append(list_unit_all[i] + '.vap_outlet')
                list_outlet_all.append(list_unit_all[i] + '.liq_outlet')
            else:
                list_outlet_all.append(list_unit_all[i] + '.outlet')

        for i in range(len(list_outlet_all)):
            if (list_outlet_all[i] == 'exhaust_1.outlet'):              
                k1=i  
        del list_outlet_all[k1]
        for i in range(len(list_outlet_all)):
            if (list_outlet_all[i] == 'exhaust_2.outlet'):              
                k2=i
        del list_outlet_all[k2]
        for i in range(len(list_outlet_all)):
            if (list_outlet_all[i] == 'product_0.outlet'):              
                k3=i
        del list_outlet_all[k3]
        for i in range(len(list_inlet_all)):
            if (list_inlet_all[i] == 'mixer_0.inlet'):              
                k4=i
        del list_inlet_all[k4]

        list_outlet_all = list(filter(None,list_outlet_all))
        list_unit_all = list(filter(None,list_unit_all))
        list_inlet_all = list(filter(None,list_inlet_all))

#        list_unit_all.insert(0, 'inlet_feed')
#        list_unit_all.insert(1, 'outlet_product')
#        list_unit_all.insert(2, 'outlet_exhaust')
#        list_inlet_all.insert(0, 'outlet_product.inlet')
#        list_inlet_all.insert(1, 'outlet_exhaust.inlet')
#        list_outlet_all.insert(0, 'inlet_feed.outlet')
#        #display(list_unit_all)
        return list_unit_all, list_inlet_all, list_outlet_all       

    w = widgets.interactive(build_unit_lists, u1=u1, u2=u2, u3=u3, u4=u4, u5=u5, u6=u6, u7=u7, u8=u8, u9=u9, u10=u10, u11=u11, u12=u12)

    return w 

def RL_options():

#    r1 = widgets.Dropdown(
#    options=['True', 'False'],
#    value='False',
#    description= 'Load saved model',
#    disabled=False,
#    style = {'description_width': 'initial'})

    r12 = widgets.Dropdown(
    options=['True', 'False'],
    value='False',
    description= 'Load saved model',
    disabled=False,
    style = {'description_width': 'initial'})	


	
    r13 = widgets.Dropdown(
    options=['True', 'False'],
    value='True',
    description= 'Save model',
    disabled=False,
    style = {'description_width': 'initial'})

    r14 = widgets.Dropdown(
    options=['True', 'False'],
    value='False',
    description= 'Visualization',
    disabled=False,
    style = {'description_width': 'initial'})

    r1 = widgets.IntSlider(
    value=10000,
    min=0,
    max=100000000,
    step=10000,
    description='Episodes:',
    style = {'description_width': 'initial'},
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='d')
	
    r15 = widgets.IntSlider(
    value=20,
    min=1,
    max=100,
    step=1,
    description='Threshold_learn:',
    style = {'description_width': 'initial'},
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='d')

    r2 = widgets.FloatSlider(
    value=0.9,
    min=0,
    max=1,
    step=0.1,
    description='E_greedy_max:',
    style = {'description_width': 'initial'},
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='.1f',)

    r3 = widgets.FloatSlider(
    value=0.0,
    min=0,
    max=1,
    step=0.1,
    description='E_greedy_min:',
    style = {'description_width': 'initial'},
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='.1f',)

    r16 = widgets.FloatSlider(
    value=0.5,
    min=0.0,
    max=1.0,
    step=0.1,
    description='Reward decay',
	)

    r17 = widgets.IntSlider(
    value=200,
    min=10,
    max=10000,
    step=10,
    description='N_hidden',
    style = {'description_width': 'initial'},
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='d')
	
	
    r18 = widgets.IntSlider(
    value=5,
    min=1,
    max=1000,
    step=5,
    description='Replace target iteration:',
    style = {'description_width': 'initial'},
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='d')
	
    r5 = widgets.IntSlider(
    value=20000,
    min=1000,
    max=1000000,
    step=1000,
    description='Memory size:',
    style = {'description_width': 'initial'},
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='d')
	
    r6 = widgets.IntSlider(
    value=3200,
    min=100,
    max=1000000,
    step=100,
    description='Batch size:',
    style = {'description_width': 'initial'},
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='d')

    r7 = widgets.IntSlider(
    value=10000,
    min=0,
    max=100000000,
    step=1000,
    description='Additional step:',
    style = {'description_width': 'initial'},
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='d')
	
    r4 = widgets.FloatSlider(
    value=0.01,
    min=0.0001,
    max=0.1,
    step=0.0001,
    description='Learning rate:',
    style = {'description_width': 'initial'},
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format= '.4f')

    r11 = widgets.Dropdown(
    options=['True', 'False'],
    value='False',
    description= 'Enable GNN',
    disabled=False,
    style = {'description_width': 'initial'})

    r8 = widgets.Dropdown(
    options=['True', 'False'],
    value='False',
    description= 'Enable IDAES',
    disabled=False,
    style = {'description_width': 'initial'})
	
    r10 = widgets.Dropdown(
    options=['True', 'False'],
    value='False',
    description= 'Enable GNN training',
    disabled=False,
    style = {'description_width': 'initial'})

    r19 = widgets.IntSlider(
    value=4,
    min=1,
    max=4,
    step=1,
    description='Complexity:',
    style = {'description_width': 'initial'},
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='d')
	
    r9 = widgets.Dropdown(
    options=['True', 'False'],
    value='True',
    description= 'Enable CNN',
    disabled=False,
    style = {'description_width': 'initial'})

    def store_values(r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18,r19):

        model_restore = r12
        model_save = r13
        visualization = r14
        Episode_max = r1
        threshold_learn = r15
        e_greedy_max=r2
        e_greedy_min=r3
        reward_decay=r16
        N_hidden=r17
        replace_target_iter=r18
        memory_size=r5
        batch_size=r6
        Additional_step=r7
        Learning_rate=r4
        GNN_enable=r11
        IDAES_enable=r8
        GNN_training=r10
        complexity=r19
        CNN_enable=r9

        return r1, r2, r3, r4, r5, r6, r7 ,r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19

    r = widgets.interactive(store_values, r1=r1, r2=r2,  r3=r3, r4=r4, r5=r5, r6=r6, r7=r7 ,r8=r8, r9=r9, r10=r10, r11=r11, r12=r12, r13=r13, r14=r14, r15=r15, r16=r16, r17=r17, r18=r18, r19=r19)


    return r

