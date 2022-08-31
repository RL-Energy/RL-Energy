#%%
from RL_ENV import fs_gen, convertobs2list
from RL_DQN import DeepQNetwork
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from timeit import default_timer as timer
# from GNN_v1 import RL2GNN_Link, GNNmodel
import HDA_IDAES_v9 as IDAES
import copy
import time

#%% Calling reinforcement learning.
def RL_run(env,RL,GNN,user_inputs,all_available,f_rec,P):

	# check time consuming for training and running IDAES
	RL_start = timer()
	IDAES_consume = 0
	GNN_consume = 0
	PreScreen_consume = 0
	constraints_consume = np.zeros((1, 12))
	constraints_satisfied_save = np.zeros((1, 14))

	tt_reward_hist=[]
	done_reward_hist=[]
	# reward_hist=[] # debug
	tt_reward_hist_R = [] # debug
	tt_reward_hist_extraR = [] # debug
	# reward_hist_R = [] # debug
	# reward_hist_extraR = [] # debug
	N_noaction_ratio_hist=[] # debug
	N_unavailable_ratio_hist=[] # debug
	end_step_hist = [] # debug

	new_obs=[]
	new_obs_reward = []
	new_obs_extra_reward = []
	new_obs_status = []
	new_obs_costing = []
	obs_runIdaes_flowsheet = []
	obs_runIdaes_reward = []
	obs_runIdaes_extra_reward = []
	obs_runIdaes_status = []
	obs_runIdaes_costing = []
	
	IDAES_status = []
	IDAES_costing = []

	i_idaes = 0
	i_idaes_unique = 0
	i_feasible_unique = 0
	i_idaes_unique_hist = []
	i_feasible_unique_hist = []
	ma_horizon = 100
	ma_tt_reward_hist = []
	ma_done_reward_hist = []
	ma_N_noaction_ratio_hist = [] # debug
	ma_N_unavailable_ratio_hist = [] # debug
	ma_tt_reward_hist_R = [] # debug
	ma_tt_reward_hist_extraR = [] # debug
	ma_end_step_hist = [] # debug
	
	# Local parameters.
	Episode_max = int(P['Episode_max'] + P['Additional_step'])
	GNN_enable = P['GNN_enable']
	IDAES_enable = P['IDAES_enable']
	model_index = P['model_index']
	threshold_learn = P['threshold_learn']
	system_complexity = P['complexity']
	dir_result = './'+'result/'

	# universal initial observation
	observation_init = np.zeros((env.MAX_Iteration, env.num_elements),dtype=float)
	temp1_row = ['.'.join(strings) for strings in all_available.list_inlet_all]
	temp2_row = ['.'.join(strings) for strings in user_inputs.list_inlet_all]
	ind_row = [temp1_row.index(i) for i in temp2_row] # available rows/inlets
	temp1_col = ['.'.join(strings) for strings in all_available.list_outlet_all]
	temp2_col = ['.'.join(strings) for strings in user_inputs.list_outlet_all]
	ind_col = [temp1_col.index(i) for i in temp2_col] # available columns/outlets

	# pre-set connection for methanol example: flash_0.liq_outlet <-> product_0.inlet
	ind_row_spe = temp1_row.index('product_0.inlet')
	ind_col_spe = temp1_col.index('flash_0.vap_outlet')
	print('hard-connection: row ', ind_row_spe, ' and column ', ind_col_spe)
	ind_col_trim = copy.deepcopy(ind_col)
	ind_col_trim.remove(ind_col_spe)
	ind_col_trim.append(env.n_actions-1)

	# mark available spots
	for i in range(env.MAX_Iteration):
		if i in ind_row:
			if i != ind_row_spe:
				observation_init[i, ind_col_trim] = 0.5
				observation_init[i, env.num_elements-1] = 0.5 # the "step counter" column always available
			else:
				observation_init[i, ind_col_spe] = 0.5
				observation_init[i, env.num_elements-1] = 0.5 # the "step counter" column always available
		else:
			observation_init[i, env.n_actions-1] = 0.5 #the "no-action" column always available
			observation_init[i, env.num_elements-1] = 0.5 # the "step counter" column always available

	# manually mark out multiple spots according to the physics constraints
	unavailable_spots = [
		[3, 4], 
		[],
		[],
		[],
		[1],
		[1],
		[2],
		[2],
		[5],
		[6],
		[7],
		[8],
		[9, 10],
		[11, 12],
		[13, 14],
		[15],
		[16],
		[17],
		[18],
		[19],
		[20]	
	]

	for i in range(env.MAX_Iteration):
		unavailable_spots_step = unavailable_spots[i]
		for j in unavailable_spots_step:
			observation_init[i, j] = 0

	print('initial observation in matrix: ')
	print(observation_init)
	observation_init_flatten = observation_init.flatten()

	# action options in each step/row according to the observation_init
	action_options = []
	for i in range(env.MAX_Iteration):
		action_options_step = []
		for j in range(env.num_elements-1):
			if observation_init[i, j] == 0.5:
				action_options_step.append(j)
		action_options.append(action_options_step)

	print('action options:')
	print(action_options)

	time.sleep(20)

	for episode in range(Episode_max):

		# observation, picked_true_fs = env.reset(episode)
		observation = np.copy(observation_init_flatten)

		i_step = 0
		tt_reward = 0
		tt_reward_R = 0 # debug
		tt_reward_extraR = 0 # debug
		N_noaction = 0 # step on no-action spots (for either empty row or filled row)
		N_unavailable = 0 # step on unavailable spots (for either empty row or filled row)
		reward = 0 # initial/empty observation is assumed to pass the pre-screen
		extra_reward = 0 # consider "no-action" penalty or reward, and the influence of # of steps
		constraints_satisfied = np.zeros((1, 14)) # debug: correspond to 12 constraints and i_step, episode
		action_hist = []
		
		while True:
		
			# RL choose action based on observation
			action_options_step = copy.deepcopy(action_options[i_step])
			# print('at ', i_step, ' step, action options: ', action_options_step)
			for action_h in action_hist:
				if action_h in action_options_step:
					action_options_step.remove(action_h)
			# print('\t action history: ', action_hist)
			# print('\t action options (available): ', action_options_step)

			if RL.epsilon < 0.0: # may add one input parameter to P later
				action, actions_value = RL.choose_action(observation)
			else:
				action, actions_value = RL.choose_action(observation, action_options_step)

			if action != env.n_actions-1: # if action is not "no-action", record it
				action_hist.append(action)
			
			# RL take action and get next observation and reward
			PS_start = timer()
			observation_, reward_, extra_reward_, episode_done, pass_pre_screen, constraints_consume, N_noaction,N_unavailable, constraints_satisfied = \
				env.step(action,observation,episode,i_step,all_available,constraints_consume,\
					reward,extra_reward,ind_row,ind_col,N_noaction,N_unavailable,system_complexity, constraints_satisfied)
			PreScreen_consume += timer()-PS_start
						
			################# GNN start #################
			if GNN_enable:
				GNN_start = timer()
				GNN_index,GNN_label,GNN_matrix = RL2GNN_Link(observation_,all_available)
				b1 = GNN_index[:,0].astype(int)
				b2 = GNN_index[:,1].astype(int)
				b1 = b1.tolist()
				b2 = b2.tolist()
				if len(b1) > 1:
					GNN_class = bool(GNNmodel(b1,b2,GNN))
				else:
					GNN_class = True
				GNN_consume += timer()-GNN_start
				# print('Time lapse for GNN: ',end - start,'s.')
			else:
				GNN_class = True
						
			################# GNN end #################
			
			################# IDAES start #################
			if GNN_class == True and pass_pre_screen == True and episode_done == True:
				
				if IDAES_enable == True:

					i_idaes += 1
					IDAES_start = timer()
					matrix_flowsheet = np.reshape(observation_,(env.MAX_Iteration, env.num_elements))
					list_flowsheet = list(matrix_flowsheet[:env.MAX_Iteration, :env.num_elements-2].flatten())

					if list_flowsheet in obs_runIdaes_flowsheet:
						# print('\nEpisode with a repeated flowsheet: ', episode, ', i_step: ', i_step, ', i_idaes: ', i_idaes)
						reward_ = obs_runIdaes_reward[obs_runIdaes_flowsheet.index(list_flowsheet)]
						extra_reward_ = obs_runIdaes_extra_reward[obs_runIdaes_flowsheet.index(list_flowsheet)]
						IDAES_status = obs_runIdaes_status[obs_runIdaes_flowsheet.index(list_flowsheet)]
						IDAES_costing = obs_runIdaes_costing[obs_runIdaes_flowsheet.index(list_flowsheet)]
					else:
						i_idaes_unique += 1
						print('\nEpisode: '+str(episode)+'/'+str(Episode_max)+', percent: '+str(round(episode/Episode_max*100,2))+'%',', i_step: ', i_step,
								', i_idaes: ', i_idaes, ', run Idaes: ', i_idaes_unique, flush = True)
						flowsheet_name = 'flowsheet_'+str(episode)+'_'+str(i_step)
						list_unit, list_inlet, list_outlet = convertobs2list(observation_, all_available, ind_row, ind_col, len(all_available.list_inlet_all), len(all_available.list_outlet_all)+2)
						
						# convert numpy array to lists
						list_inlet = ['.'.join(strings) for strings in list_inlet]
						list_outlet = ['.'.join(strings) for strings in list_outlet]
						list_unit = list(list_unit)
						reward_, extra_reward_, IDAES_status, IDAES_costing = \
							IDAES.run_optimization(flowsheet_name, list_unit, list_inlet, list_outlet)
						IDAES_status.append(i_step)
						IDAES_status.append(episode)

						obs_runIdaes_flowsheet.append(list_flowsheet)
						obs_runIdaes_reward.append(reward_)
						obs_runIdaes_extra_reward.append(extra_reward_)
						obs_runIdaes_status.append(IDAES_status)
						obs_runIdaes_costing.append(IDAES_costing)

						if reward_ >= 5000:
							i_feasible_unique += 1

						IDAES_consume += timer()-IDAES_start
				else:
					# reward_ = 5000
					IDAES_status = ['unavailable', 0.0, 0.0, i_step, episode]
					IDAES_costing = [0.0, 0.0, 0.0, 0.0, 0.0]

			################# IDAES end #################
			
			# reward_hist.append(reward_+extra_reward_) # debug
			# reward_hist_R.append(reward_) # debug
			# reward_hist_extraR.append(extra_reward_) # debug
			tt_reward=tt_reward+reward_+extra_reward_
			tt_reward_R += reward_ # debug
			tt_reward_extraR += extra_reward_ # debug
			RL.store_transition(observation, action, reward_+extra_reward_, episode_done, observation_) #Jie... add "done"
			
			# swap observation and reward
			observation = np.copy(observation_)
			reward = reward_
			extra_reward = extra_reward_
			
			# break while loop when end of this episode
			if episode_done:
				end_step_hist.append(i_step) # debug
				break
				
			i_step += 1

		if (episode % threshold_learn == 0 and episode > 0) :
			RL.learn()
		f_rec.flush()

		if episode % 100 == 0  and episode > 0: 
			print("Episode: "+str(episode)+'/'+str(Episode_max)+", percent: "+str(round(episode/Episode_max*100,2))+'%', ", Reward: ", tt_reward, flush = True)
			print("\tepsilon===", RL.epsilon, flush = True)
			time_now = timer()
			print('\tTime lapse: ', (time_now-RL_start)/3600, ' hr', flush = True)
			print('\tIDAES consume: ', IDAES_consume/3600, ' hr', flush = True)
			# print('\tGNN consume: ', GNN_consume/3600, ' hr', flush = True)
			print('\tPre-screen consume: ', PreScreen_consume/3600, ' hr', flush = True)
			print('\tconstraints consume time details: ', constraints_consume, ' s', flush = True)
			constraints_satisfied[0, 12] = i_step
			constraints_satisfied[0, 13] = episode
			print('\tEpisode ends while satisfying constraints: ', constraints_satisfied, flush = True)

		# save the pre-screen results
		if episode % 1000 == 0 and episode > 0:
			constraints_satisfied_save = np.concatenate((constraints_satisfied_save, constraints_satisfied), axis = 0)
			fmt = ",".join(["%d"] * len(constraints_satisfied_save[0]))
			np.savetxt(dir_result+'obs_pre_screen_'+str(model_index)+'.csv',np.vstack(constraints_satisfied_save),fmt=fmt,comments='')

		# save model every 10k episodes
		if episode % 10000 == 0 and episode > 0:
			save_models_to = dir_result + 'save_model/'
			if P['model_save']: 
				RL.saver.save(RL.sess, save_models_to +'model_'+str(P['model_index'])+'.ckpt')

		# save the pre-screen results in the process of RL
		if episode % 1000 == 0 and i_idaes_unique>0: 
			# convert list to numpy array and save
			tmp_array = np.array(obs_runIdaes_flowsheet)
			fmt = ",".join(["%d"] * len(tmp_array[0]))
			np.savetxt(dir_result+'obs_runIdaes_flowsheet_'+str(model_index)+'.csv',np.vstack(tmp_array),fmt=fmt,comments='')
			tmp_array = np.array(obs_runIdaes_reward)
			fmt = ",".join(["%d"])
			np.savetxt(dir_result+'obs_runIdaes_reward_'+str(model_index)+'.csv',tmp_array,fmt=fmt,comments='')
			tmp_array = np.array(obs_runIdaes_extra_reward)
			fmt = ",".join(["%d"])
			np.savetxt(dir_result+'obs_runIdaes_extra_reward_'+str(model_index)+'.csv',tmp_array,fmt=fmt,comments='')
			tmp_array = np.array(obs_runIdaes_status)
			tmp_array = np.array(tmp_array[:,1:], dtype=float)
			fmt = ",".join(["%f"]*len(tmp_array[0]))
			np.savetxt(dir_result+'obs_runIdaes_status_'+str(model_index)+'.csv',np.vstack(tmp_array),fmt=fmt,comments='')
			tmp_array = np.array(obs_runIdaes_costing)
			fmt = ",".join(["%f"]*len(tmp_array[0]))
			np.savetxt(dir_result+'obs_runIdaes_costing_'+str(model_index)+'.csv',np.vstack(tmp_array),fmt=fmt,comments='')

		# save the last 500 observations
		if episode>Episode_max-501:
			new_obs.append(observation_)
			new_obs_reward.append(reward_)
			new_obs_extra_reward.append(extra_reward_)
			new_obs_status.append(IDAES_status)
			new_obs_costing.append(IDAES_costing)

		tt_reward_hist.append(tt_reward)
		done_reward_hist.append(reward_)
		tt_reward_hist_R.append(tt_reward_R) # debug
		tt_reward_hist_extraR.append(tt_reward_extraR) # debug
		i_idaes_unique_hist.append(i_idaes_unique)
		i_feasible_unique_hist.append(i_feasible_unique)
		N_noaction_ratio_hist.append(N_noaction/(i_step+1)) # in each episode, ratio of no-action in all steps
		N_unavailable_ratio_hist.append(N_unavailable/(i_step+1)) # in each episode, ratio of un-available in all steps

		if episode % ma_horizon == 0 and episode > 0:
			ma_length = len(tt_reward_hist)
			ma_tt_reward_hist.append(sum(tt_reward_hist[ma_length-ma_horizon:ma_length])/ma_horizon)
			ma_done_reward_hist.append(sum(done_reward_hist[ma_length-ma_horizon:ma_length])/ma_horizon)
			ma_N_noaction_ratio_hist.append(sum(N_noaction_ratio_hist[ma_length-ma_horizon:ma_length])/ma_horizon) # debug
			ma_N_unavailable_ratio_hist.append(sum(N_unavailable_ratio_hist[ma_length-ma_horizon:ma_length])/ma_horizon) # debug
			ma_tt_reward_hist_R.append(sum(tt_reward_hist_R[ma_length-ma_horizon:ma_length])/ma_horizon) # debug
			ma_tt_reward_hist_extraR.append(sum(tt_reward_hist_extraR[ma_length-ma_horizon:ma_length])/ma_horizon) # debug
			ma_end_step_hist.append(sum(end_step_hist[ma_length-ma_horizon:ma_length])/ma_horizon) # debug

		# record the RL training process (debug)
		if episode % 1000 == 0: 
			# convert list to numpy array and save
			tmp_array = np.array(tt_reward_hist)
			fmt = ",".join(["%d"])
			np.savetxt(dir_result+'tt_Reward_'+str(model_index)+'.csv',tmp_array,fmt=fmt,comments='')
			tmp_array = np.array(done_reward_hist)
			fmt = ",".join(["%d"])
			np.savetxt(dir_result+'done_Reward_'+str(model_index)+'.csv',tmp_array,fmt=fmt,comments='')
			tmp_array = np.array(i_idaes_unique_hist)
			fmt = ",".join(["%d"])
			np.savetxt(dir_result+'IDAES_unique_'+str(model_index)+'.csv',tmp_array,fmt=fmt,comments='')
			tmp_array = np.array(i_feasible_unique_hist)
			fmt = ",".join(["%d"])
			np.savetxt(dir_result+'IDAES_unique_feasible_'+str(model_index)+'.csv',tmp_array,fmt=fmt,comments='')

	# save the pre-screen results at the end of RL
	if i_idaes_unique>0: 
		# convert list to numpy array and save
		tmp_array = np.array(obs_runIdaes_flowsheet)
		fmt = ",".join(["%d"] * len(tmp_array[0]))
		np.savetxt(dir_result+'obs_runIdaes_flowsheet_'+str(model_index)+'.csv',np.vstack(tmp_array),fmt=fmt,comments='')
		tmp_array = np.array(obs_runIdaes_reward)
		fmt = ",".join(["%d"])
		np.savetxt(dir_result+'obs_runIdaes_reward_'+str(model_index)+'.csv',tmp_array,fmt=fmt,comments='')
		tmp_array = np.array(obs_runIdaes_extra_reward)
		fmt = ",".join(["%d"])
		np.savetxt(dir_result+'obs_runIdaes_extra_reward_'+str(model_index)+'.csv',tmp_array,fmt=fmt,comments='')
		tmp_array = np.array(obs_runIdaes_status)
		tmp_array = np.array(tmp_array[:,1:], dtype=float)
		fmt = ",".join(["%f"]*len(tmp_array[0]))
		np.savetxt(dir_result+'obs_runIdaes_status_'+str(model_index)+'.csv',np.vstack(tmp_array),fmt=fmt,comments='')
		tmp_array = np.array(obs_runIdaes_costing)
		fmt = ",".join(["%f"]*len(tmp_array[0]))
		np.savetxt(dir_result+'obs_runIdaes_costing_'+str(model_index)+'.csv',np.vstack(tmp_array),fmt=fmt,comments='')
		
	# plot moving average reward
	plt.figure()
	plt.plot(np.arange(len(ma_tt_reward_hist))*ma_horizon, ma_tt_reward_hist)
	plt.ylabel('MA_tt_Reward (episode)')
	plt.xlabel('training steps')
	plt.savefig('./result/MA_tt_Reward_'+str(model_index)+'.png')
	if P['visualize'] == True:
		plt.show()
	tmp_array = np.array(ma_tt_reward_hist)
	fmt = ",".join(["%d"])
	np.savetxt(dir_result+'MA_tt_Reward_'+str(model_index)+'.csv',tmp_array,fmt=fmt,comments='')

	plt.figure()
	plt.plot(np.arange(len(ma_done_reward_hist))*ma_horizon, ma_done_reward_hist)
	plt.ylabel('MA_Reward (end of episode)')
	plt.xlabel('training steps')
	plt.savefig('./result/MA_done_Reward_'+str(model_index)+'.png')
	if P['visualize'] == True:
		plt.show()
	tmp_array = np.array(ma_done_reward_hist)
	fmt = ",".join(["%d"])
	np.savetxt(dir_result+'MA_done_Reward_'+str(model_index)+'.csv',tmp_array,fmt=fmt,comments='')

	# save the end reward of each episode
	tmp_array = np.array(done_reward_hist)
	fmt = ",".join(["%d"])
	np.savetxt(dir_result+'done_Reward_'+str(model_index)+'.csv',tmp_array,fmt=fmt,comments='')
	
	plt.figure()
	plt.plot(np.arange(len(ma_N_noaction_ratio_hist))*ma_horizon, ma_N_noaction_ratio_hist)
	plt.ylabel('MA_N_noaction_ratio (episode)')
	plt.xlabel('training steps')
	plt.savefig('./result/MA_N_noaction_ratio_'+str(model_index)+'.png')
	if P['visualize'] == True:
		plt.show()
	tmp_array = np.array(ma_N_noaction_ratio_hist)
	fmt = ",".join(["%d"])
	np.savetxt(dir_result+'MA_N_noaction_ratio_'+str(model_index)+'.csv',tmp_array,fmt=fmt,comments='')

	plt.figure()
	plt.plot(np.arange(len(ma_N_unavailable_ratio_hist))*ma_horizon, ma_N_unavailable_ratio_hist)
	plt.ylabel('MA_N_unavailable_ratio (episode)')
	plt.xlabel('training steps')
	plt.savefig('./result/MA_N_unavailable_ratio_'+str(model_index)+'.png')
	if P['visualize'] == True:
		plt.show()
	tmp_array = np.array(ma_N_unavailable_ratio_hist)
	fmt = ",".join(["%d"])
	np.savetxt(dir_result+'MA_N_unavailable_ratio_'+str(model_index)+'.csv',tmp_array,fmt=fmt,comments='')

	plt.figure()
	plt.plot(np.arange(len(ma_tt_reward_hist_R))*ma_horizon, ma_tt_reward_hist_R)
	plt.ylabel('MA_tt_reward_R (episode)')
	plt.xlabel('training steps')
	plt.savefig('./result/MA_tt_reward_R_'+str(model_index)+'.png')
	if P['visualize'] == True:
		plt.show()

	plt.figure()
	plt.plot(np.arange(len(ma_tt_reward_hist_extraR))*ma_horizon, ma_tt_reward_hist_extraR)
	plt.ylabel('MA_tt_reward_extraR (episode)')
	plt.xlabel('training steps')
	plt.savefig('./result/MA_tt_reward_extraR_'+str(model_index)+'.png')
	if P['visualize'] == True:
		plt.show()

	plt.figure()
	plt.plot(np.arange(len(ma_end_step_hist))*ma_horizon, ma_end_step_hist)
	plt.ylabel('MA_end_step (episode)')
	plt.xlabel('training steps')
	plt.savefig('./result/MA_end_step_'+str(model_index)+'.png')
	if P['visualize'] == True:
		plt.show()
	tmp_array = np.array(ma_end_step_hist)
	fmt = ",".join(["%d"])
	np.savetxt(dir_result+'MA_end_step_'+str(model_index)+'.csv',tmp_array,fmt=fmt,comments='')

	# # plot total rewards
	# plt.figure()
	# plt.plot(np.arange(len(tt_reward_hist)), tt_reward_hist)
	# plt.ylabel('Reward (episode)')
	# plt.xlabel('training steps')
	# plt.savefig('./result/Reward_'+str(model_index)+'.png')
	# if P['visualize'] == True:
	# 	plt.show()

	# # plot reward hist (debug)
	# plt.figure()
	# plt.plot(np.arange(len(reward_hist)), reward_hist)
	# plt.ylabel('Reward (step)')
	# plt.xlabel('count')
	# plt.savefig('./result/Step_Reward_'+str(model_index)+'.png')
	# if P['visualize'] == True:
	# 	plt.show()

	# plt.figure()
	# plt.plot(np.arange(len(reward_hist_R)), reward_hist_R)
	# plt.ylabel('Reward (R only) (step)')
	# plt.xlabel('count')
	# plt.savefig('./result/Step_Reward_R_'+str(model_index)+'.png')
	# if P['visualize'] == True:
	# 	plt.show()

	# plt.figure()
	# plt.plot(np.arange(len(reward_hist_extraR)), reward_hist_extraR)
	# plt.ylabel('Reward (extra R only) (step)')
	# plt.xlabel('count')
	# plt.savefig('./result/Step_Reward_extraR_'+str(model_index)+'.png')
	# if P['visualize'] == True:
	# 	plt.show()

	# plot unique # of IDAES and feasible cases
	plt.figure()
	plt.plot(np.arange(len(i_idaes_unique_hist)), i_idaes_unique_hist)
	plt.ylabel('Unique cases sent to IDAES')
	plt.xlabel('training steps')
	plt.savefig('./result/IDAES_unique_'+str(model_index)+'.png')
	if P['visualize'] == True:
		plt.show()
	tmp_array = np.array(i_idaes_unique_hist)
	fmt = ",".join(["%d"])
	np.savetxt(dir_result+'IDAES_unique_'+str(model_index)+'.csv',tmp_array,fmt=fmt,comments='')

	plt.figure()
	plt.plot(np.arange(len(i_feasible_unique_hist)), i_feasible_unique_hist)
	plt.ylabel('Unique feasible cases')
	plt.xlabel('training steps')
	plt.savefig('./result/IDAES_unique_feasible_'+str(model_index)+'.png')
	if P['visualize'] == True:
		plt.show()
	tmp_array = np.array(i_feasible_unique_hist)
	fmt = ",".join(["%d"])
	np.savetxt(dir_result+'IDAES_unique_feasible_'+str(model_index)+'.csv',tmp_array,fmt=fmt,comments='')

	# plot training cost
	RL.plot_cost(dir_result,model_index, P['visualize'])

	# save last 500 observations
	fmt = ",".join(["%d"] * len(new_obs[0]))
	np.savetxt(dir_result+'new_obs_'+str(model_index)+'.csv', np.vstack(new_obs),fmt=fmt,comments='')
	fmt = ",".join(["%d"])
	np.savetxt(dir_result+'new_obs_reward_'+str(model_index)+'.csv',new_obs_reward,fmt=fmt,comments='')
	fmt = ",".join(["%d"])
	np.savetxt(dir_result+'new_obs_extra_reward_'+str(model_index)+'.csv',new_obs_extra_reward,fmt=fmt,comments='')
	tmp_array = np.array(new_obs_status)
	tmp_array = np.array(tmp_array[:,1:], dtype=float)
	np.savetxt(dir_result+'new_obs_status_'+str(model_index)+'.csv',tmp_array,fmt='%10.5f',delimiter=',')
	tmp_array = np.array(new_obs_costing)
	np.savetxt(dir_result+'new_obs_status_'+str(model_index)+'.csv',tmp_array,fmt='%10.5f',delimiter=',')
	
	# # save rewards
	# tmp_array = np.array(tt_reward_hist)
	# np.savetxt(dir_result+'training_tt_reward_'+str(model_index)+'.csv',tmp_array,fmt='%10.5f',delimiter=',')
	# tmp_array = np.array(reward_hist)
	# np.savetxt(dir_result+'training_reward_'+str(model_index)+'.csv',tmp_array,fmt='%10.5f',delimiter=',')

	# save computing cost
	RL_end = timer()
	rtime = np.array((1,2))
	print('Total time consuming: ', (RL_end-RL_start)/3600, ' hr', flush = True)
	print('Pre-screen time consuming: ', PreScreen_consume/3600, ' hr', flush = True)
	print('IDAES time consuming: ', IDAES_consume/3600, ' hr', flush = True)
	rtime = np.array([(RL_end-RL_start)/3600, PreScreen_consume/3600, IDAES_consume/3600])
	np.savetxt(dir_result+'training_time_'+str(model_index)+'.csv',rtime,fmt='%10.5f',delimiter=',')

	# end of game
	print('Game Over')

#%% Calculating the maximum episode.
def calcMaxEpisode(Emax, e_max, e_min=0, R1=1, R2=20, R3=50, R4=100):
	a = 0
	b = 1
	if e_max >1 or e_min < 0 or e_max<e_min:
		os.exit("user-provided inputs must satisfy: 0<=e_min<=e_max<=1")
	if e_max == e_min:
		return 0
	while True:
		e = (a+b)/2.0
		if e_max > 0.8:
			if e_min < 0.4:
				E = round((e_max-0.8)/e*R4+0.2/e*R3+0.2/e*R2+(0.4-e_min)/e*R1)
			elif e_min < 0.6:
				E = round((e_max-0.8)/e*R4+0.2/e*R3+(0.6-e_min)/e*R2)
			elif e_min < 0.8:
				E = round((e_max-0.8)/e*R4+(0.8-e_min)/e*R3)
			else:
				E = round((e_max-e_min)/e*R4)
		elif e_max > 0.6:
			if e_min < 0.4:
				E = round((e_max-0.6)/e*R3+0.2/e*R2+(0.4-e_min)/e*R1)
			elif e_min < 0.6:
				E = round((e_max-0.6)/e*R3+(0.6-e_min)/e*R2)
			else:
				E = round((e_max-e_min)/e*R3)
		elif e_max > 0.4:
			if e_min < 0.4:
				E = round((e_max-0.4)/e*R2+(0.4-e_min)/e*R1)
			elif e_min > 0.4:
				E = round((e_max-e_min)/e*R2)
		else:
			E = round((e_max-e_min)/e*R1)
		if E<Emax:
			b = e
		else:
			a = e
		if abs(E-Emax) < 100:
			break
	return e

#%%
def RL_call(user_inputs,all_available,P):

	# Determine episilon increment
	learning_steps = round(P['Episode_max']/P['threshold_learn'])
	e_greedy_max = P['e_greedy_max']
	e_greedy_min = P['e_greedy_min']
	[R1, R2, R3, R4] = P['increment_degradation']
	if P['Episode_max_mode'].lower() == 'dynamic': 
		P['e_greedy_increment'] = calcMaxEpisode(learning_steps, \
			e_greedy_max, e_greedy_min, R1, R2, R3, R4)
		print("\nCalculated epsilon increment: "+str(P['e_greedy_increment'])+'\n', flush = True)

	# Set up Environment
	dir_result = './result/'
	if not os.path.exists(dir_result):
		os.makedirs(dir_result) 
	f_rec  = open(dir_result+'record_test.out', 'a+')

	n_inlets = len(all_available.list_inlet_all) # n_rows
	n_outlets = len(all_available.list_outlet_all) # n_cols-1
	n_features = n_inlets*(n_outlets+2) # add "no action" and "step counter" columns
	env = fs_gen(n_inlets, n_outlets+2, n_features)
	
	RL = DeepQNetwork(env.n_actions, env.n_features,
						n_inlets,
						n_outlets,
						learning_rate=P['learning_rate'],
						reward_decay=P['reward_decay'],
						e_greedy_max=P['e_greedy_max'],
						e_greedy_min=P['e_greedy_min'],
						replace_target_iter=P['replace_target_iter'],
						memory_size=P['memory_size'],
						batch_size=P['batch_size'],
						e_greedy_increment=P['e_greedy_increment'],
						randomness = 0,
						CNN_enable = P['CNN_enable'],
						N_hidden = P['N_hidden'],
						increment_degradation = P['increment_degradation']
						)

	# Load GNN model
	GNN = None
	if P['GNN_enable'] == True:
		GNN = tf.keras.models.load_model('dgcn')

	# Save model or restore model if interupted
	save_models_to = dir_result + 'save_model/'
	if not P['model_restore']:
		print("==================== train RL-GNN model ====================")
		RL_run(env,RL,GNN,user_inputs,all_available,f_rec,P)
		if P['model_save']: 
			RL.saver.save(RL.sess, save_models_to +'model_'+str(P['model_index'])+'.ckpt')
	else:
		RL.saver.restore(RL.sess,save_models_to +'model_'+str(P['model_index_restore'])+'.ckpt')
		print("==================== load saved model ====================")
		RL_run(env,RL,GNN,user_inputs,all_available,f_rec,P)
		if P['model_save']: 
			RL.saver.save(RL.sess, save_models_to +'model_'+str(P['model_index'])+'.ckpt')