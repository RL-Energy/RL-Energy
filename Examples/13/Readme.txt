v25:
Debug 8: combine debug 4 and 5
a) increased feasible reward, decreased unavailable reward
b) Fewer action options for larger step
c) RL step on "unavailable", mark as 1

Debug_8_2: combine debug 8 with 5_2
trim more columns/outlets/actions according to physics constraints

Debug_8_3: instead of trimming action options, modifiy initial observation

v24:
initialization of observation:
	make product_0.inlet only available to flash_0.liq_outlet

v23:
all_available() consistent with methanol example v6
add user_inputs_14 (consistent with methanol_v6)
make previous hidden outputs flash_0.vap_outlet, and flash_0.liq_outlet available
pre-set connection: flash_0.vap_outlet = product_0.inlet

v22:
More updates:
1. avoid savetext issue (may stop in the RL process)
2. add an opition to P for no-action penalty

More updates:
a) Two way of generating random action
	1. all spots in a row (include unavailable)
	2. only available spots in a row
b) Record and plot "end step"
c) Plot R and extraR separately 
d) Initial reward cannot be -1000, should be 0
e) extra_R cannot be accumulated

Major updates:
1. we have user_inputs and all_available, and the observation is built according to the all_available
2. the observation is n_inlets*(n_outlets+2), the last two columns denote "no-action" and "step counter"
3. reward and reward_: reward is the reward of last observation, to avoid repeated pre-screen and IDAES evaluation
4. reward_ and extra_reward_: reward_ from -1000 to 1000, extra_reward_ denotes extra reward or penalty (e.g.: penalty for no-action in partially filled row, reward for no-action in empty row, reward for high purity and flow rate)

To be noted:
1. only inlets and outlets in user_inputs are available, no-action and step-counter columns are always available
2. although the observations are n_inlets*(n_outlets+2) matrix, only the first n_outlets columns are saved in obs_runIDAES_flowsheet