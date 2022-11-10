HDA_IDAES_PostProcessing.py is for post-processing a "complete" RL test, including all observations found in the process and their status, rewards, etc.
	1. it is capable of checking if the RL test is finished, checking the performance of found observations, summurizing the amounts of feasible flowsheets, visualizing best-performance flowsheets, etc. User can visualize certain observations with minor modifications.
	2. user need to provide the test index, unit-pool associated, result directory, etc.
		
HDA_IDAES_Individual_PostProcessing.py is for post-processing "one" observation from RL test.
	this script is for re-running a specific observation and visualizing it, user need to provide the observation, associated unit-pool.