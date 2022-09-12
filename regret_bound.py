import numpy as np
from Environment import *
from scipy.special import kl_div

def TS_regret(env:Environment, time_horiz, eps):

    possible_combinations = []
    exp_rew_list = []
    distribution_list = []
    for i1 in range(4):
        for i2 in range(4):
            for i3 in range(4):
                for i4 in range(4):
                    for i5 in range(4):
                        combination = [i1, i2, i3, i4, i5]
                        possible_combinations.append(combination.copy())
                        distribution = []
                        rew = 0.
                        for prod_ind in range(5):    
                            paths_list = []
                            env.explore_path(paths_list, None, prod_ind, combination, 0)
                            for path in paths_list:
                                distribution.append(path.probability)
                                rew += path.expected_return()
                        exp_rew_list.append(rew)
                        distribution_list.append(distribution.copy())
                                

    max_rew = max(exp_rew_list)
    i_max = np.argmax(exp_rew_list)
    delta_a = []
    kl_a = []
    for i in range(len(possible_combinations)):
        if i != i_max :
            delta_a.append(max_rew-exp_rew_list[i])
            kl_a.append(np.sum(kl_div(distribution_list[i_max], distribution_list[i])))

    regret_ub = np.zeros(time_horiz)
    for t in range(time_horiz):
        regret_ub[t] = (1+eps)*np.sum(np.array(delta_a)*(np.log(t+1)+np.log(np.log(t+1)))/np.array(kl_a))
    
    return regret_ub

def ucb_regret(env:Environment, time_horiz):
    possible_combinations = []
    exp_rew_list = []
    for i1 in range(4):
        for i2 in range(4):
            for i3 in range(4):
                for i4 in range(4):
                    for i5 in range(4):
                        combination = [i1, i2, i3, i4, i5]
                        possible_combinations.append(combination.copy())
                        exp_rew_list.append(env.expected_reward(combination))

    max_rew = max(exp_rew_list)
    i_max = np.argmax(exp_rew_list)

    delta_a = []
    for i in range(len(possible_combinations)):
        if i != i_max :
            delta_a.append(max_rew-exp_rew_list[i])
    delta_a = np.array(delta_a)

    regret_ub = np.zeros(time_horiz)
    for t in range(time_horiz):
        regret_ub[t] = np.sum(4*np.log(t+1)/delta_a + 8*delta_a)
    
    return regret_ub