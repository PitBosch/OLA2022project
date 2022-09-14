import numpy as np
from Environment import *
from scipy.special import kl_div

def TS_regret(env:Environment, time_horiz, eps):

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
                        
    possible_combinations = np.array(possible_combinations)
    exp_rew_list = np.array(exp_rew_list)                    

    max_rew = np.max(exp_rew_list)
    i_max = np.argmax(exp_rew_list)
    optimal_combination = possible_combinations[i_max]
    
    delta_a = np.zeros((5,3))
    kl_a = np.zeros((5,3))

    for prod_ind in range(5):
        price_max = optimal_combination[prod_ind]
        i = 0
        for price_ind in range(4):
            if price_ind != price_max:
                comb_ind = np.where(possible_combinations[:,prod_ind] == price_ind)[0]
                mean_rew = np.mean(exp_rew_list[comb_ind])
                delta_a[prod_ind, i] = max_rew - mean_rew
                kl_a[prod_ind, i] = kl_div(max_rew, mean_rew)
                i += 1


    regret_ub = np.zeros(time_horiz)
     
    for t in range(1,time_horiz):
        regret_ub[t] = (1+eps)*np.sum(delta_a*(np.log(t+1)+np.log(np.log(t+1)))/kl_a)
    
    regret_ub[0] = 0.8*regret_ub[1]

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
                        
    possible_combinations = np.array(possible_combinations)
    exp_rew_list = np.array(exp_rew_list)                    

    max_rew = np.max(exp_rew_list)
    i_max = np.argmax(exp_rew_list)
    optimal_combination = possible_combinations[i_max]
    
    delta_a = np.zeros((5,3))
    
    for prod_ind in range(5):
        price_max = optimal_combination[prod_ind]
        i = 0
        for price_ind in range(4):
            if price_ind != price_max:
                comb_ind = np.where(possible_combinations[:,prod_ind] == price_ind)[0]
                mean_rew = np.mean(exp_rew_list[comb_ind])
                delta_a[prod_ind, i] = max_rew - mean_rew
                i += 1

    regret_ub = np.zeros(time_horiz)
    for t in range(time_horiz):
        regret_ub[t] = np.sum(4*np.log(t+1)/delta_a + 8*delta_a)
    
    return regret_ub