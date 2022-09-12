from Environment import *
from Greedy_optimizer import *
import numpy as np
from Step7_TS import *

class ContextGeneration():
    #What we need to do is to evaluate every possible partition of the space of the features, 
    #and for every one of these we need to evaluate whether partitioning is better than not doing that
    def __init__(self, env: Environment, confidence, simul_history, est_feat_prob_mat,beta_CR, beta_alpha, n_prod_data, learning_rate = 1.0):
        # Real environment
        self.env = env
        # Greedy Optimizer to compute optimal reward
        self.greedy = Greedy_optimizer(self.env)
        #confidence for lower confidence bound
        self.confidence = confidence
        # simulation history 
        self.simul_history = simul_history
        # estimate of the probability to observe a couple of feature 
        self.est_feat_prob_mat = est_feat_prob_mat
        # learning rate
        self.lr = learning_rate
        # CONVERSION RATES :
        # store informations about beta parameters and inizialize CR matrix to store estimate after a complete run
        self.initial_beta_CR = beta_CR.copy()
        # ALPHA RATIOS :
        # # store informations about beta parameters and inizialize alpha est to store estimate after a complete run
        self.initial_beta_alpha = beta_alpha.copy()             # Note beta_alpha is a 2x5 matrix (2 parameters, 5 products)
        # N PRODUCT SOLD
        # n_prod_data is a 2x5 matrix:
        # first row --> number of product sold for a specific product in the simulations
        # second row --> number of times user bought a specific product (for each product obviously)
        self.initial_n_prod_data = n_prod_data
    
    def update_history(self, simul_hist, feat_prob_mat):
        self.simul_history = simul_hist
        self.est_feat_prob_mat = feat_prob_mat

    def context_value(self, group_list):
        dim = len(group_list)
        cr, alpha, n_prod, p = self.get_group_info(group_list)
        opt_rew = self.greedy.run(conversion_rates=[cr]*dim, alphas_ratio=[alpha]*dim, n_prod=[n_prod]*dim,
                                        group_list=group_list, feat_prob_mat=self.est_feat_prob_mat)['expected_reward']
        p_lcb = p if p == 1 else self.lcb(p,group_list)
        cv = p_lcb*self.lcb(opt_rew, group_list)
        return cv

    def lcb(self, data, group_list): #vale sia per i reward che per le probabilities
        # compute the number of users observed for the tested group
        n_data = 0
        for feat_couple in group_list:
            i = feat_couple[0]
            j = feat_couple[1]
            feat_key = str(i)+str(j)
            n_data += self.simul_history[feat_key]['n_users']
        n_data = n_data * self.lr
        # return the lower confidence bound for the datum analyzed
        return max(0, data - np.sqrt(-np.log(self.confidence)/(2*n_data)))

    def get_group_info(self, group_list):
        #initialize bought an seen structure
        bought = np.zeros((5,4))
        seen = np.zeros((5,4))
        #initialize n_prod to an empty array of dim 5
        initial_prod = np.zeros(5)
        #initialize n_prod_data to a matrix 2x5 of zeros
        n_prod_data = np.zeros((2,5))
        # initialize prob estimate to 0.
        prob_est = 0.
        
        for feat_couple in group_list:
            #loop through the list
            feat1, feat2 = feat_couple
            feat_key = str(feat1)+str(feat2)
            bought += self.simul_history[feat_key]['CR_bought'].copy()
            seen += self.simul_history[feat_key]['CR_seen'].copy()
            initial_prod += self.simul_history[feat_key]['initial_prod'].copy()
            n_prod_data += self.simul_history[feat_key]['n_prod_sold'].copy()
            prob_est += self.est_feat_prob_mat[feat1, feat2]
        # conversion rates estimate
        bought += self.initial_beta_CR[0]
        seen += self.initial_beta_CR[0]+self.initial_beta_CR[1]
        cr_est = bought/seen
        # alpha ratios estimate
        initial_prod += self.initial_beta_alpha[0]
        alpha_est = initial_prod/np.sum(initial_prod)
        # number of product sold estimate
        n_prod_data += self.initial_n_prod_data
        n_prod_est = n_prod_data[0]/n_prod_data[1]        
        
        return cr_est, alpha_est, n_prod_est, prob_est

    def run(self):
        # Initial context is always all users in same group
        context = np.array([[0,0],[0,0]])
        # At first I can split on both the features
        split_var_list = [0,1]
        if self.split(context, 0, split_var_list): 
            self.split(context, 0, split_var_list)
            self.split(context, 1, split_var_list)
        
        return context

    def split(self, context, group_to_split: int, split_var_list: list[int]) :
        # At first compute the context value for the group we want to split
        group0 = feature_matrix_to_list(context)[group_to_split]
        context_value0 = self.context_value(group0)
        # Then consider possible splits and 
        values_list = np.zeros(len(split_var_list))
        split_list = []
        i_list, j_list = np.where(context == group_to_split)
        for (k, var) in enumerate(split_var_list):
            if var == 0:
                split_group1 = list(zip(i_list[i_list == 0], j_list[i_list == 0]))
                split_group2 = list(zip(i_list[i_list == 1], j_list[i_list == 1]))
            if var == 1:
                split_group1 = list(zip(i_list[j_list == 0], j_list[j_list == 0]))
                split_group2 = list(zip(i_list[j_list == 1], j_list[j_list == 1]))

            values_list[k] = self.context_value(split_group1) + self.context_value(split_group2)

            split_list.append(copy.deepcopy(split_group2))
        
        max_i = np.argmax(values_list)
        if values_list[max_i] > context_value0:

            new_group = np.max(context)+1
            split = split_list[max_i]
            for feat_couple in split:
                i, j = feat_couple
                context[i,j] = new_group 
            split_done = True
            if len(split_var_list) == 2:
                split_var_list.pop(max_i) # reference to split_var_list outside the scope 
        else:
            split_done = False

        return split_done