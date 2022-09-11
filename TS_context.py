from Step4_TS import *

class TS_context(Step4_TS):
    def __init__(self, env, beta_CR, beta_alpha, n_prod_data, group_list, learning_rate = 1.0):
        self.group_list = group_list.copy()
        self.group_dim = len(self.group_list)
        # pass learning rate to the class
        self.lr = learning_rate
        # CONVERSION RATES :
        # store informations about beta parameters and inizialize CR matrix to store estimate after a complete run
        self.initial_beta_CR = beta_CR.copy()
        self.beta_param_CR = self.initial_beta_CR.copy()
        self.cr_matrix_list = []
        # ALPHA RATIOS :
        # # store informations about beta parameters and inizialize alpha est to store estimate after a complete run
        self.initial_beta_alpha = beta_alpha.copy()             # Note beta_alpha is a 2x5 matrix (2 parameters, 5 products)
        self.beta_param_alpha = self.initial_beta_alpha.copy()  
        self.alpha_ratios_list = []
        # N PRODUCT SOLD
        # n_prod_data is a 2x5 matrix:
        # first row --> number of product sold for a specific product in the simulations
        # second row --> number of times user bought a specific product (for each product obviously)
        self.initial_n_prod_data = n_prod_data
        self.n_prod_data = n_prod_data.copy()
        # self.mean_prod_sold = n_prod_data[0]/n_prod_data[1]
        self.n_prod_list = []
        # Real environment
        self.env = env
        # Greedy optimizer to decide the price combination each day
        self.Greedy_opt = Greedy_optimizer(self.env)
        # Initialize history of theoretical rewards 
        self.reward_history = []
        # History of prices combination chosen
        self.price_comb_history = []

    def iteration(self, est_feat_prob_mat):
        """ Method to execute a single iteration of the Thompson Sampling Algorithm. Objective: choose the right price_combination
        to maximize expected reward"""

        # 1) Sample from Beta distributions the estimate for conversion rates and alpha ratios
        sampled_CR = self.sample_CR()
        sampled_alpha = self.sample_alpha()
        sampled_n_prod = self.sample_n_prod()
        # 2) Run the Greedy optimizer and select the best combination
        CR_list = [sampled_CR]*self.group_dim
        alpha_list = [sampled_alpha]*self.group_dim  
        n_prod_list = [sampled_n_prod]*self.group_dim  
        opt_prices_combination = self.Greedy_opt.run(conversion_rates=CR_list, alphas_ratio=alpha_list, n_prod=n_prod_list,
                                                        group_list = self.group_list, feat_prob_mat = est_feat_prob_mat)["combination"]
        
        return opt_prices_combination