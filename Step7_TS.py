
# from ContextGeneration import ContextGeneration
from Learner import *
from TS_context import *

class Step7_TS():

    def __init__(self, env: Environment, beta_CR, beta_alpha, n_prod_data, learning_rate = 1.0) :
        # Real environment
        self.env = env
        # Greedy optimizer to decide the price combination each day
        self.Greedy_opt = Greedy_optimizer(self.env)
        # Optimal theoretical reward
        self.opt_reward = np.sum(env.optimal_reward(Disaggregated=True)[0]*env.user_cat_prob)
        # Learning Rate
        self.lr = learning_rate
        # Initialize history of theoretical rewards 
        self.reward_history = []
        # History of prices combination chosen
        self.price_comb_history = []
        # CONTEXT INITIALIZATION
        self.context = np.array([[0,0],[0,0]])
        # SIMULATION HISTORY DIVIDED FOR COUPLE OF FEATURES (initialization)
        simul_dict = {'n_users' : 0,
                      'CR_bought' : np.zeros((5,4)),
                      'CR_seen' : np.zeros((5,4)),
                      'initial_prod' : np.zeros(5),
                      'n_prod_sold' : np.zeros((2,5))}
        self.initial_simul_history = {'00': copy.deepcopy(simul_dict),
                                      '01': copy.deepcopy(simul_dict),
                                      '10': copy.deepcopy(simul_dict),
                                      '11': copy.deepcopy(simul_dict)}
        self.simul_history = copy.deepcopy(self.initial_simul_history)
        # Initialize list to store multiple runs informations
        #self.price_comb_history = []
        self.reward_history = []
        self.context_history = []
        # PROBABILITY MATRIX FOR THE COUPLE OF FEATURES
        self.est_feat_prob_mat = np.array([[0.25, 0.25], [0.25, 0.25]])
        # CONVERSION RATES :
        # store informations about beta parameters and inizialize CR matrix to store estimate after a complete run
        self.initial_beta_CR = []
        self.initial_beta_CR.append(beta_CR[0].copy()) # Note that beta_CR is a list of 2 matrix 5x4 
        self.initial_beta_CR.append(beta_CR[1].copy()) # (2 parameters, 5 products, 4 possible prices)
        # ALPHA RATIOS :
        # # store informations about beta parameters and inizialize alpha est to store estimate after a complete run
        self.initial_beta_alpha = beta_alpha.copy()             # Note beta_alpha is a 2x5 matrix (2 parameters, 5 products)
        # N PRODUCT SOLD
        # n_prod_data is a 2x5 matrix:
        # first row --> number of product sold for a specific product in the simulations
        # second row --> number of times user bought a specific product (for each product obviously)
        self.initial_n_prod_data = n_prod_data
        # self.mean_prod_sold = n_prod_data[0]/n_prod_data[1]
        self.n_prod_list = []
        #
        self.learner_list = []
    
    def param_info(self, group):
        a = np.ones((5,4))
        b = np.ones((5,4))
        beta_alpha = np.ones((2,5))
        n_prod_data = np.ones((2,5))
        i_list,j_list = np.where(self.context == group)
        for k in range(len(i_list)):
            feat1 = i_list[k]
            feat2 = j_list[k]
            feat_key = str(feat1)+str(feat2)
            a += self.simul_history[feat_key]['CR_bought']
            b += self.simul_history[feat_key]['CR_seen'] - self.simul_history[feat_key]['CR_bought']
            beta_alpha += self.simul_history[feat_key]['initial_prod']
            n_prod_data += self.simul_history[feat_key]['n_prod_sold']
        beta_CR = [a, b]
        
        return beta_CR, beta_alpha, n_prod_data

    def update_feat_prob_mat(self):
        M = np.ones((2,2))
        for key in self.simul_history.keys():
            i = int(key[0])
            j = int(key[1])
            M[i,j] += self.simul_history['n_users']

        self.est_feat_prob_mat = M/np.sum(M)    

    def update_learner_list(self):
        self.learner_list = []
        n_groups = np.max(self.context)+1
        feature_list = feature_matrix_to_list(self.context)
        for group in range(n_groups):
            CR_beta, alpha_beta, n_prod_info = self.param_info(group)
            group_list = feature_list[group]
            self.learner_list.append(TS_context(self.env, CR_beta, alpha_beta, n_prod_info, group_list, self.lr))
        return

    def update_simul_history(self, daily_simul, price_comb_list):
    
        for key in daily_simul.keys():
            # N USERS
            self.simul_history[key]['n_users'] += daily_simul[key]['n_users']
            # CONVERSION RATES
            # simul_history store 2 matrix 5x4 (products x possible prices) for conversion rates informations
            # BUT daily simul store the informations only for the chosen price combination
            # According to the key retrieve the right combination price
            i = int(key[0])
            j = int(key[1])
            group = self.context[i,j]
            price_comb = price_comb_list[group]
            # with [np.arange(5), price_comb] for each row only the values corresponding to the chosen price combination are updated
            self.simul_history[key]['CR_bought'][np.arange(5), price_comb] += daily_simul[key]['CR_data'][0]
            self.simul_history[key]['CR_seen'][np.arange(5), price_comb] += daily_simul[key]['CR_data'][1]
            # ALPHA RATIOS
            self.simul_history[key]['initial_prod'] += daily_simul[key]['initial_prod']
            # NUMBER OF PRODUCTS SOLD
            self.simul_history[key]['n_prod_sold'] += daily_simul[key]['n_prod_sold']
        
        return
    
    def compute_info(self, simul):
        """ Method that aggregates the informations obtained by the simulation"""
        info_list = []
        n_groups = np.max(self.context)+1
        for group in range(n_groups):
            i_list,j_list = np.where(self.context == group)
            feat1 = i_list[0]
            feat2 = j_list[0]
            feat_key = str(feat1)+str(feat2)
            info_list.append(simul[feat_key])
            if len(i_list) > 1:
                for k in range(1, len(i_list)):
                    feat1 = i_list[k]
                    feat2 = j_list[k]
                    feat_key = str(feat1)+str(feat2)
                    simul_dict = simul[feat_key]
                    for key in simul_dict.keys():
                        info_list[-1][key] += simul_dict[key] 

        return info_list
                
    def iteration(self):
        # Sample from assumed distributions all parameters and retrieve the optimal prices combination
        # for each group in the context
        opt_combination_list = []
        for learner in self.learner_list:
            opt_comb = learner.iteration(self.est_feat_prob_mat)
            opt_combination_list.append(opt_comb)
        
        return opt_combination_list

    def compute_reward(self, price_comb_list):
        """ Method to compute the THEORETICAL expected reward with the context generated and the price combination chosen.
            For each group in the context compute the expected reward for the corresponding price combination and sum them"""
        exp_rew = 0.
        feature_list = feature_matrix_to_list(self.context)
        for k, group_list in enumerate(feature_list):
            # compute probability of the group:
            group_prob = np.sum(compute_group_prob(group_list, self.env.feat_prob_matrix))
            group_rew = self.env.expected_reward(price_combination=price_comb_list[k], group_list=group_list)
            exp_rew += group_rew*group_prob

        return exp_rew
        
    def run(self, n_days = 300, daily_users = 200):
        # Initialize an empty list to store the price_combination decided each day
        reward_list = []
        #price_comb_list = []
        context_list = []
        # Initialize with initial context (default: all users aggregated)
        self.context = np.array([[0,0],[0,0]])
        self.simul_history = copy.deepcopy(self.initial_simul_history)
        self.update_learner_list()
        context_dict = {0: np.array([[0,0],[0,0]]),
        1: np.array([[0,0],[1,1]]),
        2: np.array([[0,1],[2,2]]),
        3: np.array([[0,1],[2,3]]),}
        # A complete run of n_days, with context generation algorithm run every 2 weeks (14 days)
        for t in range(n_days):
            if t!=0 and t%14 == 0:
                #choice = np.random.randint(0,4) if t < 100 else 2
                choice = 2
                new_context = context_dict[choice]
                self.context = new_context.copy() #################################### <-- CONTEXT GENERATION QUI BOSCHINIIIIIII
                self.update_learner_list()
                context_list.append(self.context.copy())
            # Do a single iteration of the TS, and store the LIST of price combinations chosen for each group in the context
            opt_price_comb = self.iteration()
            reward = self.compute_reward(opt_price_comb)
            reward_list.append(reward)
            #price_comb_list.append(opt_price_comb)
            # Simulate interactions of users for a day and update simul_history
            daily_simul = self.env.simulate_day_context(daily_users, opt_price_comb, self.context,
                                                        ["conversion_rates", "alpha_ratios", "products_sold"])
            self.update_simul_history(daily_simul, opt_price_comb)
            # Update parameters for each learner
            daily_info = self.compute_info(daily_simul)
            for k, learner in enumerate(self.learner_list):
                learner.update_parameters(daily_info[k], opt_price_comb[k])
        # self.price_comb_history.append(price_comb_list.copy())
        self.reward_history.append(reward_list.copy())
        self.context_history.append(context_list.copy())
    
        return
    