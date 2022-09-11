from Learner import *
from Step5_TS import Step5_TS

class Step6_TS_sw(Step5_TS):

    def __init__(self, env: Environment, beta_CR, beta_gw, sw, changes_dict):
        # call initializer of super class
        super().__init__(env, beta_CR, beta_gw)
        # SLIDING WINDOW FOR CONVERSION RATES
        self.sw = sw
        self.CR_data_history = []
        self.changes_dict = changes_dict
        self.initial_res_price_param = copy.deepcopy(env.users[0].res_price_params)

    def update_parameters(self, simul_result, price_combination):
        if len(self.CR_data_history) == self.sw:
            to_forget = self.CR_data_history.pop(0)
            self.beta_param_CR[0] -= to_forget[0]
            self.beta_param_CR[1] -= to_forget[1]
        
        n_purchase_simul = simul_result['CR_data'][0,:]
        n_visualizations_simul = simul_result['CR_data'][1,:]
        a = np.zeros((5,4))
        b = np.zeros((5,4))
        a[np.arange(5), price_combination] = n_purchase_simul
        b[np.arange(5), price_combination] = n_visualizations_simul - n_purchase_simul
        self.CR_data_history.append([a,b])

        return super().update_parameters(simul_result, price_combination)

    def run(self, n_round=365, daily_users=200):
        
        """ Method to run Thompson Sampling algorithm given number of days to be simulated and the number of users simulated
            in each day. It updates the variable reward_history, appending the list of expected rewards obtained during the run."""

        # Initialize an empty list to store the price_combination decided each day
        rewards = []
        price_comb = []
        # Set beta_parameters to initial values for conversion rates and graph weights
        self.beta_param_CR = self.initial_beta_CR[0].copy()
        self.beta_param_gw = copy.deepcopy(self.initial_beta_gw)
        self.CR_data_history = []

        for t in range(n_round):
            if t in self.changes_dict.keys(): 
                self.env.abrupt_change_deterministic([self.changes_dict[t]])  
                self.opt_reward = self.env.optimal_reward()[0]
            # Do a single iteration of the TS, and store the price combination chosen in the iteration
            opt_price_comb = self.iteration(daily_users)
            rewards.append(self.env.expected_reward(opt_price_comb))
            price_comb.append(opt_price_comb)
        # append the list of rewards obtained through the run
        self.reward_history.append(rewards)
        # append the list of price combinations selected through the run
        self.price_comb_history.append(price_comb)
        # compute and append the matrix of conversion rates estimate after the run
        A_CR = self.beta_param_CR[0]
        B_CR = self.beta_param_CR[1]
        self.cr_matrix_list.append(A_CR/(A_CR+B_CR))
        # compute and append the graph weights estimate after the run
        A_gw = self.beta_param_gw[0]
        B_gw = self.beta_param_gw[1]
        compact_gw = A_gw/(A_gw+B_gw)
        self.graph_weights_list.append(self.generate_probailities(compact_gw))
        # Set to initial values the user reservation price parameters
        self.env.abrupt_change_deterministic([self.initial_res_price_param])