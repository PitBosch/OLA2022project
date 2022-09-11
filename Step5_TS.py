from Learner import *
from Step3_TS import Step3_TS


class Step5_TS(Step3_TS):

    def __init__(self, env: Environment, beta_CR, beta_gw, learning_rate=1.):
        # call initializer of super class
        super().__init__(env, beta_CR, learning_rate)
        # GRAPH WEIGHTS:
        # store informations about beta parameters and inizialize graph weights list to store estimate after a complete run
        # NOTE: beta_gw is a list of 2 matrices 5x2 (2 parameters, 5 products, 2 secondary)
        self.initial_beta_gw = beta_gw.copy()         
        self.beta_param_gw = self.initial_beta_gw.copy()
        self.graph_weights_list = []

    def sample_graph_weights(self):
        # initialize the data structure to store sampled alpha ratios
        sampled_gw = np.zeros((5,2))

        for prod_ind in range(5):
            for sec_ind in range(2):
                # for each product sample the daily alpha from a beta
                a = self.beta_param_gw[0, prod_ind, sec_ind]
                b = self.beta_param_gw[1, prod_ind, sec_ind]
                sampled_gw[prod_ind, sec_ind] = np.random.beta(a, b)
        
        # Generate the proability matrix of graph weigths according to sampled values
        sampled_probabilities = self.generate_probailities(sampled_gw)
        
        return sampled_probabilities

    def generate_probailities(self, compact_gw):
        # initialize the probability matrix for the graph weights
        prob = np.zeros((5,5))
        # retrieve the secondary_dict from the environment
        sec_dict = self.env.secondary_dict
        # go through secondary dict values in order to insert the graph weights values in the right place
        for i,sec_ind_list in enumerate(list(sec_dict.values())):
            prob[i, sec_ind_list] = compact_gw[i,:]
        
        return prob

    def update_parameters(self, simul_result, price_combination):
        """ Update beta parameters of arms selected (passed with price_combination) with respect 
            the results of the simulation """
        
        estimated_CR = simul_result['CR_data']
        estimated_clicks = simul_result['clicks']
        estimated_visualizations = simul_result['visualizations']

        super().update_parameters(estimated_CR, price_combination)

        for prod_ind in range(5):
            for j, sec_ind in enumerate(list(self.env.secondary_dict.values())[prod_ind]):
                # GRAPH WEIGHTS 
                # update beta parameters with the following procedure:
                # a + number of times j-th secondary product (identified by sec_ind in output of simulate_day) has been clicked for
                #   the considered primary product
                # b + number of times j-th secondary product (identified by sec_ind in output of simulate_day) has NOT been clicked for
                #   the considered primary product
                self.beta_param_gw[0][prod_ind, j] += self.lr*estimated_clicks[prod_ind, sec_ind]
                self.beta_param_gw[1][prod_ind, j] += self.lr*(estimated_visualizations[prod_ind, sec_ind]-estimated_clicks[prod_ind, sec_ind])

    def iteration(self, daily_users):
        """ Method to execute a single iteration of the Thompson Sampling Algorithm. Objective: choose the right price_combination
        to maximize expected reward"""

        # 1) Sample from Beta distributions the estimate for conversion rates and graph_weights
        sampled_CR = self.sample_CR()
        sampled_gw = self.sample_graph_weights()
        # 2) Run the Greedy optimizer and select the best combination  
        opt_prices_combination = self.Greedy_opt.run(conversion_rates=[sampled_CR], graph_weights=[sampled_gw])["combination"]
        # 3) Fixed the prices for the day simulate the daily user iterations
        simulation_result = self.env.simulate_day(daily_users, opt_prices_combination, ["conversion_rates", "graph_weights"])
        # 4) Update Beta_parameters according to the simulation done
        self.update_parameters(simulation_result, opt_prices_combination)
        
        return opt_prices_combination


    def run(self, n_round=365, daily_users=200):
        """ Method to run Thompson Sampling algorithm given number of days to be simulated and the number of users simulated
            in each day. It updates the variable reward_history, appending the list of expected rewards obtained during the run."""

        # Initialize an empty list to store the price_combination decided each day
        rewards = []
        price_comb = []
        # Set beta_parameters to initial values for conversion rates and graph weights
        self.beta_param_CR = []
        self.beta_param_CR.append(self.initial_beta_CR[0].copy())
        self.beta_param_CR.append(self.initial_beta_CR[1].copy())
        self.beta_param_gw = copy.deepcopy(self.initial_beta_gw)

        for i in range(n_round):
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

