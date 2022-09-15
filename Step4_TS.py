from Learner import *
from Step3_TS import Step3_TS


class Step4_TS(Step3_TS):

    def __init__(self, env: Environment, beta_CR, beta_alpha, n_prod_data, learning_rate=1.):
        # call initializer of super class
        super().__init__(env, beta_CR, learning_rate)
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

    def sample_alpha(self):
        # initialize the data structure to store sampled alpha ratios
        sampled_alpha = np.zeros(5)

        for prod_ind in range(5):
            # for each product sample the daily alpha from a beta
            a = self.beta_param_alpha[0,prod_ind]
            b = self.beta_param_alpha[1,prod_ind]
            sampled_alpha[prod_ind] = np.random.beta(a, b)
        
        # normalize sampled_alpha such that values sum to 1
        sampled_alpha /= np.sum(sampled_alpha)
        
        return sampled_alpha 
    
    def sample_n_prod(self):
        # initialize the data structure to store the mean number of product sold for product
        shape = self.n_prod_data[0]
        scale = 1/self.n_prod_data[1]

        return np.random.gamma(shape = shape, scale = scale)

    def update_parameters(self, simul_result, price_combination):
        """ Update beta parameters of arms selected (passed with price_combination) with respect 
            the results of the simulation """
        
        estimated_CR = simul_result['CR_data']
        estimated_alpha = simul_result['initial_prod']
        estimated_n_prod_data = simul_result['n_prod_sold']

        super().update_parameters(estimated_CR, price_combination)

        for prod_ind in range(5):
            # retrieve the price index for the considered product 
            price_ind = price_combination[prod_ind]
            # ALPHA RATIOS
            # update beta parameters with the following procedure:
            # a + number of times product i was the initial product
            # b + number of times other products where the initial product
            self.beta_param_alpha[0, prod_ind] += self.lr*estimated_alpha[prod_ind]
            self.beta_param_alpha[1, prod_ind] += self.lr*(np.sum(estimated_alpha) - estimated_alpha[prod_ind])
            # N PROD SOLD
            # simply update the collected data on past purchases adding informations of the daily simulation and
            # compute the mean of the number of products sold for each product
            self.n_prod_data += estimated_n_prod_data
            #self.mean_prod_sold = self.n_prod_data[0]/self.n_prod_data[1]

    def iteration(self, daily_users):
        """ Method to execute a single iteration of the Thompson Sampling Algorithm. Objective: choose the right price_combination
        to maximize expected reward"""

        # 1) Sample from Beta distributions the estimate for conversion rates and alpha ratios
        sampled_CR = self.sample_CR()
        sampled_alpha = self.sample_alpha()
        sampled_n_prod = self.sample_n_prod()
        # 2) Run the Greedy optimizer and select the best combination  
        opt_prices_combination = self.Greedy_opt.run(conversion_rates=[sampled_CR], alphas_ratio=[sampled_alpha], n_prod=[sampled_n_prod])["combination"]
        # 3) Fixed the prices for the day simulate the daily user iterations
        simulation_result = self.env.simulate_day(daily_users, opt_prices_combination, ["conversion_rates", "alpha_ratios", "products_sold"])
        # 4) Update Beta_parameters according to the simulation done
        self.update_parameters(simulation_result, opt_prices_combination)
        
        return opt_prices_combination


    def run(self, n_round=365, daily_users=200):
        """ Method to run Thompson Sampling algorithm given number of days to be simulated and the number of users simulated
            in each day. It updates the variable reward_history, appending the list of expected rewards obtained during the run."""

        # Initialize an empty list to store the price_combination decided each day
        rewards = []
        price_comb = []
        # Set beta_parameters to initial values for conversion rates and alpha ratios
        self.beta_param_CR = self.initial_beta_CR.copy()
        self.beta_param_alpha = self.initial_beta_alpha.copy()
        # Set data to estimate number of product sold to initial data
        self.n_prod_data = self.initial_n_prod_data.copy()
        mean_prod_sold = self.n_prod_data[0]/self.n_prod_data[1]

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
        # compute and append the alphas ratio estimate after the run
        a_alpha = self.beta_param_alpha[0,:]
        b_alpha = self.beta_param_alpha[1,:]
        self.alpha_ratios_list.append(a_alpha/(a_alpha+b_alpha))
        # compute and append estimate of number of product sold for each product
        mean_prod_sold = self.n_prod_data[0]/self.n_prod_data[1]
        self.n_prod_list.append(mean_prod_sold)