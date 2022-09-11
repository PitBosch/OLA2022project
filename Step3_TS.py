from Learner import *


class Step3_TS(Learner):

    def __init__(self, env: Environment, beta_parameters, learning_rate=1.):
        # call initializer of super class
        super().__init__(env)
        # pass learning rate to the class
        self.lr = learning_rate
        # CONVERSION RATES :
        # store informations about beta parameters and inizialize CR matrix to store estimate
        self.initial_beta = beta_parameters.copy()
        self.beta_parameters = self.initial_beta.copy()
        self.cr_matrix_list = []

    def sample_CR(self):
        # initialize the data structure to store sampled conversion rates
        sampled_CR = np.zeros((5, 4))

        for prod_ind in range(5):
            for price_ind in range(4):
                # for each product and for each possible price per product
                # sample the conversion rate from beta distributions
                a = self.beta_parameters[0, prod_ind, price_ind]
                b = self.beta_parameters[1, prod_ind, price_ind]
                sampled_CR[prod_ind, price_ind] = np.random.beta(a, b)

        return sampled_CR

    def update_parameters(self, estimated_CR, price_combination):
        """ Update beta parameters of arms selected (passed with price_combination) with respect 
            the results of the simulation """

        for prod_ind in range(5):
            # retrieve the price index for the considered product 
            price_ind = price_combination[prod_ind]
            
            # update beta parameters with the following procedure:
            # a + number of purchase
            # b + (number of time users saw product i - number of purchase)
            self.beta_parameters[0][prod_ind, price_ind] += self.lr*estimated_CR[0, prod_ind]
            self.beta_parameters[1][prod_ind, price_ind] += self.lr*(estimated_CR[1, prod_ind] - estimated_CR[0, prod_ind])

    def iteration(self, daily_users):
        """ Method to execute a single iteration of the Thompson Sampling Algorithm. Objective: choose the right price_combination
        to maximize expected reward"""

        # 1) Sample from Beta distributions the estimated conversion rate
        sampled_CR = self.sample_CR()
        # 2) Run the Greedy optimizer and select the best combination  
        opt_prices_combination = self.Greedy_opt.run(conversion_rates=[sampled_CR])["combination"]
        # 3) Fixed the prices for the day simulate the daily user iterations
        estimated_CR = self.env.simulate_day(daily_users, opt_prices_combination, ["conversion_rates"])['CR_data']
        # 4) Update Beta_parameters according to the simulation done
        self.update_parameters(estimated_CR, opt_prices_combination)
        
        return opt_prices_combination


    def run(self, n_round=365, daily_users=200):
        """ Method to run Thompson Sampling algorithm given number of days to be simulated and the number of users simulated
            in each day. It updates the variable reward_history, appending the list of expected rewards obtained during the run."""

        # Initialize an empty list to store the price_combination decided each day
        rewards = []
        price_comb = []
        # Set beta_parameters to initial values
        self.beta_parameters = []
        self.beta_parameters.append(self.initial_beta[0].copy())
        self.beta_parameters.append(self.initial_beta[1].copy())
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
        A = self.beta_parameters[0]
        B = self.beta_parameters[1]
        self.cr_matrix_list.append(A/(A+B))
