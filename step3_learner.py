from Learner import *

class TS_learner3(Learner):

    def __init__(self, beta_parameters, env : Environment):
        # call initializer of super class
        super().__init__(env)
        # list of 2 matrices n_products x n_prices (5x4 in our case)
        self.initial_beta = []
        self.initial_beta.append(beta_parameters[0].copy())
        self.initial_beta.append(beta_parameters[1].copy())
        self.beta_parameters = []

    def sample_CR(self):
        # initialize the data structure to store sampled conversion rates
        sampled_CR = np.zeros((5,4))

        for prod_ind in range(5):
            for price_ind in range(4):
                # for each product and for each possible price per product
                # sample the conversion rate from beta distributions
                a = self.beta_parameters[0][prod_ind, price_ind]
                b = self.beta_parameters[1][prod_ind, price_ind]
                sampled_CR[prod_ind, price_ind] = np.random.beta(a, b)

        return sampled_CR

    def update_parameters(self, sampled_CR, estimated_CR, price_combination):
        """ Update beta parameters of arms selected (passed with price_combination) with respect 
            the results of the simulation """

        for prod_ind in range(5) :
            # retrieve the price index for the considered product 
            price_ind = price_combination[prod_ind]
            
            # update beta parameters with the following procedure:
            # if estimated_CR[prod_ind] > sampled_CR[prod_ind, price_ind] :
                # in the simulation we have a conversion rate HIGHER than he sampled one
                # ==> increase parameter a of the corresponding beta ditribution
              #  self.beta_parameters[0][prod_ind, price_ind] += 1
            #else :
                # in the simulation we have a conversion rate HIGHER than he sampled one
                # ==> increase parameter b of the corresponding beta distribution
              #  self.beta_parameters[1][prod_ind, price_ind] += 1

            self.beta_parameters[0][prod_ind, price_ind] += estimated_CR[0, prod_ind]
            self.beta_parameters[1][prod_ind, price_ind] += estimated_CR[1, prod_ind] - estimated_CR[0, prod_ind]

    def iteration(self, daily_users):
        """ Method to execute a single iteration of the Thompson Sampling Algorithm.
            Objective: choose the right price_combination to maximize expected reward """

        # 1) Sample from Beta distributions the estimated conversion rate
        sampled_CR = self.sample_CR()
        
        # 2) Run the Greedy optimizer and select the best combination  
        opt_prices_combination = self.Greedy_opt.run(conversion_rates = [sampled_CR])["combination"]

        # 3) Fixed the prices for the day simulate the daily user iterations
        estimated_CR = self.env.simulate_day(daily_users, opt_prices_combination, ["conversion_rates"])['CR_vector']

        # 4) Update Beta_parameters according to the simulation done
        self.update_parameters(sampled_CR, estimated_CR, opt_prices_combination)

        return opt_prices_combination

    def run(self, n_round = 365, daily_users = 200) :
        """ Method to run Thompson Sampling algorithm given number of days to be simulated and 
            the number of users simulated in each day. It update the variable reward_history,
            appendign the list of expected rewards obtained during the run. """

        # Initialize an empty list to store the price_combination decided each day
        rewards = []
        # Set beta_parameters to initial values
        self.beta_parameters = []
        self.beta_parameters.append(self.initial_beta[0].copy())
        self.beta_parameters.append(self.initial_beta[1].copy())
        
        for i in range(n_round) :
            # Do a single iteration of the TS andd store the price combination chosen in the iteration
            opt_price_com = self.iteration(daily_users)
            rewards.append(self.env.expected_reward(opt_price_com))
        
        self.reward_history.append(rewards)
