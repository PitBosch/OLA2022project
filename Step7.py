
from ContextGeneration import ContextGeneration
from Step4_TS import *

class Step7(Step4_TS):

    def __init__(self, env: Environment, beta_CR, beta_alpha, learning_rate=1., context=np.array([[0,0],[0,0]])):
        super().__init__(env, beta_CR,beta_alpha,learning_rate)
        self.context=context.copy()

    def update_parameters(self, simul_result, price_combination):
        """ Update beta parameters of arms selected (passed with price_combination) with respect 
            the results of the simulation """
        
        estimated_CR = simul_result['CR_data']
        estimated_alpha = simul_result['initial_prod']
        estimated_n_prod_data = simul_result['n_prod_sold']

        for prod_ind in range(5):
            # retrieve the price index for the considered product 
            price_ind = price_combination[prod_ind]
            # CONVERSION RATES:
            # update beta parameters with the following procedure:
            # a + number of purchase
            # b + (number of time users saw product i - number of purchase)
            self.beta_param_CR[0][prod_ind, price_ind] += self.lr*estimated_CR[0, prod_ind]
            self.beta_param_CR[1][prod_ind, price_ind] += self.lr*(estimated_CR[1, prod_ind] - estimated_CR[0, prod_ind])
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
            self.mean_prod_sold = self.n_prod_data[0]/self.n_prod_data[1]

    def iteration(self, daily_users, context_matrix):
        """ Method to execute a single iteration of the Thompson Sampling Algorithm. Objective: choose the right price_combination
        to maximize expected reward"""
        list_sampled_CR=[]
        list_sampled_alpha=[]
        list_opt_prices_combination=[]
        # 1) Sample from Beta distributions the estimate for conversion rates and alpha ratios
        for i in range(len(self.context)):
            #the idea is to sample n different CR matrix based on different context 
            list_sampled_CR.append(self.sample_CR())
            list_sampled_alpha.append(self.sample_alpha())
            # 2) Run the Greedy optimizer and select the best combination  
            list_opt_prices_combination[i] = self.Greedy_opt.run(conversion_rates=[list_sampled_CR[i]], alphas_ratio=[list_sampled_alpha[i]])["combination"]
            # 3) Fixed the prices for the day simulate the daily user iterations
        simulation_result = self.env.simulate_day_context(daily_users, list_opt_prices_combination, context_matrix, ["conversion_rates", "alpha_ratios", "products_sold"])
        #This simulation result is a dictionary with "00" "01" "10" "11" as keys

        # 4) Update Beta_parameters according to the simulation done
        for i in range(len(self.context)):
            self.update_parameters(simulation_result, list_opt_prices_combination[i])
            
        return list_opt_prices_combination


    def run(self, n_round=365, daily_users=200):
        """ Method to run Thompson Sampling algorithm given number of days to be simulated and the number of users simulated
            in each day. It updates the variable reward_history, appending the list of expected rewards obtained during the run."""

        #initialize an empty list to store the daily rewards-> non la sistemiamo
        rewards = []
        # Initialize an empty list to store the price_combination decided each day-> it's a list of list since it depends on the group_list
        price_comb = []
        # Set beta_parameters to initial values for conversion rates and alpha ratios ->  QUESTO DEVE ESSERE PORTATO SUL DICT
        self.beta_param_CR = []
        self.beta_param_CR.append(self.initial_beta_CR[0].copy())
        self.beta_param_CR.append(self.initial_beta_CR[1].copy())
        self.beta_param_alpha = self.initial_beta_alpha.copy()

        for i in range(n_round):
            # Call the Context Generation Algorithm
            if i%14==0:
                dict_simulinfo=self.simulinfo()
                context_matrix=ContextGeneration.run(dict_simulinfo)
            self.iteration
            # Do a single iteration of the TS, and store the price combination chosen in the iteration
            list_opt_price_comb = self.iteration(daily_users, context_matrix)
            ##rewards.append(self.env.expected_reward(list_opt_price_comb))
            price_comb.append(list_opt_price_comb)
        # append the list of rewards obtained through the run
        ##self.reward_history.append(rewards)
        # append the list of price combinations selected through the run

def simulinfo(self, ): #da listofoptpricecomb a coversionrate + alpha ratios
        #c'è da dire che sta cosa qui  però deve essere fatta sulle diverse keys
        # compute and append the matrix of conversion rates estimate after the run
        A_CR = self.beta_param_CR[0]
        B_CR = self.beta_param_CR[1]
        self.cr_matrix_list.append(A_CR/(A_CR+B_CR))
        # compute and append the alphas ratio estimate after the run
        a_alpha = self.beta_param_alpha[0,:]
        b_alpha = self.beta_param_alpha[1,:]
        self.alpha_ratios_list.append(a_alpha/(a_alpha+b_alpha))


        return dict