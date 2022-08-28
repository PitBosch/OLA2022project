from statistics import mean
import numpy as np
import pandas as pd
import scipy.stats


class UserCat:


    """Class containing all the parameters required to simulate the behaviour of a particular category in our website"""
    def __init__(self, alphas: np.array, res_price_params: dict[str], poisson_lambda: float, probabilities: np.array, category="ciao"):
        # name of the user category
        self.category = category
        # Entry proportions between the different self.products (alpha_0: prob of visiting a competitor website)
        self.alphas = alphas
        # Proportions obtained by sampling at the start of each day
        self.sampled_alphas = [x/sum(alphas) for x in alphas]
        # Economic availability parameters
        self.res_price_params = res_price_params
        self.res_price = [0. for i in range(5)]
        # Parameter of the distribution which defines the number of purchased self.products in case of buying
        self.poisson_lambda = poisson_lambda
        # Dataframe containing all the transition probabilities that self.links the different self.products for the user
        self.probabilities = probabilities
        self.visited_products = []
        self.res_price_distr = []
        for i in range(5):
            mean_i = self.res_price_params['mean'][i]
            std_i = self.res_price_params['std'][i]
            self.res_price_distr.append(scipy.stats.norm(loc = mean_i, scale = std_i))

    def buy(self, price, prod_ind) -> bool:
        return self.res_price[prod_ind] > price


    def get_prod_number(self):
        return np.random.poisson(self.poisson_lambda) + 1


    def start_event(self):
        """Method which extract the starting point of the user visit, if it returns 0 it means that the user has decided
           to visit a competitor website. It also restores the margin, preparing it for the new coming user"""
        # self.margin = 0 #questo è da cambiare margin rimane all'interno di
        return np.random.choice(list(range(5)), p=self.sampled_alphas.reshape(-1))


    def generate_alphas(self):
        self.sampled_alphas = np.random.dirichlet(self.alphas, 1)


    def sample_res_price(self):
        # u = np.random.uniform()   
        # G_Min = 0
        # self.res_price = self.gamma.ppf(u * (G_Max-G_Min) + G_Min)
        for prod_ind in range(5):
            self.res_price[prod_ind] = self.res_price_distr[prod_ind].rvs(1)[0]


    def get_buy_prob(self, price, prod_ind):
        return 1 - self.res_price_distr[prod_ind].cdf(price)


    def empty_visited_products(self):
        self.visited_products = []

