import numpy as np
import pandas as pd
import scipy.stats


class UserCat:
    """Class containing all the informations required to simulate the behaviour of a particular category in our website"""
    def __init__(self, alphas: np.array, res_price_params: dict[str], poisson_lambda: float, probabilities: pd.DataFrame):
        # Entry proportions between the different products (alpha_0: prob of visiting a competitor website)
        self.alphas = alphas

        # Proportions obtained by sampling at the start of each day
        self.sampled_alphas = alphas

        # Economic availability parameters
        self.res_price_params = res_price_params
        self.res_price = 0

        # Parameter of the distribution which defines the number of purchased products in case of buying
        self.poisson_lambda = poisson_lambda

        # Dataframe containing all the transition probabilities that links the different products for the user
        self.probabilities = probabilities
        self.margin = 0

        self.visited_products = []

    def buy(self, price) -> bool:
        if self.res_price > price:
            self.update_res_price()
        return self.res_price > price

    def get_prod_number(self):
        return np.random.poisson(self.poisson_lambda)

    def start_event(self):
        """Method which extract the starting point of the user visit, if it returns 0 it means that the user has decided
           to visit a competitor website. It also restore the margin, preparing it for the new coming user"""
        self.margin = 0
        return np.random.choice(list(range(0, 6)), p=self.sampled_alphas.reshape(-1))

    def generate_alphas(self):
        self.sampled_alphas = np.random.dirichlet(self.alphas, 1)

    def update_res_price(self):
        print(" ")  # TODO: ragionare sul sistema di aggiornamento della disponibilità economica nel caso in cui l'utente effettui un acquisto.
    # ragionevole pensare che un acquisto abbia un impatto sul budget a disposizione.

    def sample_res_price(self):
        u = np.random.uniform()
        gamma = scipy.stats.gamma(self.res_price_params['shape'], self.res_price_params['scale'])
        G_Max = gamma.cdf(self.res_price_params['max'])
        G_Min = gamma.cdf(self.res_price_params['min'])
        self.res_price = gamma.ppf(u * (G_Max-G_Min) + G_Min)
