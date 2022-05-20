import numpy as np
import pandas as pd


class User:
    """Class containing all the informations required to simulate the behaviour of a particular user in our website"""
    def __init__(self, alphas: np.array, res_price: float, poisson_lambda: float, probabilities: pd.DataFrame):
        # Entry proportions between the different products (alpha_0: prob of visiting a competitor website)
        self.alphas = alphas
        self.sampled_alphas = alphas

        # Economic availability
        self.res_price = res_price

        # Parameter of the distribution which defines the number of purchased products in case of buying
        self.poisson_lambda = poisson_lambda

        # Dataframe containing all the transition probabilities that links the different products for the user
        self.probabilities = probabilities
        self.margin = 0

    def buy(self, price) -> bool:
        if self.res_price > price:
            self.update_res_price()
        return self.res_price > price

    def get_prod_number(self):
        return np.random.poisson(self.poisson_lambda)

    def start_event(self):
        """Method which extract the starting point of the user visit, if it returns 0 it means that the user has decided
           to visit a competitor website. """
        return np.random.multinomial(1, self.sampled_alphas)

    def update_alphas(self):
        self.sampled_alphas = np.random.dirichlet(self.alphas, len(self.alphas))

    def update_res_price(self):
        print(0)  # TODO: ragionare sul sistema di aggiornamento della disponibilità economica nel caso in cui l'utente effettui un acquisto.
    # ragionevole pensare che un acquisto abbia un impatto sul budget a disposizione.
