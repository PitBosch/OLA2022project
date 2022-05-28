import numpy as np
import pandas as pd


class UserCat:
    """Class containing all the informations required to simulate the behaviour of a particular cathegory in our website"""
    def __init__(self, alphas: np.array, res_price: float, poisson_lambda: float, probabilities: pd.DataFrame):
        # Entry proportions between the different products (alpha_0: prob of visiting a competitor website)
        self.alphas = alphas

        # Proportions obtained by sampling at the start of each day
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
        return np.random.choice(list(range(0,6)), p=self.sampled_alphas.reshape(-1))

    def generate_alphas(self):
        self.sampled_alphas = np.random.dirichlet(self.alphas, 1)

    def restore(self, original_probabilities):
        """At the end of each interaction between a user of this user class and the website we have to restore the original
           transition probabilities and the original margin since the evolution of the visit has changed them."""
        self.margin = 0
        self.probabilities = original_probabilities

    def update_res_price(self):
        print(" ")  # TODO: ragionare sul sistema di aggiornamento della disponibilità economica nel caso in cui l'utente effettui un acquisto.
    # ragionevole pensare che un acquisto abbia un impatto sul budget a disposizione.
