import numpy as np
import pandas as pd


class User:
    def __init__(self, alphas: np.array, res_price: float, poisson_lambda: float, probabilities: pd.DataFrame):
        self.alphas = alphas
        self.res_price = res_price
        self.poisson_lambda = poisson_lambda
        self.probabilities = probabilities

    def buy(self, price):
        return self.res_price > price

    def get_prod_number(self):
        return np.random.poisson(self.poisson_lambda)

    def start_event(self):
        return np.random.multinomial(1, self.alphas)
