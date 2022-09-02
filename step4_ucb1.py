import numpy as np
from ucb_learner import *
from step3_ucb1 import *
from Environment import *
from Greedy_optimizer import *


class step4_ucb1(step3_ucb1):
    def __init__(self, daily_users, n_products, n_arms, prices, env: Environment, crs_sw=np.inf, step4_only_sw=np.inf):
        super().__init__(daily_users, n_products, n_arms, prices, env, crs_sw)
        self.alphas_means = np.array([1/5, 1/5, 1/5, 1/5, 1/5])
        self.alphas = []
        self.n_products_sold_means = np.array([1, 1, 1, 1, 1])
        self.n_products_sold_means_history = []
        self.step4_only_sw = step4_only_sw

    def pull_arms(self):
        sampled_cr = np.minimum(np.array([self.means + self.widths]), 1) # limit to 1 for all the crs
        arms_pulled = self.greedy_opt.run(conversion_rates=sampled_cr, alphas_ratio=np.expand_dims(self.alphas_means, axis=0), n_prod=np.expand_dims(self.n_products_sold_means, axis=0))["combination"]
        return arms_pulled

    def update(self, arms_pulled, cr_data, alpha_data, mean_prod_sold):
        super().update(arms_pulled, cr_data)
        # updating history lists
        if len(self.alphas) < self.step4_only_sw:
            self.alphas.append(np.divide(alpha_data, np.sum(alpha_data)))
            self.n_products_sold_means_history.append(mean_prod_sold)
        else:
            self.alphas.pop(0)
            self.n_products_sold_means_history.pop(0)
            self.alphas.append(np.divide(alpha_data, np.sum(alpha_data)))
            self.n_products_sold_means_history.append(mean_prod_sold)
        # updating estimated means
        if len(self.alphas) < self.step4_only_sw:
            self.alphas_means = np.mean(self.alphas, axis=0)
            self.n_products_sold_means = np.mean(self.n_products_sold_means_history, axis=0)
        else:
            self.alphas_means = np.mean(self.alphas[-self.step4_only_sw:])
            self.n_products_sold_means = np.mean(self.n_products_sold_means_history[-self.step4_only_sw:], axis=0)
