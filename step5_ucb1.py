import numpy as np
from ucb_learner import *
from step3_ucb1 import *
from Environment import *
from Greedy_optimizer import *


class step5_ucb1(step3_ucb1):
    def __init__(self, daily_users, n_products, n_arms, prices, env: Environment, crs_sw=np.inf, step5_only_sw=np.inf):
        super().__init__(daily_users, n_products, n_arms, prices, env, crs_sw)
        self.graph_weights_means = np.ones((self.n_products, self.n_products))
        self.graph_data = []
        self.step5_only_sw = step5_only_sw

    def pull_arms(self):
        sampled_cr = np.minimum(np.array([self.means + self.widths]), 1) # limit to 1 for all the crs
        arms_pulled = self.greedy_opt.run(conversion_rates=sampled_cr, graph_weights=np.expand_dims(self.graph_weights_means, axis=0))["combination"]
        return arms_pulled

    def update(self, arms_pulled, cr_data, visualizations, clicks):
        super().update(arms_pulled, cr_data)
        graph_weights_mean = np.divide(clicks, np.maximum(visualizations, 1))
        if len(self.graph_data) < self.step5_only_sw:
            self.graph_data.append([graph_weights_mean.tolist(), visualizations.tolist()])
        else:
            self.graph_data.pop(0)
            self.graph_data.append([graph_weights_mean.tolist(), visualizations.tolist()])
        # updating estimated means
        if len(self.graph_data) <= self.step5_only_sw:
            self.graph_weights_means = np.divide(np.sum(np.multiply(np.array(self.graph_data)[:, 0], np.array(self.graph_data)[:, 1]), axis=0), np.maximum(np.sum(np.array(self.graph_data)[:, 1], axis=0), 1))
        else:
            self.graph_weights_means = np.divide(np.sum(np.multiply(np.array(self.graph_data[-self.step5_only_sw:])[:, 0], np.array(self.graph_data[-self.step5_only_sw:])[:, 1]), axis=0), np.sum(np.array(self.graph_data[-self.step5_only_sw:])[:, 1], axis=0))
