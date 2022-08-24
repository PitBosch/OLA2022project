import numpy as np
from learner2 import *
from Environment import *
from Greedy_optimizer import *


class ucb1_greedy(learner2):
    def __init__(self, n_products, n_arms, prices, env: Environment):
        super().__init__(n_products, n_arms)
        self.means = np.zeros((n_products, n_arms))
        self.widths = np.ones((n_products, n_arms)) * np.inf
        self.prices = prices
        self.env = env
        self.greedy_opt = Greedy_optimizer(self.env)

    def pull_arms(self):
        sampled_cr = np.minimum(np.array([self.means + self.widths]), 1) # limit to 1 for all the crs
        arms_pulled = self.greedy_opt.run(conversion_rates=sampled_cr)["combination"]
        print("sampled_cr", sampled_cr)
        print("arms_pulled", arms_pulled)
        print(self.t, "\n")
        return arms_pulled

    def update(self, arms_pulled, estimated_cr):
        estimated_cr[1] = np.maximum(estimated_cr[1], 1) # we ensure the division will never be done by zero (min number of count for visualized = 1)
        estimated_cr = estimated_cr[0]/estimated_cr[1]
        super().update(arms_pulled, estimated_cr)
        for product_idx in range(self.n_products):
            self.means[product_idx, arms_pulled[product_idx]] = np.mean(self.rewards_per_arm[product_idx][arms_pulled[product_idx]])
        for product_idx in range(self.n_products):
            for arm_idx in range(self.n_arms):
                n = len(self.rewards_per_arm[product_idx][arm_idx])
                # n = np.count_nonzero(np.array(ucb1.pulled)[:, product_idx] == arm_idx) ;for sw-ucb1 we can simply use: np.count_nonzero(np.array(ucb1.pulled[-sw_size])[:, product_idx] == arm_idx)
                if n > 0:
                    self.widths[product_idx, arm_idx] = np.sqrt(2 * np.log(self.t) / n)
                else:
                    self.widths[product_idx, arm_idx] = np.inf
