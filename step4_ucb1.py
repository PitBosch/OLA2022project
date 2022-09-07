import numpy as np
from ucb_learner import *
from step3_ucb1 import *
from Environment import *
from Greedy_optimizer import *


class step4_ucb1(step3_ucb1):
    def __init__(self, n_products, n_arms, prices, env: Environment, crs_sw=np.inf, step4_only_sw=np.inf):
        super().__init__(n_products, n_arms, prices, env, crs_sw)
        self.alphas_means = np.array([1/5, 1/5, 1/5, 1/5, 1/5])
        self.alphas = []
        self.n_products_sold_means = np.array([1, 1, 1, 1, 1])
        self.n_products_sold_means_history = []
        self.step4_only_sw = step4_only_sw

    def pull_arms(self):
        sampled_cr = np.minimum(np.array([self.means + self.widths]), 1) # limit to 1 for all the crs
        arms_pulled = self.greedy_opt.run(conversion_rates=sampled_cr, alphas_ratio=np.expand_dims(self.alphas_means, axis=0), n_prod=np.expand_dims(self.n_products_sold_means, axis=0))["combination"]
        return arms_pulled

    def update(self, arms_pulled, cr_data, n_users, alpha_data, mean_prod_sold):
        super().update(arms_pulled, cr_data, n_users)
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

    def run(self, n_days, daily_users):
        self.reset()
        collected_rewards_temp = []
        opt_reward = self.env.optimal_reward()[0]
        instant_regret = []
        for t in range(n_days):
            pulled_arms = self.pull_arms()
            day_data = self.env.simulate_day(daily_users, pulled_arms, ["conversion_rates", "alpha_ratios", "products_sold"])
            cr_data = day_data["CR_data"]
            n_users = np.sum(day_data["initial_prod"])
            alpha_data = day_data["initial_prod"]
            mean_prod_sold = day_data["mean_prod_sold"]
            self.update(pulled_arms, cr_data, n_users, alpha_data, mean_prod_sold)
            reward = self.env.expected_reward(pulled_arms)
            collected_rewards_temp.append(reward)
            instant_regret.append(opt_reward - reward)
        self.collected_rewards.append(collected_rewards_temp)
        cumulative_regret = np.cumsum(instant_regret)
        self.regret.append(cumulative_regret)

    def reset(self):
        super().reset()
        self.alphas_means = np.array([1/5, 1/5, 1/5, 1/5, 1/5])
        self.alphas = []
        self.n_products_sold_means = np.array([1, 1, 1, 1, 1])
        self.n_products_sold_means_history = []

    def print_estimations(self):
        print("Estimated alpha ratios:\n", self.alphas_means)
        print("\nEstimated n of products sold:\n\n", self.alphas_means)
        print("Conversion rates - estimated means:\n", self.means)
        print("\nConversion rates - estimated widths:\n", self.means)
