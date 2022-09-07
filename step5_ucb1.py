import numpy as np
from ucb_learner import *
from step3_ucb1 import *
from Environment import *
from Greedy_optimizer import *


class step5_ucb1(step3_ucb1):
    def __init__(self, n_products, n_arms, prices, env: Environment, crs_sw=np.inf, step5_only_sw=np.inf):
        super().__init__(n_products, n_arms, prices, env, crs_sw)
        self.graph_weights_means = np.ones((self.n_products, self.n_products))
        # self.graph_data is a history; each element contains 2 matrix: the estimated graph matrix for that day, and the number of visualizations on that edge
        self.graph_data = []
        # sliding window on the graph weights
        self.step5_only_sw = step5_only_sw

    def pull_arms(self):
        sampled_cr = np.minimum(np.array([self.means + self.widths]), 1) # limit to 1 for all the crs
        arms_pulled = self.greedy_opt.run(conversion_rates=sampled_cr, graph_weights=np.expand_dims(self.graph_weights_means, axis=0))["combination"]
        return arms_pulled

    def update(self, arms_pulled, cr_data, n_users, visualizations, clicks):
        super().update(arms_pulled, cr_data, n_users)
        graph_weights_mean = np.divide(clicks, np.maximum(visualizations, 1))
        # updating graph data according to the sliding window length
        if len(self.graph_data) < self.step5_only_sw:
            self.graph_data.append([graph_weights_mean.tolist(), visualizations.tolist()])
        else:
            self.graph_data.pop(0) # delete the first element
            self.graph_data.append([graph_weights_mean.tolist(), visualizations.tolist()]) # adding a new tuple to the history
        # updating estimated means (useless if else, only for robustness)
        if len(self.graph_data) <= self.step5_only_sw:
            self.graph_weights_means = np.divide(np.sum(np.multiply(np.array(self.graph_data)[:, 0], np.array(self.graph_data)[:, 1]), axis=0), np.maximum(np.sum(np.array(self.graph_data)[:, 1], axis=0), 1))
        else:
            self.graph_weights_means = np.divide(np.sum(np.multiply(np.array(self.graph_data[-self.step5_only_sw:])[:, 0], np.array(self.graph_data[-self.step5_only_sw:])[:, 1]), axis=0), np.sum(np.array(self.graph_data[-self.step5_only_sw:])[:, 1], axis=0))

    def run(self, n_days, daily_users):
        self.reset()
        collected_rewards_temp = []
        opt_reward = self.env.optimal_reward()[0]
        instant_regret = []
        for t in range(n_days):
            pulled_arms = self.pull_arms()
            day_data = self.env.simulate_day(daily_users, pulled_arms, ["conversion_rates", "alpha_ratios", "graph_weights"])
            cr_data = day_data["CR_data"]
            n_users = np.sum(day_data["initial_prod"])
            visualizations = day_data["visualizations"]
            clicks = day_data["clicks"]
            self.update(pulled_arms, cr_data, n_users, visualizations, clicks)
            reward = self.env.expected_reward(pulled_arms)
            collected_rewards_temp.append(reward)
            instant_regret.append(opt_reward - reward)
        self.collected_rewards.append(collected_rewards_temp)
        cumulative_regret = np.cumsum(instant_regret)
        self.regret.append(cumulative_regret)

    def reset(self):
        super().reset()
        self.graph_weights_means = np.ones((self.n_products, self.n_products))
        self.graph_data = []

    def print_estimations(self):
        print("Estimated graph weights (lambda included):\n\n", self.graph_weights_means)
        print("\nConversion rates - estimated means:\n", self.means)
        print("\nConversion rates - estimated widths:\n", self.means)
