import numpy as np
from ucb_learner import *
from step3_ucb1 import *
from Environment import *
from Greedy_optimizer import *


class step5_ucb1(step3_ucb1):
    def __init__(self, n_products, n_arms, prices, env: Environment, crs_sw=np.inf, step5_only_sw=np.inf):
        super().__init__(n_products, n_arms, prices, env, crs_sw)
        self.graph_weights_means = np.ones((self.n_products, self.n_products))
        self.graph_weights_widths = np.ones((self.n_products, self.n_products)) * np.inf
        # self.graph_data is a history; each element contains 2 matrix: the estimated graph matrix for that day, and the number of visualizations on that edge
        self.graph_data = []
        # sliding window on the graph weights
        self.step5_only_sw = step5_only_sw
        # for Elisa's printing estimations
        self.crs_estimations_over_n_experiments = []
        self.graph_weights_estimations_over_n_experiments = []

    def pull_arms(self):
        sampled_cr = np.minimum(np.array([self.means + self.widths]), 1) # limit to 1 for all the crs
        sampled_graph_weights = np.minimum(np.sum([self.graph_weights_means, self.graph_weights_widths], axis=0), 1)
        arms_pulled = self.greedy_opt.run(conversion_rates=sampled_cr, graph_weights=np.expand_dims(sampled_graph_weights, axis=0))["combination"]
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
        # start: graph weights widths update
        # calculation of graph_weights time, as minimum withs the sliding window for the step 5 (that is different from the sw for the crs)
        graph_weights_t = np.amin([len(self.graph_data), self.step5_only_sw], axis=0) + 1
        for product_idx_1 in range(self.n_products):
            for product_idx_2 in range(self.n_products):
                # total number of samples on the secondary product [product_idx_2] for [prod_idx_1] as primary
                n = np.sum(np.array(self.graph_data)[:, 1][:, product_idx_1, product_idx_2])
                if n > 0:
                    self.graph_weights_widths[product_idx_1, product_idx_2] = np.sqrt(np.divide(2 * np.log(graph_weights_t), (n * (graph_weights_t - 1))))
                else:
                    self.graph_weights_widths[product_idx_1, product_idx_2] = np.inf
        # end: graph weights widths update

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
        # updating Elisa's parameters'
        self.crs_estimations_over_n_experiments.append([self.means, self.widths])
        self.graph_weights_estimations_over_n_experiments.append([self.graph_weights_means, self.graph_weights_widths])

    def reset(self):
        super().reset()
        self.graph_weights_means = np.ones((self.n_products, self.n_products))
        self.graph_data = []

    def print_estimations(self):
        print("Estimated graph weights (means + widths, over n experiment, lambda included):\n", np.mean(np.sum([np.array(self.graph_weights_estimations_over_n_experiments)[:, 0], np.array(self.graph_weights_estimations_over_n_experiments)[:, 1]], axis=0), axis=0), "\n\n")
        super().print_estimations()
