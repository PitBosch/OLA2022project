import numpy as np
from ucb_learner import *
from Environment import *
from Greedy_optimizer import *


class step3_ucb1(ucb_learner):
    def __init__(self, n_products, n_arms, prices, env: Environment, crs_sw=np.inf):
        super().__init__(n_products, n_arms, crs_sw)
        self.means = np.zeros((n_products, n_arms))
        self.widths = np.ones((n_products, n_arms)) * np.inf
        self.prices = prices
        self.greedy_opt = Greedy_optimizer(env)
        self.env = env
        #
        self.collected_rewards = []
        self.regret = []

    def pull_arms(self):
        sampled_cr = np.minimum(np.array([self.means + self.widths]), 1) # limit to 1 for all the crs
        arms_pulled = self.greedy_opt.run(conversion_rates=sampled_cr)["combination"]
        return arms_pulled

    def update(self, arms_pulled, cr_data, n_users):
        super().update(arms_pulled, cr_data, n_users)
        for product_idx in range(self.n_products):
            # weighted mean (on the n°_of_collected_samples so far) for the arm pulled
            mask = np.array(self.pulled)[:, 0, product_idx] == arms_pulled[product_idx]
            n_clicks = np.array(self.pulled)[:, 1, product_idx]
            n_conv = np.array(self.pulled)[:, 2, product_idx]
            n_clicks_masked = np.multiply(mask, n_clicks)
            n_conv_masked = np.multiply(mask, n_conv)
            n_clicks_masked_sum = np.sum(n_clicks_masked)
            product_x_arm_pulled_weighted_mean = np.sum(np.multiply(n_clicks_masked, n_conv_masked))/n_clicks_masked_sum
            #
            self.means[product_idx, arms_pulled[product_idx]] = product_x_arm_pulled_weighted_mean
        for product_idx in range(self.n_products):
            for arm_idx in range(self.n_arms):
                # below: number of visualization for product x with arm x, divided by the estimated number of daily users
                n = np.sum(np.multiply(np.array(self.pulled, dtype=np.int32)[:, 0, product_idx] == arm_idx, np.array(self.pulled, dtype=np.int32)[:, 1, product_idx]))/np.mean(self.daily_users)
                if n > 0:
                    self.widths[product_idx, arm_idx] = np.sqrt(2 * np.log(self.t) / (n*(self.t - 1)))
                else:
                    self.widths[product_idx, arm_idx] = np.inf

    def run(self, n_days, daily_users):
        self.reset()
        collected_rewards_temp = []
        opt_reward = self.env.optimal_reward()[0]
        instant_regret = []
        for t in range(n_days):
            pulled_arms = self.pull_arms()
            day_data = self.env.simulate_day(daily_users, pulled_arms, ["conversion_rates", "alpha_ratios"])
            n_users = np.sum(day_data["initial_prod"])
            cr_data = day_data["CR_data"]
            self.update(pulled_arms, cr_data, n_users)
            reward = self.env.expected_reward(pulled_arms)
            collected_rewards_temp.append(reward)
            instant_regret.append(opt_reward - reward)
        self.collected_rewards.append(collected_rewards_temp)
        cumulative_regret = np.cumsum(instant_regret)
        self.regret.append(cumulative_regret)

    def reset(self):
        super().reset()
        self.means = np.zeros((self.n_products, self.n_arms))
        self.widths = np.ones((self.n_products, self.n_arms)) * np.inf

    def print_estimations(self):
        print("Conversion rates - estimated means:\n", self.means)
        print("\nConversion rates - estimated widths:\n", self.means)
