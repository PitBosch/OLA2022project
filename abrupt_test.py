import numpy as np
from ucb_learner import *
from Environment import *
from Greedy_optimizer import *


class Abrupt_learner(ucb_learner):
    def __init__(self, n_products, n_arms, prices, env: Environment, changes_dict, crs_sw=np.inf):
        super().__init__(n_products, n_arms, crs_sw)
        self.means = np.zeros((n_products, n_arms))
        self.widths = np.ones((n_products, n_arms)) * np.inf
        self.prices = prices
        self.greedy_opt = Greedy_optimizer(env)
        self.env = env
        #
        self.collected_rewards = []
        self.regret = []
        self.opt_reward_history = []
        self.changes_dict = changes_dict
        self.initial_res_price_param = copy.deepcopy(env.users[0].res_price_params) ###################################################################################

    def pull_arms(self):
        # limit to 1 for all the crs
        sampled_cr = np.minimum(np.array([self.means + self.widths]), 1)
        arms_pulled = self.greedy_opt.run(conversion_rates=sampled_cr)["combination"]
        return arms_pulled

    def update(self, arms_pulled, cr_data, n_users):
        super().update(arms_pulled, cr_data, n_users)
        for product_idx in range(self.n_products):
            # start: computation of the weighted mean (on the n°_of_collected_samples so far) for the arm pulled
            mask = np.array(self.pulled)[:, 0, product_idx] == arms_pulled[product_idx] # mask matrix to select all the datas of a specified arm
            n_clicks = np.array(self.pulled)[:, 1, product_idx] # n° of click vector for a specific arm
            n_conv = np.array(self.pulled)[:, 2, product_idx] # conversion rates mean (vector) for a specific arm
            n_clicks_masked = np.multiply(mask, n_clicks) # n° of clicks for a specific arm (vector)
            n_conv_masked = np.multiply(mask, n_conv) # cr vector for a specific arm (vector)
            n_clicks_masked_sum = np.sum(n_clicks_masked) # n° of total samples collected for a specific arm
            product_x_arm_pulled_weighted_mean = np.sum(np.multiply(n_clicks_masked, n_conv_masked))/n_clicks_masked_sum # weighted mean computation on the estimated cr for each day
            # end: weighted mean computed
            self.means[product_idx, arms_pulled[product_idx]] = product_x_arm_pulled_weighted_mean
        for product_idx in range(self.n_products):
            for arm_idx in range(self.n_arms):
                # (below) n = number of visualization for product x with arm x, divided by the estimated mean number of daily users
                n = np.sum(np.multiply(np.array(self.pulled, dtype=np.int32)[:, 0, product_idx] == arm_idx, np.array(self.pulled, dtype=np.int32)[:, 1, product_idx]))/np.mean(self.daily_users)
                if n > 0:
                    self.widths[product_idx, arm_idx] = np.sqrt(2 * np.log(self.t) / (n * (self.t - 1)))
                else:
                    self.widths[product_idx, arm_idx] = np.inf

    def run(self, n_days, daily_users):
        self.reset()
        collected_rewards_temp = []
        opt_reward = self.env.optimal_reward()[0]
        opt_reward_list = []
        instant_regret = []
        
        for t in range(n_days):
            if t in self.changes_dict.keys(): ###################################################################################
                self.env.abrupt_change_deterministic([self.changes_dict[t]]) ###################################################################################
                opt_reward = self.env.optimal_reward()[0] ###################################################################################
            pulled_arms = self.pull_arms()
            day_data = self.env.simulate_day(daily_users, pulled_arms, ["conversion_rates", "alpha_ratios"])
            n_users = np.sum(day_data["initial_prod"])
            cr_data = day_data["CR_data"]
            self.update(pulled_arms, cr_data, n_users)
            reward = self.env.expected_reward(pulled_arms)
            collected_rewards_temp.append(reward)
            instant_regret.append(opt_reward - reward)
            opt_reward_list.append(opt_reward) ###################################################################################
        self.collected_rewards.append(collected_rewards_temp)
        cumulative_regret = np.cumsum(instant_regret)
        self.regret.append(cumulative_regret)
        self.opt_reward_history.append(opt_reward_list) ###################################################################################

    def reset(self):
        super().reset()
        self.means = np.zeros((self.n_products, self.n_arms))
        self.widths = np.ones((self.n_products, self.n_arms)) * np.inf
        self.env.users[0].res_price_params = copy.deepcopy(self.initial_res_price_param) ###################################################################################

    def print_estimations(self):
        print("Conversion rates - estimated means:\n", self.means)
        print("\nConversion rates - estimated widths:\n", self.means)

