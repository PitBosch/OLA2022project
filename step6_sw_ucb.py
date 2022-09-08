import numpy as np
from ucb_learner import *
from step5_ucb1 import *
from Environment import *
from Greedy_optimizer import *


class step6_sw_ucb(step5_ucb1):
    def __init__(self, n_products, n_arms, prices, env: Environment, changes_dict, crs_sw=np.inf, step5_only_sw=np.inf):
        super().__init__(n_products, n_arms, prices, env, crs_sw, step5_only_sw)
        self.changes_dict = changes_dict ###################
        self.initial_res_price_param = copy.deepcopy(env.users[0].res_price_params) ###################

    def run(self, n_days, daily_users):
        self.reset()
        collected_rewards_temp = []
        opt_reward = self.env.optimal_reward()[0]
        instant_regret = []
        for t in range(n_days):
            if t in self.changes_dict.keys():  ###################
                self.env.abrupt_change_deterministic([self.changes_dict[t]])  ###################
                opt_reward = self.env.optimal_reward()[0]  ###################
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
        self.env.abrupt_change_deterministic([self.initial_res_price_param]) ###################
