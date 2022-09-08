import numpy as np
from ucb_learner import *
from step3_ucb1 import *
from Environment import *
from Greedy_optimizer import *
N_PROD_SOLD_MINIMUM = 10000


class step4_ucb1(step3_ucb1):
    def __init__(self, n_products, n_arms, prices, env: Environment, crs_sw=np.inf, step4_only_sw=np.inf):
        super().__init__(n_products, n_arms, prices, env, crs_sw)
        self.alphas_means = np.array([1/5, 1/5, 1/5, 1/5, 1/5])
        self.alphas_widths = np.ones(self.n_products) * np.inf
        # history of alphas: contains all the alpha ratios parameters estimated each day
        self.alphas = []
        self.n_products_sold_means = np.array([1, 1, 1, 1, 1])
        self.n_products_sold_widths = np.ones(self.n_products) * np.inf
        # history of n° items sold; contains the n° of items sold each day
        self.n_products_sold_means_history = []
        # sliding window applied on alpha ratios and n° items sold estimations
        self.step4_only_sw = step4_only_sw
        # for Elisa's printing estimations
        self.crs_estimations_over_n_experiments = []
        self.alphas_estimations_over_n_experiments = []
        self.n_products_sold_over_n_experiments = []

    def pull_arms(self):
        sampled_cr = np.minimum(np.array([self.means + self.widths]), 1) # limit to 1 for all the crs
        alphas_ratio = np.divide(np.sum([self.alphas_means, self.alphas_widths], axis=0), np.sum([self.alphas_means, self.alphas_widths]))
        n_prod = np.minimum(np.sum([self.n_products_sold_means, self.n_products_sold_widths], axis=0), N_PROD_SOLD_MINIMUM)
        arms_pulled = self.greedy_opt.run(conversion_rates=sampled_cr, alphas_ratio=np.expand_dims(alphas_ratio, axis=0), n_prod=np.expand_dims(n_prod, axis=0))["combination"]
        return arms_pulled

    def update(self, arms_pulled, cr_data, n_users, alpha_data, mean_prod_sold):
        super().update(arms_pulled, cr_data, n_users)
        # updating history lists, according to the sliding window length
        if len(self.alphas) < self.step4_only_sw:
            self.alphas.append(np.divide(alpha_data, np.sum(alpha_data)))
            self.n_products_sold_means_history.append(mean_prod_sold)
        else:
            self.alphas.pop(0) # delete the first element
            self.n_products_sold_means_history.pop(0) # delete the first element
            self.alphas.append(np.divide(alpha_data, np.sum(alpha_data)))
            self.n_products_sold_means_history.append(mean_prod_sold)
        # updating estimated means (useless if-else, only for robustness)
        if len(self.alphas) < self.step4_only_sw:
            self.alphas_means = np.mean(self.alphas, axis=0)
            self.n_products_sold_means = np.mean(self.n_products_sold_means_history, axis=0)
        else:
            self.alphas_means = np.mean(self.alphas[-self.step4_only_sw:], axis=0)
            self.n_products_sold_means = np.mean(self.n_products_sold_means_history[-self.step4_only_sw:], axis=0)
        # start: step 4 only widths update
        # calculation of step 4 only time, as minimum with the sliding window for the step 4 (that is different from the sw for the crs)
        step4_t = np.amin([len(self.alphas), len(self.n_products_sold_means_history), self.step4_only_sw], axis=0) + 1
        for product_idx in range(self.n_products):
            # total number of samples on the secondary product [product_idx_2] for [prod_idx_1] as primary
            alphas_n = len(self.alphas)
            if alphas_n > 0:
                self.alphas_widths[product_idx] = np.sqrt(np.divide(2 * np.log(step4_t), (alphas_n * (step4_t - 1))))
            else:
                self.alphas_widths[product_idx] = np.inf
            n_products_sold_n = len(self.n_products_sold_means_history)
            if n_products_sold_n > 0:
                self.n_products_sold_widths[product_idx] = np.sqrt(np.divide(2 * np.log(step4_t), (n_products_sold_n * (step4_t - 1))))
            else:
                self.n_products_sold_widths[product_idx] = np.inf
        # end: step 4 only widths update

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
        # updating Elisa's parameters
        self.crs_estimations_over_n_experiments.append([self.means, self.widths])
        self.alphas_estimations_over_n_experiments.append(np.sum([self.alphas_means, self.alphas_widths], axis=0))
        self.n_products_sold_over_n_experiments.append(np.sum([self.n_products_sold_means, self.n_products_sold_widths], axis=0))

    def reset(self):
        super().reset()
        self.alphas_means = np.array([1/5, 1/5, 1/5, 1/5, 1/5])
        self.alphas = []
        self.n_products_sold_means = np.array([1, 1, 1, 1, 1])
        self.n_products_sold_means_history = []

    def print_estimations(self):
        print("Estimated alpha ratios (means + widths, over n experiments):\n", np.mean(self.alphas_estimations_over_n_experiments, axis=0), "\n")
        print("Estimated number of products sold (means + widths, over n experiments):\n", np.mean(self.n_products_sold_over_n_experiments, axis=0), "\n\n")
        super().print_estimations()
