from Cusum import *
from ucb_learner import *
from step5_ucb1 import *
from Environment import *
from Greedy_optimizer import *


class Step6_CD(step5_ucb1):
    def __init__(self, n_products, n_arms, prices, env: Environment, changes_dict, M, eps, h):
        super().__init__(n_products, n_arms, prices, env)
        # Dictionary defining the changes in demand curve
        self.changes_dict = changes_dict
        self.initial_res_price_param = copy.deepcopy(env.users[0].res_price_params)
        # Matrix of cusum (chenge detection algorithms), 1 for each conversion rates value
        self.M = M
        self.eps = eps
        self.h = h
        self.CD_matrix = [[Cusum(M, eps, h) for _ in range(4)] for _ in range(5)]
        # Conversion Rates info initialized
        self.cr_info = np.array([np.zeros((5,4)), 1e-6*np.ones((5,4))])

    def run(self, n_days, daily_users):
        self.reset()
        collected_rewards_temp = []
        opt_reward = self.env.optimal_reward()[0]
        instant_regret = []
        for t in range(n_days):
            if t in self.changes_dict.keys():  
                self.env.abrupt_change_deterministic([self.changes_dict[t]])  
                opt_reward = self.env.optimal_reward()[0]
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
        # Restore initial setting of the environment
        self.env.abrupt_change_deterministic([self.initial_res_price_param])
        # updating Elisa's parameters'
        self.crs_estimations_over_n_experiments.append([self.means, self.widths])
        self.graph_weights_estimations_over_n_experiments.append([self.graph_weights_means, self.graph_weights_widths])

    def reset(self):
        super().reset()
        self.env.abrupt_change_deterministic([self.initial_res_price_param]) ###################
        self.CD_matrix = [[Cusum(self.M, self.eps, self.h) for _ in range(4)] for _ in range(5)]
        self.cr_info = np.array([np.zeros((5,4)), 1e-6*np.ones((5,4))])

    def reset_cr_info(self, prod_ind, price_ind):
        self.means[prod_ind, price_ind] = 0.
        self.widths[prod_ind, price_ind] = np.inf
        self.cr_info[:, prod_ind, price_ind] = np.array([0, 1e-6])

    def update(self, arms_pulled, cr_data, n_users, visualizations, clicks):
        self.daily_users.append(n_users)
        n_of_purchase_for_product = cr_data[0].astype(int)
        n_of_clicks_for_product = cr_data[1].astype(int)
        crs_estimation = np.divide(n_of_purchase_for_product, n_of_clicks_for_product)
        
        self.pulled.append([arms_pulled, n_of_clicks_for_product.tolist(), crs_estimation.tolist()])
        self.t += 1
        
        # Verify if change detection algorithms activate for one of the conversion rates estimated
        cr_est = cr_data[0]/(cr_data[1]+1e-6)
        for prod_ind, price_ind in zip(range(5), arms_pulled):
            cd = self.CD_matrix[prod_ind][price_ind].update(cr_est[prod_ind])
            if cd :
                for price_i in range(4):
                    self.CD_matrix[prod_ind][price_i].reset(self.t)
                    self.reset_cr_info(prod_ind, price_i)
        self.cr_info[0, np.arange(5), arms_pulled ] += cr_data[0]
        self.cr_info[1, np.arange(5), arms_pulled ] += cr_data[1]
        # CONVERSION RATES UPDATE
        for prod_ind, price_ind in zip(range(5), arms_pulled):
            # update mean values
            self.means[prod_ind, price_ind] = self.cr_info[0,prod_ind,price_ind]/self.cr_info[1,prod_ind,price_ind]
            # (below) n = number of visualization for product x with arm x, divided by the estimated mean number of daily users
            n = self.cr_info[1, prod_ind, price_ind]/(np.mean(self.daily_users)/DIVISION_LEARNING_NUMBER)
            t = self.t #- self.CD_matrix[prod_ind][price_ind].last_change_t + 1
            if n>0 and t>0:
                self.widths[prod_ind, price_ind] = np.sqrt(2 * np.log(t) / (n * (t - 1)))
            else:
                self.widths[prod_ind, price_ind] = np.inf
        # GRAPH WEIGHTS UPDATE
        graph_weights_mean = np.divide(clicks, np.maximum(visualizations, 1))
        # updating graph data according to the sliding window length
        
        self.graph_data.append([graph_weights_mean.tolist(), visualizations.tolist()])
        
        # updating estimated means (useless if else, only for robustness)
        if len(self.graph_data) <= self.step5_only_sw:
            self.graph_weights_means = np.divide(np.sum(np.multiply(np.array(self.graph_data)[:, 0], np.array(self.graph_data)[:, 1]), axis=0), np.maximum(np.sum(np.array(self.graph_data)[:, 1], axis=0), 1))
        else:
            self.graph_weights_means = np.divide(np.sum(np.multiply(np.array(self.graph_data[-self.step5_only_sw:])[:, 0], np.array(self.graph_data[-self.step5_only_sw:])[:, 1]), axis=0), np.sum(np.array(self.graph_data[-self.step5_only_sw:])[:, 1], axis=0))
        # start: graph weights widths update
        # calculation of graph_weights time, as minimum withs the sliding window for the step 5 (that is different from the sw for the crs)
        graph_weights_t = np.amin([len(self.graph_data) + 1, self.step5_only_sw], axis=0)
        for product_idx_1 in range(self.n_products):
            for product_idx_2 in range(self.n_products):
                # total number of samples on the secondary product [product_idx_2] for [prod_idx_1] as primary
                n = np.sum(np.array(self.graph_data)[:, 1][:, product_idx_1, product_idx_2])
                if n > 0:
                    self.graph_weights_widths[product_idx_1, product_idx_2] = np.sqrt(np.divide(2 * np.log(graph_weights_t), (n * (graph_weights_t - 1))))
                else:
                    self.graph_weights_widths[product_idx_1, product_idx_2] = np.inf
        # end: graph weights widths update

