import numpy as np


class ucb_learner:
    def __init__(self, n_products, n_arms, crs_sw=np.inf):
        self.t = 1
        self.n_products = n_products
        self.n_arms = n_arms
        # each row of self.pulled is the history;
        # each element is composed by 3 vectors: the arms pulled | the n° of visualizations for each arm pulled | the cr estimation for that day for each arm pulled
        self.pulled = []
        self.crs_sw = crs_sw # sliding window for the crs
        self.daily_users = [] # list of int; the number of users observed each day

    def update(self, arms_pulled, cr_data, n_users):
        self.daily_users.append(n_users)
        n_of_purchase_for_product = cr_data[0].astype(int)
        n_of_clicks_for_product = cr_data[1].astype(int)
        crs_estimation = np.divide(n_of_purchase_for_product, n_of_clicks_for_product)
        if len(self.pulled) < self.crs_sw: # self.pulled (the history), is updated to contain always #[sliding window] samples
            self.pulled.append([arms_pulled, n_of_clicks_for_product.tolist(), crs_estimation.tolist()])
            self.t += 1
        else:
            self.pulled.pop(0)
            self.pulled.append([arms_pulled, n_of_clicks_for_product.tolist(), crs_estimation.tolist()])

    # reset for a new experiment
    def reset(self):
        self.t = 1
        self.pulled = []
        self.daily_users = []
