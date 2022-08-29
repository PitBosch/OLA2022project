import numpy as np


class ucb_learner:
    def __init__(self, n_products, n_arms):
        self.t = 0
        self.n_products = n_products
        self.n_arms = n_arms
        self.crs_estimation_day_x = []
        self.rewards_per_arm = [[[] for i in range(n_arms)] for j in range(n_products)]
        self.pulled = []

    def update(self, arms_pulled, cr_data):
        n_of_purchase_for_product = cr_data[0].astype(int)
        n_of_clicks_for_product = cr_data[1].astype(int)
        n_of_purchase = np.sum(n_of_purchase_for_product, dtype=np.int32)
        n_of_clicks = np.sum(n_of_clicks_for_product, dtype=np.int32)
        crs_estimation = np.divide(n_of_purchase_for_product, n_of_clicks_for_product)
        self.t += 1 * np.sum(n_of_clicks) # cr_data[1]: number of user in a day
        self.crs_estimation_day_x.append(crs_estimation)
        for product_idx in range(self.n_products):
            self.rewards_per_arm[product_idx][arms_pulled[product_idx]].append([crs_estimation[product_idx], n_of_clicks_for_product[product_idx]])
        self.pulled.append([arms_pulled, n_of_clicks_for_product.tolist()])
