import numpy as np


class ucb_learner:
    def __init__(self, n_products, n_arms, crs_sw=np.inf):
        self.t = 0
        self.n_products = n_products
        self.n_arms = n_arms
        self.pulled = []
        self.crs_sw = crs_sw

    def update(self, arms_pulled, cr_data):
        n_of_purchase_for_product = cr_data[0].astype(int)
        n_of_clicks_for_product = cr_data[1].astype(int)
        n_of_purchase = np.sum(n_of_purchase_for_product, dtype=np.int32)
        n_of_clicks = np.sum(n_of_clicks_for_product, dtype=np.int32)
        crs_estimation = np.divide(n_of_purchase_for_product, n_of_clicks_for_product)
        if len(self.pulled) < self.crs_sw:
            self.pulled.append([arms_pulled, n_of_clicks_for_product.tolist(), crs_estimation.tolist()])
            self.t = np.sum(np.array(self.pulled)[:, 1])
        else:
            self.pulled.pop(0)
            self.pulled.append([arms_pulled, n_of_clicks_for_product.tolist(), crs_estimation.tolist()])
            self.t = np.sum(np.array(self.pulled[-self.crs_sw:])[:, 1])
