from step4_ucb1 import *

class ucb_context(step4_ucb1):
    def __init__(self, env: Environment, prices, CR_info, alpha_info, n_prod_info, group_list):

        super().__init__(5, 5, prices, env)
        self.cr_info = CR_info.copy()
        self.alpha_info = alpha_info.copy()
        self.n_prod_info = n_prod_info.copy()
        self.daily_users = np.sum(alpha_info)
        self.group_list = group_list.copy()
        self.group_dim = len(self.group_list)

    def pull_arms(self, est_feat_prob_mat):
        sampled_cr = np.minimum(np.array([self.means + self.widths]), 1) # limit to 1 for all the crs
        alphas_ratio = np.divide(np.sum([self.alphas_means, self.alphas_widths], axis=0), np.sum([self.alphas_means, self.alphas_widths]))
        n_prod = np.minimum(np.sum([self.n_products_sold_means, self.n_products_sold_widths], axis=0), N_PROD_SOLD_MINIMUM)
        cr_list = list(sampled_cr)*self.group_dim
        alpha_list = [alphas_ratio]*self.group_dim
        n_prod_list = [n_prod]*self.group_dim
        arms_pulled = self.greedy_opt.run(conversion_rates=cr_list, alphas_ratio=alpha_list, n_prod=n_prod_list,
                                            group_list=self.group_list, feat_prob_mat=est_feat_prob_mat)["combination"]
        return arms_pulled
    
    def update(self, arms_pulled, cr_data, alpha_data, n_prod_data):
        n_users = np.sum(alpha_data)
        # daily users update
        self.daily_users.append(n_users)
        # conversion rates info update
        n_of_purchase_for_product = cr_data[0].astype(int)
        n_of_clicks_for_product = cr_data[1].astype(int)
        crs_estimation = np.divide(n_of_purchase_for_product, n_of_clicks_for_product)
        self.cr_info[0, np.arange(5), arms_pulled ] += cr_data[0]
        self.cr_info[1, np.arange(5), arms_pulled ] += cr_data[1]
        # alpha info update
        self.alpha_info += alpha_data
        # n_prod info update
        self.n_prod_info += n_prod_data
        self.pulled.append([arms_pulled, n_of_clicks_for_product.tolist(), crs_estimation.tolist()])
        self.t += 1
        
        # CONVERSION RATES UPDATE
        t = self.t
        for prod_ind, price_ind in zip(range(5), arms_pulled):
            # update mean values
            self.means[prod_ind, price_ind] = self.cr_info[0,prod_ind,price_ind]/self.cr_info[1,prod_ind,price_ind]
            # (below) n = number of visualization for product x with arm x, divided by the estimated mean number of daily users
            n = self.cr_info[1, prod_ind, price_ind]/(np.mean(self.daily_users)/DIVISION_NUMBER)
            if n>0 and t>0:
                self.widths[prod_ind, price_ind] = np.sqrt(2 * np.log(t) / (n * (t - 1)))
            else:
                self.widths[prod_ind, price_ind] = np.inf
        # ALPHA RATIO and  NUMBER OF PRODUCT SOLD UPDATE
        # means update
        self.alphas_means = self.alpha_info/n_users
        self.n_products_sold_means = self.n_prod_info[0]/self.n_prod_info[1]
        for product_idx in range(self.n_products):
            # total number of samples on the secondary product [product_idx_2] for [prod_idx_1] as primary
            alphas_n = self.alpha_info[product_idx]
            if alphas_n > 0:
                self.alphas_widths[product_idx] = np.sqrt(np.divide(2 * np.log(t), (alphas_n * (t - 1))))
            else:
                self.alphas_widths[product_idx] = np.inf
            n_products_sold_n = self.n_prod_info[1, product_idx]
            if n_products_sold_n > 0:
                self.n_products_sold_widths[product_idx] = np.sqrt(np.divide(2 * np.log(t), (n_products_sold_n * (t - 1))))
            else:
                self.n_products_sold_widths[product_idx] = np.inf

        