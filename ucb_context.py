from step4_ucb1 import *

class ucb_context(step4_ucb1):

    def __init__(self, env: Environment,n_products, n_arms, prices, CR_info, alpha_info, n_prod_info, group_list, t):
        self.n_products = n_products
        self.n_arms = n_arms
        self.prices = prices
        self.env = env
        self.greedy_opt = Greedy_optimizer(env)
            # widhts
        self.widths = np.ones((n_products, n_arms)) * np.inf
        self.alphas_widths = np.ones(self.n_products) * np.inf
        self.n_products_sold_widths = np.ones(self.n_products) * np.inf
        
        
        self.daily_users = [] # list of int; the number of users observed each day
        # time
        self.t = t
        # INITIALIZE UNCERTAIN PARAMETERS WITH OLD VALUES
        self.cr_info = CR_info.copy()
        self.alpha_info = alpha_info.copy()
        self.n_prod_info = n_prod_info.copy()
        # INITIALIZE INFORMATIONS ABOUT GROUP
        self.group_list = group_list.copy()
        self.group_dim = len(self.group_list)
        # INITIALIZE MEAN AND WIDTHS OF THE UNCERTAIN PARAMETERS
            # means
        self.means = self.cr_info[0, :, :]/self.cr_info[1, :, :]
        self.alphas_means = self.alpha_info/np.sum(self.alpha_info)
        self.n_products_sold_means = self.n_prod_info[0]/self.n_prod_info[1]
            # widhts
        self.widths = np.ones((n_products, n_arms)) * np.inf
        self.alphas_widths = np.ones(self.n_products) * np.inf
        self.n_products_sold_widths = np.ones(self.n_products) * np.inf
        for prod_ind in range(5):
            for price_ind in range(4):
                # conversion rates
                n = self.cr_info[1, prod_ind, price_ind]
                if n>=1 and t>1:
                    self.widths[prod_ind, price_ind] = np.sqrt(2 * np.log(t) / (n * (t - 1)))
                else:
                    self.widths[prod_ind, price_ind] = np.inf
            # alpha
            n_alpha = self.alpha_info[prod_ind]
            if n_alpha>=1 and t>1:
                    self.alphas_widths[prod_ind] = np.sqrt(2 * np.log(t) / (n_alpha * (t - 1)))
            else:
                self.alphas_widths[prod_ind] = np.inf
            # number of product solf
            n_prod = self.n_prod_info[1, prod_ind]
            if n_prod>=1 and t>1:
                    self.n_products_sold_widths[prod_ind] = np.sqrt(2 * np.log(t) / (n_prod * (t - 1)))
            else:
                self.n_products_sold_widths[prod_ind] = np.inf

    def pull_arms(self, est_feat_prob_mat):
        sampled_cr = np.minimum(np.array([self.means + self.widths]), 1) # limit to 1 for all the crs
        if np.sum([self.alphas_means, self.alphas_widths]) == np.inf:
            alphas_ratio = [0.2, 0.2, 0.2, 0.2, 0.2]
        else:
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
        self.cr_info[0, np.arange(5), arms_pulled ] += cr_data[0]
        self.cr_info[1, np.arange(5), arms_pulled ] += cr_data[1]
        # alpha info update
        self.alpha_info += alpha_data
        # n_prod info update
        self.n_prod_info += n_prod_data
        self.t += 1
        
        # CONVERSION RATES UPDATE
        t = self.t
        for prod_ind, price_ind in zip(range(5), arms_pulled):
            # update mean values
            self.means[prod_ind, price_ind] = self.cr_info[0,prod_ind,price_ind]/self.cr_info[1,prod_ind,price_ind]
            # (below) n = number of visualization for product x with arm x, divided by the estimated mean number of daily users
            n = self.cr_info[1, prod_ind, price_ind]/(np.mean(self.daily_users)/DIVISION_LEARNING_NUMBER)
            if n>=1 and t>1:
                self.widths[prod_ind, price_ind] = np.sqrt(2 * np.log(t) / (n * (t - 1)))
            else:
                self.widths[prod_ind, price_ind] = np.inf
        # ALPHA RATIO and  NUMBER OF PRODUCT SOLD UPDATE
        # means update
        self.alphas_means = self.alpha_info/np.sum(self.alpha_info)
        self.n_products_sold_means = self.n_prod_info[0]/self.n_prod_info[1]
        for product_idx in range(self.n_products):
            # total number of samples on the secondary product [product_idx_2] for [prod_idx_1] as primary
            alphas_n = self.alpha_info[product_idx]
            if alphas_n >= 1 and t>1:
                self.alphas_widths[product_idx] = np.sqrt(np.divide(2 * np.log(t), (alphas_n * (t - 1))))
            else:
                self.alphas_widths[product_idx] = np.inf
            n_products_sold_n = self.n_prod_info[1, product_idx]
            if n_products_sold_n >= 1 and t>1:
                self.n_products_sold_widths[product_idx] = np.sqrt(np.divide(2 * np.log(t), (n_products_sold_n * (t - 1))))
            else:
                self.n_products_sold_widths[product_idx] = np.inf

        