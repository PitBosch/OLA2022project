from UserCat import *
from Product import *
import copy

def lambda_correct(prob_matrix, sec_dict : dict, lambda_q):
    """ Function to apply the correction on the graph weights values due the decided value of lambda_q.
        In our setting for each product i the probability of click on second secondary j is given by the
        product of value (i,j) of the defined probability matrix and lambda.
        This function apply this correction to the user defined probability matrix describing the graph weights"""
        
    out_matrix = prob_matrix.copy()

    for i in range(5):
        j = list(sec_dict.values())[i][1]
        out_matrix[i,j] *= lambda_q
    
    return out_matrix

def feature_matrix_to_list(feat_mat):
    feature_list = []
    n_groups = int(np.max(feat_mat)) + 1

    for group in range(n_groups):
        i_list, j_list = np.where(feat_mat == group)
        group_list = []
        for k in range(len(i_list)):
            group_list.append([i_list[k], j_list[k]])
        feature_list.append(group_list.copy())

    return feature_list

def generate_users_prob(feat_matrix, feat_prob):
    feat_list = feature_matrix_to_list(feat_matrix)
    prob_list = []
    n_groups = np.max(feat_matrix)+1
    if n_groups == 1:
        prob_list.append(1)
    else:
        for user_feat in feat_list:
            prob = 0.
            for feat_comb in user_feat:
                p1 = feat_prob[0] if feat_comb[0] == 1 else 1-feat_prob[0]
                p2 = feat_prob[1] if feat_comb[1] == 1 else 1-feat_prob[1]
                prob += p1*p2
            prob_list.append(prob)
    
    return prob_list

def generate_feat_prob_matrix(feat_prob):
    feat_prob_matrix = np.zeros((2,2))
    for i in range(2):
        for j in range(2):
            p1 = feat_prob[0] if i == 1 else 1-feat_prob[0]
            p2 = feat_prob[1] if j == 1 else 1-feat_prob[1]
            feat_prob_matrix[i,j] = p1*p2
    return feat_prob_matrix

class Environment:
    """Class containing all the parameters that characterize the problem, from the classes of users to the list of available products."""

    def __init__(self, users: list[UserCat], products: list[Product],secondary_dict: dict[list[int]], feat_matrix, feat_prob):
        # List of different categories of users considered. If len(users) == 1 --> AGGREGATED DEMAND         
        self.users = users
        # List of available products: each of them has available the information of its position in the list -> attribute index
        self.products = products
        # dictionary of lists of secondary products
        self.secondary_dict = secondary_dict
        # matrix defining the partition based on the users features
        self.feature_matrix = feat_matrix
        # List of probability to have feature_i = 1
        self.feature_prob = feat_prob
        # relative frequency of the users category
        self.user_cat_prob = generate_users_prob(feat_matrix, feat_prob)
        # Matrix of probability for the possible couple of features
        self.feat_prob_matrix = generate_feat_prob_matrix(feat_prob)
        """ 4 parameters can be certain or uncertain in our simulations :
            - conversion rate: list of matrices representing the probabilities to buy a product i at a price j for the user k
            - alpha_ratios : list of lists of probabilities to land on product i at first for user j
            - n_prod : list of average number of product sold for a user
            - graph_weights : list of matrices describing the probabilities of a click on a
                secondary product given that we have bought a primary product (a matrix for each user) """
        # Theoretical values for conversion_rates, alpha_ratios, n_prod_sold and graph_weights
        self.theoretical_values = {'conversion_rates': []}
        # CONVERSION RATES
        # compute theoretical values for conversion rate when them are certain
        self.conversion_rates = []
        CR_matrix = []
        CR_list = []
        for user in self.users:
            for product in self.products:
                for price in product.prices:
                    prod_ind = product.index
                    CR_list.append(user.get_buy_prob(price, prod_ind))
                CR_matrix.append(CR_list.copy())
                CR_list = []
            CR_matrix = np.matrix(CR_matrix)
            self.theoretical_values['conversion_rates'].append(CR_matrix.copy())  # EXPECTED
            self.conversion_rates.append(CR_matrix.copy())
            CR_matrix = []
        # ALPHA RATIOS
        # retrieve theoretical alpha ratios for each user from the UserCat's class variable
        self.theoretical_values['alpha_ratios'] = []
        self.alpha_ratios = []
        for user in self.users:
            alpha_distr = [x / sum(user.alphas) for x in user.alphas]
            self.theoretical_values['alpha_ratios'].append(alpha_distr.copy())
            self.alpha_ratios.append(alpha_distr.copy())
        # NUMBER OF PRODUCT SOLD FOR A FIXED PRICE
        # when number of product sold is certain is given by the poisson parameter of each user 
        # (+1 because 0 products bought makes no sense, i.e. we are considered a translated poisson)
        self.theoretical_values["n_prod_sold"] = []
        self.n_prod_sold = []
        for user in self.users:
            self.theoretical_values["n_prod_sold"].append(user.poisson_lambda+1) # EXPECTED
            self.n_prod_sold.append(user.poisson_lambda+1)
        # GRAPH WEIGHTS
        self.theoretical_values["graph_weights"] = []
        self.graph_weights = []
        for user in self.users:
            self.theoretical_values["graph_weights"].append(user.probabilities)
            self.graph_weights.append(user.probabilities.copy())


    def user_simulation(self, user: UserCat, price_combination, product_index, to_save_dict: dict):
        # retrieve the price of the product indicated by product_index for the current price_combination
        price_ind = price_combination[product_index]
        product_price = self.products[product_index].prices[price_ind]
        # check if the primary product is bought
        primary_bought = user.buy(product_price, product_index)
        # if the conversion_rate are uncertain, update information retrieved from the simulation
        # NOTICE: we update the conversion rates values only if we are on the FIRST product, to avoid
        #         the overestimate of the conversion rates due to the reservation price mechanism
        #if "CR_data" in to_save_dict.keys() and len(user.visited_products) == 1: # <-------------- FIRST PRODUCT ONLY VERSION FOR CR_data UPDATE
        if "CR_data" in to_save_dict.keys(): # <--------------------------------------------------- "COMPLETE" VERSION FOR CR_data UPDATE
            # update number of times users has visualized the product 
            to_save_dict["CR_data"][1][product_index] += 1
            if primary_bought:
                # only if the product is bought the number of sales are increased
                to_save_dict["CR_data"][0][product_index] += 1
        
        if not primary_bought:
            return
        # User bought the object, so i sample how many products He bought and compute the margin on the sale
        n_prod_bought = user.get_prod_number(product_index)
        # profit = self.products[product_index].margins[price_ind] * n_prod_bought
        # if the numbers of product sold are uncertain, update information retrieved from the simulation
        if "n_prod_sold" in to_save_dict.keys():
            to_save_dict["n_prod_sold"][0][product_index] += n_prod_bought
            to_save_dict["n_prod_sold"][1][product_index] += 1
        #"""The margin of the user is updated recursively every time he proceeds with a purchase, considering the margin of 
        #   that product and the number of items bought by the user (random number)"""
        # GET THE PRODUCT FROM THE DICT -> POSSO FARLI DIVENTARE DEI METODI
        first_secondary_index = self.secondary_dict.get(self.products[product_index].name)[0]
        first_secondary = self.products[first_secondary_index]
        second_secondary_index = self.secondary_dict.get(self.products[product_index].name)[1]
        second_secondary = self.products[second_secondary_index]
        """To simulate the random behaviour of the user we sample from a random distribution and we use it to evaluate whether an event has occurred or not. """
        # the user clicks on the first secondary if it has never been shown before and with a probability
        # defined by user.probabilities
        first_click = (np.random.uniform() < user.probabilities[self.products[product_index].index, first_secondary.index]) and first_secondary not in user.visited_products
        # if the graph weights are uncertain we update the information store in to_save_dict
        # with respect to the result of the simulation
        if "graph_weights" in to_save_dict.keys():
            # update visualizations for the pairs primary-secondary only if we have not already seen the product
            if first_secondary not in user.visited_products:
                to_save_dict["visualizations"][product_index][first_secondary_index] += 1
            # if in the simulation the user has clicked we update also clicks values
            if first_click:
                to_save_dict["clicks"][product_index][first_secondary_index] += 1
        # click sul primo e non l'ho ancora visitato
        if first_click:
            user.visited_products.append(first_secondary)  # add visited product to list
            self.user_simulation(user, price_combination, first_secondary.index, to_save_dict)
        # the user clicks on the second secondary if it has never been shown before and with a probability
        # defined by user.probabilities
        second_click = np.random.uniform() < user.probabilities[self.products[product_index].index, second_secondary.index] and second_secondary not in user.visited_products
        # if the graph weights are uncertain we update the information store in to_save_dict
        # with respect to the result of the simulation
        if "graph_weights" in to_save_dict.keys():
            # update visualizations for the pairs primary-secondary only if we have not already seen the product
            if second_secondary not in user.visited_products:
                to_save_dict["visualizations"][product_index][second_secondary_index] += 1
            # if in the simulation the user has clicked we update also clicks values
            if second_click:
                to_save_dict["clicks"][product_index][second_secondary_index] += 1
        #click sul secondo e non l'ho ancora visitato
        if second_click:
            user.visited_products.append(second_secondary)  # add visited product to list
            self.user_simulation(user, price_combination, second_secondary.index, to_save_dict)
        return


    def execute(self, user: UserCat, price_combination, to_save_dict: dict):
        """Method which simulates the entire interaction (considering even the case in which the user doesn't visit the website)
           of a user with our infrastructure. It returns the cumulative amount of margin generated by this interaction."""
        # sample the reservation price of the user considered
        user.sample_res_price()
        # sample which is the first product showed to the user
        page_index = user.start_event()
        # if alpha ratios are uncertain count each time a product is open as first
        if "initial_prod" in to_save_dict.keys():
            to_save_dict["initial_prod"][page_index] += 1
        # svuoto i prodotti visitati
        user.empty_visited_products()
        user.visited_products = [self.products[page_index]]
        self.user_simulation(user, price_combination, page_index, to_save_dict)
        return
        
    def simulate_day_context(self, daily_users, price_combination_list, context_matrix, to_save: list):
        """Method which simulates the usage of our website into an entire working day. Each day the alphas of each class of users
           are updated according to a Dirichlet distribution, it takes as input number of users, user probability (that now will be
           inside user_cat and the price combination of today"""
        # We may need to return also approximation for:
        #   - conversion rates
        #   - alpha_ratios
        #   - number of products sold
        #   - graph weights
        # In fact, when one of these variable is uncertain we need its approximation from the simulation
        # For this reason we need to create some structure to store these values
        d = len(price_combination_list[0])
        to_save_dict = {}
        to_save_dict['n_users'] = 0
        if "conversion_rates" in to_save:
            to_save_dict["CR_data"] = np.zeros((2, d))
        if "alpha_ratios" in to_save:
            to_save_dict["initial_prod"] = np.zeros(d)
        if "products_sold" in to_save:
            to_save_dict["n_prod_sold"] = np.zeros((2, d))
        if "graph_weights" in to_save:
            to_save_dict["graph_weights"] = np.zeros((d, d))
            to_save_dict["visualizations"] = np.zeros((d, d))
            to_save_dict["clicks"] = np.zeros((d, d))
        # We save data 
        to_save_data = {
            '00' : copy.deepcopy(to_save_dict),
            '01' : copy.deepcopy(to_save_dict),
            '10' : copy.deepcopy(to_save_dict),
            '11' : copy.deepcopy(to_save_dict)
        }
        # Generate daily alpha ratios for each user category for the new day
        for user in self.users:
            user.generate_alphas()
        # We simulate the interactions of "users_number" users
        n_users = np.random.poisson(lam = daily_users)
        for i in range(n_users):
            # FEATURE 1 SAMPLE
            feat1 = np.random.binomial(1, self.feature_prob[0])
            # FEATURE 2 SAMPLE
            feat2 = np.random.binomial(1, self.feature_prob[0])
            # Retrieve right key to access to_sava_data according to features sampled
            feat_key = str(feat1)+str(feat2)
            # Increase number of users appeared for the couple of features
            to_save_data[feat_key]['n_users'] += 1
            # user category
            feat_ind = [int(feat1), int(feat2)]
            user_ind = int(self.feature_matrix[feat_ind[0]][feat_ind[1]])
            comb_ind = int(context_matrix[feat_ind[0]][feat_ind[1]])
            # we increment the daily profit of the website by the profit done with the simulated user
            self.execute(self.users[user_ind], price_combination_list[comb_ind], to_save_data[feat_key])
            # notice that we have passed only the dictionary for the specific user category sampled
        
        for data_dict in list(to_save_data.values()):
            # if conversion rates are uncertain save the result obtained by the daily simulation
            if "conversion_rates" in to_save:
                data_dict["CR_vector"] = data_dict["CR_data"][0]/(data_dict["CR_data"][1]+1e-6)
                # +1e-6 at denominator to avoid 0/0 division
            # if alpha ratios are uncertain save the result obtained by the daily simulation
            if "alpha_ratios" in to_save:
                data_dict["alpha_ratios"] = data_dict["initial_prod"]/np.sum(data_dict["initial_prod"])
            # if number of product sold per product are uncertain save the result obtained by the daily simulation
            if "products_sold" in to_save:
                data_dict["mean_prod_sold"] = data_dict["n_prod_sold"][0]/(data_dict["n_prod_sold"][1]+1e-6)
            # if number of product sold per product are uncertain save the result obtained by the daily simulation
            if "graph_weights" in to_save:
                data_dict["graph_weights"] = data_dict["clicks"]/(data_dict["visualizations"]+1e-6)
        
        return to_save_data
    
    def simulate_day(self, daily_users, price_combination, to_save: list, aggregated = True):
        """Method which simulates the usage of our website into an entire working day. Each day the alphas of each class of users
           are updated according to a Dirichlet distribution, it takes as input number of users, user probability (that now will be
           inside user_cat and the price combination of today"""
        # We may need to return also approximation for:
        #   - conversion rates
        #   - alpha_ratios
        #   - number of products sold
        #   - graph weights
        # In fact, when one of these variable is uncertain we need its approximation from the simulation
        # For this reason we need to create some structure to store these values
        d = len(price_combination)
        to_save_dict = {}
        if "conversion_rates" in to_save:
            to_save_dict["CR_data"] = np.zeros((2, d))
        if "alpha_ratios" in to_save:
            to_save_dict["initial_prod"] = np.zeros(d)
        if "products_sold" in to_save:
            to_save_dict["n_prod_sold"] = np.zeros((2, d))
        if "graph_weights" in to_save:
            to_save_dict["graph_weights"] = np.zeros((d, d))
            to_save_dict["visualizations"] = np.zeros((d, d))
            to_save_dict["clicks"] = np.zeros((d, d))
        # We have to deal with the case of multiple categories of users :
        # let's create a list of dictionary (1 for each user category) of data to save
        to_save_data = []
        for i in range(len(self.users)):
            to_save_data.append(copy.deepcopy(to_save_dict))
        # Generate daily alpha ratios for each user category for the new day
        for user in self.users:
            user.generate_alphas()
        # We simulate the interactions of "users_number" users
        n_users = np.random.poisson(lam = daily_users)
        for i in range(n_users):
            # extract the category of the simulated user
            if len(self.users) == 1:
                # if we have only a user we don't need to extract the category
                user_kind = 0
            else:
                user_indices = list(range(len(self.users)))
                user_kind = np.random.choice(user_indices, p=self.user_cat_prob)
            # we increment the daily profit of the website by the profit done with the simulated user
            self.execute(self.users[user_kind], price_combination, to_save_data[user_kind])
            # notice that we have passed only the dictionary for the specific user category sampled
        

        if aggregated:
            # if data are aggregated return a single dictionary containing all needed informations
            final_dict = {}
            if "conversion_rates" in to_save:
                final_dict["CR_data"] = np.sum([tsd["CR_data"] for tsd in to_save_data], axis = 0)
            if "alpha_ratios" in to_save:
                final_dict["initial_prod"] = np.sum([tsd["initial_prod"] for tsd in to_save_data], axis = 0)
            if "products_sold" in to_save:
                final_dict["n_prod_sold"] = np.sum([tsd["n_prod_sold"] for tsd in to_save_data], axis = 0)
            if "graph_weights" in to_save:
                final_dict["clicks"] = np.sum([tsd["clicks"] for tsd in to_save_data], axis = 0)
                final_dict["visualizations"] = np.sum([tsd["visualizations"] for tsd in to_save_data], axis = 0)
            to_save_data = [final_dict]
        
        for data_dict in to_save_data:
            # if conversion rates are uncertain save the result obtained by the daily simulation
            if "conversion_rates" in to_save:
                data_dict["CR_vector"] = data_dict["CR_data"][0]/(data_dict["CR_data"][1]+1e-6)
                # +1e-6 at denominator to avoid 0/0 division
            # if alpha ratios are uncertain save the result obtained by the daily simulation
            if "alpha_ratios" in to_save:
                data_dict["alpha_ratios"] = data_dict["initial_prod"]/np.sum(data_dict["initial_prod"])
            # if number of product sold per product are uncertain save the result obtained by the daily simulation
            if "products_sold" in to_save:
                data_dict["mean_prod_sold"] = data_dict["n_prod_sold"][0]/(data_dict["n_prod_sold"][1]+1e-6)
            # if number of product sold per product are uncertain save the result obtained by the daily simulation
            if "graph_weights" in to_save:
                data_dict["graph_weights"] = data_dict["clicks"]/(data_dict["visualizations"]+1e-6)

        if len(to_save_data) == 1:
            to_save_data = to_save_data[0]
        
        return to_save_data

    def get_secondary(self, primary: Product):
        """ Support method to retrieve the secondary products associated to the primary product considered. The output is a list of 2 object
        of the class Product (i.e. the 2 secondary products)"""
        secondary_indices = self.secondary_dict[primary.name]
        secondary_list = [self.products[secondary_indices[0]], self.products[secondary_indices[1]]]
        return secondary_list


    class Graph_path:


        def __init__(self):
            self.primary_seen = []
            self.probability = 1
            self.collected_margin = 0
            self.link = [-1]
            self.link_prob = [0]


        def __init__(self, info_dict=None):
            if info_dict is None:
                self.primary_seen = []
                self.primary_bought = []
                self.probability = 1
                self.collected_margin = 0
                self.link = [-1]
                self.link_prob = [0]
            else:
                self.primary_seen = info_dict['primary_seen']
                self.primary_bought = info_dict['primary_bought']
                self.probability = info_dict['probability']
                self.collected_margin = info_dict['collected_margin']
                self.link = info_dict['link']
                self.link_prob = info_dict['link_prob']


        def __str__(self):
            output = ""
            output += "primary seen : " + str(self.primary_seen) + "\n"
            output += "primary bought : " + str(self.primary_bought) + "\n"
            output += "probability : " + str(self.probability) + "\n"
            output += "collected_margin : " + str(self.collected_margin) + "\n"
            return output


        def copy_info(self):
            info_dict = {'primary_seen': self.primary_seen.copy(),
                         'primary_bought': self.primary_bought.copy(),
                         'probability': self.probability,
                         'collected_margin': self.collected_margin,
                         'link': self.link.copy(),
                         'link_prob': self.link_prob.copy()}
            return info_dict


        def expected_return(self):
            return self.probability*self.collected_margin


    def explore_path(self, paths_list: list[Graph_path], path: Graph_path, primary_index, price_combination, user_index):
        
        # initialization of paths in the case it is None
        if path is None:
            path = self.Graph_path()
        # if primary_index = -1 it means the path has jumped to an unexplored second secondary product
        if primary_index == -1:
            # first at all check if the second secondary is in already visited product
            path0 = self.Graph_path(path.copy_info())
            if path0.link[-1] in path0.primary_seen:
                path0.link.pop()
                path0.link_prob.pop()
                if path0.link[-1] == -1:
                    paths_list.append(path0)
                else:
                    self.explore_path(paths_list, path0, -1, price_combination, user_index)
                return
            # now explore tha case where second secondary can be clicked
            # we have 2 possible new paths:
            # 1) we click on the second secondary stored as last element ok path.link
            # 2) we do NOT click
            path1 = self.Graph_path(path.copy_info())
            path2 = self.Graph_path(path.copy_info())
            # 1) we click, so we explore a new path with the second secondary considered as primary
            new_primary1 = path1.link.pop()
            path1.probability *= path1.link_prob.pop()
            self.explore_path(paths_list, path1, new_primary1, price_combination, user_index)
            # 2) we do NOT click, so we have 2 possible situations:
            # a) there are not other second secondary to jump to
            # b) there is at least another second secondary to be explored, and we explore it
            path2.link.pop()
            path2.probability *= 1 - path2.link_prob.pop()
            # a)
            if path2.link[-1] == -1:
                paths_list.append(path2)
            # b)
            else:
                self.explore_path(paths_list, path2, -1, price_combination, user_index)
            return
        # add the primary index to the list of seen primary of the path
        path.primary_seen.append(primary_index)
        # retrieve secondary products indexes
        primary_name = self.products[primary_index].name
        sec1_ind = self.secondary_dict[primary_name][0]
        sec2_ind = self.secondary_dict[primary_name][1]
        # compute b_i, so the probability to buy the primary product considered
        b_i = self.conversion_rates[user_index][primary_index, price_combination[primary_index]]
        # compute expected margin
        margin = self.products[primary_index].get_daily_margin(price_combination[primary_index])
        exp_margin = margin * (self.n_prod_sold[user_index][primary_index]) # margin * expected number of items bought, that is the poisson parameter
        # compute probabilities to click on the secondary given that the primary is bought
        q_1 = self.graph_weights[user_index][primary_index, sec1_ind]
        q_2 = self.graph_weights[user_index][primary_index, sec2_ind]
        ######################
        # PRIMARY NOT BOUGHT #
        ######################
        # Path where user does NOT buy the product
        # create new path as copy of the previous one
        path1 = self.Graph_path(path.copy_info())
        # update the probability according to the event: "product not bought"
        path1.probability *= 1 - b_i
        # 2 possibilities:
        # a) we cannot explore the path anymore, so we append the path to paths_list and terminate the exploration
        # b) we have a second secondary to be clicked --> pass -1 as new primary to deal with this case
        if path1.link[-1] == -1:
            paths_list.append(path1)
        else:
            self.explore_path(paths_list, path1, -1, price_combination, user_index)
        #####################
        # PRIMARY IS BOUGHT #
        #####################
        path2 = self.Graph_path(path.copy_info())
        path2.probability *= b_i
        path2.collected_margin += exp_margin
        path2.primary_bought.append(primary_index)
        # first we check if is possible to click on the secondary, otherwise we have no path to explore
        sec1_seen = sec1_ind in path.primary_seen
        sec2_seen = sec2_ind in path.primary_seen
        if sec1_seen and sec2_seen:
            if path2.link[-1] == -1:
                paths_list.append(path2)
            else :
                self.explore_path(paths_list, path2, -1, price_combination, user_index)
            return
        if sec1_seen:
            # we know for sure that sec2 can be seen, so we can't click on sec1
            # 1) we click on sec2 and explore the related path
            path3 = self.Graph_path(path2.copy_info())
            path3.probability *= q_2
            self.explore_path(paths_list, path3, sec2_ind, price_combination, user_index)
            # 2) we do NOT click on sec2 and there two possible situation:
            # a) NO past second secondary to return to : stop exploration
            # b) past second secondary to explore : explore this path
            path4 = self.Graph_path(path2.copy_info())
            path4.probability *= 1-q_2
            if path4.link[-1] == -1:
                paths_list.append(path4)
            else:
                self.explore_path(paths_list, path4, -1, price_combination, user_index)
            return
        if sec2_seen:
            # we know for sure that sec1 can be seen, so we can't click on sec2
            # 1) we click on sec1 and explore the related path
            path3 = self.Graph_path(path2.copy_info())
            path3.probability *= q_1
            self.explore_path(paths_list, path3, sec1_ind, price_combination, user_index)
            # 2) we do NOT click on sec1 and there two possible situation:
            # a) NO past second secondary to return to : path is closed
            # b) past second secondary to explore : explore this path
            path4 = self.Graph_path(path2.copy_info())
            path4.probability *= 1-q_1
            if path4.link[-1] == -1:
                paths_list.append(path4)
            else:
                self.explore_path(paths_list, path4, -1, price_combination, user_index)
            return
        # if we arrive at this point we know for sure that both secondary products can be seen
        # in this case we have 3 new possible paths:
        # 1) click on first secondary --> append secondo secondary to link and explore the new path
        # 2) do NOT click on first secondary and click on second secondary
        # 3) no clicks
        
        # 1)
        path3 = self.Graph_path(path2.copy_info())
        path3.probability *= q_1
        path3.link.append(sec2_ind)
        path3.link_prob.append(q_2)
        self.explore_path(paths_list, path3, sec1_ind, price_combination, user_index)
        # 2)
        path4 = self.Graph_path(path2.copy_info())
        path4.probability *= (1-q_1)*q_2
        self.explore_path(paths_list, path4, sec2_ind, price_combination, user_index)
        # 3)
        path5 = self.Graph_path(path2.copy_info())
        path5.probability *= (1-q_1)*(1-q_2)
        # know we have 2 possible scenarios:    
        # a) NO past second secondary to return to : path is closed
        # b) past second secondary to explore : explore this path
        if path5.link[-1] == -1:
            paths_list.append(path5)
        else:
            self.explore_path(paths_list, path5, -1, price_combination, user_index)
        return


    def product_reward(self, prod_index, user_index, price_combination):
        paths_list = []
        self.explore_path(paths_list, None, prod_index, price_combination, user_index)
        product_reward = 0. 
        for path in paths_list:
            product_reward += path.expected_return()
        return product_reward


    def user_reward(self, user_index, price_combination):
        # initialize reward and index i for the starting page
        reward = 0.
        for i in range(len(self.products)):
            alpha_i = self.alpha_ratios[user_index][i]
            # product_reward compute the expected return starting from a specific product, so we have to multiply it 
            # for the probability of starting from that product (alpha_i)
            reward += alpha_i * self.product_reward(i, user_index, price_combination)
        return reward

    def expected_reward(self, price_combination, conversion_rates=None, alpha_ratios=None, n_prod=None,
                        graph_weights=None, user_index=None, group_list=None, feat_prob_mat=None):
        """ Method that compute the expected reward related to the prices' combination passed to the function.
            If the only argument passed is the price combination the function returns the theoretical expected reward.
            The method can receive 4 optional arguments:
                - conversion rate: list of matrices representing the probabilities to buy a product i at a price j for the user k
                - alpha_ratios : list of lists of probabilities to land on product i at first for user j
                - n_prod : list of average number of product sold for a user
                - graph_weights : list of matrices describing the probabilities of a click on a secondary product given that we have
                                  bought a primary product (a matrix for each user)
            If at least one of the optional argument is passed, the function returns an expected reward different
            from the theoretical one. In this case the output is the expected reward when argument passed are assumed
            uncertain, so we are guessing their true value (e.g. with a bandit algorithm) """
        
        # At first, we have to deal with the argument passed and initialize the variable conversion_rate, alpha_ratios,
        # n_prod_sold and graph_weights of the environment accordingly
        #------------------#
        # CONVERSION RATES #
        #------------------#
        if conversion_rates is None:
            # conversion rate is certain, so we consider the theoretical values given by the parameters chosen for
            # the gamma distribution of each user
            # BUT if we are passing a group list we must reorder the theoretical_values
            if group_list is not None:
                self.conversion_rates = []
                for feat_couple in group_list:
                    # Adapt graph_weights to the specific case
                    i,j = feat_couple
                    user_index = self.feature_matrix[i,j]
                    self.conversion_rates.append(copy.deepcopy(self.theoretical_values["conversion_rates"][user_index]))
            else:
                self.conversion_rates = copy.deepcopy(self.theoretical_values["conversion_rates"])
        else :
            # conversion rate are uncertain, so we consider the guess passed to the function
            self.conversion_rates = conversion_rates
        #--------------#
        # ALPHA RATIOS #
        #--------------#
        if alpha_ratios is None:
            # alpha_ratios are assumed to be certain
            # BUT if we are passing a group list we must reorder the theoretical_values
            if group_list is not None:
                self.alpha_ratios = []
                for feat_couple in group_list:
                    # Adapt graph_weights to the specific case
                    i,j = feat_couple
                    user_index = self.feature_matrix[i,j]
                    self.alpha_ratios.append(copy.deepcopy(self.theoretical_values["alpha_ratios"][user_index]))
            else:
                self.alpha_ratios = copy.deepcopy(self.theoretical_values["alpha_ratios"])
        else:
            # alpha_ratios uncertain, so we consider the guess passed to the function
            self.alpha_ratios = alpha_ratios
        #------------------------#
        # NUMBER OF PRODUCT SOLD #
        #------------------------#
        if n_prod is None:
            # number of product sold is certain, so we retrieve the theoretical values
            # BUT if we are passing a group list we must reorder the theoretical_values
            if group_list is not None:
                self.n_prod_sold = []
                for feat_couple in group_list:
                    # Adapt graph_weights to the specific case
                    i,j = feat_couple
                    user_index = self.feature_matrix[i,j]
                    self.n_prod_sold.append(copy.deepcopy(self.theoretical_values["n_prod_sold"][user_index]))
            else:
                self.n_prod_sold = copy.deepcopy(self.theoretical_values["n_prod_sold"])
        else:
            # number of product sold is uncertain, so we consider a guess of the mean value
            self.n_prod_sold = n_prod
        #-–––-----------#
        # GRAPH WEIGTHS #
        #---------------#
        if graph_weights is None:
            # Graph weights are considered certain, so we simply use the values stored in the user classes
            # BUT if we are passing a group list we must reorder the theoretical_values
            if group_list is not None:
                self.graph_weights = []
                for feat_couple in group_list:
                    # Adapt graph_weights to the specific case
                    i,j = feat_couple
                    user_index = self.feature_matrix[i,j]
                    self.graph_weights.append(copy.deepcopy(self.theoretical_values['graph_weights'][user_index]))
            else:
                self.graph_weights = copy.deepcopy(self.theoretical_values["graph_weights"])
        else:
            self.graph_weights = graph_weights
        
        # initialize final reward 
        reward = 0.

        # if in the environment we have only 1 user we simply return the single_reward linked to the user
        if len(self.users) == 1:
            reward = self.user_reward(0, price_combination)
        
        # STEP7 (CONTEXT GENERATION) CASE ONLY!
        if group_list is not None:
            prob_list = []
            for feat_couple in group_list:
                # Adapt graph_weights to the specific case
                i,j = feat_couple
                if feat_prob_mat is None:
                    # Retrieve theoretica values
                    prob_list.append(self.feat_prob_matrix[i,j])
                else:
                    # Retrieve the frequency estimate of couple of features needed
                    prob_list.append(feat_prob_mat[i,j])
            # Compute the reward for each couple of features and weight the result for the corresponding probability
            for user_index in range(len(group_list)):
                reward += prob_list[user_index]*self.user_reward(user_index, price_combination)
        else:        
            # The default value for user_index is None, representing the case of aggregated demand curve
            if user_index is None :
                # if we have more than one user we have to weight the reward linked to the users with the 
                # theoretical frequencies of the user categories (user_cat_prob)
                for i in range(len(self.users)):
                    user_reward = self.user_reward(i, price_combination)
                    reward += self.user_cat_prob[i] * user_reward
            # Otherwise we return the expected reward for the specified user
            else :
                reward = self.user_reward(user_index, price_combination)
            
        return reward

    def optimal_reward(self, user_index=None, Disaggregated = False):

        """This method explores all the possible combination with a brute force approach to determine which is the price combination
            that returns the highest expected reward. It returns both the optimal price combination and optimal expected reward"""
        optimal_combination = [0, 0, 0, 0, 0]
        reward_max = 0.
        reward = 0.
        # enumerate all possible combinations of prices (4^5, 1024)
        possible_combinations = []
        for i1 in range(4):
            for i2 in range(4):
                for i3 in range(4):
                    for i4 in range(4):
                        for i5 in range(4):
                            possible_combinations.append([i1, i2, i3, i4, i5])

        if Disaggregated :
            rewards_list = np.zeros(len(self.users))
            opt_combination_list = [[]]*3
            for i in range(len(self.users)) :
                rewards_list[i], opt_combination_list[i] = self.optimal_reward(user_index = i)
            return rewards_list, opt_combination_list

        for price_combination in possible_combinations:
            # compute the reward for the price combination considered
            reward = self.expected_reward(price_combination = price_combination, user_index = user_index)
            # update if actual  reward is greater than best past  reward
            if reward > reward_max:
                reward_max = reward
                optimal_combination = price_combination.copy()
        
        return reward_max, optimal_combination

    def abrupt_change_random(self, mean_sigma, std_lambda):
        """ Method to call a ranodm abrupt change in the demand curve for each user. The abrupt change is modeled with the 
            following criterion:
            1) Sample from a gaussian with specified standard deviation the variation to be applied to the mean; Sample
                a random value between (std/lambda, lambda*std) for the new standard deviation of reservation price
            2) Modify the reservation price parameters in the user accordingly
            3) Set reservation price distributions in user class for the new parameters"""
        for i, user in enumerate(self.users):
            for i in range(5):
                mean_variation = np.random.randn()*mean_sigma
                user.res_price_params['mean'][i] += mean_variation
                old_std = user.res_price_params['std'][i]
                new_std = old_std/std_lambda + np.random.rand()*(old_std*std_lambda - old_std/std_lambda)
                user.res_price_params['std'][i] = new_std
            user.set_res_price_distr()
            # update the theoretical_values for the conversion rates
            CR_matrix = []
            CR_list = []
            for product in self.products:
                for price in product.prices:
                    prod_ind = product.index
                    CR_list.append(user.get_buy_prob(price, prod_ind))
                CR_matrix.append(CR_list.copy())
                CR_list = []
            CR_matrix = np.matrix(CR_matrix)
            self.theoretical_values['conversion_rates'][i] = CR_matrix.copy()
            CR_matrix = []
    
    def abrupt_change_deterministic(self, new_res_price_param):
        for i in range(len(self.users)):
            # Change the reservation prices' parameters
            self.users[i].res_price_params = new_res_price_param[i].copy()
            self.users[i].set_res_price_distr()
            # update the theoretical_values for the conversion rates
            CR_matrix = []
            CR_list = []
            for product in self.products:
                for price in product.prices:
                    prod_ind = product.index
                    CR_list.append(self.users[i].get_buy_prob(price, prod_ind))
                CR_matrix.append(CR_list.copy())
                CR_list = []
            CR_matrix = np.matrix(CR_matrix)
            self.theoretical_values['conversion_rates'][i] = CR_matrix.copy()
            CR_matrix = []
