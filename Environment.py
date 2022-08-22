from zmq import NULL
from UserCat import *
from Product import *
import copy




class Environment:
    """Class containing all the informations that characterize the problem, from the classes of users to the list of available
       products. """

    def __init__(self, users: list[UserCat], products: list[Product], lambda_q, Secondary_dict,
                 user_cat_prob):
        # List of different categories of users considered. If len(users) == 1 --> AGGREGATED DEMAND         
        self.users = users
        # List of available products: each of them has available the information of its position in the list -> attribute label
        self.products = products
        # lambda_q is the parameter that determines how much the second secondary is less probable to be clicked
        self.lambda_q = lambda_q
        # dictionary of lists of secondary products
        self.Secondary_dict = Secondary_dict
        # relative frequency of the users category
        self.user_cat_prob = user_cat_prob # TODO:valutare se inserirlo come membro della classe userCat

        """ 4 informations can be certain or uncertain in our simulations :
            - conversion rate: list of matrices representig the probabilities to buy a product i at a price j for the user k
            - alpha_ratios : list of lists of probabilities to land on product i at first for user j
            - nprod : list of average number of product sold for a user
            - graph_weights : list of matrices describing the probabilities of a click on a
                secondary product given that we have bought a primary product (a matrix for each user) """

        # Theoretical values for conversion_rates, alpha_ratios, n_prod_sold and graph_weights
        self.theoretical_values = {}

        # CONVERSION RATES
        # compute theoretical values for conversion rate when them are certain
        self.theoretical_values['conversion_rates'] = []
        self.conversion_rates = []
        CR_matrix = []
        CR_list = []
        for user in self.users:
            for product in self.products:
                for price in product.prices:
                    CR_list.append(user.get_buy_prob(price))
                
                CR_matrix.append(CR_list.copy())
                CR_list = []
            
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
        for user in self.users :
                self.theoretical_values["n_prod_sold"].append(user.poisson_lambda + 1) # EXPECTED
                self.n_prod_sold.append(user.poisson_lambda + 1)

        # GRAPH WEIGHTS
        self.theoretical_values["graph_weights"] = []
        self.graph_weights = []
        
        for user in self.users :
            self.theoretical_values["graph_weights"].append(user.probabilities)
            self.graph_weights.append(user.probabilities.copy())

    def user_profit(self, user : UserCat, price_combination, product_index, to_save_dict: dict):
        
        # retrieve the price of the product indicated by product_index for the current price_combination
        price_ind = price_combination[product_index]
        product_price = self.products[product_index].prices[price_ind]
        
        # check if the primary product is bought
        primary_bought = user.buy(product_price)

        # if the conversion_rate are uncertain, update information retrieved from the simulation
        if "CR_vector" in to_save_dict.keys() :
            # update number of times users has visualized the product 
            to_save_dict["CR_vector"][1][product_index] +=1
            if primary_bought:
                # only if the product is bought the number of sales are increased
                to_save_dict["CR_vector"][0][product_index] +=1

        if not(primary_bought) : 
            return

        # User bought the object, so i sample how many products He bought and compute the margin on the sale
        n_prod_bought = user.get_prod_number()
        profit = self.products[product_index].margins[price_ind] * n_prod_bought

        # if the numbers of product sold are uncertain, update information retrieved from the simulation
        if "n_prod_sold" in to_save_dict.keys() :
            to_save_dict["n_prod_sold"][0][product_index] += n_prod_bought
            to_save_dict["n_prod_sold"][1][product_index] += 1
        
        """The margin of the user is updated recursively every time he proceeds with a purchase, considering the margin of 
           that product and the number of items bought by the user (random number)"""
        
        #GET THE PRODUCT FROM THE DICT -> POSSO FARLI DIVENTARE DEI METODI
        first_secondary_index = self.Secondary_dict.get(self.products[product_index].name)[0]
        first_secondary = self.products[first_secondary_index]
        second_secondary_index = self.Secondary_dict.get(self.products[product_index].name)[1]
        second_secondary = self.products[second_secondary_index]

        """To simulate the random behaviour of the user we sample from a random distribution and we use it to evaluate whether
           an event has occurred or not. """
        
        # the user clicks on the first secondary if it has never been shown before and with a probability
        # defined by user.probabilities

        first_click = (np.random.uniform() < user.probabilities[
            self.products[product_index].label, first_secondary.label]) and first_secondary not in user.visited_products
        
        # if the graph weights are uncertain we update the information store in to_save_dict
        # with respect to the result of the simulation
        if "graph_weights" in to_save_dict.keys() :
            # update visualizations for the pairs primary-secondary only if we have not already seen the product
            if first_secondary not in user.visited_products :
                to_save_dict["visualizations"][product_index][first_secondary_index] += 1

            # if in the simulation the user has clicked we update also clicks values
            if first_click :
                to_save_dict["clicks"][product_index][first_secondary_index] += 1

        # click sul primo e non l'ho ancora visitato
        if first_click :
            user.visited_products.append(first_secondary)  # add visited product to list
            self.user_profit(user, price_combination, first_secondary.label, to_save_dict)
        
        # the user clicks on the second secondary if it has never been shown before and with a probability
        # defined by user.probabilities
        second_click = np.random.uniform() < self.lambda_q * user.probabilities[
            self.products[product_index].label, second_secondary.label] and second_secondary not in user.visited_products
        
        # if the graph weights are uncertain we update the information store in to_save_dict
        # with respect to the result of the simulation
        if "graph_weights" in to_save_dict.keys() :
            # update visualizations for the pairs primary-secondary only if we have not already seen the product
            if second_secondary not in user.visited_products :
                to_save_dict["visualizations"][product_index][second_secondary_index] += 1
            # if in the simulation the user has clicked we update also clicks values
            if second_click :
                to_save_dict["clicks"][product_index][second_secondary_index] += 1

        #click sul secondo e non l'ho ancora visitato
        if second_click :
            user.visited_products.append(second_secondary)  # add visited product to list
            self.user_profit(user, price_combination, second_secondary.label, to_save_dict)
        
        return


    def execute(self, user: UserCat, price_combination, to_save_dict: dict):
        """Method which simulates the entire interaction (considering even the case in which the user doesn't visit the website)
           of a user with our infrastructure. It returns the cumulative amount of margin generated by this interaction. """
        
        # sample the reservation price of the user considered
        user.sample_res_price()
        # sample which is the first product showed to the user
        page_index = user.start_event()
        # if alpha ratios are uncertain count each time a product is open as first
        if "alpha_ratios" in to_save_dict.keys() :
            to_save_dict["alpha_ratios"][page_index] += 1
        
        # svuoto i prodotti visitati
        user.empty_visited_products()
        user.visited_products = [self.products[page_index]]
        
        self.user_profit(user, price_combination, page_index, to_save_dict)
        
        return
        

    def simulate_day(self, daily_users, price_combination, to_save: list):
        """Method which simulates the usage of our website into an entire working day. Each day the alphas of each class of users
           are updated according to a Dirichlet distribution, it takes as input number of users, user probability (that now will be
           inside usercat and the price combination of today"""

        # We may need to return also approximation for:
        #   - conversion rates
        #   - alpha_ratios
        #   - number of products sold
        #   - graph weights
        # In fact, when one of these variable is uncertain we need its approximation from the simulation
        # For this reason we need to create some structure to store these values
        
        d = len(price_combination)
        to_save_dict = {}
        if "conversion_rates" in to_save :
            to_save_dict["CR_vector"] = np.zeros((2,d))

        if "alpha_ratios" in to_save :
            to_save_dict["alpha_ratios"] = np.zeros(d)

        if "products_sold" in to_save :
            to_save_dict["n_prod_sold"] = np.zeros((2,d))
            
        if "graph_weights" in to_save :
            to_save_dict["graph_weights"] = np.zeros((d,d))
            to_save_dict["visualizations"] = np.zeros((d,d))
            to_save_dict["clicks"] = np.zeros((d,d))

        # We have to deal with the case of multiple categories of users :
        # let's create a list of dictionary (1 for each user category) of data to save
        to_save_data = []
        for i in range(len(self.users)) :
            to_save_data.append(copy.deepcopy(to_save_dict))

        # Generate daily alpha ratios for each user category for the new day
        for user in self.users:
            user.generate_alphas()

        # We simulate the interactions of "users_number" users
        for i in range(daily_users):

            # extract the category of the simulated user
            if len(self.users) == 1 :
                # if we have only a user we don't need to extract the category
                user_kind = 0
            else :
                user_indices = list(range(len(self.users)))
                user_kind = np.random.choice(user_indices, p = self.user_cat_prob)

            # incremente the daily profit of the website by the profit done with the simulated user
            self.execute(self.users[user_kind], price_combination, to_save_data[user_kind])
            # notice that we have passed only the dictionary for the specific user category sampled

        for i in range(len(self.users)) :
            to_save_dict = to_save_data[i]
            
            # if conversion rates are uncertain save the result obtained by the daily simulation
            if "conversion_rates" in to_save :
                to_save_dict["CR_vector"] = to_save_dict["CR_vector"][0]/(to_save_dict["CR_vector"][1]+0.01)
                # +0.01 at denominator to avoid 0/0 division

            # if alpha ratios are uncertain save the result obtained by the daily simulation
            if "alpha_ratios" in to_save :
                to_save_dict["alpha_ratios"] = to_save_dict["alpha_ratios"]/np.sum(to_save_dict["alpha_ratios"])
            
            # if number of product sold per product are uncertain save the result obtained by the daily simulation
            if "products_sold" in to_save :
                to_save_dict["n_prod_sold"] = to_save_dict["n_prod_sold"][0]/(to_save_dict["n_prod_sold"][1]+0.01)
            
            # if number of product sold per product are uncertain save the result obtained by the daily simulation
            if "graph_weights" in to_save :
                to_save_dict["graph_weights"] = to_save_dict["clicks"]/(to_save_dict["visualizations"] + 0.01)
                to_save_dict.pop("clicks")
                to_save_dict.pop("visualizations") 
            
            # we store the daily profit (USELESS)
            # to_save_dict["daily_profit"] = daily_profit/daily_users
        
        if len(self.users) == 1:
            to_save_data = to_save_data[0]

        return to_save_data

    def get_secondary(self, primary: Product) :
        """ Support method to retrieve the secondary products associated to the primary product considered. The output is a 
            a list of 2 object of the class Product (i.e. the 2 secondary products) """

        secondary_indices = self.Secondary_dict[primary.name]

        secondary_list = [ self.products[secondary_indices[0]], self.products[secondary_indices[1]] ]

        return secondary_list


    def expected_reward111(self, price_combination, conversion_rates = None, alpha_ratios = None, n_prod = None, graph_weights = None):
        """ Method that compute the expected reward related to the prices combination passed to the function.
            If the only argument passed is the price combination the function returns the theoretical expected
            reward.
            The method can receive 4 optional arguments:
                - conversion rate: list of matrices representig the probabilities to buy a product i at a price j for the user k
                - alpha_ratios : list of lists of probabilities to land on product i at first for user j
                - nprod : list of average number of product sold for a user
                - graph_weights : list of matrices describing the probabilities of a click on a
                    secondary product given that we have bought a primary product (a matrix for each user)
            If at least one of the optional argument is passed, the function returns an expected reward different
            from the theoretical one. In this case the output is the expected reward when argument passed are assumed
            uncertain and we are guessing their true value (e.g. with a bandit algorithm) """
        
        # At first we have to deal with the argument passed and initialize the variable conversion_rate, alpha_ratios,
        # n_prod_sold and graph_weights of the environment accordingly

        # CONVERSION RATES
        if conversion_rates == None:
            # conversion rate is certain, so we consider the theoretical values given by the parameters chosen for
            # the gamma distribution of each user
            self.conversion_rates = copy.deepcopy(self.theoretical_values["conversion_rates"])
        else :
            # conversion rate are uncertain, so we consider the guess passed to the function
            self.conversion_rates = conversion_rates

        # ALPHA RATIOS
        if alpha_ratios == None :
            # alpha_ratios are assumed to be certain
            self.alpha_ratios = copy.deepcopy(self.theoretical_values["alpha_ratios"])
        
        else :
            # alpha_ratios uncertain, so we consider the guess passed to the function
            self.alpha_ratios = alpha_ratios

        # NUMBER OF PRODUCT SOLD
        if n_prod == None :
            # number of product sold is certain and we retrieve it bu theoretical values
            self.n_prod_sold = copy.deepcopy(self.theoretical_values["n_prod_sold"])
        else :
            # number of product sold is uncertain, so we consider a guess of the mean value
            self.n_prod_sold = n_prod

        # GRAPH WEIGTHS
        if graph_weights == None:
            # Graph weights are considered certain, so we simply use the values stored in the user classes
            self.graph_weights = copy.deepcopy(self.theoretical_values["graph_weights"])
            
        else :
            self.graph_weights = graph_weights

        # initialize final reward 
        reward = 0.
        
        # if in the environment we have only 1 user we simply return the single_reward linked to the user
        if len(self.users) == 1 :
            reward = self.single_reward(0, price_combination)

        else :
            # if we have more than one user we have to weight the reward linked to the users with the 
            # theoretical frequencies of the user categories (user_cat_prob)
            for i in range(len(self.users)) :
                user_reward = self.single_reward(i, price_combination)
                reward += self.user_cat_prob[i] * user_reward
            
        return  reward
          

    def single_reward(self, user_index, price_combination):
        """ Method to compute the expected reward for a single user category given a specific price combination for the 5 
            products """

        # select the right user
        user = self.users[user_index]

        # initialize    reward and index i for the starting page
        reward = 0.
        i = 0

        for product in self.products:
            alpha_i = self.theoretical_values['alpha_ratios'][user_index][i]
            i = i+1
            # product_reward compute the expected return starting from a specific product, so we have to multiply it 
            # for the probability of starting from that product (alpha_i)
            reward += alpha_i * self.product_reward(product, [], [0], [Product([], -1, "null", [])], price_combination, user_index)
        
        return  reward

    def product_reward(self, primary: Product, primary_history: list[Product], q_link : list[float], 
                    link: list[Product], price_combination, user_index):
        """ Method to compute the expected reward for a single product. The method is thought to give the priority to the 
            fist secondary product related to the primary product."""
        
        # first check if we have to stop the function, this is the case if:
        # 1) primary is in primary_history --> the probability of the click is zero so the expected return
        # 2) primary is the "null" product, i.e. we cannot explore further a certain path
        # if not, add primary to the primary history  
        
        if primary in primary_history or primary.label == -1 :
            return 0
        else:
            primary_history.append(primary)

        # retrieve the margin for the primary product
        i = primary.label
        margin = primary.get_daily_margin(price_combination[i])
        
        # compute b_i, i.e the probability to buy the primary product considered
        b_i = self.conversion_rates[user_index][i][price_combination[i]]
        
        # compute expected margin
        exp_margin = margin * (self.n_prod_sold[user_index]) # margin * expected number of items bought, that is the poisson parameter

        # if both secondary items have been already seen we simply return the expected margin and go back to the link
        secondary_list = self.get_secondary(primary)

        s_1 = secondary_list[0] # for better reading we store the secondary products and their labels
        s_2 = secondary_list[1]
        
        j_1 = s_1.label
        j_2 = s_2.label
        
        # if all the secondary are in the primary history, we return the expected margin linked to the purchase of the primary
        # and return to the most recent "link"
        if s_1 in primary_history and s_2 in primary_history:
            return b_i * exp_margin + q_link.copy()[-1] * self.product_reward(link.copy()[-1], primary_history, q_link.copy()[:-1],
                                                                                link.copy()[:-1], price_combination, user_index)

        # all exceptions have been treated, let's now compute the expected return in the basic case
        
        # compute probabilities to click on the secondary given that the primary is bought
        q_1 = self.graph_weights[user_index][i, j_1]
        q_2 = self.graph_weights[user_index][i, j_2] * self.lambda_q
        
        link1 = link.copy()
        link1.append(s_2)
        link2 = link.copy()
        link3 = link.copy()
        q_link1 = q_link.copy()
        q_link1.append(q_2)
        q_link2 = q_link.copy()
        q_link3 = q_link.copy()

        prim_hist1 = primary_history.copy()
        prim_hist2 = primary_history.copy()
        prim_hist3 = primary_history.copy()

        return (b_i*(exp_margin + q_1 * self.product_reward(s_1, prim_hist1, q_link1, link1, price_combination, user_index) +
                    (1-q_1) * q_2 * self.product_reward(s_2, prim_hist2, q_link2, link2, price_combination, user_index)) +
                (1 - b_i + b_i * (1-q_1) * (1-q_2) )* q_link3[-1] * self.product_reward(link3[-1], prim_hist3, q_link3[:-1], link3[:-1], price_combination, user_index) )
    

    def optimal_reward(self, user_index = -1):
        """ This method explores all the possible combination with a brute force approach to determine which is the price combination
            that returns the highest expected reward.
            It returns both the optimal price combination and optimal expected  reward """
            
        optimal_combination = [0, 0, 0, 0, 0]
        reward_max = 0
        reward = 0
        
        # enumerate all possible combinations of prices (4^5, 1024)
        possible_combinations = []

        # TODO: PENSARE UN MODO PIU' INTELLIGENTE!
        for i1 in range(4):
            for i2 in range(4):
                for i3 in range(4):
                    for i4 in range(4):
                        for i5 in range(4):
                            possible_combinations.append([i1, i2, i3, i4, i5])

        for price_combination in possible_combinations:       
            
            # compute the reward for the price combination considered
            reward = self.expected_reward(price_combination)
            
            # update if actual  reward is greater than best past  reward
            if  reward > reward_max:
                reward_max = reward
                optimal_combination = price_combination.copy()

        return  reward_max, optimal_combination

    class Graph_path:
        def __init__(self) :
            self.primary_seen = []
            self.probability = 1
            self.collected_margin = 0
            self.link = [-1]
            self.link_prob = [0]
        
        def __init__(self, info_dict = None):
            if info_dict == None :
                self.primary_seen = []
                self.primary_bought = []
                self.probability = 1
                self.collected_margin = 0
                self.link = [-1]
                self.link_prob = [0]
            else :
                self.primary_seen = info_dict['primary_seen']
                self.primary_bought = info_dict['primary_bought']
                self.probability = info_dict['probability']
                self.collected_margin = info_dict['collected_margin']
                self.link = info_dict['link']
                self.link_prob = info_dict['link_prob']

        def __str__(self) :
            output = ""
            output += "primary seen : " + str(self.primary_seen) + "\n"
            output += "primary bought : " + str(self.primary_bought) + "\n"
            output += "probability : " + str(self.probability) + "\n"
            output += "collected_margin : " + str(self.collected_margin) + "\n"
            return output


        def copy_info(self) :
            info_dict = {}
            info_dict['primary_seen'] = self.primary_seen.copy()
            info_dict['primary_bought'] = self.primary_bought.copy() 
            info_dict['probability'] = self.probability
            info_dict['collected_margin'] = self.collected_margin
            info_dict['link'] = self.link.copy()
            info_dict['link_prob'] = self.link_prob.copy()
            return info_dict

        def expected_return(self):
            return self.probability*self.collected_margin

    def explore_path(self, paths_list: list[Graph_path], path : Graph_path, primary_index, price_combination, user_index) :
        
        # initialization of paths in the case it is None
        if path == None :
            path = self.Graph_path()

        # if primary_index = -1 it means the path has jumped to an unexplored second secondary product
        if primary_index == -1:
            
            # first at all check if the second secondary is in already visited product
            path0 = self.Graph_path(path.copy_info())
            if path0.link[-1] in path0.primary_seen :
                path0.link.pop()
                path0.link_prob.pop()
                if path0.link[-1] == -1:
                    paths_list.append(path0)
                else :
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
            # b) there is at least another second secondary to be explored and we explore it
            path2.link.pop()
            path2.probability *= 1 - path2.link_prob.pop()

            # a)
            if path2.link[-1] == -1 :
                paths_list.append(path2)
            # b)
            else :
                self.explore_path(paths_list, path2, -1, price_combination, user_index)

            return

        # add the primary index to the list of seen primary of the path
        path.primary_seen.append(primary_index)

        # retrieve secondary products indeces
        primary_name = self.products[primary_index].name
        sec1_ind = self.Secondary_dict[primary_name][0]
        sec2_ind = self.Secondary_dict[primary_name][1]

        # compute b_i, i.e the probability to buy the primary product considered
        b_i = self.conversion_rates[user_index][primary_index][price_combination[primary_index]]
        
        # compute expected margin
        margin = self.products[primary_index].get_daily_margin(price_combination[primary_index])
        exp_margin = margin * (self.n_prod_sold[user_index]) # margin * expected number of items bought, that is the poisson parameter

        # compute probabilities to click on the secondary given that the primary is bought
        q_1 = self.graph_weights[user_index][primary_index, sec1_ind]
        q_2 = self.graph_weights[user_index][primary_index, sec2_ind] * self.lambda_q

        ######################
        # PRIMARY NOT BOUGHT #
        ######################
        # Path where user does NOT buy the product
            # create new path as copy of the prrevious one
        path1 = self.Graph_path(path.copy_info())
            # update the probability according to the event: "product not bought"
        path1.probability *= 1 - b_i
        
            # 2 possibilities:
            # a) we cannot explore the path anymore, so we append the path to paths_list and terminate the exploration
            # b) we have a second secondary to be clicked --> pass -1 as new primary to deal with this case
        if path1.link[-1] == -1 : 
            paths_list.append(path1)
        else :
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

        if sec1_seen and sec2_seen :
            if path2.link[-1] == -1 :
                paths_list.append(path2)
            else :
                self.explore_path(paths_list, path2, -1, price_combination, user_index)
            return

        if sec1_seen:
            # we know for sure that sec2 can be seen and we can't click on sec1

            # 1) we click on sec2 and explore the related path
            path3 = self.Graph_path(path2.copy_info())
            path3.probability *= q_2
            self.explore_path(paths_list, path3, sec2_ind, price_combination, user_index)

            # 2) we do NOT click on sec2 and there two possible situation:
            # a) NO past second secondary to return to : stop exploration
            # b) past second secondary to explore : explore this path
            path4 = self.Graph_path(path2.copy_info())
            path4.probability *= 1-q_2

            if path4.link[-1] == -1 :
                paths_list.append(path4)
            else :
                self.explore_path(paths_list, path4, -1, price_combination, user_index)
            
            return


        if sec2_seen:
            # we know for sure that sec1 can be seen and we can't click on sec2

            # 1) we click on sec1 and explore the related path
            path3 = self.Graph_path(path2.copy_info())
            path3.probability *= q_1
            self.explore_path(paths_list, path3, sec1_ind, price_combination, user_index)

            # 2) we do NOT click on sec1 and there two possible situation:
            # a) NO past second secondary to return to : path is closed
            # b) past second secondary to explore : explore this path
            path4 = self.Graph_path(path2.copy_info())
            path4.probability *= 1-q_1

            if path4.link[-1] == -1 :
                paths_list.append(path4)
            else :
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
        if path5.link[-1] == -1 :
            paths_list.append(path5)
        else :
            self.explore_path(paths_list, path5, -1, price_combination, user_index)
                  
        return

    def product_reward_path(self, prod_index, user_index, price_combination) :

        paths_list = []
        self.explore_path(paths_list, None, prod_index, price_combination, user_index)
        product_reward = 0. 
        for path in paths_list:
            product_reward += path.expected_return()

        return product_reward

    
    def user_reward_path(self, user_index, price_combination) :
        # initialize    reward and index i for the starting page
        reward = 0.

        for i in range(len(self.products)) :
            alpha_i = self.alpha_ratios[user_index][i]
            # product_reward compute the expected return starting from a specific product, so we have to multiply it 
            # for the probability of starting from that product (alpha_i)
            reward += alpha_i * self.product_reward_path(i, user_index, price_combination)
        
        return reward

    def expected_reward(self, price_combination, conversion_rates = None, alpha_ratios = None, n_prod = None, graph_weights = None) :

        """ Method that compute the expected reward related to the prices combination passed to the function.
            If the only argument passed is the price combination the function returns the theoretical expected
            reward.
            The method can receive 4 optional arguments:
                - conversion rate: list of matrices representig the probabilities to buy a product i at a price j for the user k
                - alpha_ratios : list of lists of probabilities to land on product i at first for user j
                - nprod : list of average number of product sold for a user
                - graph_weights : list of matrices describing the probabilities of a click on a
                    secondary product given that we have bought a primary product (a matrix for each user)
            If at least one of the optional argument is passed, the function returns an expected reward different
            from the theoretical one. In this case the output is the expected reward when argument passed are assumed
            uncertain and we are guessing their true value (e.g. with a bandit algorithm) """
        
        # At first we have to deal with the argument passed and initialize the variable conversion_rate, alpha_ratios,
        # n_prod_sold and graph_weights of the environment accordingly

        # CONVERSION RATES
        if conversion_rates == None:
            # conversion rate is certain, so we consider the theoretical values given by the parameters chosen for
            # the gamma distribution of each user
            self.conversion_rates = copy.deepcopy(self.theoretical_values["conversion_rates"])
        else :
            # conversion rate are uncertain, so we consider the guess passed to the function
            self.conversion_rates = conversion_rates

        # ALPHA RATIOS
        if alpha_ratios == None :
            # alpha_ratios are assumed to be certain
            self.alpha_ratios = copy.deepcopy(self.theoretical_values["alpha_ratios"])
        
        else :
            # alpha_ratios uncertain, so we consider the guess passed to the function
            self.alpha_ratios = alpha_ratios

        # NUMBER OF PRODUCT SOLD
        if n_prod == None :
            # number of product sold is certain and we retrieve it bu theoretical values
            self.n_prod_sold = copy.deepcopy(self.theoretical_values["n_prod_sold"])
        else :
            # number of product sold is uncertain, so we consider a guess of the mean value
            self.n_prod_sold = n_prod

        # GRAPH WEIGTHS
        if graph_weights == None:
            # Graph weights are considered certain, so we simply use the values stored in the user classes
            self.graph_weights = copy.deepcopy(self.theoretical_values["graph_weights"])
            
        else :
            self.graph_weights = graph_weights

        # initialize final reward 
        reward = 0.
        
        # if in the environment we have only 1 user we simply return the single_reward linked to the user
        if len(self.users) == 1 :
            reward = self.user_reward_path(0, price_combination)

        else :
            # if we have more than one user we have to weight the reward linked to the users with the 
            # theoretical frequencies of the user categories (user_cat_prob)
            for i in range(len(self.users)) :
                user_reward = self.user_reward_path(i, price_combination)
                reward += self.user_cat_prob[i] * user_reward
            
        return  reward
