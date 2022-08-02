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

        # variable to compute expected reward 
        self.conversion_rates = []
        self.alphas_ratio = []
        self.n_prod_sold = []
        self.graph_weights = []

    def user_profit(self, user : UserCat, price_combination, product_index, to_save_dict: dict):
        
        # passo una price_combination che passo dal main e un product index
        profit = 0.
        
        # retrieve the price of the product indicated by product_index for the current price_combination
        price_ind = price_combination[product_index]
        product_price = self.products[product_index].prices[price_ind]
        # check if the primary product is bought
        primary_bought = user.buy(product_price)

        # if the conversion_rate are uncertain, update information retrieved from the simulation
        if "CR_vector" in to_save_dict.keys() :
            to_save_dict["CR_vector"][1][product_index] +=1
            if primary_bought:
                to_save_dict["CR_vector"][0][product_index] +=1

        if not(primary_bought) or (self.products[product_index] in user.visited_products) : #TODO in teoria la seconda condizione non dovrebbe mai verificarsi
            return profit

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
        
        # the user clicks on a secondary if it has never been shown before and with a probability
        # defined by user.probabilities

        first_click = (np.random.uniform() < user.probabilities[
            self.products[product_index].label, first_secondary.label]) and first_secondary not in user.visited_products
        
        second_click = np.random.uniform() < self.lambda_q * user.probabilities[
            self.products[product_index].label, second_secondary.label] and second_secondary not in user.visited_products
        
        # if the graph weights are uncertain we update the information store in to_save_dict
        # with respect to the result of the simulation
        if "graph_weights" in to_save_dict.keys() :
            # update visualizations for the pairs primary-secondary
            to_save_dict["visualizations"][product_index][first_secondary_index] += 1
            to_save_dict["visualizations"][product_index][second_secondary_index] += 1
            # if in the simulation the user has clicked we update also clicks values
            if first_click :
                to_save_dict["clicks"][product_index][first_secondary_index] += 1
            if second_click :
                to_save_dict["clicks"][product_index][second_secondary_index] += 1
        
        # click sul primo e non l'ho ancora visitato
        if first_click :
            user.visited_products.append(first_secondary)  # add visited product to list
            return profit + self.user_profit(user, price_combination, first_secondary.label, to_save_dict)
        
        #click sul secondo e non l'ho ancora visitato
        if second_click :
            user.visited_products.append(second_secondary)  # add visited product to list
            return profit + self.user_profit(user, price_combination, second_secondary.label, to_save_dict)
        
        return profit


    def execute(self, user: UserCat, price_combination, to_save_dict: dict):
        """Method which simulates the entire interaction (considering even the case in which the user doesn't visit the website)
           of a user with our infrastructure. It returns the cumulative amount of margin generated by this interaction. """
        
        # sample the reservation price of the user considered
        user.sample_res_price()
        # sample which is the first product showed to the user
        page_index = user.start_event()
        # if alphas ratios are uncertain count each time a product is open as first
        if "alpha_ratios" in to_save_dict.keys() :
            to_save_dict["alpha_ratios"][page_index] += 1
        
        # svuoto i prodotti visitati
        user.empty_visited_products()
        
        return self.user_profit(user,price_combination, page_index, to_save_dict)
        

    def simulate_day(self, daily_users, price_combination, to_save: list):
        """Method which simulates the usage of our website into an entire working day. Each day the alphas of each class of users
           are updated according to a Dirichlet distribution, it takes as input number of users, user probability (that now will be
           inside usercat and the price combination of today"""

        # We may need to return also approximation for:
        #   - conversion rates
        #   - alphas_ratios
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

        # Generate daily alpha ratio for each user category for the new day
        for user in self.users:
            user.generate_alphas()
        
        # If we have more than 1 user we store the daily profit for all users
        if len(self.users) == 1:
            daily_profit = [0.]
        else :
            daily_profit = np.zeros(len(self.users))
        

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
            daily_profit[user_kind] += self.execute(self.users[user_kind], price_combination, to_save_data[user_kind])
            # notice that we have passed only the dictionary for the specific user category sampled

        for i in range(len(self.users)) :
            to_save_dict = to_save_data[i]
            
            # if conversion rates are uncertain save the result obtained by the daily simulation
            if "conversion_rates" in to_save :
                to_save_dict["CR_vector"] = to_save_dict["CR_vector"][0]/(to_save_dict["CR_vector"][1]+0.01)
                # +0.01 at denominator to avoid 0/0 division

            # if alphas ratio are uncertain save the result obtained by the daily simulation
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

    def product_reward(self, primary: Product, primary_history: list[Product], q_link : list[float], 
                    link: list[Product], price_combination, user_index):
        """ Method to compute the expected reward for a single product. The method is thought to give the priority to the 
            fist secondary product related to the primary product."""
        
        # first check if we have to stop the function, this is the case if:
        # 1) primary is in primary_history --> the probability of the click is zero so the expected return
        # 2) primary is the "null" product, i.e. we cannot explore further a certain path
        # if not, add primary to the primary history  
         
        if primary in primary_history or primary.label == "null" or link == None:
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
            return b_i * exp_margin + q_link[-1] * self.product_reward(link[-1], primary_history, q_link[:-1], link[:-1], price_combination, user_index)

        # all exceptions have been treated, let's now compute the expected return in the basic case
        
        
        # compute probabilities to click on the secondary given that the primary is bought
        q_1 = self.graph_weights[user_index][i, j_1]
        q_2 = self.graph_weights[user_index][i, j_2] * self.lambda_q

        new_link = link.append(s_2)
        new_q_link = q_link.append(q_2)

        return b_i*(exp_margin + q_1*self.product_reward(s_1, primary_history, new_q_link, new_link, price_combination, user_index) +
                      (1-q_1) * q_2 * self.product_reward(s_2, primary_history, q_link, link, price_combination, user_index) +
                      (1-q_1) * (1-q_2) * q_link[-1] * self.product_reward(link[-1], primary_history, q_link[:-1], link[:-1], price_combination, user_index))
    
    def expected_reward(self, price_combination, conversion_rates = None, alphas_ratio = None, n_prod = None, graph_weights = None):
        """ Method that compute the expected reward related to the prices combination passed to the function.
            If the only argument passed is the price combination the function returns the theoretical expected
            reward.
            The method can receive 4 optional arguments:
                - conversion rate: list of matrices representig the probabilities to buy a product i at a price j for the user k
                - alphas_ratio : list of lists of probabilities to land on product i at first for user j
                - nprod : list of average number of product sold for a user
                - graph_weights : list of matrices describing the probabilities of a click on a
                    secondary product given that we have bought a primary product (a matrix for each user)
            If at least one of the optional argument is passed, the function returns an expected reward different
            from the theoretical one. In this case the output is the expected reward when argument passed are assumed
            uncertain and we are guessing their true value (e.g. with a bandit algorithm) """
        
        # At first we have to deal with the argument passed and initialize the variable conversion_rate, alphas_ratio,
        # n_prod_sold and graph_weights of the environment accordingly

        # CONVERSION RATES
        if conversion_rates == None:
            # conversion rate is certain, so we consider the theoretical values given by the parameters chosen for
            # the gamma distribution of each user
            self.conversion_rates = []
            CR_matrix = []
            CR_list = []
            for user in self.users:
                for product in self.products:
                    for price in product.prices:
                        CR_list.append(user.get_buy_prob(price))
                    
                    CR_matrix.append(CR_list.copy())
                    CR_list = []
                
                self.conversion_rates.append(CR_matrix.copy())
                CR_matrix = []
        else :
            # conversion rate are uncertain, so we consider the guess passed to the function
            self.conversion_rates = conversion_rates

        # ALPHAS RATIO
        if alphas_ratio == None :
            # alphas_ratio are assumed to be certain
            self.alphas_ratio = []

            for user in self.users:
                self.alphas_ratio.append(user.alphas)
        
        else :
            # alphas_ratio uncertain, so we consider the guess passed to the function
            self.alphas_ratio = alphas_ratio

        # NUMBER OF PRODUCT SOLD
        if n_prod == None :
            # number of product sold is certain and given by the poisson parameter of each user 
            # (+1 because 0 products bought makes no sense, i.e. we are considered a translated poisson)

            self.n_prod_sold = []
            for user in self.users :
                self.n_prod_sold.append(user.poisson_lambda + 1)
        else :
            # number of product sold is uncertain, so we consider a guess of the mean value
            self.n_prod_sold = n_prod

        # GRAPH WEIGTHS
        if graph_weights == None:
            # Graph weights are considered certain, so we simply use the values stored in the user classes
            self.graph_weights = []
            for user in self.users :
                self.graph_weights.append(user.probabilities)
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
            alpha_i = user.alphas[i]
            i = i+1
            # product_reward compute the expected return starting from a specific product, so we have to multiply it 
            # for the probability of starting from that product (alpha_i)
            reward += alpha_i * self.product_reward(product, [], [0], [Product([], -1, "null", [])], price_combination, user_index)
        
        return  reward

    

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
