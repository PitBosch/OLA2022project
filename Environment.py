from zmq import NULL
from UserCat import *
from Product import *


class Environment:
    """Class containing all the informations that characterize the problem, from the classes of users to the list of available
       products. """

    def __init__(self, users: list[UserCat], products: list[Product], lambda_q, Secondary_dict,
                 user_cat_prob):
        self.users = users
        # List of available products: each of them has available the information of its position in the list -> attribute label
        self.products = products
        # lambda_q is the parameter that determines how much the second secondary is less probable to be clicked
        self.lambda_q = lambda_q
        # dictionary of lists of secondary products
        self.Secondary_dict = Secondary_dict
        # relative frequency of the users category
        self.user_cat_prob = user_cat_prob # TODO:valutare se inserirlo come membro della classe userCat

    def user_profit(self, user : UserCat, price_combination, product_index):
        
        # passo una price_combination che passo dal main e un product index
        margin = 0.
        
        # retrieve the price of the product indicated by product_index for the current price_combination
        price_ind = price_combination[product_index]
        product_price = self.products[product_index].prices[price_ind]

        if not(user.buy(product_price)) or (self.products[product_index] in user.visited_products) :
            return margin

        # ho comprato e calcolo quanto ho guadagnato
        margin = self.products[product_index].margins[price_ind] * user.get_prod_number()
        
        """The margin of the user is updated recursively every time he proceeds with a purchase, considering the margin of 
           that product and the number of items bought by the user (random number)"""
        
        #GET THE PRODUCT FROM THE DICT -> POSSO FARLI DIVENTARE DEI METODI
        first_secondary = self.products[self.Secondary_dict.get(self.products[product_index].name)[0]]
        second_secondary = self.products[self.Secondary_dict.get(self.products[product_index].name)[1]]

        """To simulate the random behaviour of the user we sample from a random distribution and we use it to evaluate whether
           an event has occurred or not. """
        
        first_click = (np.random.uniform() < user.probabilities[
            self.products[product_index].label, first_secondary.label])
        
        second_click = np.random.uniform() < self.lambda_q * user.probabilities[
            self.products[product_index].label, second_secondary.label]
        
        # click sul primo e non l'ho ancora visitato
        if first_click and first_secondary not in user.visited_products:
            user.visited_products.append(first_secondary)  # add visited product to list
            return margin + self.user_profit(user, price_combination, first_secondary.label)
        
        #click sul secondo e non l'ho ancora visitato
        if second_click and second_secondary not in user.visited_products:
            user.visited_products.append(second_secondary)  # add visited product to list
            return margin + self.user_profit(user, price_combination, second_secondary.label)
        
        return margin


    def execute(self, user: UserCat, price_combination):
        """Method which simulates the entire interaction (considering even the case in which the user doesn't visit the website)
           of a user with our infrastructure. It returns the cumulative amount of margin generated by this interaction. """
        
        # sample the reservation price of the user considered
        user.sample_res_price()
        # sample which is the first product showed to the user
        page_index = user.start_event()
        
        #svuoto i prodotti visitati
        user.empty_visited_products()
        
        return self.user_profit(user,price_combination,page_index)
        

    def simulate_day(self, users_number, price_combination):
        """Method which simulates the usage of our website into an entire working day. Each day the alphas of each class of users
           are updated according to a Dirichlet distribution, it takes as input number of users, user probability (that now will be
           inside usercat and the price combination of today"""

        # Generate the alpha ratio for each user category for the new day
        for user in self.users:
            user.generate_alphas()

        daily_profit = [0., 0., 0.] # we divide the daily profit for each type of user

        # We simulate the interactions of "users_number" users
        for i in range(users_number):
            # extract the category of the simulated user
            user_kind = np.random.choice([0, 1, 2], p = self.user_cat_prob)

            # incremente the daily profit of the website by the profit done with the simulated user
            daily_profit[user_kind] += self.execute(self.users[user_kind], price_combination)

        return daily_profit # list of profit divided for the user category

    def get_secondary(self, primary: Product) :
        """ Support method to retrieve the secondary products associated to the primary product considered. The output is a 
            a list of 2 object of the class Product (i.e. the 2 secondary products) """

        secondary_indices = self.Secondary_dict[primary.name]

        secondary_list = [ self.products[secondary_indices[0]], self.products[secondary_indices[1]] ]

        return secondary_list

    def exp_return(self, primary: Product, primary_history: list[Product], q_link : list[float], link: list[Product], price_combination, user: UserCat):
        """ Method to compute the expected return for a singel product. The method is thought to give the priority to the 
            fist secondary product related to the primary product."""
        
        # first check if we have to stop the function, this is the case if:
        # 1) primary is in primary_history --> the probability of the click is zero so the expected return
        # 2) primary is the "null" product, i.e. we cannot explore further a certain path
        # if not, add primary to the primary history  
         
        if primary in primary_history or primary.label == "null" or link == None:
            return 0
        else:
            primary_history.append(primary)

        # retrieve price and margin for the primary product
        i = primary.label
        price = primary.get_daily_price(price_combination[i])
        margin = primary.get_daily_margin(price_combination[i])
        
        # compute b_i, i.e the probability to buy the primary product considered
        b_i = user.get_buy_prob(price)
        
        # compute expected margin
        exp_margin = margin * (user.poisson_lambda + 1) # margin * expected number of items bought, that is the poisson parameter

        # if both secondary items have been already seen we simply return the expected margin and go back to the link
        secondary_list = self.get_secondary(primary)

        s_1 = secondary_list[0] # for better reading we store the secondary products and their labels
        s_2 = secondary_list[1]
        
        j_1 = s_1.label
        j_2 = s_2.label
        
        if s_1 in primary_history and s_2 in primary_history:
            return b_i * exp_margin + q_link[-1] * self.exp_return(link[-1], primary_history, q_link[:-1], link[:-1], price_combination, user)

        # all exceptions have been treated, let's now compute the expected return in the basic case
        
        
        # compute probabilities to click on the secondary given that the primary is bought
        q_1 = user.probabilities[i, j_1]
        q_2 = user.probabilities[i, j_2] * self.lambda_q

        new_link = link.append(s_2)
        new_q_link = q_link.append(q_2)

        return b_i*(exp_margin + q_1*self.exp_return(s_1, primary_history, new_q_link, new_link, price_combination, user) +
                      (1-q_1) * q_2 * self.exp_return(s_2, primary_history, q_link, link, price_combination, user) +
                      (1-q_1) * (1-q_2) * q_link[-1] * self.exp_return(link[-1], primary_history, q_link[:-1], link[:-1], price_combination, user))
    
    def aggregated_reward(self, price_combination):
        """ Method that compute the expected rewrd related to the prices compbination passed to the function"""
        # initialize final regret and index j indicating the user category
        regret = 0.
        j = 0

        for user_cat in self.users:
            # intialize the regret relative to the specific user considered and index i for the products
            i = 0
            user_regret = 0.
            
            # explore the possibility of starting at each product
            for product in self.products:
                alpha_i = user_cat.alphas[i]
                i += 1
                # exp_return compute the expected return starting from a specific product, so we have to multiply it 
                # for the probability of starting from that product (alpha_i)
                user_regret += alpha_i * self.exp_return(product, [], [0], [Product([], -1, "null", [])], price_combination, user_cat)

            # the regret is weighted to the mean probability of having a client of a specific user category
            regret += self.user_cat_prob[j] * user_regret
            j += 1
        
        return regret

    def single_reward(self, user_index, price_combination):
        """ Method to compute the expected reward for a single user category given a specific price combination for the 5 
            products """

        # select the right user
        user = self.users[user_index]

        # initialize regret and index i for the starting page
        regret = 0.
        i = 0

        for product in self.products:
            alpha_i = user.alphas[i]
            i = i+1
            # exp_return compute the expected return starting from a specific product, so we have to multiply it 
            # for the probability of starting from that product (alpha_i)
            regret += alpha_i * self.exp_return(product, [], [0], [Product([], -1, "null", [])], price_combination, user)
        
        return regret

    

    def optimal_reward(self, user_index = -1):
        """ This method explores all the possible combination with a brute force approac to determine which is the price combination
            that returns the highest expected reward.
            It returns both the optimal price combination and optimal expected regret """
            
        optimal_combination = [0, 0, 0, 0, 0]
        regret_max = 0
        regret = 0
        
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
            # compute regret for the price combination considered
            if user_index == -1:
                regret = self.aggregated_reward(price_combination)
            else :
                regret = self.single_reward(user_index, price_combination)
            
            # update if actual regret is greater than best past regret
            if regret > regret_max:
                regret_max = regret
                optimal_combination = price_combination.copy()

        return regret_max, optimal_combination
