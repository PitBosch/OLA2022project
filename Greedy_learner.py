from Environment import *

class Greedy_learner: 

    def __init__(self, env: Environment, n: int) -> None :
        self.env = env
        self.daily_user_n = n

    def greedy_iteration(self, price_combination: list[int], actual_margin) -> list[int]:
        """ Method for a single iteration of the greedy algorithm. Try to increase the prices for the products
            one at the time and decide, if it is possible, to increase the price for the product with the 
            highest positive increase"""

        return_dict = { "updated" : False,
                        "combination" : price_combination,
                        "margin" : actual_margin }

        # initialize a list of new margins for the price combination
        new_margins = [0 for x in self.env.products]

        for i in range(len(self.env.products)) :
            # Compute the value of new_margins for product i IF we are not considering the highest price already
            if price_combination[i] < 3:
                new_combination = price_combination.copy() 
                new_combination[i] += 1
                disaggregated_margins = self.env.simulate_day(self.daily_user_n, new_combination)
                new_margins[i] = sum(disaggregated_margins)
        print(new_margins)
        print(actual_margin)
        diff = np.array(new_margins) - np.array(actual_margin)
        if max(diff) > 0:
            i_opt = np.argmax(new_margins)
            return_dict["updated"] = True
            return_dict["combination"][i_opt] += 1
            return_dict["margin"] = new_margins[i_opt]
        
        return return_dict

    def run(self):
        """ Method for the complete run of the greedy algorithm.
            The starting point is the combination with all the lowest prices"""    
        
        updated = True
        # Initially we consider all the lowest prices *Recall that prices are stored in ascending order
        price_combination = [0 for x in self.env.products]
        # Compute the margin for the initial price combination
        margin = sum(self.env.simulate_day(self.daily_user_n, price_combination))
        # Create a list of list to store the evolution of the algorithm
        history = []

        while(updated):
            history.append(price_combination.copy())
            iter_result = self.greedy_iteration(price_combination,  margin)
            updated = iter_result["updated"]
            margin = iter_result["margin"]



        iter_result.pop("updated")
        iter_result["history"] = history

        return iter_result

        

