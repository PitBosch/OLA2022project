from Environment import *


class Greedy_optimizer: 

    def __init__(self, env: Environment):
        self.env = env

    def greedy_iteration(self, price_combination: list[int], actual_reward, conversion_rates=None, alphas_ratio=None, n_prod=None,
                            graph_weights=None, user_index=None, group_list=None, feat_prob_mat=None):
        """ Method for a single iteration of the greedy algorithm. Try to increase the prices for the products
            one at the time and decide, if it is possible, to increase the price for the product with the 
            highest positive increase"""

        return_dict = {"updated": False,
                       "combination": price_combination,
                       "expected_reward": actual_reward}
        # initialize a list of new rewards for the price combination
        new_rewards = [0 for x in self.env.products]
        # try for each product to increase the price, if possible
        for i in range(len(self.env.products)):
            # Compute the value of new_rewards for product i IF we are not considering the highest price already
            if price_combination[i] < 3:
                new_combination = price_combination.copy() 
                new_combination[i] += 1
                new_rewards[i] = self.env.expected_reward(new_combination, conversion_rates, alphas_ratio, n_prod,
                                                            graph_weights, user_index, group_list, feat_prob_mat)
        diff = np.array(new_rewards) - np.array(actual_reward)
        if max(diff) > 0:
            i_opt = np.argmax(new_rewards)
            return_dict["updated"] = True
            return_dict["combination"][i_opt] += 1
            return_dict["expected_reward"] = new_rewards[i_opt]
        return return_dict


    def run(self, conversion_rates=None, alphas_ratio=None, n_prod=None, graph_weights=None, user_index=None, group_list=None, feat_prob_mat=None):
        """ Method for the complete run of the greedy algorithm. The starting point is the combination with all the lowest prices"""
        updated = True
        # Initially we consider all the lowest prices *Recall that prices are stored in ascending order
        price_combination = [0 for x in self.env.products]
        # Initialize the optimal reward with the value linked to the expected reward for lowest prices
        optimal_reward = self.env.expected_reward(price_combination, conversion_rates, alphas_ratio, n_prod, graph_weights, user_index, group_list, feat_prob_mat)
        # Create a list of list to store the evolution of the algorithm
        # history = []
        while updated:
            # history.append(price_combination.copy())
            iter_result = self.greedy_iteration(price_combination,  optimal_reward, conversion_rates, alphas_ratio, n_prod,
                                                graph_weights, user_index, group_list, feat_prob_mat)
            updated = iter_result["updated"]
            optimal_reward = iter_result["expected_reward"]
        iter_result.pop("updated")
        
        return_dict = {
            "expected_reward": iter_result["expected_reward"],
            "combination": iter_result['combination']
        }

        return return_dict

        

