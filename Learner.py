from Environment import Environment
from Greedy_optimizer import Greedy_optimizer
import numpy as np
import copy

class Learner:
    def __init__(self, env: Environment):
        # Real environment
        self.env = env
        # Greedy optimizer to decide the price combination each day
        self.Greedy_opt = Greedy_optimizer(self.env)
        # Optimal theoretical reward
        self.opt_reward = env.optimal_reward()[0]
        # Initialize history of theoretical rewards 
        self.reward_history = []
        # History of prices combination chosen
        self.price_comb_history = []
