class ucb_learner:
    def __init__(self, n_products, n_arms):
        self.t = 0
        self.n_products = n_products
        self.n_arms = n_arms
        self.rewards = []
        self.rewards_per_arm = [[[] for i in range(n_arms)] for j in range(n_products)]
        self.pulled = []

    def update(self, arms_pulled, rewards):
        self.t += 1
        self.rewards.append(rewards)
        for product_idx in range(self.n_products):
            self.rewards_per_arm[product_idx][arms_pulled[product_idx]].append(rewards[product_idx])
        self.pulled.append(arms_pulled)
