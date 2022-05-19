from WebPage import *


class Environment:
    def __init__(self, user: User, products: [Product], n_arms):
        self.user = user
        self.n_arms = n_arms
        self.products = products
        self.margin = 0

    # def round(self, pulled_arm):
    #     reward = np.random.binomial(1, self.probabilities[pulled_arm])
    #     return reward

    def execute(self, links, lambda_prob):
        page_kind = self.user.start_event()
        if page_kind != 0:
            wp = WebPage(self.products[page_kind], self.user, self.margin)
            wp.interact(links, lambda_prob)

