from User import *
from Product import *


class WebPage:
    def __init__(self, product: Product, user: User, margin):
        self.product = product
        self.user = user
        self.margin = margin

    def interact(self, links: dict[Product], lambda_prob):
        if self.user.buy(self.product.price):
            self.margin += self.product.price * self.user.get_prod_number()
            self.user.probabilities.loc[:, self.product.label] = 0
            first_secondary = links.get(self.product)[0]
            second_secondary = links.get(self.product)[1]
            if np.random.uniform() < self.user.probabilities.loc[self.product.label, first_secondary.label]:
                new_WebPage = WebPage(first_secondary, self.user, self.margin)
                new_WebPage.interact(links, lambda_prob)

            if np.random.uniform() < lambda_prob * self.user.probabilities.loc[self.product.label, second_secondary.label]:
                second_new_WebPage = WebPage(second_secondary, self.user, self.margin)
                second_new_WebPage.interact(links, lambda_prob)

        return self.margin
