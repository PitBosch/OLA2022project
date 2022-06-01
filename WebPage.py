from UserCat import *
from Product import *


class WebPage:
    """Class characterized by a user and a product: contains all the steps required to simulate the behaviour of a user in
       a particular page. Through a recursive structure it allows to simulate the evolution of a visit of the user to our website,
       returning the cumulative margin generated by such event. """

    def __init__(self, product: Product, user: UserCat):
        self.product = product
        self.user = user

    def interact(self, links: dict[Product], products, lambda_prob, price_pos):

        if self.user.buy(self.product.get_daily_price(price_pos)):
            self.user.margin += self.product.get_daily_margin(price_pos) * self.user.get_prod_number()
            """The margin of the user is updated recursively every time he proceeds with a purchase, considering the margin of 
               that product and the number of items bought by the user (random number)"""
            #devo accedere in qualche modo all'array di prodotti
            first_secondary = products[links.get(self.product.name)[0]]  # TODO: i collegamenti tra i diversi oggetti sono fissi o tipici per classe di utente?
            second_secondary = products[links.get(self.product.name)[1]]

            """To simulate the random behaviour of the user we sample from a random distribution and we use it to evaluate whether
               an event has occurred or not. """
            first_click = (np.random.uniform() < self.user.probabilities[self.product.label, first_secondary.label])
            second_click = np.random.uniform() < lambda_prob * self.user.probabilities[self.product.label, second_secondary.label]

            if first_click and first_secondary not in self.user.visited_products:
                self.user.visited_products.append(first_secondary)
                new_WebPage = WebPage(first_secondary, self.user)
                new_WebPage.interact(links, products, lambda_prob, price_pos)

            if second_click and second_secondary not in self.user.visited_products:
                self.user.visited_products.append(second_secondary)
                second_new_WebPage = WebPage(second_secondary, self.user)
                second_new_WebPage.interact(links, products, lambda_prob, price_pos)

        """Since python passes by reference the objects, the margin of the user is updated recursively. At the end of the
           recursion, once all the paths in our website have been completed we remain with the total margin accumulated by the
           visit. """

        return self.user.margin
