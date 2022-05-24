from UserCat import *
from Product import *


class WebPage:
    """Class characterized by a user and a product: contains all the steps required to simulate the behaviour of a user in
       a particular page. Through a recursive structure it allows to simulate the evolution of a visit of the user to our website,
       returning the cumulative margin generated by such event. """

    def __init__(self, product: Product, user: UserCat):
        self.product = product
        self.user = user

    def interact(self, links: dict[Product], lambda_prob, price_pos):

        if self.user.buy(self.product.get_daily_price(price_pos)):
            self.user.margin += self.product.get_daily_margin(price_pos) * self.user.get_prod_number()
            """The margin of the user is updated recursively every time he proceeds with a purchase, considering the margin of 
               that product and the number of items bought by the user (random number)"""

            first_secondary = links.get(self.product)[0]  # TODO: i collegamenti tra i diversi oggetti sono fissi o tipici per classe di utente?
            second_secondary = links.get(self.product)[1]

            """To simulate the random behaviour of the user we sample from a random distribution and we use it to evaluate whether
               an event has occurred or not. """
            first_click = np.random.uniform() < self.user.probabilities.loc[self.product.label, first_secondary.label]
            second_click = np.random.uniform() < lambda_prob * self.user.probabilities.loc[self.product.label, second_secondary.label]

            # First case-> the user clicks on both the available links
            if first_click and second_click:

                """When a product appears as primary for a user, we should set all the probabilities to go back to it to 0
                   since after a first look the user will never open again that product. In this case, since the paths are 
                   considered independent, the information that the two products are being displayed as primary in a webpage
                   is immediately shared. """
                self.user.probabilities.loc[:, [first_secondary.label, second_secondary.label]] = 0
                # TODO: al momento questo aggiornamento cambia per tutta la classe di utenti le probabilità, trovare una fix che
                # TODO: resetti le probabilità al termine della visita di un utente.
                new_WebPage = WebPage(first_secondary, self.user)
                new_WebPage.interact(links, lambda_prob, price_pos)

                second_new_WebPage = WebPage(second_secondary, self.user)
                second_new_WebPage.interact(links, lambda_prob, price_pos)

            # Second case -> only the first link is clicked
            elif first_click:
                self.user.probabilities.loc[:, first_secondary.label] = 0
                new_WebPage = WebPage(first_secondary, self.user)
                new_WebPage.interact(links, lambda_prob, price_pos)

            # Third case -> only the second link is clicked
            elif second_click:
                self.user.probabilities.loc[:, second_secondary.label] = 0
                second_new_WebPage = WebPage(second_secondary, self.user)
                second_new_WebPage.interact(links, lambda_prob, price_pos)

        """Since python passes by reference the objects, the margin of the user is updated recursively. At the end of the
           recursion, once all the paths in our website have been completed we remain with the total margin accumulated by the
           visit. """

        return self.user.margin
