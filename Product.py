import string


class Product:


    """ Data structure containing all the necessary information about each product"""
    def __init__(self, prices: list[float], label: int, name: string, margins: list[float]):
        # List containing all the possible price values for the product
        self.prices = prices
        # List containing all the possible margins (margin[i] = price[i]- prod_cost) values for the product
        self.margins = margins
        # Position of the product in the product list of the environment
        self.label = label
        self.name = name


    def get_daily_price(self, pos):
        return self.prices[pos]


    def get_daily_margin(self, pos):
        return self.margins[pos]
