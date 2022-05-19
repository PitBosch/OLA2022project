import string


class Product:
    def __init__(self, price: float, label: int, name: string):
        self.price = price
        self.label = label
        self.name = name
