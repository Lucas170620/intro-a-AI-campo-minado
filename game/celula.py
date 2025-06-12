class Celula:
    def __init__(self):
        self.tem_bomba = False
        self.revelada = False
        self.bombas_vizinhas = 0

    def __str__(self):
        if not self.revelada:
            return "#"
        elif self.tem_bomba:
            return "*"
        elif self.bombas_vizinhas > 0:
            return str(self.bombas_vizinhas)
        else:
            return " "
