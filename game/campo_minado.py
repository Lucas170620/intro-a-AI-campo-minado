import random
from game.celula import Celula

class CampoMinado:
    def __init__(self, linhas, colunas, num_bombas):
        self.linhas = linhas
        self.colunas = colunas
        self.num_bombas = num_bombas
        self.campo = [[Celula() for _ in range(colunas)] for _ in range(linhas)]
        self.jogo_ativo = True
        self._plantar_bombas()
        self._calcular_vizinhas()

    def _plantar_bombas(self):
        bombas_plantadas = 0
        while bombas_plantadas < self.num_bombas:
            linha = random.randint(0, self.linhas - 1)
            coluna = random.randint(0, self.colunas - 1)
            celula = self.campo[linha][coluna]
            if not celula.tem_bomba:
                celula.tem_bomba = True
                bombas_plantadas += 1

    def _calcular_vizinhas(self):
        for l in range(self.linhas):
            for c in range(self.colunas):
                if not self.campo[l][c].tem_bomba:
                    self.campo[l][c].bombas_vizinhas = self._contar_bombas_vizinhas(l, c)

    def _contar_bombas_vizinhas(self, linha, coluna):
        contador = 0
        for i in range(max(0, linha - 1), min(self.linhas, linha + 2)):
            for j in range(max(0, coluna - 1), min(self.colunas, coluna + 2)):
                if self.campo[i][j].tem_bomba:
                    contador += 1
        return contador

    def revelar(self, linha, coluna):
        if not (0 <= linha < self.linhas and 0 <= coluna < self.colunas):
            print("Coordenadas inválidas.")
            return

        celula = self.campo[linha][coluna]

        if celula.revelada:
            print("Célula já revelada!")
            return

        celula.revelada = True

        if celula.tem_bomba:
            self.jogo_ativo = False
            print("BOOM! Você perdeu.")
            self._revelar_todas()
            return

        if celula.bombas_vizinhas == 0:
            self._revelar_vizinhas(linha, coluna)

        if self._verificar_vitoria():
            self.jogo_ativo = False
            self._revelar_todas()
            print("Parabéns! Você venceu!")

    def _revelar_vizinhas(self, linha, coluna):
        for i in range(max(0, linha - 1), min(self.linhas, linha + 2)):
            for j in range(max(0, coluna - 1), min(self.colunas, coluna + 2)):
                vizinha = self.campo[i][j]
                if not vizinha.revelada and not vizinha.tem_bomba:
                    vizinha.revelada = True
                    if vizinha.bombas_vizinhas == 0:
                        self._revelar_vizinhas(i, j)

    def _revelar_todas(self):
        for linha in self.campo:
            for celula in linha:
                celula.revelada = True
        self.mostrar_campo()

    def _verificar_vitoria(self):
        for linha in self.campo:
            for celula in linha:
                if not celula.tem_bomba and not celula.revelada:
                    return False
        return True

    def mostrar_campo(self):
        print("\n   " + " ".join(str(i) for i in range(self.colunas)))
        for idx, linha in enumerate(self.campo):
            print(f"{idx:2} " + " ".join(str(celula) for celula in linha))