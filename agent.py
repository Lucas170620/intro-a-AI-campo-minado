import torch
import random
import numpy as np
from collections import deque
from model import Linear_QNet, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class MinesweeperAgent:
    def __init__(self, linhas, colunas, n_bombas):
        self.linhas = linhas
        self.colunas = colunas
        self.n_bombas = n_bombas
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        # Estado = 2 canais: mascara (0=oculto,1=aberto), valores (0-8)
        # Entrada = linhas*colunas*2
        # Saida = linhas*colunas (probabilidade de clicar em cada célula)
        self.model = Linear_QNet(linhas * colunas * 2, 128, linhas * colunas)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, campo):
        # Canal 1: mascara (0=oculto, 1=aberto)
        # Canal 2: valores (0-8 para abertas, -1 para ocultas)
        mascara = []
        valores = []
        for l in range(self.linhas):
            for c in range(self.colunas):
                cel = campo.campo[l][c]
                if cel.revelada:
                    mascara.append(1)
                    if cel.tem_bomba:
                        valores.append(9)  # bomba aberta
                    else:
                        valores.append(cel.bombas_vizinhas)
                else:
                    mascara.append(0)
                    valores.append(-1)
        # Estado = [mascara ...] + [valores ...]
        return np.array(mascara + valores, dtype=float)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state, campo):
        self.epsilon = max(2, 40 - self.n_games // 10)
        final_move = np.zeros(self.linhas * self.colunas)
        # Só pode clicar em células ocultas
        choices = [(i, j) for i in range(self.linhas) for j in range(self.colunas) if not campo.campo[i][j].revelada]
        if not choices:
            return final_move
        if random.randint(0, 100) < self.epsilon:
            # Explora: escolhe célula oculta aleatória
            l, c = random.choice(choices)
            pos = l * self.colunas + c
            final_move[pos] = 1
        else:
            # Exploita: rede neural
            state0 = torch.tensor(state, dtype=torch.float)
            pred = self.model(state0)
            # zera as células já abertas
            for l in range(self.linhas):
                for c in range(self.colunas):
                    if campo.campo[l][c].revelada:
                        pred[l*self.colunas+c] = -float('inf')
            move = torch.argmax(pred).item()
            # se a célula já está aberta por algum motivo, pega uma aleatória válida
            l, c = move // self.colunas, move % self.colunas
            if (l, c) not in choices:
                l, c = random.choice(choices)
                move = l * self.colunas + c
            final_move[move] = 1
        return final_move
