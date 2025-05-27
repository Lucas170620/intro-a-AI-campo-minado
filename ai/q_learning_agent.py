import numpy as np
import random
from collections import defaultdict

class QLearningAgent:
    def __init__(self, learning_rate=0.2, discount_factor=0.99, epsilon=0.3):
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.min_epsilon = 0.01
        self.epsilon_decay = 0.995
        self.previous_state = None
        self.previous_action = None

    def get_state_representation(self, campo):
        """Converte o campo atual em uma representação de estado mais informativa."""
        state = []
        for i in range(len(campo)):
            for j in range(len(campo[0])):
                if campo[i][j].revelada:
                    state.append(str(campo[i][j].bombas_vizinhas) if not campo[i][j].tem_bomba else '*')
                else:
                    # Adiciona informação sobre células vizinhas reveladas
                    vizinhas_reveladas = self._contar_vizinhas_reveladas(campo, i, j)
                    state.append(f"#{vizinhas_reveladas}")
        return tuple(state)

    def _contar_vizinhas_reveladas(self, campo, linha, coluna):
        """Conta o número de células vizinhas reveladas."""
        contador = 0
        for i in range(max(0, linha - 1), min(len(campo), linha + 2)):
            for j in range(max(0, coluna - 1), min(len(campo[0]), coluna + 2)):
                if campo[i][j].revelada:
                    contador += 1
        return contador

    def get_valid_actions(self, campo):
        """Retorna ações válidas priorizando células com mais informação."""
        actions = []
        for i in range(len(campo)):
            for j in range(len(campo[0])):
                if not campo[i][j].revelada:
                    vizinhas_reveladas = self._contar_vizinhas_reveladas(campo, i, j)
                    actions.append((i, j, vizinhas_reveladas))
        
        # Ordena ações por número de vizinhas reveladas
        actions.sort(key=lambda x: x[2], reverse=True)
        return [(a[0], a[1]) for a in actions]

    def choose_action(self, campo):
        """Escolhe uma ação usando a política epsilon-greedy com decaimento."""
        state = self.get_state_representation(campo)
        valid_actions = self.get_valid_actions(campo)
        
        if not valid_actions:
            return None

        # Decai epsilon ao longo do tempo
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        if random.random() < self.epsilon:
            return random.choice(valid_actions)

        # Escolhe a ação com maior valor Q
        q_values = {action: self.q_table[state][action] for action in valid_actions}
        max_q = max(q_values.values()) if q_values else 0
        best_actions = [action for action, q in q_values.items() if q == max_q]
        
        return random.choice(best_actions) if best_actions else random.choice(valid_actions)

    def learn(self, state, action, reward, next_state):
        """Atualiza a Q-table com double learning."""
        old_q = self.q_table[state][action]
        
        # Implementa Double Q-learning
        if random.random() < 0.5:
            next_max_q = max(self.q_table[next_state].values()) if self.q_table[next_state] else 0
        else:
            next_actions = list(self.q_table[next_state].keys())
            if next_actions:
                next_action = max(next_actions, key=lambda a: self.q_table[next_state][a])
                next_max_q = self.q_table[next_state][next_action]
            else:
                next_max_q = 0
        
        new_q = old_q + self.learning_rate * (reward + self.discount_factor * next_max_q - old_q)
        self.q_table[state][action] = new_q

    def reset(self):
        """Reseta o estado do agente para uma nova partida."""
        self.previous_state = None
        self.previous_action = None