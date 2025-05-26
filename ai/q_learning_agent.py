import numpy as np
import random
from collections import defaultdict

class QLearningAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.previous_state = None
        self.previous_action = None

    def get_state_representation(self, campo):
        """Converte o campo atual em uma representação de estado."""
        state = []
        for i in range(len(campo)):
            for j in range(len(campo[0])):
                if campo[i][j].revelada:
                    state.append(str(campo[i][j].bombas_vizinhas) if not campo[i][j].tem_bomba else '*')
                else:
                    state.append('#')
        return tuple(state)

    def get_valid_actions(self, campo):
        """Retorna todas as ações válidas (células não reveladas)."""
        actions = []
        for i in range(len(campo)):
            for j in range(len(campo[0])):
                if not campo[i][j].revelada:
                    actions.append((i, j))
        return actions

    def choose_action(self, campo):
        """Escolhe uma ação usando a política epsilon-greedy."""
        state = self.get_state_representation(campo)
        valid_actions = self.get_valid_actions(campo)
        
        if not valid_actions:
            return None

        if random.random() < self.epsilon:
            return random.choice(valid_actions)

        # Escolhe a ação com maior valor Q
        q_values = {action: self.q_table[state][action] for action in valid_actions}
        max_q = max(q_values.values()) if q_values else 0
        best_actions = [action for action, q in q_values.items() if q == max_q]
        
        return random.choice(best_actions) if best_actions else random.choice(valid_actions)

    def learn(self, state, action, reward, next_state):
        """Atualiza a Q-table baseado na experiência."""
        old_q = self.q_table[state][action]
        next_max_q = max(self.q_table[next_state].values()) if self.q_table[next_state] else 0
        
        new_q = old_q + self.learning_rate * (reward + self.discount_factor * next_max_q - old_q)
        self.q_table[state][action] = new_q

    def reset(self):
        """Reseta o estado do agente para uma nova partida."""
        self.previous_state = None
        self.previous_action = None