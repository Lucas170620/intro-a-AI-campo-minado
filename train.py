import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from game.campo_minado import CampoMinado
from ai.q_learning_agent import QLearningAgent
import numpy as np
import json
import signal
import sys

class CampoMinadoTrainer:
    def __init__(self, linhas, colunas, num_bombas):
        self.linhas = linhas
        self.colunas = colunas
        self.num_bombas = num_bombas
        self.agent = QLearningAgent()
        self.scores = []
        self.episodes = []
        self.episode_count = 0
        self.running = True
        self.best_score = float('-inf')
        
        # Configuração do plot
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [])
        self.ax.set_xlabel('Episódios')
        self.ax.set_ylabel('Pontuação Média (últimos 100)')
        self.ax.set_title('Evolução do Treinamento')
        
        # Configurar handler para CTRL+C
        signal.signal(signal.SIGINT, self.handle_interrupt)

    def handle_interrupt(self, signum, frame):
        print("\nInterrompendo treinamento...")
        self.running = False
        self.save_agent()

    def save_agent(self):
        # Convertendo defaultdict para dict regular com chaves e valores serializáveis
        q_table_serializable = {}
        for state, actions in self.agent.q_table.items():
            state_str = json.dumps(list(state))
            q_table_serializable[state_str] = {}
            for action, value in actions.items():
                action_str = json.dumps(list(action))
                q_table_serializable[state_str][action_str] = value

        with open('q_table.json', 'w') as f:
            json.dump(q_table_serializable, f)
        print("Q-table salva em 'q_table.json'")

    def update_plot(self):
        if len(self.scores) > 0:
            self.line.set_xdata(self.episodes)
            self.line.set_ydata(self.scores)
            self.ax.relim()
            self.ax.autoscale_view()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def train(self):
        """Treina o agente continuamente até receber sinal de parada."""
        window_size = 100
        scores_window = []
        
        print("Iniciando treinamento... (Pressione CTRL+C para parar)")
        
        while self.running:
            self.episode_count += 1
            score = self._play_episode()
            scores_window.append(score)
            
            if len(scores_window) > window_size:
                scores_window.pop(0)
            
            avg_score = np.mean(scores_window)
            self.scores.append(avg_score)
            self.episodes.append(self.episode_count)
            
            # Salva o melhor modelo
            if avg_score > self.best_score:
                self.best_score = avg_score
                self.save_agent()
            
            if self.episode_count % 10 == 0:
                print(f"Episódio {self.episode_count}, Pontuação Média: {avg_score:.2f}, Epsilon: {self.agent.epsilon:.3f}")
                self.update_plot()

    def _play_episode(self):
        """Joga um episódio completo e retorna a pontuação."""
        jogo = CampoMinado(self.linhas, self.colunas, self.num_bombas)
        score = 0
        cells_revealed = 0
        total_safe_cells = (self.linhas * self.colunas) - self.num_bombas
        
        while jogo.jogo_ativo:
            current_state = self.agent.get_state_representation(jogo.campo)
            action = self.agent.choose_action(jogo.campo)
            
            if action is None:
                break

            linha, coluna = action
            celula_anterior = jogo.campo[linha][coluna].revelada
            vizinhas_antes = self._contar_celulas_reveladas(jogo.campo)
            jogo.revelar(linha, coluna)
            vizinhas_depois = self._contar_celulas_reveladas(jogo.campo)
            
            # Sistema de recompensas melhorado
            if not jogo.jogo_ativo and jogo.campo[linha][coluna].tem_bomba:
                reward = -200  # Penalidade maior por encontrar bomba
                score -= 200
            else:
                celulas_reveladas = vizinhas_depois - vizinhas_antes
                if celulas_reveladas > 0:
                    reward = 20 * celulas_reveladas  # Recompensa por revelar múltiplas células
                    score += 20 * celulas_reveladas
                    cells_revealed += celulas_reveladas
                else:
                    reward = -5  # Penalidade por não revelar nada novo
                    score -= 5
                
                # Bônus por progresso
                if cells_revealed == total_safe_cells:
                    reward += 500  # Bônus grande por vencer
                    score += 500
            
            next_state = self.agent.get_state_representation(jogo.campo)
            self.agent.learn(current_state, action, reward, next_state)
        
        return score

    def _contar_celulas_reveladas(self, campo):
        """Conta o número total de células reveladas."""
        return sum(1 for linha in campo for celula in linha if celula.revelada)

if __name__ == "__main__":
    trainer = CampoMinadoTrainer(8, 8, 10)
    trainer.train()