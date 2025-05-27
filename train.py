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
        # Convertendo defaultdict para dict regular
        q_table = {str(state): dict(actions) for state, actions in self.agent.q_table.items()}
        with open('q_table.json', 'w') as f:
            json.dump(q_table, f)
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
            
            if self.episode_count % 10 == 0:
                print(f"Episódio {self.episode_count}, Pontuação Média: {avg_score:.2f}")
                self.update_plot()

    def _play_episode(self):
        """Joga um episódio completo e retorna a pontuação."""
        jogo = CampoMinado(self.linhas, self.colunas, self.num_bombas)
        score = 0
        
        while jogo.jogo_ativo:
            current_state = self.agent.get_state_representation(jogo.campo)
            action = self.agent.choose_action(jogo.campo)
            
            if action is None:
                break

            linha, coluna = action
            celula_anterior = jogo.campo[linha][coluna].revelada
            jogo.revelar(linha, coluna)
            
            if not jogo.jogo_ativo and jogo.campo[linha][coluna].tem_bomba:
                reward = -100
                score -= 100
            elif not celula_anterior and jogo.campo[linha][coluna].revelada:
                reward = 10
                score += 10
            else:
                reward = -1
                score -= 1
            
            next_state = self.agent.get_state_representation(jogo.campo)
            self.agent.learn(current_state, action, reward, next_state)
        
        return score

if __name__ == "__main__":
    trainer = CampoMinadoTrainer(8, 8, 10)
    trainer.train()