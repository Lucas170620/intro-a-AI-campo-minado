from game.campo_minado import CampoMinado
from ai.q_learning_agent import QLearningAgent

class CampoMinadoTrainer:
    def __init__(self, linhas, colunas, num_bombas):
        self.linhas = linhas
        self.colunas = colunas
        self.num_bombas = num_bombas
        self.agent = QLearningAgent()

    def train(self, num_episodes):
        """Treina o agente por um número específico de episódios."""
        for episode in range(num_episodes):
            jogo = CampoMinado(self.linhas, self.colunas, self.num_bombas)
            self._play_episode(jogo)
            
            if (episode + 1) % 100 == 0:
                print(f"Episódio {episode + 1} completado")

    def _play_episode(self, jogo):
        """Joga um episódio completo e atualiza o agente."""
        while jogo.jogo_ativo:
            current_state = self.agent.get_state_representation(jogo.campo)
            action = self.agent.choose_action(jogo.campo)
            
            if action is None:
                break

            # Executa a ação e obtém a recompensa
            linha, coluna = action
            celula_anterior = jogo.campo[linha][coluna].revelada
            jogo.revelar(linha, coluna)
            
            # Calcula a recompensa
            if not jogo.jogo_ativo and jogo.campo[linha][coluna].tem_bomba:
                reward = -100  # Penalidade por encontrar bomba
            elif not celula_anterior and jogo.campo[linha][coluna].revelada:
                reward = 10  # Recompensa por revelar célula segura
            else:
                reward = -1  # Pequena penalidade por ações redundantes
            
            # Aprende com a experiência
            next_state = self.agent.get_state_representation(jogo.campo)
            self.agent.learn(current_state, action, reward, next_state)

    def play_game(self, show_steps=True):
        """Joga uma partida usando o agente treinado."""
        jogo = CampoMinado(self.linhas, self.colunas, self.num_bombas)
        moves = 0
        
        while jogo.jogo_ativo:
            if show_steps:
                print(f"\nMovimento {moves + 1}")
                jogo.mostrar_campo()
            
            action = self.agent.choose_action(jogo.campo)
            if action is None:
                break
                
            linha, coluna = action
            jogo.revelar(linha, coluna)
            moves += 1
        
        print("\nJogo finalizado!")
        jogo.mostrar_campo()
        return jogo.jogo_ativo  # Retorna True se venceu, False se perdeu