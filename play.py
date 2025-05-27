from game.campo_minado import CampoMinado
from ai.q_learning_agent import QLearningAgent
import json
import time

def load_agent():
    agent = QLearningAgent()
    try:
        with open('q_table.json', 'r') as f:
            q_table_dict = json.load(f)
            # Convertendo as strings de volta para tuplas
            for state_str, actions in q_table_dict.items():
                state = eval(state_str)  # Converte string de volta para tupla
                for action_str, value in actions.items():
                    action = eval(action_str)  # Converte string de volta para tupla
                    agent.q_table[state][action] = value
        print("Q-table carregada com sucesso!")
    except FileNotFoundError:
        print("Arquivo q_table.json não encontrado. O agente começará sem treinamento.")
    return agent

def play_game(agent, delay=1.0):
    """Joga uma partida usando o agente treinado."""
    jogo = CampoMinado(8, 8, 10)
    moves = 0
    
    while jogo.jogo_ativo:
        print(f"\nMovimento {moves + 1}")
        jogo.mostrar_campo()
        
        action = agent.choose_action(jogo.campo)
        if action is None:
            break
            
        linha, coluna = action
        print(f"Agente escolheu: linha {linha}, coluna {coluna}")
        jogo.revelar(linha, coluna)
        moves += 1
        
        time.sleep(delay)  # Pausa para melhor visualização
    
    print("\nJogo finalizado!")
    jogo.mostrar_campo()
    return jogo.jogo_ativo

if __name__ == "__main__":
    print("Carregando agente treinado...")
    agent = load_agent()
    
    while True:
        input("\nPressione Enter para jogar uma nova partida (ou CTRL+C para sair)...")
        won = play_game(agent)
        if won:
            print("O agente venceu!")
        else:
            print("O agente perdeu!")