import time
import torch
from game.campo_minado import CampoMinado
from agent import MinesweeperAgent

def play_game(agent, delay=1.0):
    """Joga uma partida usando o agente treinado."""
    BOARD_SIZE = 8
    NUM_BOMBS = 10
    
    jogo = CampoMinado(BOARD_SIZE, BOARD_SIZE, NUM_BOMBS)
    moves = 0
    
    while jogo.jogo_ativo:
        print(f"\nMovimento {moves + 1}")
        jogo.mostrar_campo()
        
        # Obtém o estado e a ação do agente
        state = agent.get_state(jogo)
        action = agent.get_action(state)
        move = torch.argmax(torch.tensor(action)).item()
        
        # Converte o índice em coordenadas
        linha, coluna = move // BOARD_SIZE, move % BOARD_SIZE
        
        print(f"Agente escolheu: linha {linha}, coluna {coluna}")
        jogo.revelar(linha, coluna)
        moves += 1
        
        time.sleep(delay)  # Pausa para melhor visualização
    
    print("\nJogo finalizado!")
    jogo.mostrar_campo()
    return jogo.jogo_ativo

if __name__ == "__main__":
    print("Carregando agente treinado...")
    
    # Inicializa o agente com as mesmas dimensões usadas no treinamento
    BOARD_SIZE = 8
    state_size = BOARD_SIZE * BOARD_SIZE
    action_size = state_size
    
    agent = MinesweeperAgent(state_size, action_size)
    agent.load_model()  # Carrega o modelo treinado
    
    # Define epsilon para 0 durante o jogo (sem exploração)
    agent.epsilon = 0
    
    while True:
        input("\nPressione Enter para jogar uma nova partida (ou CTRL+C para sair)...")
        won = play_game(agent)
        if won:
            print("O agente venceu!")
        else:
            print("O agente perdeu!")