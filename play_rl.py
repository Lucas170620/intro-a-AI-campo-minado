# play_rl.py

import torch
import os
import matplotlib.pyplot as plt
from reinforcement_learning.agent import MinesweeperAgent
from game.campo_minado import CampoMinado

LINHAS = 4
COLUNAS = 4
N_BOMBAS = 2
MODEL_PATH = 'reinforcement_learning/models/model_10000_partidas_4_x_4_2M.pth'

RESULTADOS_DIR = 'reinforcement_learning/resultados_finais'
os.makedirs(RESULTADOS_DIR, exist_ok=True)

def carregar_modelo(agent, path):
    agent.model.load_state_dict(torch.load(path, map_location='cpu'))
    agent.model.eval()

def print_campo(campo):
    print("\n   " + " ".join(str(i) for i in range(COLUNAS)))
    for idx, linha in enumerate(campo.campo):
        print(f"{idx:2} " + " ".join(str(celula) for celula in linha))

def main():
    agent = MinesweeperAgent(LINHAS, COLUNAS, N_BOMBAS)
    carregar_modelo(agent, MODEL_PATH)
    agent.epsilon = 0  # Garante modo exploit puro

    vitorias = 0
    derrotas = 0
    partidas = 1000

    for partida in range(1, partidas+1):
        campo = CampoMinado(LINHAS, COLUNAS, N_BOMBAS)
        state = agent.get_state(campo)
        log = []
        passos = 0

        resultado = None  # 'VITÓRIA' ou 'DERROTA'

        while campo.jogo_ativo:
            action = agent.get_action(state, campo)
            move = action.argmax()
            linha, coluna = move // COLUNAS, move % COLUNAS
            celula = campo.campo[linha][coluna]

            prev_revelada = celula.revelada
            campo.revelar(linha, coluna)
            state = agent.get_state(campo)
            passos += 1

            if partida == 1:
                log.append(f"Passo {passos}: Jogando em ({linha}, {coluna}) | Revelada antes? {prev_revelada} | Bomba? {celula.tem_bomba}")
                print_campo(campo)
                print('-'*35)

            if celula.tem_bomba:
                resultado = "DERROTA"
                derrotas += 1
                if partida == 1:
                    print("\n".join(log))
                    print(f"Resultado da 1ª partida: {resultado}\n")
                break
            elif campo._verificar_vitoria():
                resultado = "VITÓRIA"
                vitorias += 1
                if partida == 1:
                    print("\n".join(log))
                    print(f"Resultado da 1ª partida: {resultado}\n")
                break

    print(f"\nTotal de partidas: {partidas}")
    print(f"Vitórias: {vitorias}")
    print(f"Derrotas: {derrotas}")

    # Gráfico de pizza (Pie Chart)
    labels = ['Vitórias', 'Derrotas']
    values = [vitorias, derrotas]
    colors = ['#4CAF50', '#F44336']

    plt.figure(figsize=(6, 6))
    plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, explode=(0.07, 0))
    plt.title(f'Resultado IA RL - {partidas} partidas')
    plt.tight_layout()
    output_path = os.path.join(RESULTADOS_DIR, f'resultado_{LINHAS}x{COLUNAS}_{N_BOMBAS}M_{partidas}partidas.png')
    plt.savefig(output_path)
    print(f"Gráfico salvo em: {output_path}")

if __name__ == '__main__':
    main()
