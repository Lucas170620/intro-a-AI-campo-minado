from reinforcement_learning.agent import MinesweeperAgent
from game.campo_minado import CampoMinado
from reinforcement_learning.helper import plot
import numpy as np

def avaliar_jogada(campo, linha, coluna, prev_revelada):
    celula = campo.campo[linha][coluna]
    jogada_invalida = prev_revelada
    bomba = celula.tem_bomba
    vitoria = False
    fim_de_jogo = False

    if jogada_invalida:
        fim_de_jogo = False
    elif bomba:
        fim_de_jogo = True
    elif not campo.jogo_ativo:
        fim_de_jogo = True
        vitoria = True

    return jogada_invalida, bomba, vitoria, fim_de_jogo

def calcular_recompensa(celula, vitoria=False, derrota=False, prev_revelada=False):
    """
    Sistema de recompensas:
    1. Abertura de célula segura: +0.1
    2. Abertura de célula com bomba: -1.0
    3. Célula já revelada: -0.5
    4. Vitória: +5.0
    5. Penalização por arriscada: -0.1*n_bombas_vizinho
    6. Bônus por célula com 0 bombas vizinhas: +0.2
    """
    reward = 0.0

    if prev_revelada:
        reward = -0.5
    elif celula.tem_bomba:
        reward = -1.0
    else:
        # Aberta célula segura
        reward = 0.1
        reward -= 0.1 * celula.bombas_vizinhas
        if celula.bombas_vizinhas == 0:
            reward += 0.2

    if vitoria:
        reward += 5.0
    if derrota:
        reward -= 5.0  # Se preferir, pode aumentar para -10, etc.

    return reward

def train():
    linhas, colunas, n_bombas = 10,10,10
    scores = []
    mean_scores = []
    victories = []
    winrates = []
    total_score = 0
    total_victory = 0
    record = 0
    agent = MinesweeperAgent(linhas, colunas, n_bombas)
    N_EPISODES = 100000


    for game in range(N_EPISODES):
        campo = CampoMinado(linhas, colunas, n_bombas)
        score = 0
        done = False
        state_old = agent.get_state(campo)
        vitoria = False

        while campo.jogo_ativo:
            action = agent.get_action(state_old, campo)
            move = action.argmax()
            linha, coluna = move // colunas, move % colunas

            celula = campo.campo[linha][coluna]
            prev_revelada = celula.revelada

            campo.revelar(linha, coluna)
            state_new = agent.get_state(campo)

            jogada_invalida, bomba, vitoria, fim_de_jogo = avaliar_jogada(campo, linha, coluna, prev_revelada)
            derrota = bomba and fim_de_jogo

            # Passando prev_revelada para checagem de penalidade correta
            reward = calcular_recompensa(celula, vitoria, derrota, prev_revelada=prev_revelada)

            if not prev_revelada and not bomba:
                score += 1

            agent.train_short_memory(state_old, action, reward, state_new, fim_de_jogo)
            agent.remember(state_old, action, reward, state_new, fim_de_jogo)
            state_old = state_new

            if fim_de_jogo:
                break

        agent.n_games += 1
        agent.train_long_memory()

        if score > record:
            record = score
            model_name = f"model_{N_EPISODES}_partidas_{linhas}_x_{colunas}_{n_bombas}M_final_version.pth"
            print(model_name)
            agent.model.save(file_name=model_name)
        scores.append(score)
        victories.append(int(vitoria))
        total_score += score
        total_victory += int(vitoria)
        mean_scores.append(total_score / (game+1))
        winrate = 100 * total_victory / (game+1)
        winrates.append(winrate)

        # Média móvel dos scores (500)
        if len(scores) >= 500:
            mov_avg_scores = [np.mean(scores[max(0, i-499):i+1]) for i in range(len(scores))]
        else:
            mov_avg_scores = mean_scores

        # Média móvel do winrate (200)
        if len(victories) >= 200:
            mov_avg_winrates = [100 * np.mean(victories[max(0, i-199):i+1]) for i in range(len(victories))]
        else:
            mov_avg_winrates = [100 * np.mean(victories[:i+1]) for i in range(len(victories))]

        # Plota normal durante o treino
        # plot(scores, mean_scores, winrates, mov_avg_scores, mov_avg_winrates, save_final=False)
        print(f'Game {game+1} | Score: {score} | Record: {record} | Mean: {mean_scores[-1]:.2f} | MovingAvgWinrate(200): {mov_avg_winrates[-1]:.2f}% | WinRate: {winrate:.1f}%')

    # Ao final do treinamento, salva a imagem
    plot(scores, mean_scores, winrates, mov_avg_scores, mov_avg_winrates, save_final=True,
         plays=N_EPISODES, num_mines=n_bombas, tab_len=linhas)
    print("Gráfico salvo como campo_minado_rl_training.png")

if __name__ == '__main__':
    train()
