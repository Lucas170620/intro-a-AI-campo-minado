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

def detectar_padroes_campo(campo):
    linhas = campo.linhas
    colunas = campo.colunas
    resultado = {}

    # Padrão 1: Sequência horizontal 1-1-1 na borda
    for l in range(linhas):
        for c in range(colunas-2):
            if (c == 0 or c+2 == colunas-1):
                if all(campo.campo[l][c2].revelada and campo.campo[l][c2].bombas_vizinhas == 1 for c2 in range(c, c+3)):
                    # Borda esquerda
                    if c == 0 and c+3 < colunas and not campo.campo[l][c+3].revelada:
                        resultado[(l, c+3)] = 'safe_check'
                    # Borda direita
                    if c+2 == colunas-1 and c-1 >= 0 and not campo.campo[l][c-1].revelada:
                        resultado[(l, c-1)] = 'safe_check'
                    if not campo.campo[l][c].revelada:
                        resultado[(l, c)] = 'unsafe_bandeira'
                    if not campo.campo[l][c+2].revelada:
                        resultado[(l, c+2)] = 'unsafe_bandeira'

    # Padrão 2: Sequência vertical 1-1-1 na borda
    for c in range(colunas):
        for l in range(linhas-2):
            if (l == 0 or l+2 == linhas-1):
                if all(campo.campo[l2][c].revelada and campo.campo[l2][c].bombas_vizinhas == 1 for l2 in range(l, l+3)):
                    if l == 0 and l+3 < linhas and not campo.campo[l+3][c].revelada:
                        resultado[(l+3, c)] = 'safe_check'
                    if l+2 == linhas-1 and l-1 >= 0 and not campo.campo[l-1][c].revelada:
                        resultado[(l-1, c)] = 'safe_check'
                    if not campo.campo[l][c].revelada:
                        resultado[(l, c)] = 'unsafe_bandeira'
                    if not campo.campo[l+2][c].revelada:
                        resultado[(l+2, c)] = 'unsafe_bandeira'

    # Padrão 3: 1-1-1-1 horizontal (bombas nas pontas)
    for l in range(linhas):
        for c in range(colunas-3):
            if all(campo.campo[l][c2].revelada and campo.campo[l][c2].bombas_vizinhas == 1 for c2 in range(c, c+4)):
                if not campo.campo[l][c].revelada:
                    resultado[(l, c)] = 'unsafe_bandeira'
                if not campo.campo[l][c+3].revelada:
                    resultado[(l, c+3)] = 'unsafe_bandeira'
                if not campo.campo[l][c+1].revelada:
                    resultado[(l, c+1)] = 'incerta'
                if not campo.campo[l][c+2].revelada:
                    resultado[(l, c+2)] = 'incerta'

    # Padrão 4: Diagonal 1-? | ?-safe (borda)
    for l in range(linhas-1):
        for c in range(colunas-1):
            if campo.campo[l][c].revelada and campo.campo[l][c].bombas_vizinhas == 1:
                # Diagonal para baixo-direita
                if not campo.campo[l][c+1].revelada and not campo.campo[l+1][c].revelada and \
                    (l+1 == linhas-1 or c+1 == colunas-1):
                    resultado[(l+1, c+1)] = 'safe_check'
    return resultado

def calcular_recompensa(celula, vitoria=False, derrota=False, prev_revelada=False, tipo_padrao='neutro'):
    """
    Mescla sistema clássico + patterns:
    - safe_check: +10 (abrir), -5 (se já aberta)
    - unsafe_bandeira: -10 (abrir), +4 (se já marcada como bomba)
    - incerta: -1 (abrir)
    - neutro: sistema padrão minado
    """
    if prev_revelada:
        return -0.5
    if tipo_padrao == 'safe_check':
        return 10.0
    elif tipo_padrao == 'unsafe_bandeira':
        if celula.revelada:
            return -10.0  # Abriu uma bomba (deveria marcar bandeira)
        else:
            return 4.0   # (Poderia premiar se marcar bandeira, mas aqui é só pra abrir)
    elif tipo_padrao == 'incerta':
        return -1.0
    else:
        # Sistema clássico
        if celula.tem_bomba:
            reward = -1.0
        else:
            reward = 0.1
            reward -= 0.1 * celula.bombas_vizinhas
            if celula.bombas_vizinhas == 0:
                reward += 0.2
        if vitoria:
            reward += 5.0
        if derrota:
            reward -= 5.0
        return reward

def train():
    linhas, colunas, n_bombas = 4, 4, 2
    scores = []
    mean_scores = []
    victories = []
    winrates = []
    total_score = 0
    total_victory = 0
    record = 0
    agent = MinesweeperAgent(linhas, colunas, n_bombas)
    N_EPISODES = 5000

    for game in range(N_EPISODES):
        campo = CampoMinado(linhas, colunas, n_bombas)
        score = 0
        state_old = agent.get_state(campo)
        vitoria = False

        while campo.jogo_ativo:
            action = agent.get_action(state_old, campo)
            move = action.argmax()
            linha, coluna = move // colunas, move % colunas

            celula = campo.campo[linha][coluna]
            prev_revelada = celula.revelada

            padroes = detectar_padroes_campo(campo)
            tipo_padrao = padroes.get((linha, coluna), 'neutro')

            campo.revelar(linha, coluna)
            state_new = agent.get_state(campo)

            jogada_invalida, bomba, vitoria, fim_de_jogo = avaliar_jogada(campo, linha, coluna, prev_revelada)
            derrota = bomba and fim_de_jogo

            reward = calcular_recompensa(
                celula, vitoria, derrota, prev_revelada=prev_revelada, tipo_padrao=tipo_padrao
            )

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
            model_name = f"model_{N_EPISODES}_partidas_{linhas}_x_{colunas}_{n_bombas}M_V10.pth"
            print(model_name)
            agent.model.save(file_name=model_name)
        scores.append(score)
        victories.append(int(vitoria))
        total_score += score
        total_victory += int(vitoria)
        mean_scores.append(total_score / (game+1))
        winrate = 100 * total_victory / (game+1)
        winrates.append(winrate)

        if len(scores) >= 500:
            mov_avg_scores = [np.mean(scores[max(0, i-499):i+1]) for i in range(len(scores))]
        else:
            mov_avg_scores = mean_scores
        if len(victories) >= 200:
            mov_avg_winrates = [100 * np.mean(victories[max(0, i-199):i+1]) for i in range(len(victories))]
        else:
            mov_avg_winrates = [100 * np.mean(victories[:i+1]) for i in range(len(victories))]
        plot(scores, mean_scores, winrates, mov_avg_scores, mov_avg_winrates, save_final=False)
        print(f'Game {game+1} | Score: {score} | Record: {record} | Mean: {mean_scores[-1]:.2f} | MovingAvgWinrate(200): {mov_avg_winrates[-1]:.2f}% | WinRate: {winrate:.1f}%')

    plot(scores, mean_scores, winrates, mov_avg_scores, mov_avg_winrates, save_final=True,
         plays=N_EPISODES, num_mines=n_bombas, tab_len=linhas)
    print("Gráfico salvo como campo_minado_rl_training.png")

if __name__ == '__main__':
    train()
