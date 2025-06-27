from reinforcement_learning.agent import MinesweeperAgent
from game.campo_minado import CampoMinado
from reinforcement_learning.helper import plot
import numpy as np

def train():
    linhas, colunas, n_bombas = 4 , 4, 2
    scores = []
    mean_scores = []
    victories = []
    winrates = []
    total_score = 0
    total_victory = 0
    record = 0
    agent = MinesweeperAgent(linhas, colunas, n_bombas)
    N_EPISODES = 1000

    for game in range(N_EPISODES):
        campo = CampoMinado(linhas, colunas, n_bombas)
        score = 0
        done = False
        state_old = agent.get_state(campo)
        victory = 0

        while campo.jogo_ativo:
            action = agent.get_action(state_old, campo)
            move = action.argmax()
            linha, coluna = move // colunas, move % colunas

            celula = campo.campo[linha][coluna]
            prev_revelada = celula.revelada

            campo.revelar(linha, coluna)
            state_new = agent.get_state(campo)

            if prev_revelada:
                reward = -1  # penalidade leve, pode ocorrer por erro de clique ou revelação em cascata
                done = False
            elif celula.tem_bomba:
                reward = -10  # ainda penaliza muito
                done = True
            elif campo._verificar_vitoria():
                reward = +30  # recompensa pela vitória
                score += 1
                victory = 1
                done = True
            elif not celula.tem_bomba:
                # célula segura aberta
                if celula.bombas_vizinhas == 0:
                    reward = +4  # abrir área segura continua valioso
                elif celula.bombas_vizinhas == 1:
                    reward = +2
                else:
                    reward = +1
                score += 1
                done = False
            else:
                reward = 0
                done = False

            agent.train_short_memory(state_old, action, reward, state_new, done)
            agent.remember(state_old, action, reward, state_new, done)
            state_old = state_new

            if done:
                break

        agent.n_games += 1
        agent.train_long_memory()

        if score > record:
            record = score
            model_name = f"model_{N_EPISODES}_partidas_{linhas}_x_{colunas}_{n_bombas}M_v7.pth"
            print(model_name)
            agent.model.save(file_name=model_name)
        scores.append(score)
        victories.append(victory)
        total_score += score
        total_victory += victory
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
        plot(scores, mean_scores, winrates, mov_avg_scores, mov_avg_winrates, save_final=False)
        print(f'Game {game+1} | Score: {score} | Record: {record} | Mean: {mean_scores[-1]:.2f} | MovingAvgWinrate(200): {mov_avg_winrates[-1]:.2f}% | WinRate: {winrate:.1f}%')

    # Ao final do treinamento, salva a imagem
    plot(scores, mean_scores, winrates, mov_avg_scores, mov_avg_winrates, save_final=True
         ,plays=N_EPISODES, num_mines=n_bombas, tab_len= linhas)
    print("Gráfico salvo como campo_minado_rl_training.png")

if __name__ == '__main__':
    train()
