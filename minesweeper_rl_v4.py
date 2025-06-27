
from reinforcement_learning.agent_v4 import MinesweeperAgent
from game.campo_minado import CampoMinado
from reinforcement_learning.helper import plot
import numpy as np

def train():
    linhas, colunas, n_bombas = 4, 4 , 2
    scores = []
    mean_scores = []
    victories = []
    winrates = []
    total_score = 0
    total_victory = 0
    record = 0
    agent = MinesweeperAgent(linhas, colunas, n_bombas)
    epsilon_start = 80
    epsilon_end = 2
    epsilon_decay = 0.001
    N_EPISODES = 1000

    for game in range(N_EPISODES):
        campo = CampoMinado(linhas, colunas, n_bombas)
        score = 0
        done = False
        state_old = agent.get_state(campo)
        victory = 0

        while campo.jogo_ativo:
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-epsilon_decay * agent.n_games)
            action = agent.get_action(state_old, campo, epsilon=epsilon)
            move = action.argmax()
            linha, coluna = move // colunas, move % colunas

            celula = campo.campo[linha][coluna]
            prev_revelada = celula.revelada

            campo.revelar(linha, coluna)
            state_new = agent.get_state(campo)

            if prev_revelada:
                reward = -2  # Penalização menor por clique redundante
                done = False

            elif celula.tem_bomba:
                reward = -100
                done = True

            elif not celula.tem_bomba and not prev_revelada:
                risco_local = 0
                for dl in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        nl, nc = linha + dl, coluna + dc
                        if 0 <= nl < campo.linhas and 0 <= nc < campo.colunas:
                            vizinha = campo.campo[nl][nc]
                            if vizinha.revelada and not vizinha.tem_bomba:
                                risco_local += vizinha.bombas_vizinhas

                # Ajuste: Redução da penalização, ampliação da recompensa para jogadas boas
                base_reward = 10 if celula.bombas_vizinhas == 0 else 6
                risco_penalty = 0.2 * risco_local
                vizinhanca_penalty = 0.3 * celula.bombas_vizinhas

                reward = base_reward - risco_penalty - vizinhanca_penalty
                score += 1

                if campo._verificar_vitoria():
                    reward = 150
                    victory = 1
                    done = True
                else:
                    done = False

            else:
                reward = -1
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
            model_name = f"model_{N_EPISODES}_partidas_{linhas}_x_{colunas}_{n_bombas}M_V4.pth"
            print(model_name)
            agent.model.save(file_name=model_name)

        scores.append(score)
        victories.append(victory)
        total_score += score
        total_victory += victory
        mean_scores.append(total_score / (game + 1))
        winrate = 100 * total_victory / (game + 1)
        winrates.append(winrate)

        mov_avg_scores = [np.mean(scores[max(0, i - 499):i + 1]) for i in range(len(scores))]
        mov_avg_winrates = [100 * np.mean(victories[max(0, i - 199):i + 1]) for i in range(len(victories))]

        plot(scores, mean_scores, winrates, mov_avg_scores, mov_avg_winrates, save_final=False)
        print(f'Game {game + 1} | Score: {score} | Record: {record} | Mean: {mean_scores[-1]:.2f} | MovingAvgWinrate(200): {mov_avg_winrates[-1]:.2f}% | WinRate: {winrate:.1f}%')

    plot(scores, mean_scores, winrates, mov_avg_scores, mov_avg_winrates, save_final=True,
         plays=N_EPISODES, num_mines=n_bombas, tab_len=linhas)
    print("Gráfico salvo como campo_minado_rl_training.png")

if __name__ == '__main__':
    train()
