import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores, mean_scores, winrates=None, mov_avg_scores=None, mov_avg_winrates=None, save_final=False, plays=0,
         tab_len=0, num_mines=0):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training Progress')
    plt.xlabel('Games')
    plt.ylabel('Score / WinRate')
    plt.plot(scores, label='Score')
    plt.plot(mean_scores, label='Mean Score')
    if mov_avg_scores is not None:
        plt.plot(mov_avg_scores, label='Moving Avg Score (500)')
    if winrates is not None:
        plt.plot(winrates, label='Win Rate (%)')
    if mov_avg_winrates is not None:
        plt.plot(mov_avg_winrates, label='Moving WinRate (200)')
    plt.ylim(ymin=0)
    plt.legend()
    if len(scores):
        plt.text(len(scores)-1, scores[-1], str(scores[-1]))
        plt.text(len(mean_scores)-1, mean_scores[-1], f"{mean_scores[-1]:.2f}")
        if mov_avg_scores is not None:
            plt.text(len(mov_avg_scores)-1, mov_avg_scores[-1], f"{mov_avg_scores[-1]:.2f}")
        if winrates is not None:
            plt.text(len(winrates)-1, winrates[-1], f"{winrates[-1]:.1f}%")
        if mov_avg_winrates is not None:
            plt.text(len(mov_avg_winrates)-1, mov_avg_winrates[-1], f"{mov_avg_winrates[-1]:.1f}%")
    plt.show(block=False)
    plt.pause(.1)
    # Salva a imagem se solicitado
    if save_final:
        plt.savefig(f"reinforcement_learning/results/campo_minado_rl_training_{plays}_partidas_{tab_len}_x_{tab_len}_{num_mines}M_V7.png")
