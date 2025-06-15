# play.py

import torch
import torch.nn.functional as F
import numpy as np
from game.campo_minado import CampoMinado
from consts import PATCH_SIZE
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Funções utilitárias ----
def extract_patch(board, x, y, patch_size=PATCH_SIZE):
    context = patch_size // 2
    patch = np.full((patch_size, patch_size), -1, dtype=np.float32)
    bx, by = board.shape
    for dx in range(-context, context + 1):
        for dy in range(-context, context + 1):
            xi, yj = x + dx, y + dy
            if 0 <= xi < bx and 0 <= yj < by:
                patch[dx + context, dy + context] = board[xi, yj]
    return patch

def get_visible_board(campo):
    """Retorna uma matriz numpy do ponto de vista do jogador (-1: não revelado, 0-8: valor, 9: bomba descoberta)"""
    linhas, colunas = campo.linhas, campo.colunas
    board = np.full((linhas, colunas), -1, dtype=np.int32)
    for i in range(linhas):
        for j in range(colunas):
            celula = campo.campo[i][j]
            if celula.revelada:
                if celula.tem_bomba:
                    board[i, j] = 9
                else:
                    board[i, j] = celula.bombas_vizinhas
    return board

def print_board_with_probs(visible, probs):
    print("\nTabuleiro do jogador (sua visão):")
    for i in range(visible.shape[0]):
        row = ""
        for j in range(visible.shape[1]):
            if visible[i, j] == -1:
                row += "# "
            elif visible[i, j] == 9:
                row += "* "
            elif visible[i, j] == 0:
                row += "  "
            else:
                row += f"{visible[i,j]} "
        print(row)
    print("\nProbabilidade de não ser bomba nas casas não reveladas:")
    for i in range(probs.shape[0]):
        row = ""
        for j in range(probs.shape[1]):
            if visible[i, j] == -1:
                row += f"{probs[i, j]*100:7.3f}% "
            else:
                row += "         "
        print(row)
    print()

# ---- Carrega modelos ----
from train import SSLNet  # Reutiliza arquitetura

model_bin = SSLNet(out_classes=2).to(device)
model_bin.load_state_dict(torch.load("models/ssl_net_partial.pth", map_location=device))
model_bin.eval()

# (Opcional, para visualização de predição multi-classe)
model_full = SSLNet(out_classes=10).to(device)
if os.path.exists("models/ssl_net_full.pth"):
    model_full.load_state_dict(torch.load("models/ssl_net_full.pth", map_location=device))
    model_full.eval()
else:
    model_full = None

# ---- Jogo ----
tam = 10
bombas = 10
jogo = CampoMinado(tam, tam, bombas)

max_moves = tam * tam  # só para evitar loop infinito

for move in range(max_moves):
    # 1. Board visível
    visible = get_visible_board(jogo)

    # 2. Calcular probabilidade de NÃO ser bomba em cada célula não revelada
    probs = np.zeros((tam, tam))
    candidates = []
    for i in range(PATCH_SIZE//2, tam - PATCH_SIZE//2):
        for j in range(PATCH_SIZE//2, tam - PATCH_SIZE//2):
            if visible[i, j] == -1:  # não revelada
                patch = extract_patch(visible, i, j)
                patch = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device) / 9.0
                with torch.no_grad():
                    out = F.softmax(model_bin(patch), dim=1)
                prob_not_bomb = out[0, 0].item()  # índice 0 = seguro
                probs[i, j] = prob_not_bomb
                candidates.append((prob_not_bomb, i, j))

    if not candidates:
        print("Nenhuma célula disponível. Fim de jogo.")
        break

    # 3. Escolher melhor jogada (maior probabilidade de não ser bomba)
    max_prob = max([c[0] for c in candidates])
    if max_prob == 0:
        # Escolhe aleatório entre as opções
        import random
        prob, x, y = random.choice(candidates)
        print("Todas as probabilidades são 0%. Escolhendo jogada aleatória!\n")
    else:
        # Mantém escolha padrão (maior probabilidade)
        candidates.sort(reverse=True)  # maior probabilidade primeiro
        prob, x, y = candidates[0]

    # 4. (Opcional) Visualização extra usando modelo multiclasse
    if model_full:
        pred_board = np.full((tam, tam), -1, dtype=np.int32)
        for i in range(PATCH_SIZE//2, tam - PATCH_SIZE//2):
            for j in range(PATCH_SIZE//2, tam - PATCH_SIZE//2):
                if visible[i, j] == -1:
                    patch = extract_patch(visible, i, j)
                    patch = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device) / 9.0
                    with torch.no_grad():
                        pred = model_full(patch).argmax(dim=1).item()
                    pred_board[i, j] = pred
        print("\nPredição do modelo multiclasse (0-8 ou 9=bomba):")
        for i in range(tam):
            row = ""
            for j in range(tam):
                if visible[i, j] == -1 and pred_board[i, j] != -1:
                    if pred_board[i, j] == 9:
                        row += "B "
                    else:
                        row += f"{pred_board[i,j]} "
                else:
                    row += "  "
            print(row)

    # 5. Exibe tudo no terminal
    print_board_with_probs(visible, probs)
    print(f"Jogada {move+1}: escolhendo célula ({x},{y}) com prob. de NÃO ser bomba: {prob*100:.6f}%\n")

    # 6. Executa jogada
    if visible[x, y] != -1:
        print("Jogada inválida, célula já revelada. Encerrando.")
        break
    jogo.revelar(x, y)

    if not jogo.jogo_ativo:
        print("\nEstado final:")
        jogo.mostrar_campo()
        break

else:
    print("Número máximo de jogadas atingido.")

