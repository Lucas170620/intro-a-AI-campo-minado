# train.py

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from consts import TRAINING_NUMBER

# ----- 1. Geração do Board Real -----
def generate_board(size=10, num_mines=10):
    board = np.zeros((size, size), dtype=int)
    # Coloca minas
    mines = np.random.choice(size * size, num_mines, replace=False)
    for mine in mines:
        x, y = divmod(mine, size)
        board[x, y] = 9
        for i in range(x - 1, x + 2):
            for j in range(y - 1, y + 2):
                if 0 <= i < size and 0 <= j < size and board[i, j] != 9:
                    board[i, j] += 1
    return board

# ----- 2. Patches -----
def extract_patch(board, x, y, patch_size=5):
    context = patch_size // 2
    return board[x - context:x + context + 1, y - context:y + context + 1]

# ----- 3. Dataset inicial: Board Completo -----
def generate_ssl_dataset(num_samples=10000, board_size=10, num_mines=10, patch_size=5):
    context = patch_size // 2
    X, y = [], []
    while len(X) < num_samples:
        board = generate_board(board_size, num_mines)
        for i in range(context, board_size - context):
            for j in range(context, board_size - context):
                patch = extract_patch(board, i, j, patch_size)
                X.append(patch)
                y.append(board[i, j])
                if len(X) >= num_samples:
                    break
            if len(X) >= num_samples:
                break
    X = np.array(X).astype(np.float32) / 9.0
    y = np.array(y)
    return torch.tensor(X).unsqueeze(1), torch.tensor(y)

# ----- 4. Dataset secundário: Jogo Parcial (apenas células reveladas) -----
def simulate_partial_board(board, reveal_steps=15):
    # Estado inicial: tudo coberto (-1)
    partial = np.full_like(board, -1)
    size = board.shape[0]
    revealed = set()
    for _ in range(reveal_steps):
        # Sorteia uma célula não revelada e que não seja bomba
        safe_cells = [(i, j) for i in range(size) for j in range(size) if (board[i, j] != 9 and (i, j) not in revealed)]
        if not safe_cells:
            break
        i, j = safe_cells[np.random.randint(len(safe_cells))]
        partial[i, j] = board[i, j]
        revealed.add((i, j))
    return partial

def generate_partial_ssl_dataset(num_samples=5000, board_size=10, num_mines=10, patch_size=5):
    context = patch_size // 2
    X, y = [], []
    while len(X) < num_samples:
        board = generate_board(board_size, num_mines)
        # Simula jogo real: só algumas células estão reveladas
        partial = simulate_partial_board(board, reveal_steps=np.random.randint(8, 20))
        for i in range(context, board_size - context):
            for j in range(context, board_size - context):
                if partial[i, j] == -1:  # Só gera exemplos para células não reveladas
                    patch = extract_patch(partial, i, j, patch_size)
                    # Label: 0 = seguro, 1 = bomba
                    label = int(board[i, j] == 9)
                    X.append(patch)
                    y.append(label)
                    if len(X) >= num_samples:
                        break
            if len(X) >= num_samples:
                break
    X = np.array(X).astype(np.float32) / 9.0
    y = np.array(y)
    return torch.tensor(X).unsqueeze(1), torch.tensor(y)

# ----- 5. Modelo Neural (mesmo para as duas fases) -----
class SSLNet(nn.Module):
    def __init__(self, out_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, out_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----- 6. Treinamento Fase 1: Board Completo (multiclasse) -----
    print("Treinando Fase 1: board completo (multiclasse 0-8 ou bomba)")
    X_full, y_full = generate_ssl_dataset(10000)
    dataset_full = TensorDataset(X_full, y_full)
    loader_full = DataLoader(dataset_full, batch_size=64, shuffle=True)

    model = SSLNet(out_classes=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    losses_full = []

    for epoch in range(50):
        total_loss = 0
        for xb, yb in tqdm(loader_full):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        losses_full.append(total_loss)
        print(f"[Fase 1] Epoch {epoch+1}, Loss: {total_loss:.2f}")

    # Salva modelo da primeira fase
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/ssl_net_full.pth")

    # ----- 7. Treinamento Fase 2: Board Parcial (binário seguro/bomba) -----
    print("Treinando Fase 2: board parcial (binário seguro/bomba)")
    X_part, y_part = generate_partial_ssl_dataset(5000)
    dataset_part = TensorDataset(X_part, y_part)
    loader_part = DataLoader(dataset_part, batch_size=64, shuffle=True)

    # Troca a última camada para binária
    model.fc2 = nn.Linear(128, 2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    losses_part = []

    for epoch in range(TRAINING_NUMBER):
        total_loss = 0
        for xb, yb in tqdm(loader_part):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        losses_part.append(total_loss)
        print(f"[Fase 2] Epoch {epoch+1}, Loss: {total_loss:.2f}")

    # Salva modelo final (apto para jogar com informações reais do jogador)
    torch.save(model.state_dict(), "models/ssl_net_partial.pth")

    # ----- 8. Visualização da curva de perda -----
    plt.plot(losses_full, label="Fase 1 - Board Completo")
    plt.plot(losses_part, label="Fase 2 - Board Parcial")
    plt.xlabel("Epoch")
    plt.ylabel("Perda (Loss)")
    plt.legend()
    plt.title("Curva de treinamento - Campo Minado SSL")
    plt.show()
