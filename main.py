import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from consts import TRAINING_NUMBER


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

# 3. Dataset SSL
def extract_patch(board, x, y, patch_size=5):
    context = patch_size // 2
    return board[x - context:x + context + 1, y - context:y + context + 1]

def generate_ssl_dataset(num_samples=10000, board_size=10, num_mines=10):
    PATCH_SIZE = 5
    CONTEXT = PATCH_SIZE // 2
    X, y = [], []
    while len(X) < num_samples:
        board = generate_board(board_size, num_mines)
        for i in range(CONTEXT, board_size - CONTEXT):
            for j in range(CONTEXT, board_size - CONTEXT):
                patch = extract_patch(board, i, j, PATCH_SIZE)
                X.append(patch)
                y.append(board[i, j])
                if len(X) >= num_samples:
                    break
            if len(X) >= num_samples:
                break
    X = np.array(X).astype(np.float32) / 9.0
    y = np.array(y)
    return torch.tensor(X).unsqueeze(1), torch.tensor(y)

# 4. Modelo
class SSLNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)  # 0-8 e mina (9)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# 5. Treinamento
X, y = generate_ssl_dataset(10000)
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SSLNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(TRAINING_NUMBER):
    total_loss = 0
    for xb, yb in tqdm(loader):
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.2f}")

# 6. Salvar modelo
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/ssl_net.pth")

# 7. Função de previsão

def predict_board(board, model, patch_size=5):
    CONTEXT = patch_size // 2
    revealed = np.full_like(board, -1)
    model.eval()
    with torch.no_grad():
        for i in range(CONTEXT, board.shape[0] - CONTEXT):
            for j in range(CONTEXT, board.shape[1] - CONTEXT):
                patch = extract_patch(board, i, j, patch_size).astype(np.float32) / 9.0
                tensor = torch.tensor(patch).unsqueeze(0).unsqueeze(0).to(device)
                pred = model(tensor).argmax(dim=1).item()
                revealed[i, j] = pred
    return revealed

# 8. Teste de previsão
sample_board = generate_board()
predicted = predict_board(sample_board, model)

print("Tabuleiro original:")
print(sample_board)
print("\nTabuleiro previsto:")
print(predicted)

# 9. Função iterativa de revelação progressiva

def progressive_reveal(board, model, threshold=0.9):
    CONTEXT = 2
    revealed = np.full_like(board, -1)
    confidence = np.zeros_like(board, dtype=float)
    model.eval()

    for _ in range(3):  # número de passes
        for i in range(CONTEXT, board.shape[0] - CONTEXT):
            for j in range(CONTEXT, board.shape[1] - CONTEXT):
                if revealed[i, j] != -1:
                    continue
                patch = extract_patch(revealed, i, j, 5).astype(np.float32) / 9.0
                tensor = torch.tensor(patch).unsqueeze(0).unsqueeze(0).to(device)
                out = F.softmax(model(tensor), dim=1)
                prob, pred = out.max(dim=1)
                if prob.item() > threshold:
                    revealed[i, j] = pred.item()
                    confidence[i, j] = prob.item()
    return revealed, confidence

revealed_board, conf = progressive_reveal(sample_board, model)
print("\nTabuleiro revelado progressivamente:")
print(revealed_board)