import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from game.campo_minado import CampoMinado

from consts import TRAINING_NUMBER


def generate_board(size: int = 10, num_mines: int = 10) -> np.ndarray:
    """Create a random Minesweeper board."""
    board = np.zeros((size, size), dtype=int)
    mines = np.random.choice(size * size, num_mines, replace=False)
    for mine in mines:
        x, y = divmod(mine, size)
        board[x, y] = 9
        for i in range(x - 1, x + 2):
            for j in range(y - 1, y + 2):
                if 0 <= i < size and 0 <= j < size and board[i, j] != 9:
                    board[i, j] += 1
    return board


def board_from_game(game: CampoMinado) -> np.ndarray:
    """Convert a ``CampoMinado`` instance to a numeric board."""
    board = np.zeros((game.linhas, game.colunas), dtype=int)
    for i in range(game.linhas):
        for j in range(game.colunas):
            cell = game.campo[i][j]
            board[i, j] = 9 if cell.tem_bomba else cell.bombas_vizinhas
    return board


def board_from_view(game: CampoMinado) -> np.ndarray:
    """Return the current visible state of the game board."""
    board = np.full((game.linhas, game.colunas), -1, dtype=int)
    for i in range(game.linhas):
        for j in range(game.colunas):
            cell = game.campo[i][j]
            if cell.revelada:
                board[i, j] = 9 if cell.tem_bomba else cell.bombas_vizinhas
    return board


def extract_patch(board: np.ndarray, x: int, y: int, patch_size: int = 5) -> np.ndarray:
    context = patch_size // 2
    return board[x - context:x + context + 1, y - context:y + context + 1]


def generate_ssl_dataset(num_samples: int = 10000, board_size: int = 10, num_mines: int = 10):
    patch_size = 5
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


class SSLNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)  # 0-8 and mine (9)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def train(model: nn.Module, device: torch.device, epochs: int = TRAINING_NUMBER):
    X, y = generate_ssl_dataset(10000)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    losses = []
    for epoch in range(epochs):
        total_loss = 0.0
        for xb, yb in tqdm(loader):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        losses.append(total_loss)
        print(f"Epoch {epoch + 1}, Loss: {total_loss:.2f}")
    return losses


def plot_loss(losses, path):
    plt.figure()
    plt.plot(range(1, len(losses) + 1), losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.savefig(path)
    plt.close()


def predict_board(board: np.ndarray, model: nn.Module, device: torch.device, patch_size: int = 5) -> np.ndarray:
    context = patch_size // 2
    revealed = np.full_like(board, -1)
    model.eval()
    with torch.no_grad():
        for i in range(context, board.shape[0] - context):
            for j in range(context, board.shape[1] - context):
                patch = extract_patch(board, i, j, patch_size).astype(np.float32) / 9.0
                tensor = torch.tensor(patch).unsqueeze(0).unsqueeze(0).to(device)
                pred = model(tensor).argmax(dim=1).item()
                revealed[i, j] = pred
    return revealed


def plot_boards(original: np.ndarray, predicted: np.ndarray, path: str):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(original, cmap="viridis", vmin=-1, vmax=9)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(predicted, cmap="viridis", vmin=-1, vmax=9)
    axes[1].set_title("Predicted")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def progressive_reveal(board: np.ndarray, model: nn.Module, device: torch.device, threshold: float = 0.9):
    context = 2
    revealed = board.copy()
    confidence = np.zeros_like(board, dtype=float)
    model.eval()

    for _ in range(3):  # number of passes
        for i in range(context, board.shape[0] - context):
            for j in range(context, board.shape[1] - context):
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


def evaluate(model: nn.Module, device: torch.device, board: np.ndarray) -> tuple:
    """Predict an entire board and return accuracy and the prediction."""
    predicted = predict_board(board, model, device)
    mask = predicted != -1
    if mask.sum() == 0:
        return 0.0, predicted
    correct = (predicted[mask] == board[mask]).sum()
    accuracy = float(correct) / float(mask.sum())
    return accuracy, predicted


def play_game(model: nn.Module, device: torch.device, size: int = 10, mines: int = 10) -> bool:
    """Play a single game using the model's predictions."""
    game = CampoMinado(size, size, mines)
    step = 0
    while game.jogo_ativo:
        step += 1
        view = board_from_view(game)
        predicted, conf = progressive_reveal(view, model, device, threshold=0.8)

        best = None
        best_conf = -1.0
        for i in range(size):
            for j in range(size):
                if not game.campo[i][j].revelada and predicted[i, j] != 9:
                    if conf[i, j] > best_conf:
                        best = (i, j)
                        best_conf = conf[i, j]

        if best is None:
            import random
            options = [
                (i, j)
                for i in range(size)
                for j in range(size)
                if not game.campo[i][j].revelada
            ]
            best = random.choice(options)

        print(f"Jogada {step}: revelando {best}")
        game.revelar(*best)
        game.mostrar_campo()

    return game._verificar_vitoria()


def train_and_play(cycles: int = 3, epochs_per_cycle: int = 5) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SSLNet().to(device)

    os.makedirs("models", exist_ok=True)

    for cycle in range(1, cycles + 1):
        print(f"--- Cycle {cycle}/{cycles} ---")
        losses = train(model, device, epochs=epochs_per_cycle)
        torch.save(model.state_dict(), f"models/ssl_net_cycle{cycle}.pth")
        plot_loss(losses, f"models/loss_cycle{cycle}.png")

        game = CampoMinado(10, 10, 10)
        board = board_from_game(game)
        accuracy, predicted = evaluate(model, device, board)
        plot_boards(board, predicted, f"models/prediction_cycle{cycle}.png")
        print(f"Accuracy after cycle {cycle}: {accuracy * 100:.2f}%")

        win = play_game(model, device)
        print("Resultado da partida:", "Vitoria" if win else "Derrota")



def main():
    """Run a few train-and-play cycles demonstrating improvement."""
    train_and_play()


if __name__ == "__main__":
    main()
