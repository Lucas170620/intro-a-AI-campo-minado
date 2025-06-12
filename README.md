# Minesweeper SSL

This project trains a simple self-supervised neural network to predict the values of a Minesweeper board.

## Requirements

- Python 3.8+
- [PyTorch](https://pytorch.org/)
- `numpy`
- `matplotlib`
- `tqdm`

Install the dependencies with pip:

```bash
pip install torch numpy matplotlib tqdm
```

## Training and Running

Execute `main.py` to train the network and automatically evaluate it over several game rounds:

```bash
python main.py
```

Each training cycle stores the weights and plots in the `models/` directory, e.g.:

- `models/ssl_net_cycle1.pth`
- `models/loss_cycle1.png`
- `models/prediction_cycle1.png`

## Further Ideas

- Experiment with different network architectures or training parameters to improve accuracy.
- Explore more advanced strategies for selecting cells based on the model's confidence.

