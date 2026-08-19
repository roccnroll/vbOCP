"""CLI: allena la FFNN mu -> coefficienti POD(u), a partire dalla base gia' costruita.

Solo la fase di training - separata dalla POD (build_control_pod.py) cosi'
si puo' riallenare con parametri diversi senza ricostruire la base.

Uso:
    python -m src.rom.train_control_nn --pod-model data/snapshots/test1_control_pod.npz \
        --output data/snapshots/test1_control_nn.pt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.dl.common import FFNN, train_ffnn


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pod-model", required=True, help="path al .npz da build_control_pod.py")
    parser.add_argument("--epochs", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=30)
    parser.add_argument("--n-hidden-layers", type=int, default=4)
    parser.add_argument("--output", required=True, help="path dove salvare i pesi della rete (.pt)")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Caricamento base POD da {args.pod_model} ...")
    pod_data = np.load(args.pod_model)
    coeffs = pod_data["coeffs"]  # (n_modes, n_samples)
    mu1, mu2, mu_u = pod_data["mu1"], pod_data["mu2"], pod_data["mu_u"]
    n_modes = int(pod_data["n_modes"])

    print("Training FFNN mu -> coefficienti POD(u) ...")
    x_train = torch.tensor(np.stack([mu1, mu2, mu_u], axis=1), dtype=torch.float32)
    y_train = torch.tensor(coeffs.T, dtype=torch.float32)  # (n_samples, n_modes)

    net = FFNN(input_dim=3, output_dim=n_modes, hidden_dim=args.hidden_dim, n_hidden_layers=args.n_hidden_layers)
    train_ffnn(net, x_train, y_train, epochs=args.epochs, lr=args.lr, lr_drop_epoch=args.epochs // 2)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), args.output)
    print(f"Pesi rete salvati in {args.output}")


if __name__ == "__main__":
    main()
