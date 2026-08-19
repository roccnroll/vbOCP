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

from src.dl.common import (
    FFNN, train_ffnn,
    compute_minmax_stats, normalize_minmax,
    compute_standard_stats, normalize_standard,
)


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

    x_raw = np.stack([mu1, mu2, mu_u], axis=1)
    y_raw = coeffs.T  # (n_samples, n_modes)

    # normalizzazione: input in [-1,1] (si abbina a Tanh), output standardizzati
    # (i coefficienti POD hanno scale molto diverse tra i modi, senza normalizzare
    # la loss sarebbe dominata dai coefficienti piu' grandi)
    x_stats = compute_minmax_stats(x_raw)
    y_stats = compute_standard_stats(y_raw)
    x_norm = normalize_minmax(x_raw, x_stats)
    y_norm = normalize_standard(y_raw, y_stats)

    print("Training FFNN mu -> coefficienti POD(u) (normalizzati) ...")
    x_train = torch.tensor(x_norm, dtype=torch.float32)
    y_train = torch.tensor(y_norm, dtype=torch.float32)

    net = FFNN(input_dim=3, output_dim=n_modes, hidden_dim=args.hidden_dim, n_hidden_layers=args.n_hidden_layers)
    train_ffnn(net, x_train, y_train, epochs=args.epochs, lr=args.lr, lr_drop_epoch=args.epochs // 2)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), args.output)

    # statistiche di normalizzazione + architettura salvate accanto ai pesi - servono a
    # evaluate per de-normalizzare e per ricostruire la STESSA rete (stesse dimensioni)
    stats_path = str(Path(args.output).with_suffix(".norm.npz"))
    np.savez_compressed(
        stats_path,
        x_min=x_stats["min"], x_max=x_stats["max"],
        y_mean=y_stats["mean"], y_std=y_stats["std"],
        hidden_dim=args.hidden_dim, n_hidden_layers=args.n_hidden_layers,
    )
    print(f"Pesi rete salvati in {args.output}")
    print(f"Statistiche di normalizzazione salvate in {stats_path}")


if __name__ == "__main__":
    main()
