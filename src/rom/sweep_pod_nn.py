"""CLI: sweep di iperparametri per le PODNN di stato/aggiunto (numero di modi o hidden_dim).

Carica mesh/operatori/snapshot una sola volta (non ricarica ad ogni punto
dello sweep). Per ogni valore del parametro: ricostruisce base POD e/o
rete, allena, valuta errore L2/H1 sul test set. Salva risultati in .csv
e un plot errore-vs-parametro.

Uso (sweep su N, hidden_dim fisso):
    python -m src.rom.sweep_pod_nn --config configs/test1.yaml \
        --snapshots data/snapshots/test1_300.npz \
        --test-snapshots data/snapshots/test1_test150.npz \
        --sweep-param n_modes --values 1,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150 \
        --epochs 50000 --output data/snapshots/sweep_n_modes.csv

Uso (sweep su hidden_dim, N fisso):
    python -m src.rom.sweep_pod_nn --config configs/test1.yaml \
        --snapshots data/snapshots/test1_300.npz \
        --test-snapshots data/snapshots/test1_test150.npz \
        --sweep-param hidden_dim --values 16,32,64,128 --n-modes 50 \
        --epochs 50000 --output data/snapshots/sweep_hidden_dim.csv
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.full_order.mesh import load_mesh
from src.full_order.assembly import assemble_operators
from src.rom.inner_product import assemble_full_mass_matrix
from src.rom.pod import (
    compute_correlation_eigenvalues, select_n_modes, build_pod_basis,
    project_onto_basis, plot_eigenvalue_decay_curves,
)
from src.dl.common import (
    FFNN, train_ffnn,
    compute_minmax_stats, normalize_minmax,
    compute_standard_stats, normalize_standard, denormalize_standard,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--snapshots", required=True, help="snapshot di training")
    parser.add_argument("--test-snapshots", required=True)
    parser.add_argument("--inner-product", choices=["seminorm", "full"], default="seminorm")
    parser.add_argument("--sweep-param", required=True, choices=["n_modes", "hidden_dim"])
    parser.add_argument("--values", required=True, help="valori separati da virgola, es. 1,10,20,30")
    parser.add_argument("--n-modes", type=int, default=50,
                         help="N fisso quando si fa lo sweep su hidden_dim")
    parser.add_argument("--hidden-dim", type=int, default=30,
                         help="hidden_dim fisso quando si fa lo sweep su n_modes")
    parser.add_argument("--n-hidden-layers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", required=True, help="path del .csv con i risultati")
    return parser.parse_args()


def train_and_evaluate(Y, P, Y_test, P_test, X, n_modes, hidden_dim, n_hidden_layers,
                        mu1, mu2, mu_u, mu1_test, mu2_test, mu_u_test, epochs, lr):
    """Costruisce base+rete per y e p con dati parametri, valuta errore L2/H1 sul test set."""
    results = {}
    for field, Y_train_field, Y_test_field in [("y", Y, Y_test), ("p", P, P_test)]:
        basis, _ = build_pod_basis(Y_train_field, X, n_modes)
        coeffs = project_onto_basis(Y_train_field, basis, X)  # (n_modes, n_samples)

        x_raw = np.stack([mu1, mu2, mu_u], axis=1)
        y_raw = coeffs.T
        x_stats = compute_minmax_stats(x_raw)
        y_stats = compute_standard_stats(y_raw)
        x_train = torch.tensor(normalize_minmax(x_raw, x_stats), dtype=torch.float32)
        y_train = torch.tensor(normalize_standard(y_raw, y_stats), dtype=torch.float32)

        net = FFNN(input_dim=3, output_dim=n_modes, hidden_dim=hidden_dim, n_hidden_layers=n_hidden_layers)
        train_ffnn(net, x_train, y_train, epochs=epochs, lr=lr, lr_drop_epoch=epochs // 2, print_every=epochs + 1)

        x_test_raw = np.stack([mu1_test, mu2_test, mu_u_test], axis=1)
        x_test = torch.tensor(normalize_minmax(x_test_raw, x_stats), dtype=torch.float32)
        with torch.no_grad():
            coeffs_pred = denormalize_standard(net(x_test).numpy(), y_stats)
        Y_pred = basis @ coeffs_pred.T

        diff = Y_pred - Y_test_field
        err_sq = np.sum(diff * (X @ diff), axis=0)
        true_sq = np.sum(Y_test_field * (X @ Y_test_field), axis=0)
        rel_err = np.sqrt(np.abs(err_sq)) / np.where(np.sqrt(np.abs(true_sq)) > 0, np.sqrt(np.abs(true_sq)), 1.0)
        results[field] = rel_err.mean()

    return results


def main():
    args = parse_args()
    values = [int(v) for v in args.values.split(",")]

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print(f"Caricamento mesh da {config['mesh']['path']} ...")
    mesh_data = load_mesh(config["mesh"]["path"], config["boundary_markers"])
    operators = assemble_operators(mesh_data, config["problem"]["omega_obs"])

    if args.inner_product == "seminorm":
        X = operators["A_diff"]
    else:
        X = operators["A_diff"] + assemble_full_mass_matrix(mesh_data)

    print(f"Caricamento snapshot da {args.snapshots} ...")
    data = np.load(args.snapshots)
    Y, P = data["Y"], data["P"]
    mu1, mu2, mu_u = data["mu1"], data["mu2"], data["mu_u"]

    print(f"Caricamento test set da {args.test_snapshots} ...")
    test_data = np.load(args.test_snapshots)
    Y_test, P_test = test_data["Y"], test_data["P"]
    mu1_test, mu2_test, mu_u_test = test_data["mu1"], test_data["mu2"], test_data["mu_u"]

    rows = []
    for i, value in enumerate(values):
        print(f"\n=== [{i + 1}/{len(values)}] {args.sweep_param} = {value} ===")
        start = time.time()

        if args.sweep_param == "n_modes":
            n_modes, hidden_dim = value, args.hidden_dim
        else:
            n_modes, hidden_dim = args.n_modes, value

        errs = train_and_evaluate(
            Y, P, Y_test, P_test, X, n_modes, hidden_dim, args.n_hidden_layers,
            mu1, mu2, mu_u, mu1_test, mu2_test, mu_u_test, args.epochs, args.lr,
        )
        elapsed = time.time() - start
        print(f"  err_y={errs['y']:.4e}  err_p={errs['p']:.4e}  ({elapsed:.1f}s)")
        rows.append({args.sweep_param: value, "err_y": errs["y"], "err_p": errs["p"]})

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[args.sweep_param, "err_y", "err_p"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nRisultati salvati in {args.output}")

    plot_path = str(Path(args.output).with_suffix(".png"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    param_values = [r[args.sweep_param] for r in rows]
    ax.semilogy(param_values, [r["err_y"] for r in rows], "o-", label="Stato (y)")
    ax.semilogy(param_values, [r["err_p"] for r in rows], "s-", label="Aggiunto (p)")
    ax.set_xlabel(args.sweep_param)
    ax.set_ylabel(f"errore relativo ({args.inner_product})")
    ax.set_title(f"Sweep {args.sweep_param}")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=80)
    print(f"Plot salvato in {plot_path}")


if __name__ == "__main__":
    main()
