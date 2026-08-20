"""CLI: errore di ricostruzione della SOLA POD (proiezione di Galerkin) al variare di N,
con plot combinato contro un CSV di PODNN gia' calcolato (src/rom/sweep_pod_nn.py).

Nessun training qui - solo algebra lineare, quindi veloce anche per molti
valori di N. Se gli viene passato --podnn-csv (il csv gia' prodotto da
sweep_pod_nn.py, NON rifatto qui), il plot combina le due curve (solo POD
vs PODNN) sulla stessa griglia di N - il confronto va fatto sulla stessa
lista di N passata a entrambi gli script, ma il training della PODNN resta
un run separato e non viene mai ripetuto da questo script.

Uso (solo curva POD):
    python -m src.rom.sweep_pod_error --config configs/test1.yaml \
        --snapshots data/snapshots/test1_300.npz \
        --test-snapshots data/snapshots/test1_test150.npz \
        --values 5,7,9,11,13,15,17,20 \
        --output data/snapshots/sweep_pod_error.csv

Uso (con confronto PODNN, gia' calcolato separatamente con sweep_pod_nn.py
sulla STESSA lista di --values):
    python -m src.rom.sweep_pod_error --config configs/test1.yaml \
        --snapshots data/snapshots/test1_300.npz \
        --test-snapshots data/snapshots/test1_test150.npz \
        --values 5,7,9,11,13,15,17,20 \
        --podnn-csv data/snapshots/sweep_n_modes.csv \
        --output data/snapshots/sweep_pod_error.csv
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.full_order.mesh import load_mesh
from src.full_order.assembly import assemble_operators
from src.rom.inner_product import assemble_full_mass_matrix
from src.rom.pod import build_pod_basis, project_onto_basis


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--snapshots", required=True, help="snapshot di training (per costruire la base)")
    parser.add_argument("--test-snapshots", required=True)
    parser.add_argument("--inner-product", choices=["seminorm", "full"], default="seminorm")
    parser.add_argument("--values", required=True, help="valori di N separati da virgola")
    parser.add_argument("--podnn-csv", default=None,
                         help="opzionale: csv gia' prodotto da sweep_pod_nn.py (colonne n_modes, err_y, "
                              "err_p) sulla STESSA lista di --values - se dato, il plot combina le due "
                              "curve; il training della PODNN NON viene rifatto qui")
    parser.add_argument("--output", required=True, help="path del .csv con i risultati")
    return parser.parse_args()


def relative_error(true, pred, norm_matrix):
    """Errore relativo per campione in norma indotta da norm_matrix (sparse-safe)."""
    diff = pred - true
    err_sq = np.sum(diff * (norm_matrix @ diff), axis=0)
    true_sq = np.sum(true * (norm_matrix @ true), axis=0)
    errors = np.sqrt(np.abs(err_sq))
    norms = np.sqrt(np.abs(true_sq))
    return errors / np.where(norms > 0, norms, 1.0)


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


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

    print(f"Caricamento test set da {args.test_snapshots} ...")
    test_data = np.load(args.test_snapshots)
    Y_test, P_test = test_data["Y"], test_data["P"]

    rows = []
    for n_modes in values:
        basis_y, _ = build_pod_basis(Y, X, n_modes)
        basis_p, _ = build_pod_basis(P, X, n_modes)

        coeffs_y_test = project_onto_basis(Y_test, basis_y, X)
        coeffs_p_test = project_onto_basis(P_test, basis_p, X)

        Y_reconstructed = basis_y @ coeffs_y_test
        P_reconstructed = basis_p @ coeffs_p_test

        err_y = relative_error(Y_test, Y_reconstructed, X).mean()
        err_p = relative_error(P_test, P_reconstructed, X).mean()

        print(f"N={n_modes}  err_y_pod={err_y:.4e}  err_p_pod={err_p:.4e}")
        rows.append({"n_modes": n_modes, "err_y_pod": err_y, "err_p_pod": err_p})

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["n_modes", "err_y_pod", "err_p_pod"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nRisultati salvati in {args.output}")

    plot_path = str(Path(args.output).with_suffix(".png"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    podnn_rows = None
    if args.podnn_csv is not None:
        podnn_rows = read_csv(args.podnn_csv)
        n_podnn = [int(r["n_modes"]) for r in podnn_rows]
        if n_podnn != values:
            print(f"ATTENZIONE: --values ({values}) diverso da n_modes nel podnn-csv ({n_podnn}) - "
                  f"il confronto sul grafico potrebbe non essere sulla stessa griglia di N.")

    if podnn_rows is not None:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
        n_podnn = [int(r["n_modes"]) for r in podnn_rows]

        axes[0].semilogy(values, [r["err_y_pod"] for r in rows], "o--", label="solo POD")
        axes[0].semilogy(n_podnn, [float(r["err_y"]) for r in podnn_rows], "s-", label="PODNN")
        axes[0].set_title("Stato (y)")
        axes[0].set_xlabel("N modi")
        axes[0].set_ylabel(f"errore relativo ({args.inner_product})")
        axes[0].legend()
        axes[0].grid(True, which="both", alpha=0.3)

        axes[1].semilogy(values, [r["err_p_pod"] for r in rows], "o--", label="solo POD")
        axes[1].semilogy(n_podnn, [float(r["err_p"]) for r in podnn_rows], "s-", label="PODNN")
        axes[1].set_title("Aggiunto (p)")
        axes[1].set_xlabel("N modi")
        axes[1].legend()
        axes[1].grid(True, which="both", alpha=0.3)
    else:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.semilogy(values, [r["err_y_pod"] for r in rows], "o-", label="Stato (y) - solo POD")
        ax.semilogy(values, [r["err_p_pod"] for r in rows], "s-", label="Aggiunto (p) - solo POD")
        ax.set_xlabel("N modi")
        ax.set_ylabel(f"errore relativo ricostruzione ({args.inner_product})")
        ax.set_title("Errore di ricostruzione POD (proiezione di Galerkin, senza rete)")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=80)
    print(f"Plot salvato in {plot_path}")


if __name__ == "__main__":
    main()
