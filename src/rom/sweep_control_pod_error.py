"""CLI: errore di ricostruzione della SOLA POD del controllo (traccia 1D sul bordo) al variare di N.

Stesso pattern di sweep_pod_error.py (stato/aggiunto), applicato alla traccia 1D
del controllo u estratta da extract_boundary_control_trace - nessun training,
solo proiezione di Galerkin (prodotto scalare euclideo, coerente con
build_control_pod.py).

Uso:
    python -m src.rom.sweep_control_pod_error --config configs/test1.yaml \
        --snapshots data/snapshots/test1_300.npz \
        --test-snapshots data/snapshots/test1_test150.npz \
        --values 1,2,3,5,7,10,15,20 \
        --output data/snapshots/sweep_control_pod_error.csv
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
from src.rom.control import extract_boundary_control_trace
from src.rom.pod import build_pod_basis, project_onto_basis


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--snapshots", required=True, help="snapshot di training (per costruire la base)")
    parser.add_argument("--test-snapshots", required=True)
    parser.add_argument("--values", required=True, help="valori di N separati da virgola")
    parser.add_argument("--build-markers", default="8,10",
                         help="marker dei lati usati per COSTRUIRE la base (default 8,10 - entrambi i bordi "
                              "simmetrici; usare es. 8 per costruirla solo su un bordo)")
    parser.add_argument("--eval-markers", default=None,
                         help="marker dei lati su cui VALUTARE la ricostruzione (default: uguale a "
                              "--build-markers). Usare per testare se una base costruita su un bordo "
                              "ricostruisce bene anche l'altro (verifica della simmetria)")
    parser.add_argument("--output", required=True, help="path del .csv con i risultati")
    return parser.parse_args()


def relative_error(true, pred):
    """Errore relativo per campione in norma euclidea (coerente con build_control_pod.py)."""
    errors = np.linalg.norm(pred - true, axis=0)
    norms = np.linalg.norm(true, axis=0)
    return errors / np.where(norms > 0, norms, 1.0)


def main():
    args = parse_args()
    values = [int(v) for v in args.values.split(",")]
    build_markers = tuple(int(m) for m in args.build_markers.split(","))
    eval_markers = (tuple(int(m) for m in args.eval_markers.split(","))
                     if args.eval_markers is not None else build_markers)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print(f"Caricamento mesh da {config['mesh']['path']} ...")
    mesh_data = load_mesh(config["mesh"]["path"], config["boundary_markers"])
    operators = assemble_operators(mesh_data, config["problem"]["omega_obs"])
    node_to_dof = operators["node_to_dof"]

    print(f"Caricamento snapshot da {args.snapshots} ...")
    data = np.load(args.snapshots)
    mu_u, U = data["mu_u"], data["U"]
    boundary_x_build, U_boundary = extract_boundary_control_trace(
        mesh_data, node_to_dof, U, mu_u, control_markers=build_markers)
    print(f"Base costruita su marker {build_markers}  nodi di bordo: {U_boundary.shape[0]}")

    print(f"Caricamento test set da {args.test_snapshots} ...")
    test_data = np.load(args.test_snapshots)
    mu_u_test, U_test = test_data["mu_u"], test_data["U"]
    boundary_x_eval, U_boundary_test = extract_boundary_control_trace(
        mesh_data, node_to_dof, U_test, mu_u_test, control_markers=eval_markers)
    print(f"Valutata su marker {eval_markers}  nodi di bordo: {U_boundary_test.shape[0]}")

    # richiede stesse coordinate x (stesso numero e ordine di nodi) tra base e valutazione -
    # vale per costruzione se i due marker sono bordi simmetrici sulla stessa mesh (stesso range x)
    if not np.allclose(boundary_x_build, boundary_x_eval):
        raise ValueError(
            "I nodi di bordo usati per costruire la base e quelli su cui si valuta non coincidono "
            "(coordinate x diverse) - la proiezione non avrebbe senso nodo per nodo."
        )

    inner_product = np.eye(U_boundary.shape[0])

    rows = []
    for n_modes in values:
        basis, _ = build_pod_basis(U_boundary, inner_product, n_modes)
        coeffs_test = project_onto_basis(U_boundary_test, basis, inner_product)
        U_reconstructed = basis @ coeffs_test

        err = relative_error(U_boundary_test, U_reconstructed).mean()
        print(f"N={n_modes}  err_u_pod={err:.4e}")
        rows.append({"n_modes": n_modes, "err_u_pod": err})

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["n_modes", "err_u_pod"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nRisultati salvati in {args.output}")

    plot_path = str(Path(args.output).with_suffix(".png"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(values, [r["err_u_pod"] for r in rows], "o-", label="Controllo (u) - solo POD")
    ax.set_xlabel("N modi")
    ax.set_ylabel("errore relativo ricostruzione (euclidea)")
    ax.set_title("Errore di ricostruzione POD del controllo (bordo 1D)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=80)
    print(f"Plot salvato in {plot_path}")


if __name__ == "__main__":
    main()
