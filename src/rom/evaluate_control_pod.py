"""CLI: valuta l'errore di ricostruzione della SOLA base POD del controllo (senza rete).

Proietta gli snapshot di test sulla base (proiezione di Galerkin, il
miglior risultato possibile con N modi) e calcola l'errore di
ricostruzione - isola quanto e' brava la base POD da quanto impara bene
la FFNN (vedi evaluate_control_podnn.py per l'errore end-to-end con NN).

Uso:
    python -m src.rom.evaluate_control_pod --config configs/test1.yaml \
        --pod-model data/snapshots/test1_control_pod.npz \
        --test-snapshots data/snapshots/test1_test150.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.full_order.mesh import load_mesh
from src.full_order.assembly import assemble_operators
from src.rom.control import extract_boundary_control_trace
from src.rom.pod import project_onto_basis


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--pod-model", required=True, help="path al .npz da build_control_pod.py")
    parser.add_argument("--test-snapshots", required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print(f"Caricamento mesh da {config['mesh']['path']} ...")
    mesh_data = load_mesh(config["mesh"]["path"], config["boundary_markers"])
    operators = assemble_operators(mesh_data, config["problem"]["omega_obs"])
    node_to_dof = operators["node_to_dof"]

    print(f"Caricamento base POD da {args.pod_model} ...")
    pod_data = np.load(args.pod_model)
    basis = pod_data["basis"]

    print(f"Caricamento test set da {args.test_snapshots} ...")
    test_data = np.load(args.test_snapshots)
    mu_u = test_data["mu_u"]
    U_true = test_data["U"]

    print("Estrazione traccia di controllo vera (test set) ...")
    boundary_x_test, U_boundary_true = extract_boundary_control_trace(mesh_data, node_to_dof, U_true, mu_u)

    if not np.allclose(boundary_x_test, pod_data["boundary_x"]):
        raise ValueError("I nodi di bordo del test set non coincidono con quelli del modello - mesh diversa?")

    inner_product = np.eye(U_boundary_true.shape[0])

    print("Proiezione di Galerkin sulla base (miglior ricostruzione possibile con N modi) ...")
    coeffs = project_onto_basis(U_boundary_true, basis, inner_product)  # (n_modes, n_samples)
    U_boundary_reconstructed = basis @ coeffs

    errors = np.linalg.norm(U_boundary_reconstructed - U_boundary_true, axis=0)
    norms = np.linalg.norm(U_boundary_true, axis=0)
    relative_errors = errors / np.where(norms > 0, norms, 1.0)

    print(f"N modi: {basis.shape[1]}")
    print(f"Errore relativo medio (solo POD, senza rete): {relative_errors.mean():.4e}")
    print(f"Errore relativo mediano: {np.median(relative_errors):.4e}")
    print(f"Errore relativo max: {relative_errors.max():.4e}")


if __name__ == "__main__":
    main()
