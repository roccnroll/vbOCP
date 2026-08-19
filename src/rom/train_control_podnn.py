"""CLI: allena la PODNN per il controllo u (traccia 1D sul bordo Gamma_C/Gamma_N).

Pipeline (pattern del notebook del prof, Lab9/PODnn.ipynb):
1. estrae la traccia di u sui nodi di bordo, mascherata a zero fuori Gamma_C
2. base POD sulla traccia (prodotto scalare euclideo, curva 1D)
3. proiezione di Galerkin degli snapshot sulla base (target di training)
4. FFNN mu -> coefficienti proiettati

Uso:
    python -m src.rom.train_control_podnn --config configs/test1.yaml \
        --snapshots data/snapshots/test1_300.npz \
        --output data/snapshots/test1_control_podnn.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.full_order.mesh import load_mesh
from src.full_order.assembly import assemble_operators
from src.rom.control import extract_boundary_control_trace
from src.rom.pod import compute_correlation_eigenvalues, select_n_modes, build_pod_basis, project_onto_basis
from src.dl.common import FFNN, train_ffnn


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--snapshots", required=True)
    parser.add_argument("--energy-threshold", type=float, default=0.9999)
    parser.add_argument("--max-modes", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=20000)
    parser.add_argument("--output", required=True, help="path del .npz con base, coefficienti, pesi rete")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print(f"Caricamento mesh da {config['mesh']['path']} ...")
    mesh_data = load_mesh(config["mesh"]["path"], config["boundary_markers"])
    operators = assemble_operators(mesh_data, config["problem"]["omega_obs"])
    node_to_dof = operators["node_to_dof"]

    print(f"Caricamento snapshot da {args.snapshots} ...")
    data = np.load(args.snapshots)
    mu1, mu2, mu_u, U = data["mu1"], data["mu2"], data["mu_u"], data["U"]

    print("Estrazione traccia di controllo sul bordo ...")
    boundary_x, U_boundary = extract_boundary_control_trace(mesh_data, node_to_dof, U, mu_u)
    print(f"Traccia: {U_boundary.shape[0]} nodi di bordo, {U_boundary.shape[1]} snapshot")

    # prodotto scalare euclideo: curva 1D, nessuna matrice FEM da assemblare
    inner_product = np.eye(U_boundary.shape[0])

    eigenvalues, _ = compute_correlation_eigenvalues(U_boundary, inner_product)
    n_modes = select_n_modes(eigenvalues, args.energy_threshold, args.max_modes)
    print(f"N modi scelto: {n_modes}")

    basis, _ = build_pod_basis(U_boundary, inner_product, n_modes)
    coeffs = project_onto_basis(U_boundary, basis, inner_product)  # (n_modes, n_samples)

    print("Training FFNN mu -> coefficienti POD(u) ...")
    x_train = torch.tensor(np.stack([mu1, mu2, mu_u], axis=1), dtype=torch.float32)
    y_train = torch.tensor(coeffs.T, dtype=torch.float32)  # (n_samples, n_modes)

    net = FFNN(input_dim=3, output_dim=n_modes)
    train_ffnn(net, x_train, y_train, epochs=args.epochs, lr_drop_epoch=args.epochs // 2)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        boundary_x=boundary_x, basis=basis, n_modes=n_modes,
    )
    torch.save(net.state_dict(), str(Path(args.output).with_suffix(".pt")))
    print(f"Base + coefficienti salvati in {args.output}")
    print(f"Pesi rete salvati in {Path(args.output).with_suffix('.pt')}")


if __name__ == "__main__":
    main()
