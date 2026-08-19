"""CLI: costruisce la base POD del controllo u (traccia 1D sul bordo Gamma_C/Gamma_N).

Solo la fase POD - separata dal training della rete (train_control_nn.py)
cosi' i parametri delle due fasi si controllano indipendentemente.

Uso:
    python -m src.rom.build_control_pod --config configs/test1.yaml \
        --snapshots data/snapshots/test1_300.npz \
        --output data/snapshots/test1_control_pod.npz
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
from src.rom.pod import (
    compute_correlation_eigenvalues, select_n_modes, build_pod_basis,
    project_onto_basis, plot_eigenvalue_decay_curves,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--snapshots", required=True)
    parser.add_argument("--energy-threshold", type=float, default=0.9999)
    parser.add_argument("--max-modes", type=int, default=50)
    parser.add_argument("--output", required=True, help="path del .npz con base, coefficienti, mu di training")
    parser.add_argument("--plot", action="store_true",
                         help="mostra il plot di decadimento autovalori (salvato in un path di default)")
    parser.add_argument("--save-plot", default=None,
                         help="path dove salvare permanentemente il plot (opzionale, oltre a --plot)")
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

    if args.plot:
        import tempfile
        default_path = str(Path(tempfile.gettempdir()) / "control_pod_eigenvalue_decay.png")
        plot_eigenvalue_decay_curves(
            {"Controllo (u)": eigenvalues}, output_path=default_path,
            title="Decadimento autovalori POD - controllo",
        )
        print(f"PLOT_PATH={default_path}")
        if args.save_plot:
            import shutil
            Path(args.save_plot).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(default_path, args.save_plot)
            print(f"Plot salvato anche in {args.save_plot}")

    basis, _ = build_pod_basis(U_boundary, inner_product, n_modes)
    coeffs = project_onto_basis(U_boundary, basis, inner_product)  # (n_modes, n_samples)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        boundary_x=boundary_x, basis=basis, n_modes=n_modes,
        coeffs=coeffs, mu1=mu1, mu2=mu2, mu_u=mu_u,
    )
    print(f"Base POD + coefficienti + mu di training salvati in {args.output}")


if __name__ == "__main__":
    main()
