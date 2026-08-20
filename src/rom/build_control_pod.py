"""CLI: costruisce la base POD del controllo u (traccia 1D sul bordo Gamma_C/Gamma_N).

Solo la fase POD - separata dal training della rete (train_reduced_nn.py)
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
    plot_eigenvalue_decay_with_error, compute_reconstruction_error_curve,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--snapshots", required=True)
    parser.add_argument("--test-snapshots", default=None,
                         help="opzionale: se dato, il plot include anche l'errore di ricostruzione "
                              "solo-POD su questo set, accanto al decadimento autovalori")
    parser.add_argument("--energy-threshold", type=float, default=0.9999)
    parser.add_argument("--max-modes", type=int, default=50)
    parser.add_argument("--no-normalize-correlation", action="store_true",
                         help="non dividere la matrice di correlazione per M (eq. 16 invece di 17 del "
                              "paper) - non cambia la base ne' N scelto, solo la scala di lambda nel plot")
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

    normalize_correlation = not args.no_normalize_correlation
    eigenvalues, eigenvectors = compute_correlation_eigenvalues(U_boundary, inner_product, normalize_correlation)
    n_modes = select_n_modes(eigenvalues, args.energy_threshold, args.max_modes)
    print(f"N modi scelto: {n_modes}")

    if args.plot:
        import tempfile
        default_path = str(Path(tempfile.gettempdir()) / "control_pod_eigenvalue_decay.png")

        if args.test_snapshots is not None:
            print(f"Caricamento test set da {args.test_snapshots} per l'errore di ricostruzione ...")
            test_data = np.load(args.test_snapshots)
            mu_u_test, U_test = test_data["mu_u"], test_data["U"]
            boundary_x_test, U_boundary_test = extract_boundary_control_trace(
                mesh_data, node_to_dof, U_test, mu_u_test)
            if not np.allclose(boundary_x, boundary_x_test):
                raise ValueError("I nodi di bordo di training e test non coincidono - mesh diversa?")

            max_n = min(80, len(eigenvalues))
            n_values = range(1, max_n + 1)
            err_u = compute_reconstruction_error_curve(
                U_boundary, U_boundary_test, inner_product, eigenvectors, n_values)

            plot_eigenvalue_decay_with_error(
                {"Controllo (u)": eigenvalues}, {"Controllo (u)": err_u},
                output_path=default_path, max_n=max_n,
                title="Decadimento autovalori POD - controllo",
            )
        else:
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

    basis, _ = build_pod_basis(U_boundary, inner_product, n_modes, normalize_correlation)
    coeffs = project_onto_basis(U_boundary, basis, inner_product)  # (n_modes, n_samples)

    # nomi delle chiavi con suffisso _u, stesso schema di stato/aggiunto (basis_y/coeffs_y/n_modes_y
    # in train_pod.py) - permette di riusare uno script di training generico per y, p, u
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        boundary_x=boundary_x, basis_u=basis, n_modes_u=n_modes,
        coeffs_u=coeffs, mu1=mu1, mu2=mu2, mu_u=mu_u,
    )
    print(f"Base POD + coefficienti + mu di training salvati in {args.output}")


if __name__ == "__main__":
    main()
