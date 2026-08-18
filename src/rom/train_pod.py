"""CLI: costruisce la base POD (stato e aggiunto) da uno snapshot .npz.

Uso:
    python -m src.rom.train_pod --config configs/test1.yaml \
        --snapshots data/snapshots/test1_300.npz \
        --inner-product seminorm --n-modes 30 \
        --output data/snapshots/test1_pod.npz --plot
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.full_order.mesh import load_mesh
from src.full_order.assembly import assemble_operators
from src.rom.inner_product import assemble_full_mass_matrix
from src.rom.pod import build_pod_basis, plot_eigenvalue_decay


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path al file YAML del caso")
    parser.add_argument("--snapshots", required=True, help="path allo snapshot .npz")
    parser.add_argument("--inner-product", choices=["seminorm", "full"], default="seminorm",
                         help="seminorm = solo A_diff (H1-seminorma); full = A_diff + M_full (H1 completa)")
    parser.add_argument("--n-modes", type=int, required=True, help="numero di modi POD da tenere")
    parser.add_argument("--output", required=True, help="path del .npz con basi e autovalori")
    parser.add_argument("--plot", action="store_true", help="salva anche il plot di decadimento autovalori")
    parser.add_argument("--plot-output", default="pod_eigenvalue_decay.png", help="path del PNG (solo se --plot)")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print(f"Caricamento mesh da {config['mesh']['path']} ...")
    mesh_data = load_mesh(config["mesh"]["path"], config["boundary_markers"])
    operators = assemble_operators(mesh_data, config["problem"]["omega_obs"])

    if args.inner_product == "seminorm":
        X = operators["A_diff"]
    else:
        M_full = assemble_full_mass_matrix(mesh_data)
        X = operators["A_diff"] + M_full
    print(f"Prodotto scalare: {args.inner_product}")

    print(f"Caricamento snapshot da {args.snapshots} ...")
    data = np.load(args.snapshots)
    Y, P = data["Y"], data["P"]
    print(f"Y shape: {Y.shape}, P shape: {P.shape}")

    print(f"Costruzione base POD (N={args.n_modes}) ...")
    basis_y, eig_y = build_pod_basis(Y, X, args.n_modes)
    basis_p, eig_p = build_pod_basis(P, X, args.n_modes)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        basis_y=basis_y, basis_p=basis_p,
        eigenvalues_y=eig_y, eigenvalues_p=eig_p,
        inner_product=args.inner_product, n_modes=args.n_modes,
    )
    print(f"Base POD salvata in {args.output}")

    if args.plot:
        plot_eigenvalue_decay(eig_y, eig_p, output_path=args.plot_output)


if __name__ == "__main__":
    main()
