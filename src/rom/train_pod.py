"""CLI: costruisce la base POD (stato e aggiunto) da uno snapshot .npz.

N scelto automaticamente in base a una soglia di energia cumulata,
indipendentemente per stato e aggiunto (non piu' forzati allo stesso N -
approccio non intrusivo, ogni base ha la propria dimensione naturale).

Uso:
    python -m src.rom.train_pod --config configs/test1.yaml \
        --snapshots data/snapshots/test1_300.npz \
        --inner-product seminorm --energy-threshold 0.9999 \
        --output data/snapshots/test1_pod.npz --plot
"""
import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.full_order.mesh import load_mesh
from src.full_order.assembly import assemble_operators
from src.rom.inner_product import assemble_full_mass_matrix
from src.rom.pod import (
    build_pod_basis, compute_correlation_eigenvalues, select_n_modes,
    plot_eigenvalue_decay, project_onto_basis,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path al file YAML del caso")
    parser.add_argument("--snapshots", required=True, help="path allo snapshot .npz")
    parser.add_argument("--inner-product", choices=["seminorm", "full"], default="seminorm",
                         help="seminorm = solo A_diff (H1-seminorma); full = A_diff + M_full (H1 completa)")
    parser.add_argument("--energy-threshold", type=float, default=0.9999,
                         help="soglia di energia cumulata relativa per scegliere N automaticamente")
    parser.add_argument("--max-modes", type=int, default=150,
                         help="tetto massimo di modi, anche se la soglia non e' raggiunta")
    parser.add_argument("--output", required=True, help="path del .npz con basi e autovalori")
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

    if args.inner_product == "seminorm":
        X = operators["A_diff"]
    else:
        M_full = assemble_full_mass_matrix(mesh_data)
        X = operators["A_diff"] + M_full
    print(f"Prodotto scalare: {args.inner_product}")

    print(f"Caricamento snapshot da {args.snapshots} ...")
    data = np.load(args.snapshots)
    Y, P = data["Y"], data["P"]
    mu1, mu2, mu_u = data["mu1"], data["mu2"], data["mu_u"]
    print(f"Y shape: {Y.shape}, P shape: {P.shape}")

    # autovalori (non dipendono da N): servono per scegliere N e per il plot
    eig_y, _ = compute_correlation_eigenvalues(Y, X)
    eig_p, _ = compute_correlation_eigenvalues(P, X)

    n_y = select_n_modes(eig_y, args.energy_threshold, args.max_modes)
    n_p = select_n_modes(eig_p, args.energy_threshold, args.max_modes)
    print(f"N scelto: stato={n_y}, aggiunto={n_p} (indipendenti)")

    basis_y, _ = build_pod_basis(Y, X, n_y)
    basis_p, _ = build_pod_basis(P, X, n_p)

    # proiezione di Galerkin degli snapshot sulle basi (target di training per la PODNN,
    # stesso pattern gia' usato per il controllo)
    coeffs_y = project_onto_basis(Y, basis_y, X)  # (n_y, n_samples)
    coeffs_p = project_onto_basis(P, basis_p, X)  # (n_p, n_samples)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        basis_y=basis_y, basis_p=basis_p,
        eigenvalues_y=eig_y, eigenvalues_p=eig_p,
        coeffs_y=coeffs_y, coeffs_p=coeffs_p,
        mu1=mu1, mu2=mu2, mu_u=mu_u,
        inner_product=args.inner_product, n_modes_y=n_y, n_modes_p=n_p,
    )
    print(f"Base POD salvata in {args.output}")

    if args.plot:
        # --plot da solo: salva in un path di default (non serve specificarlo)
        # --save-plot: copia aggiuntiva in un path scelto, se la vuoi tenere
        default_path = str(Path(tempfile.gettempdir()) / "pod_eigenvalue_decay.png")
        plot_eigenvalue_decay(eig_y, eig_p, output_path=default_path)
        print(f"PLOT_PATH={default_path}")

        if args.save_plot:
            import shutil
            Path(args.save_plot).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(default_path, args.save_plot)
            print(f"Plot salvato anche in {args.save_plot}")


if __name__ == "__main__":
    main()
