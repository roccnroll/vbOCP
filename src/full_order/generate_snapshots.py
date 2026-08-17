"""CLI: genera N snapshot FOM campionando mu1, mu2, mu_u uniformemente dai range del config.

Uso:
    python -m src.full_order.generate_snapshots --config configs/test1.yaml --n-samples 300 --output data/snapshots/test1_300.npz
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.full_order.mesh import load_mesh
from src.full_order.assembly import (
    assemble_operators,
    assemble_control_matrix,
    assemble_dirichlet_and_source,
)
from src.full_order.solve import solve_otd


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path al file YAML del caso (es. configs/test1.yaml)")
    parser.add_argument("--n-samples", type=int, required=True, help="numero di snapshot da generare")
    parser.add_argument("--seed", type=int, default=42, help="seed per il campionamento casuale")
    parser.add_argument("--output", required=True, help="path del file .npz di output")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    mesh_dir = config["mesh"]["path"]
    boundary_markers = config["boundary_markers"]
    omega_obs_regions = config["problem"]["omega_obs"]
    alpha = config["problem"]["alpha"]
    p = config["parameters"]

    # campionamento uniforme casuale sui range del config (stesso approccio del paper)
    rng = np.random.default_rng(args.seed)
    mu1_samples = rng.uniform(p["mu1"]["min"], p["mu1"]["max"], args.n_samples)
    mu2_samples = rng.uniform(p["mu2"]["min"], p["mu2"]["max"], args.n_samples)
    mu_u_samples = rng.uniform(p["mu_u"]["min"], p["mu_u"]["max"], args.n_samples)

    print(f"Caricamento mesh da {mesh_dir} ...")
    mesh_data = load_mesh(mesh_dir, boundary_markers)
    Nh = mesh_data["Nh"]
    print(f"Nh = {Nh}")

    # assembly indipendente da mu: fatto una volta sola per tutti gli N campioni
    print("Assembly operatori (indipendenti da mu) ...")
    operators = assemble_operators(mesh_data, omega_obs_regions)
    dirichlet_data = assemble_dirichlet_and_source(mesh_data, omega_obs_regions)
    node_to_dof = operators["node_to_dof"]

    Y = np.zeros((Nh, args.n_samples))
    P = np.zeros((Nh, args.n_samples))
    U = np.zeros((Nh, args.n_samples))

    start_time = time.time()

    for i in range(args.n_samples):
        mu1, mu2, mu_u = mu1_samples[i], mu2_samples[i], mu_u_samples[i]

        # solo questi due passi dipendono da mu: assemblaggio C(mu_u) e solve OTD
        C = assemble_control_matrix(mesh_data, node_to_dof, mu_u)
        y, pp, u = solve_otd(operators, C, dirichlet_data, mu1, mu2, alpha)

        Y[:, i] = y
        P[:, i] = pp
        U[:, i] = u

        # barra di progresso su un'unica riga (\r): x/N, tempo trascorso, tempo medio per snapshot
        elapsed = time.time() - start_time
        avg = elapsed / (i + 1)
        print(f"\r  {i + 1}/{args.n_samples}  elapsed={elapsed:.1f}s  avg={avg:.2f}s/snapshot",
              end="", flush=True)

    print()  # a capo dopo l'ultima riga di progresso
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        mu1=mu1_samples, mu2=mu2_samples, mu_u=mu_u_samples,
        Y=Y, P=P, U=U,
    )
    print(f"Snapshot salvati in {args.output}")


if __name__ == "__main__":
    main()
