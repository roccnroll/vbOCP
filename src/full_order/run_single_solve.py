"""CLI di test: carica un config, risolve il FOM per una singola terna di parametri.

Uso:
    python -m src.full_order.run_single_solve --config configs/test1.yaml --mu1 12 --mu2 2.5 --mu_u 0.99
"""
import argparse
import sys
from pathlib import Path

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
    parser.add_argument("--mu1", type=float, default=12.0)
    parser.add_argument("--mu2", type=float, default=2.5)
    parser.add_argument("--mu_u", type=float, default=0.99)
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    mesh_dir = config["mesh"]["path"]
    boundary_markers = config["boundary_markers"]
    omega_obs_regions = config["problem"]["omega_obs"]
    alpha = config["problem"]["alpha"]

    print(f"Caricamento mesh da {mesh_dir} ...")
    mesh_data = load_mesh(mesh_dir, boundary_markers)
    print(f"Nh = {mesh_data['Nh']}")

    print("Assembly operatori (indipendenti da mu) ...")
    operators = assemble_operators(mesh_data, omega_obs_regions)
    dirichlet_data = assemble_dirichlet_and_source(mesh_data, omega_obs_regions)

    print(f"Assembly matrice di controllo C(mu_u={args.mu_u}) ...")
    C = assemble_control_matrix(mesh_data, operators["node_to_dof"], args.mu_u)

    print(f"Solve OTD per mu1={args.mu1}, mu2={args.mu2}, mu_u={args.mu_u} ...")
    y, p, u = solve_otd(operators, C, dirichlet_data, args.mu1, args.mu2, alpha)

    print("Soluzione OTD:")
    print(f"  stato     y: min={y.min():.4f}, max={y.max():.4f}")
    print(f"  aggiunto  p: min={p.min():.4f}, max={p.max():.4f}")
    print(f"  controllo u: min={u.min():.4f}, max={u.max():.4f}")


if __name__ == "__main__":
    main()
