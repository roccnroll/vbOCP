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
    parser.add_argument("--plot", action="store_true", help="salva un plot PNG di stato e aggiunto")
    parser.add_argument("--output", default="fom_solution.png", help="path del PNG (solo se --plot)")
    return parser.parse_args()


def plot_solution(mesh_data, operators, dirichlet_data, y, p, mu1, mu2, mu_u, output_path):
    """Riporta y/p su tutti i nodi e salva un plot stato+aggiunto su disco."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")  # nessun display richiesto, solo salvataggio su file
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from pypolydim import polydim

    mesh = mesh_data["mesh"]
    trial_dofs_data = mesh_data["trial_dofs_data"]
    assemble = polydim.pde_tools.assembler_utilities.pcc_2_d

    sol_y = assemble.extract_solution_on_cell0_ds(mesh, trial_dofs_data, y, dirichlet_data["u_D_y"])
    sol_p = assemble.extract_solution_on_cell0_ds(mesh, trial_dofs_data, p, dirichlet_data["u_D_p"])
    y_plot = sol_y.numeric_solution
    p_plot = sol_p.numeric_solution

    x_nodes = np.array([mesh.cell0_d_coordinate_x(i) for i in range(mesh.cell0_d_total_number())])
    y_nodes = np.array([mesh.cell0_d_coordinate_y(i) for i in range(mesh.cell0_d_total_number())])
    triangles = np.array([
        [mesh.cell2_d_vertex(t, 0), mesh.cell2_d_vertex(t, 1), mesh.cell2_d_vertex(t, 2)]
        for t in range(mesh.cell2_d_total_number())
    ])
    triang = mtri.Triangulation(x_nodes, y_nodes, triangles)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    ax = axes[0]
    tc = ax.tricontourf(triang, y_plot, levels=200, cmap="jet")
    plt.colorbar(tc, ax=ax, label="y")
    ax.set_title(f"Stato y - mu1={mu1}, mu2={mu2}, mu_u={mu_u}")
    ax.set_aspect("equal")

    ax = axes[1]
    tc = ax.tricontourf(triang, p_plot, levels=200, cmap="jet")
    plt.colorbar(tc, ax=ax, label="p")
    ax.set_title(f"Aggiunto p - mu1={mu1}, mu2={mu2}, mu_u={mu_u}")
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Plot salvato in {output_path}")


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

    if args.plot:
        plot_solution(mesh_data, operators, dirichlet_data, y, p,
                      args.mu1, args.mu2, args.mu_u, args.output)


if __name__ == "__main__":
    main()
