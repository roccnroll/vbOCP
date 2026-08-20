"""CLI: genera snapshot FOM campionando mu1, mu2, mu_u dai range del config.

Due modi di campionamento:
- random (default): uniforme casuale indipendente per ciascun parametro, --n-samples totali
- grid: griglia fattoriale equispaziata, --grid-shape n1,n2,n3 (mu1 x mu2 x mu_u) - ordine
  di iterazione con mu_u che varia piu' velocemente (stessa convenzione di
  itertools.product usata da gca-rom's pde.problem()+product(mu_space))

Uso (random):
    python -m src.full_order.generate_snapshots --config configs/test1.yaml --n-samples 300 --output data/snapshots/test1_300.npz

Uso (griglia):
    python -m src.full_order.generate_snapshots --config configs/test1.yaml --sampling grid --grid-shape 6,6,9 --output data/snapshots/test1_grid.npz
"""
import argparse
import itertools
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


def get_boundary_mu_u_candidates(mesh_data, control_markers=(8, 10)):
    """Coordinate x (traslate in mu_u = x-1) dei nodi di bordo su cui puo' cadere x_ctrl.

    Stesso pattern di scansione marker usato in assemble_control_matrix, ma qui
    serve solo per elencare i nodi candidati, non per assemblare nulla.

    Args:
        mesh_data: dict restituito da load_mesh()
        control_markers: marker dei lati dove passa la soglia x_ctrl (8, 10 per Test_1)

    Returns:
        array di mu_u candidati (x_nodo - 1), ordinati, senza duplicati
    """
    mesh = mesh_data["mesh"]

    node_coords_x = np.array([mesh.cell0_d_coordinate_x(i)
                               for i in range(mesh.cell0_d_total_number())])

    candidate_x = set()
    for e in range(mesh.cell1_d_total_number()):
        if mesh.cell1_d_marker(e) not in control_markers:
            continue
        for n in mesh.cell1_d_extremes(e):
            candidate_x.add(node_coords_x[n])

    mu_u_candidates = np.sort(np.array(sorted(candidate_x)) - 1.0)
    return mu_u_candidates


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path al file YAML del caso (es. configs/test1.yaml)")
    parser.add_argument("--sampling", choices=["random", "grid"], default="random",
                         help="random (default): uniforme casuale, --n-samples totali; "
                              "grid: griglia fattoriale equispaziata, --grid-shape n1,n2,n3")
    parser.add_argument("--n-samples", type=int, default=None,
                         help="numero di snapshot da generare (richiesto se --sampling random)")
    parser.add_argument("--grid-shape", default=None,
                         help="n1,n2,n3 - punti equispaziati per mu1,mu2,mu_u (richiesto se --sampling grid)")
    parser.add_argument("--seed", type=int, default=42, help="seed per il campionamento casuale (solo random)")
    parser.add_argument("--output", required=True, help="path del file .npz di output")
    parser.add_argument("--align-mu-u-to-mesh", action="store_true",
                         help="campiona mu_u solo dai nodi di bordo esistenti (niente inconsistenza O(h) "
                              "in assemble_control_matrix) - usare per il training set, non per il test set "
                              "(solo con --sampling random)")
    args = parser.parse_args()

    if args.sampling == "random" and args.n_samples is None:
        parser.error("--n-samples e' richiesto con --sampling random")
    if args.sampling == "grid" and args.grid_shape is None:
        parser.error("--grid-shape e' richiesto con --sampling grid")
    if args.sampling == "grid" and args.align_mu_u_to_mesh:
        parser.error("--align-mu-u-to-mesh non ha senso con --sampling grid (i punti sono gia' fissi)")

    return args


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    mesh_dir = config["mesh"]["path"]
    boundary_markers = config["boundary_markers"]
    omega_obs_regions = config["problem"]["omega_obs"]
    alpha = config["problem"]["alpha"]
    p = config["parameters"]

    print(f"Caricamento mesh da {mesh_dir} ...")
    mesh_data = load_mesh(mesh_dir, boundary_markers)
    Nh = mesh_data["Nh"]
    print(f"Nh = {Nh}")

    if args.sampling == "grid":
        n1, n2, nu = (int(v) for v in args.grid_shape.split(","))
        mu1_grid = np.linspace(p["mu1"]["min"], p["mu1"]["max"], n1)
        mu2_grid = np.linspace(p["mu2"]["min"], p["mu2"]["max"], n2)
        mu_u_grid = np.linspace(p["mu_u"]["min"], p["mu_u"]["max"], nu)

        # mu_u varia piu' velocemente (ultimo indice) - stessa convenzione di
        # itertools.product(*mu_space) usata dal tutorial di gca-rom
        grid = np.array(list(itertools.product(mu1_grid, mu2_grid, mu_u_grid)))
        mu1_samples, mu2_samples, mu_u_samples = grid[:, 0], grid[:, 1], grid[:, 2]
        n_samples = len(grid)
        print(f"Griglia {n1}x{n2}x{nu} = {n_samples} campioni")
    else:
        # campionamento: uniforme casuale sui range del config (stesso approccio del paper),
        # oppure mu_u allineato ai nodi di bordo se richiesto (--align-mu-u-to-mesh)
        rng = np.random.default_rng(args.seed)
        n_samples = args.n_samples
        mu1_samples = rng.uniform(p["mu1"]["min"], p["mu1"]["max"], n_samples)
        mu2_samples = rng.uniform(p["mu2"]["min"], p["mu2"]["max"], n_samples)

        if args.align_mu_u_to_mesh:
            mu_u_candidates = get_boundary_mu_u_candidates(mesh_data)
            in_range = (mu_u_candidates >= p["mu_u"]["min"]) & (mu_u_candidates <= p["mu_u"]["max"])
            mu_u_candidates = mu_u_candidates[in_range]
            print(f"mu_u allineato alla mesh: {len(mu_u_candidates)} nodi candidati nel range")
            mu_u_samples = rng.choice(mu_u_candidates, n_samples, replace=True)
        else:
            mu_u_samples = rng.uniform(p["mu_u"]["min"], p["mu_u"]["max"], n_samples)

    # assembly indipendente da mu: fatto una volta sola per tutti gli N campioni
    print("Assembly operatori (indipendenti da mu) ...")
    operators = assemble_operators(mesh_data, omega_obs_regions)
    dirichlet_data = assemble_dirichlet_and_source(mesh_data, omega_obs_regions)
    node_to_dof = operators["node_to_dof"]

    Y = np.zeros((Nh, n_samples))
    P = np.zeros((Nh, n_samples))
    U = np.zeros((Nh, n_samples))

    start_time = time.time()

    for i in range(n_samples):
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
        print(f"\r  {i + 1}/{n_samples}  elapsed={elapsed:.1f}s  avg={avg:.2f}s/snapshot",
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
