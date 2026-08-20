"""CLI: converte uno snapshot .npz (nostro formato) nel formato .mat atteso da gca-rom.

Le nostre Y/P sono indicizzate per DOF libero (Nh = numero di DOF, esclude i
nodi Dirichlet forti su Gamma_D dove y=p=0 e' imposto) - non si allineano
direttamente con le coordinate di tutti i nodi mesh. Questo script ricostruisce
il campo su TUTTI i nodi mesh (num_nodes >= Nh), riempiendo con 0 dove
node_to_dof == -1 (corretto: e' proprio dove y=p=0 per costruzione), cosi'
coordinate e valori hanno la stessa lunghezza/ordine come richiesto da gca-rom.

Formato .mat prodotto (vedi gca_rom/loader.py):
    xx, yy: (num_nodes, n_samples) - coordinate nodali, ripetute per colonna
    T: (n_triangoli, 3) - connettivita' triangoli, 1-indicizzata (solo plotting)
    E: (n_archi, 2) - lista archi, 1-indicizzata (usata per il grafo)
    U: (num_nodes, n_samples) - SOLO se --n-comp 1, il campo scelto (y o p)
    VX, VY: (num_nodes, n_samples) - SOLO se --n-comp 2, y e p come due canali
    params: (n_samples, 3) - mu1, mu2, mu_u veri (non a griglia - bypassa
        pde.problem()/product(mu_space) del tutorial gca-rom)

Uso (un solo campo):
    python -m src.gnn.convert_to_gca_rom --config configs/test1.yaml \
        --snapshots data/snapshots/test1_300.npz --field y --n-comp 1 \
        --output data/gnn/test1_300_y.mat

Uso (combinato, y e p come due canali):
    python -m src.gnn.convert_to_gca_rom --config configs/test1.yaml \
        --snapshots data/snapshots/test1_300.npz --n-comp 2 \
        --output data/gnn/test1_300_yp.mat
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.io
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.full_order.mesh import load_mesh
from src.full_order.assembly import assemble_operators


def reconstruct_full_field(dof_field, node_to_dof):
    """Espande un campo indicizzato per DOF (Nh,) o (Nh, n) a tutti i nodi mesh
    (num_nodes,) o (num_nodes, n), mettendo 0 sui nodi Dirichlet (node_to_dof == -1)."""
    num_nodes = len(node_to_dof)
    if dof_field.ndim == 1:
        full = np.zeros(num_nodes)
        free = node_to_dof >= 0
        full[free] = dof_field[node_to_dof[free]]
    else:
        n_samples = dof_field.shape[1]
        full = np.zeros((num_nodes, n_samples))
        free = node_to_dof >= 0
        full[free, :] = dof_field[node_to_dof[free], :]
    return full


def build_mesh_arrays(mesh_data):
    """Estrae coordinate, triangoli (T) e archi (E) dalla mesh, 1-indicizzati per gca-rom."""
    mesh = mesh_data["mesh"]
    num_nodes = mesh.cell0_d_total_number()

    x = np.array([mesh.cell0_d_coordinate_x(i) for i in range(num_nodes)])
    y = np.array([mesh.cell0_d_coordinate_y(i) for i in range(num_nodes)])

    n_tri = mesh.cell2_d_total_number()
    T = np.array([
        [mesh.cell2_d_vertex(t, 0), mesh.cell2_d_vertex(t, 1), mesh.cell2_d_vertex(t, 2)]
        for t in range(n_tri)
    ]) + 1  # 1-indicizzato

    n_edges = mesh.cell1_d_total_number()
    E = np.array([mesh.cell1_d_extremes(e) for e in range(n_edges)]) + 1  # 1-indicizzato

    return x, y, T, E, num_nodes


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--snapshots", required=True, help="path allo snapshot .npz")
    parser.add_argument("--n-comp", type=int, choices=[1, 2], required=True,
                         help="1 = un solo campo (--field), 2 = y e p combinati come VX/VY")
    parser.add_argument("--field", choices=["y", "p"], default=None,
                         help="richiesto se --n-comp 1: quale campo esportare")
    parser.add_argument("--output", required=True, help="path del .mat di output")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.n_comp == 1 and args.field is None:
        raise ValueError("--field e' richiesto con --n-comp 1")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print(f"Caricamento mesh da {config['mesh']['path']} ...")
    mesh_data = load_mesh(config["mesh"]["path"], config["boundary_markers"])
    operators = assemble_operators(mesh_data, config["problem"]["omega_obs"])
    node_to_dof = operators["node_to_dof"]

    print(f"Caricamento snapshot da {args.snapshots} ...")
    data = np.load(args.snapshots)
    mu1, mu2, mu_u = data["mu1"], data["mu2"], data["mu_u"]
    n_samples = len(mu1)

    print("Estrazione coordinate/connettivita' mesh ...")
    x, y, T, E, num_nodes = build_mesh_arrays(mesh_data)
    xx = np.tile(x[:, None], (1, n_samples))
    yy = np.tile(y[:, None], (1, n_samples))

    mat_dict = {
        "xx": xx, "yy": yy, "T": T, "E": E,
        "params": np.stack([mu1, mu2, mu_u], axis=1),
    }

    print(f"Ricostruzione campo(i) su tutti i {num_nodes} nodi mesh (Nh DOF = {mesh_data['Nh']}) ...")
    if args.n_comp == 1:
        field = data["Y"] if args.field == "y" else data["P"]
        mat_dict["U"] = reconstruct_full_field(field, node_to_dof)
    else:
        mat_dict["VX"] = reconstruct_full_field(data["Y"], node_to_dof)
        mat_dict["VY"] = reconstruct_full_field(data["P"], node_to_dof)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    scipy.io.savemat(args.output, mat_dict)
    print(f"Salvato in {args.output}")


if __name__ == "__main__":
    main()
