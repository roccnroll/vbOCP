"""CLI: valuta errore (L2, H1) e speedup della PODNN per stato e aggiunto.

Carica le basi POD (train_pod.py) e i pesi delle due reti (train_reduced_nn.py
--field y / --field p), predice y e p per i parametri del test set, confronta
con i valori veri (FOM) in norma L2 e H1, e misura lo speedup rispetto al
solve FOM online (stesso pattern errore/speedup dei notebook del prof).

Uso:
    python -m src.rom.evaluate_pod_nn --config configs/test1.yaml \
        --pod-model data/snapshots/test1_pod.npz \
        --weights-y data/snapshots/test1_y_nn.pt \
        --weights-p data/snapshots/test1_p_nn.pt \
        --test-snapshots data/snapshots/test1_test150.npz
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.full_order.mesh import load_mesh
from src.full_order.assembly import assemble_operators, assemble_control_matrix, assemble_dirichlet_and_source
from src.full_order.solve import solve_otd
from src.rom.inner_product import assemble_full_mass_matrix
from src.dl.common import FFNN, normalize_minmax, denormalize_standard


def load_field_net(weights_path):
    """Carica rete + statistiche. n_modes letto dal .norm.npz (quanti ne ha effettivamente
    usati il training, puo' essere < di quelli disponibili nel pod-model se --n-modes
    e' stato passato esplicitamente a train_reduced_nn.py)."""
    norm_path = str(Path(weights_path).with_suffix(".norm.npz"))
    norm_data = np.load(norm_path)
    x_stats = {"min": norm_data["x_min"], "max": norm_data["x_max"]}
    y_stats = {"mean": norm_data["y_mean"], "std": norm_data["y_std"]}
    hidden_dim = int(norm_data["hidden_dim"])
    n_hidden_layers = int(norm_data["n_hidden_layers"])
    n_modes = int(norm_data["n_modes"])

    net = FFNN(input_dim=3, output_dim=n_modes, hidden_dim=hidden_dim, n_hidden_layers=n_hidden_layers)
    net.load_state_dict(torch.load(weights_path))
    net.eval()
    return net, x_stats, y_stats, n_modes


def relative_error(true, pred, norm_matrix):
    """Errore relativo per campione in norma indotta da norm_matrix, colonne = campioni.

    norm_matrix e' sparse (scipy) - niente np.einsum (non supporta operandi
    sparse); si usa moltiplicazione sparse@denso seguita da somma element-wise,
    che calcola la stessa forma quadratica diff[:,k]^T @ norm_matrix @ diff[:,k].
    """
    diff = pred - true
    err_sq = np.sum(diff * (norm_matrix @ diff), axis=0)
    true_sq = np.sum(true * (norm_matrix @ true), axis=0)
    errors = np.sqrt(np.abs(err_sq))
    norms = np.sqrt(np.abs(true_sq))
    return errors / np.where(norms > 0, norms, 1.0)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--pod-model", required=True, help="path al .npz da train_pod.py")
    parser.add_argument("--weights-y", required=True, help="pesi rete per lo stato (.pt)")
    parser.add_argument("--weights-p", required=True, help="pesi rete per l'aggiunto (.pt)")
    parser.add_argument("--test-snapshots", required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print(f"Caricamento mesh da {config['mesh']['path']} ...")
    mesh_data = load_mesh(config["mesh"]["path"], config["boundary_markers"])
    operators = assemble_operators(mesh_data, config["problem"]["omega_obs"])
    dirichlet_data = assemble_dirichlet_and_source(mesh_data, config["problem"]["omega_obs"])
    node_to_dof = operators["node_to_dof"]
    alpha = config["problem"]["alpha"]

    print("Assemblaggio matrici per le norme L2 (M_full) e H1 (A_diff) ...")
    M_full = assemble_full_mass_matrix(mesh_data)
    A_diff = operators["A_diff"]

    print(f"Caricamento base POD da {args.pod_model} ...")
    pod_data = np.load(args.pod_model)
    basis_y, basis_p = pod_data["basis_y"], pod_data["basis_p"]

    print(f"Caricamento rete stato da {args.weights_y} ...")
    net_y, x_stats_y, y_stats_y, n_modes_y = load_field_net(args.weights_y)
    print(f"Caricamento rete aggiunto da {args.weights_p} ...")
    net_p, x_stats_p, y_stats_p, n_modes_p = load_field_net(args.weights_p)

    # tronca la base ai primi N modi usati in training (puo' essere < dei modi disponibili
    # nel pod-model se e' stato passato --n-modes esplicito a train_reduced_nn.py)
    basis_y, basis_p = basis_y[:, :n_modes_y], basis_p[:, :n_modes_p]
    print(f"N modi usati - stato: {n_modes_y}  aggiunto: {n_modes_p}")

    print(f"Caricamento test set da {args.test_snapshots} ...")
    test_data = np.load(args.test_snapshots)
    mu1, mu2, mu_u = test_data["mu1"], test_data["mu2"], test_data["mu_u"]
    Y_true, P_true = test_data["Y"], test_data["P"]
    n_test = len(mu1)

    x_raw = np.stack([mu1, mu2, mu_u], axis=1)

    print("Predizione con le due PODNN ...")
    start_rom = time.time()
    x_norm_y = torch.tensor(normalize_minmax(x_raw, x_stats_y), dtype=torch.float32)
    x_norm_p = torch.tensor(normalize_minmax(x_raw, x_stats_p), dtype=torch.float32)
    with torch.no_grad():
        coeffs_y_pred = denormalize_standard(net_y(x_norm_y).numpy(), y_stats_y)
        coeffs_p_pred = denormalize_standard(net_p(x_norm_p).numpy(), y_stats_p)
    Y_pred = basis_y @ coeffs_y_pred.T
    P_pred = basis_p @ coeffs_p_pred.T
    time_rom = time.time() - start_rom

    print("Errore stato (y):")
    err_y_l2 = relative_error(Y_true, Y_pred, M_full)
    err_y_h1 = relative_error(Y_true, Y_pred, A_diff)
    print(f"  L2 - medio: {err_y_l2.mean():.4e}  mediano: {np.median(err_y_l2):.4e}  max: {err_y_l2.max():.4e}")
    print(f"  H1 - medio: {err_y_h1.mean():.4e}  mediano: {np.median(err_y_h1):.4e}  max: {err_y_h1.max():.4e}")

    print("Errore aggiunto (p):")
    err_p_l2 = relative_error(P_true, P_pred, M_full)
    err_p_h1 = relative_error(P_true, P_pred, A_diff)
    print(f"  L2 - medio: {err_p_l2.mean():.4e}  mediano: {np.median(err_p_l2):.4e}  max: {err_p_l2.max():.4e}")
    print(f"  H1 - medio: {err_p_h1.mean():.4e}  mediano: {np.median(err_p_h1):.4e}  max: {err_p_h1.max():.4e}")

    # speedup: tempo FOM online (assembla C(mu_u) + risolve) per lo stesso numero di campioni,
    # confrontato col tempo delle due reti (gia' misurato sopra, time_rom, sull'intero batch)
    print("Misurazione tempo FOM online (assemble_control_matrix + solve_otd) per confronto speedup ...")
    start_fom = time.time()
    for i in range(n_test):
        C = assemble_control_matrix(mesh_data, node_to_dof, mu_u[i])
        solve_otd(operators, C, dirichlet_data, mu1[i], mu2[i], alpha)
    time_fom = time.time() - start_fom

    print(f"Tempo FOM (online, {n_test} campioni): {time_fom:.3f} s")
    print(f"Tempo PODNN (online, {n_test} campioni): {time_rom:.3f} s")
    print(f"Speedup: {time_fom / time_rom:.1f}x")


if __name__ == "__main__":
    main()
