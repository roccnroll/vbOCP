"""CLI: valuta la PODNN del controllo su un test set (mai visto in training).

Carica base POD (build_control_pod.py) e pesi rete (train_control_nn.py)
separatamente, predice u per i parametri del test set, confronta con u
vero (FOM) - stesso pattern di validazione errore/speedup dei notebook
del prof (Lab4/Lab9).

Uso:
    python -m src.rom.evaluate_control_podnn --config configs/test1.yaml \
        --pod-model data/snapshots/test1_control_pod.npz \
        --weights data/snapshots/test1_control_nn.pt \
        --test-snapshots data/snapshots/test1_test150.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.full_order.mesh import load_mesh
from src.full_order.assembly import assemble_operators
from src.rom.control import extract_boundary_control_trace
from src.dl.common import FFNN, normalize_minmax, denormalize_standard


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--pod-model", required=True, help="path al .npz da build_control_pod.py")
    parser.add_argument("--weights", required=True, help="path ai pesi rete (.pt) da train_control_nn.py")
    parser.add_argument("--test-snapshots", required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print(f"Caricamento mesh da {config['mesh']['path']} ...")
    mesh_data = load_mesh(config["mesh"]["path"], config["boundary_markers"])
    operators = assemble_operators(mesh_data, config["problem"]["omega_obs"])
    node_to_dof = operators["node_to_dof"]

    print(f"Caricamento base POD da {args.pod_model} ...")
    model_data = np.load(args.pod_model)
    basis = model_data["basis"]
    n_modes = int(model_data["n_modes"])

    norm_path = str(Path(args.weights).with_suffix(".norm.npz"))
    print(f"Caricamento statistiche di normalizzazione e architettura da {norm_path} ...")
    norm_data = np.load(norm_path)
    x_stats = {"min": norm_data["x_min"], "max": norm_data["x_max"]}
    y_stats = {"mean": norm_data["y_mean"], "std": norm_data["y_std"]}
    hidden_dim = int(norm_data["hidden_dim"])
    n_hidden_layers = int(norm_data["n_hidden_layers"])

    print(f"Caricamento pesi rete da {args.weights} (hidden_dim={hidden_dim}, n_hidden_layers={n_hidden_layers}) ...")
    net = FFNN(input_dim=3, output_dim=n_modes, hidden_dim=hidden_dim, n_hidden_layers=n_hidden_layers)
    net.load_state_dict(torch.load(args.weights))
    net.eval()

    print(f"Caricamento test set da {args.test_snapshots} ...")
    test_data = np.load(args.test_snapshots)
    mu1, mu2, mu_u, U_true = test_data["mu1"], test_data["mu2"], test_data["mu_u"], test_data["U"]

    print("Estrazione traccia di controllo vera (test set) ...")
    boundary_x_test, U_boundary_true = extract_boundary_control_trace(mesh_data, node_to_dof, U_true, mu_u)

    # coerenza: la traccia di bordo deve avere gli stessi nodi (stessa mesh) del training
    if not np.allclose(boundary_x_test, model_data["boundary_x"]):
        raise ValueError("I nodi di bordo del test set non coincidono con quelli del modello - mesh diversa?")

    print("Predizione con la PODNN ...")
    x_raw = np.stack([mu1, mu2, mu_u], axis=1)
    x_norm = normalize_minmax(x_raw, x_stats)
    x_test = torch.tensor(x_norm, dtype=torch.float32)
    with torch.no_grad():
        coeffs_pred_norm = net(x_test).numpy()  # (n_samples, n_modes), normalizzati

    coeffs_pred = denormalize_standard(coeffs_pred_norm, y_stats)

    U_boundary_pred = basis @ coeffs_pred.T  # (n_boundary_nodes, n_samples)

    # errore relativo per campione, norma euclidea (stesso prodotto scalare usato in training)
    errors = np.linalg.norm(U_boundary_pred - U_boundary_true, axis=0)
    norms = np.linalg.norm(U_boundary_true, axis=0)
    relative_errors = errors / np.where(norms > 0, norms, 1.0)

    print(f"Errore relativo medio: {relative_errors.mean():.4e}")
    print(f"Errore relativo mediano: {np.median(relative_errors):.4e}")
    print(f"Errore relativo max: {relative_errors.max():.4e}")


if __name__ == "__main__":
    main()
