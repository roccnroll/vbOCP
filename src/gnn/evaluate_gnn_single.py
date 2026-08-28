"""CLI: valuta un GCA-ROM gia' allenato su UN SOLO punto (mu1, mu2, mu_u) scelto a mano,
invece che sulla media di un test set intero (per quello vedi evaluate_gnn.py).

Risolve il FOM per la terna richiesta (come run_single_solve.py), costruisce al volo
un .mat "di test" con un solo campione (stesso formato di convert_to_gca_rom.py) e lo
passa alla stessa pipeline di valutazione di evaluate_gnn.py.

Uso:
    python -m src.gnn.evaluate_gnn_single --config configs/test1.yaml \
        --gca-rom-path /path/to/gca-rom \
        --train-mat data/gnn/train_yp.mat \
        --net-dir data/gnn/models/test1_gnn_yp \
        --mu1 12 --mu2 2.5 --mu_u 0.5
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.io
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.full_order.mesh import load_mesh
from src.full_order.assembly import assemble_operators, assemble_control_matrix, assemble_dirichlet_and_source
from src.full_order.solve import solve_otd
from src.rom.inner_product import assemble_full_mass_matrix
from src.gnn.train_gnn import build_combined_dataset, inverse_scale_channel, relative_error, build_hyperparams
from src.gnn.convert_to_gca_rom import build_mesh_arrays, reconstruct_full_field, restrict_to_dof


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--gca-rom-path", required=True)
    parser.add_argument("--train-mat", required=True,
                         help=".mat di training (stesso usato per allenare i pesi - serve per rifare lo scaling)")
    parser.add_argument("--net-dir", required=True, help="cartella con i pesi + train_meta.json da train_gnn.py")
    parser.add_argument("--mu1", type=float, required=True)
    parser.add_argument("--mu2", type=float, required=True)
    parser.add_argument("--mu_u", type=float, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    sys.path.insert(0, args.gca_rom_path)
    from gca_rom import network, preprocessing, testing, initialization

    meta_path = Path(args.net_dir) / "train_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{meta_path} non trovato - il training in questa cartella non risulta completato")
    with open(meta_path) as f:
        meta = json.load(f)
    print(f"Iperparametri caricati da {meta_path}: {meta}")

    net_name = Path(args.net_dir.rstrip("/")).name
    train_args = SimpleNamespace(
        net_name=net_name,
        comp=meta["comp"], field=meta["field"],
        scaling_type=meta["scaling_type"], scaler_number=meta["scaler_number"],
        ffn=meta["ffn"], map_nodes=meta["map_nodes"], bottleneck_dim=meta["bottleneck_dim"],
        lambda_map=meta["lambda_map"], in_channels=meta["in_channels"],
        epochs=meta["epochs"], batch_size=meta["batch_size"], minibatch=meta["minibatch"],
        lr=None,
    )

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print(f"Caricamento mesh da {config['mesh']['path']} ...")
    mesh_data = load_mesh(config["mesh"]["path"], config["boundary_markers"])
    operators = assemble_operators(mesh_data, config["problem"]["omega_obs"])
    dirichlet_data = assemble_dirichlet_and_source(mesh_data, config["problem"]["omega_obs"])
    node_to_dof = operators["node_to_dof"]
    alpha = config["problem"]["alpha"]
    M_full = assemble_full_mass_matrix(mesh_data)
    A_diff = operators["A_diff"]

    print(f"Solve FOM per mu1={args.mu1}, mu2={args.mu2}, mu_u={args.mu_u} ...")
    C = assemble_control_matrix(mesh_data, node_to_dof, args.mu_u)
    y_true_dof, p_true_dof, _ = solve_otd(operators, C, dirichlet_data, args.mu1, args.mu2, alpha)

    print("Costruzione .mat temporaneo con questo singolo campione ...")
    x, y_coord, T, E, num_nodes = build_mesh_arrays(mesh_data)
    mat_dict = {
        "xx": x[:, None], "yy": y_coord[:, None], "T": T, "E": E,
        "params": np.array([[args.mu1, args.mu2, args.mu_u]]),
    }
    if train_args.comp == 1:
        field_dof = y_true_dof if train_args.field == "y" else p_true_dof
        mat_dict["U"] = reconstruct_full_field(field_dof[:, None], node_to_dof)
    else:
        mat_dict["VX"] = reconstruct_full_field(y_true_dof[:, None], node_to_dof)
        mat_dict["VY"] = reconstruct_full_field(p_true_dof[:, None], node_to_dof)

    with tempfile.TemporaryDirectory() as tmp_dir:
        test_mat_path = str(Path(tmp_dir) / "single_test.mat")
        scipy.io.savemat(test_mat_path, mat_dict)

        print(f"Caricamento train (per lo scaling) da {args.train_mat} ...")
        dataset, params_np, train_snapshots, test_snapshots = build_combined_dataset(
            args.train_mat, test_mat_path, train_args.comp)
        n_param = params_np.shape[1]

        variable = train_args.field if train_args.comp == 1 else "yp"
        HyperParams = build_hyperparams(network, train_args, variable, n_param, net_dir=args.net_dir)

        device = initialization.set_device()
        initialization.set_reproducibility(HyperParams)

        processor = preprocessing.SteadyDataProcessor()
        xx, yy, zz, xyz, var, var1, var2, num_graphs, _ = processor.prepare_var(dataset, HyperParams)
        VAR_all, VAR_test, scaler_all, scaler_test = processor.scale(
            HyperParams, dataset, test_snapshots, var, var1, var2)
        graphs, train_dataset, test_dataset = processor.append_graphs(
            HyperParams, VAR_all, dataset, num_graphs, xx, yy, train_snapshots, test_snapshots, zz)
        _, _, _, val_loader = processor.return_loaders(
            HyperParams, graphs, train_dataset, test_dataset, len(train_snapshots), len(test_snapshots))

        weights_path = Path(args.net_dir.rstrip("/") + "/") / f"{net_name}{HyperParams.net_run}.pt"
        print(f"Caricamento pesi da {weights_path} ...")
        model = network.Net(HyperParams).to(device)
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.to("cpu")

        params = torch.tensor(params_np, dtype=torch.get_default_dtype())

        print("Predizione GNN sul punto richiesto ...")
        results_test, _, _ = testing.evaluate(VAR_test, model, val_loader, params, HyperParams, test_snapshots)

    # M_full/A_diff sono in spazio DOF ridotto (Nh, come la POD) - pred/true della GNN
    # vivono su tutti i nodi mesh, quindi vanno ristretti ai soli DOF liberi prima del confronto
    if train_args.comp == 1:
        pred_full = inverse_scale_channel(results_test[:, :, 0], scaler_test, train_args.scaling_type).numpy()
        pred = restrict_to_dof(pred_full, node_to_dof)
        true = restrict_to_dof(mat_dict["U"], node_to_dof)
        err_l2 = relative_error(true, pred, M_full)
        err_h1 = relative_error(true, pred, A_diff)
        label = train_args.field or "campo"
        print(f"Errore {label} nel punto (mu1={args.mu1}, mu2={args.mu2}, mu_u={args.mu_u}):")
        print(f"  L2: {err_l2[0]:.4e}")
        print(f"  H1: {err_h1[0]:.4e}")
    else:
        pred_y_full = inverse_scale_channel(results_test[:, :, 0], scaler_test[0], train_args.scaling_type).numpy()
        pred_p_full = inverse_scale_channel(results_test[:, :, 1], scaler_test[1], train_args.scaling_type).numpy()
        pred_y, pred_p = restrict_to_dof(pred_y_full, node_to_dof), restrict_to_dof(pred_p_full, node_to_dof)
        true_y = restrict_to_dof(mat_dict["VX"], node_to_dof)
        true_p = restrict_to_dof(mat_dict["VY"], node_to_dof)
        err_y_l2 = relative_error(true_y, pred_y, M_full)
        err_y_h1 = relative_error(true_y, pred_y, A_diff)
        err_p_l2 = relative_error(true_p, pred_p, M_full)
        err_p_h1 = relative_error(true_p, pred_p, A_diff)
        print(f"Errore nel punto (mu1={args.mu1}, mu2={args.mu2}, mu_u={args.mu_u}):")
        print(f"  y - L2: {err_y_l2[0]:.4e}  H1: {err_y_h1[0]:.4e}")
        print(f"  p - L2: {err_p_l2[0]:.4e}  H1: {err_p_h1[0]:.4e}")


if __name__ == "__main__":
    main()
