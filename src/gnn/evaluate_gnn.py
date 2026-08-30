"""CLI: valuta errore (L2, H1) e speedup di un GCA-ROM gia' allenato (train_gnn.py),
senza ri-allenare.

Ricostruisce HyperParams/rete dagli iperparametri salvati in train_meta.json (scritto
da train_gnn.py a fine training), carica i pesi, e valuta su un test set esplicito
(puo' essere diverso da quello usato in training, purche' stessa mesh/comp).

Uso:
    python -m src.gnn.evaluate_gnn --config configs/test1.yaml \
        --gca-rom-path /path/to/gca-rom \
        --train-mat data/gnn/train_y.mat --test-mat data/gnn/test_y.mat \
        --net-dir data/gnn/models/test1_gnn_y
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.full_order.mesh import load_mesh
from src.full_order.assembly import assemble_operators, assemble_control_matrix, assemble_dirichlet_and_source
from src.full_order.solve import solve_otd
from src.rom.inner_product import assemble_full_mass_matrix
from src.gnn.train_gnn import build_combined_dataset, inverse_scale_channel, relative_error, build_hyperparams
from src.gnn.convert_to_gca_rom import restrict_to_dof


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--gca-rom-path", required=True, help="path alla repo gca-rom clonata (per il sys.path)")
    parser.add_argument("--train-mat", required=True,
                         help=".mat di training (stesso usato per allenare i pesi - serve per rifare lo scaling)")
    parser.add_argument("--test-mat", required=True, help=".mat di test su cui valutare")
    parser.add_argument("--net-dir", required=True, help="cartella con i pesi + train_meta.json da train_gnn.py")
    parser.add_argument("--save-csv", default=None,
                         help="opzionale: salva un .csv con l'errore PER CAMPIONE (mu1,mu2,mu_u,err_*) "
                              "invece che solo le statistiche aggregate - utile per mappare l'errore "
                              "sullo spazio dei parametri")
    return parser.parse_args()


def main():
    args = parse_args()
    sys.path.insert(0, args.gca_rom_path)
    from gca_rom import network, preprocessing, testing, initialization

    meta_path = Path(args.net_dir) / "train_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{meta_path} non trovato - il training in questa cartella non risulta completato "
            "(train_meta.json viene scritto da train_gnn.py solo a fine training riuscito)")
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

    print(f"Caricamento mesh da {config['mesh']['path']} (per le norme L2/H1) ...")
    mesh_data = load_mesh(config["mesh"]["path"], config["boundary_markers"])
    operators = assemble_operators(mesh_data, config["problem"]["omega_obs"])
    dirichlet_data = assemble_dirichlet_and_source(mesh_data, config["problem"]["omega_obs"])
    node_to_dof = operators["node_to_dof"]
    alpha = config["problem"]["alpha"]
    M_full = assemble_full_mass_matrix(mesh_data)
    A_diff = operators["A_diff"]

    print(f"Caricamento train/test da {args.train_mat} / {args.test_mat} ...")
    dataset, params_np, train_snapshots, test_snapshots = build_combined_dataset(
        args.train_mat, args.test_mat, train_args.comp)
    n_param = params_np.shape[1]
    print(f"Campioni: {len(train_snapshots)} train (per lo scaling), {len(test_snapshots)} test, {n_param} parametri")

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

    print("Valutazione sul test set ...")
    start_rom = time.time()
    results_test, _, _ = testing.evaluate(VAR_test, model, val_loader, params, HyperParams, test_snapshots)
    time_rom = time.time() - start_rom

    # M_full/A_diff sono in spazio DOF ridotto (Nh, come la POD) - pred/true della GNN
    # vivono su tutti i nodi mesh (convert_to_gca_rom.py li ricostruisce cosi'), quindi
    # vanno ristretti ai soli DOF liberi prima del confronto
    if train_args.comp == 1:
        pred_full = inverse_scale_channel(results_test[:, :, 0], scaler_test, train_args.scaling_type).numpy()
        true_full = dataset.U[:, test_snapshots].numpy()
        pred = restrict_to_dof(pred_full, node_to_dof)
        true = restrict_to_dof(true_full, node_to_dof)
        err_l2 = relative_error(true, pred, M_full)
        err_h1 = relative_error(true, pred, A_diff)
        label = train_args.field or "campo"
        print(f"Errore {label}:")
        print(f"  L2 - medio: {err_l2.mean():.4e}  mediano: {np.median(err_l2):.4e}  max: {err_l2.max():.4e}")
        print(f"  H1 - medio: {err_h1.mean():.4e}  mediano: {np.median(err_h1):.4e}  max: {err_h1.max():.4e}")
    else:
        pred_y_full = inverse_scale_channel(results_test[:, :, 0], scaler_test[0], train_args.scaling_type).numpy()
        pred_p_full = inverse_scale_channel(results_test[:, :, 1], scaler_test[1], train_args.scaling_type).numpy()
        true_y_full = dataset.VX[:, test_snapshots].numpy()
        true_p_full = dataset.VY[:, test_snapshots].numpy()
        pred_y, true_y = restrict_to_dof(pred_y_full, node_to_dof), restrict_to_dof(true_y_full, node_to_dof)
        pred_p, true_p = restrict_to_dof(pred_p_full, node_to_dof), restrict_to_dof(true_p_full, node_to_dof)
        err_y_l2 = relative_error(true_y, pred_y, M_full)
        err_y_h1 = relative_error(true_y, pred_y, A_diff)
        err_p_l2 = relative_error(true_p, pred_p, M_full)
        err_p_h1 = relative_error(true_p, pred_p, A_diff)
        print("Errore y:")
        print(f"  L2 - medio: {err_y_l2.mean():.4e}  mediano: {np.median(err_y_l2):.4e}  max: {err_y_l2.max():.4e}")
        print(f"  H1 - medio: {err_y_h1.mean():.4e}  mediano: {np.median(err_y_h1):.4e}  max: {err_y_h1.max():.4e}")
        print("Errore p:")
        print(f"  L2 - medio: {err_p_l2.mean():.4e}  mediano: {np.median(err_p_l2):.4e}  max: {err_p_l2.max():.4e}")
        print(f"  H1 - medio: {err_p_h1.mean():.4e}  mediano: {np.median(err_p_h1):.4e}  max: {err_p_h1.max():.4e}")

    if args.save_csv is not None:
        mu_test = params_np[test_snapshots, :]
        Path(args.save_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_csv, "w", newline="") as f:
            if train_args.comp == 1:
                writer = csv.writer(f)
                writer.writerow(["mu1", "mu2", "mu_u", "err_l2", "err_h1"])
                for i in range(len(test_snapshots)):
                    writer.writerow([mu_test[i, 0], mu_test[i, 1], mu_test[i, 2], err_l2[i], err_h1[i]])
            else:
                writer = csv.writer(f)
                writer.writerow(["mu1", "mu2", "mu_u", "err_y_l2", "err_y_h1", "err_p_l2", "err_p_h1"])
                for i in range(len(test_snapshots)):
                    writer.writerow([mu_test[i, 0], mu_test[i, 1], mu_test[i, 2],
                                      err_y_l2[i], err_y_h1[i], err_p_l2[i], err_p_h1[i]])
        print(f"Errore per campione salvato in {args.save_csv}")

    # speedup: tempo FOM online (assembla C(mu_u) + risolve) per lo stesso numero di campioni
    # di test, confrontato col tempo di inferenza della GNN misurato sopra (stesso pattern
    # di evaluate_pod_nn.py, cosi' i due speedup sono confrontabili)
    mu1_test = params_np[test_snapshots, 0]
    mu2_test = params_np[test_snapshots, 1]
    mu_u_test = params_np[test_snapshots, 2]
    print("Misurazione tempo FOM online (assemble_control_matrix + solve_otd) per confronto speedup ...")
    start_fom = time.time()
    for mu1_i, mu2_i, mu_u_i in zip(mu1_test, mu2_test, mu_u_test):
        C = assemble_control_matrix(mesh_data, node_to_dof, mu_u_i)
        solve_otd(operators, C, dirichlet_data, mu1_i, mu2_i, alpha)
    time_fom = time.time() - start_fom

    n_test = len(test_snapshots)
    print(f"Tempo FOM (online, {n_test} campioni): {time_fom:.3f} s")
    print(f"Tempo GNN (online, {n_test} campioni): {time_rom:.3f} s")
    print(f"Speedup: {time_fom / time_rom:.1f}x")


if __name__ == "__main__":
    main()
