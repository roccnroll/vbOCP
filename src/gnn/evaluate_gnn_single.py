"""CLI: valuta un GCA-ROM gia' allenato su UN SOLO punto (mu1, mu2, mu_u) scelto a mano,
invece che sulla media di un test set intero (per quello vedi evaluate_gnn.py).

Risolve il FOM per la terna richiesta (come run_single_solve.py) e AGGIUNGE il nuovo
campione al test set vero (--test-mat), invece di valutare su un .mat con un solo
campione isolato. Motivo: gca_rom.preprocessing.scale() fitta lo scaler di
normalizzazione SUI campioni di test passati (non su quelli di training) - con un solo
campione la statistica (media/std per nodo, stadio "feature" dello scaling_type=4) e'
degenere (popolazione di 1), e dava risultati diversi/sbagliati rispetto alla stessa
valutazione fatta dentro il test set intero. Aggiungendo il punto al test set vero la
popolazione usata per la normalizzazione e' la stessa (151 campioni) di una valutazione
normale, e si riporta l'errore solo per l'ultimo indice (quello nuovo).

Uso:
    python -m src.gnn.evaluate_gnn_single --config configs/test1.yaml \
        --gca-rom-path /path/to/gca-rom \
        --train-mat data/gnn/train_yp.mat --test-mat data/gnn/test_yp.mat \
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
from src.dl.common import normalize_minmax


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--gca-rom-path", required=True)
    parser.add_argument("--train-mat", required=True,
                         help=".mat di training (stesso usato per allenare i pesi - serve per rifare lo scaling)")
    parser.add_argument("--test-mat", required=True,
                         help=".mat di test vero (stesso usato in evaluate_gnn.py) - il nuovo punto viene "
                              "aggiunto a questo set invece di essere valutato isolato, per non rompere la "
                              "normalizzazione (vedi docstring del modulo)")
    parser.add_argument("--net-dir", required=True, help="cartella con i pesi + train_meta.json da train_gnn.py")
    parser.add_argument("--mu1", type=float, required=True)
    parser.add_argument("--mu2", type=float, required=True)
    parser.add_argument("--mu_u", type=float, required=True)
    parser.add_argument("--plot", action="store_true",
                         help="mostra il plot FOM/GNN/errore (salvato in un path di default, come run_single_solve.py)")
    parser.add_argument("--save-plot", default=None, help="path dove salvare permanentemente il plot (opzionale)")
    return parser.parse_args()


def plot_comparison(mesh_data, fields, mu1, mu2, mu_u, output_path):
    """fields: lista di (label, true_full, pred_full), array su tutti i nodi mesh (num_nodes,).
    Una riga per campo: FOM (verita'), GNN (predizione), errore assoluto - stesso stile di
    run_single_solve.plot_solution ma con matplotlib.tri diretto (i campi sono gia' su tutti
    i nodi mesh, nessun bisogno di extract_solution_on_cell0_ds)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    mesh = mesh_data["mesh"]
    x_nodes = np.array([mesh.cell0_d_coordinate_x(i) for i in range(mesh.cell0_d_total_number())])
    y_nodes = np.array([mesh.cell0_d_coordinate_y(i) for i in range(mesh.cell0_d_total_number())])
    triangles = np.array([
        [mesh.cell2_d_vertex(t, 0), mesh.cell2_d_vertex(t, 1), mesh.cell2_d_vertex(t, 2)]
        for t in range(mesh.cell2_d_total_number())
    ])
    triang = mtri.Triangulation(x_nodes, y_nodes, triangles)

    n_rows = len(fields)
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4 * n_rows), squeeze=False)
    for row, (label, true_full, pred_full) in enumerate(fields):
        true_flat = np.asarray(true_full).reshape(-1)
        pred_flat = np.asarray(pred_full).reshape(-1)
        diff_flat = np.abs(pred_flat - true_flat)
        # range preso dal FOM (verita') e applicato anche al plot GNN, cosi' i colori sono
        # confrontabili a colpo d'occhio e si vede subito se la GNN esce dal range vero
        vmin, vmax = true_flat.min(), true_flat.max()
        levels = np.linspace(vmin, vmax, 200)

        ax = axes[row][0]
        tc = ax.tricontourf(triang, true_flat, levels=levels, cmap="jet", vmin=vmin, vmax=vmax)
        plt.colorbar(tc, ax=ax, label=label)
        ax.set_title(f"FOM (verita') - {label}")
        ax.set_aspect("equal")

        ax = axes[row][1]
        # extend='both': se la GNN esce dal range del FOM, i valori fuori scala restano
        # visibili (colore piu' estremo) invece di sparire/troncare silenziosamente
        tc = ax.tricontourf(triang, pred_flat, levels=levels, cmap="jet", vmin=vmin, vmax=vmax, extend="both")
        plt.colorbar(tc, ax=ax, label=label)
        ax.set_title(f"GNN (predizione) - {label}")
        ax.set_aspect("equal")

        ax = axes[row][2]
        tc = ax.tricontourf(triang, diff_flat, levels=200, cmap="jet")
        plt.colorbar(tc, ax=ax, label=f"|errore {label}|")
        ax.set_title(f"Errore assoluto - {label}")
        ax.set_aspect("equal")

    fig.suptitle(f"mu1={mu1}, mu2={mu2}, mu_u={mu_u}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Plot salvato in {output_path}")


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

    print("Costruzione del campione singolo ...")
    x, y_coord, T, E, num_nodes = build_mesh_arrays(mesh_data)
    single_dict = {
        "xx": x[:, None], "yy": y_coord[:, None], "T": T, "E": E,
        "params": np.array([[args.mu1, args.mu2, args.mu_u]]),
    }
    # Dirichlet: y=1 su Gamma_D, p=0 su Gamma_D (assemble_dirichlet_and_source)
    if train_args.comp == 1:
        field_dof = y_true_dof if train_args.field == "y" else p_true_dof
        dirichlet_value = 1.0 if train_args.field == "y" else 0.0
        single_dict["U"] = reconstruct_full_field(field_dof[:, None], node_to_dof, dirichlet_value)
    else:
        single_dict["VX"] = reconstruct_full_field(y_true_dof[:, None], node_to_dof, dirichlet_value=1.0)
        single_dict["VY"] = reconstruct_full_field(p_true_dof[:, None], node_to_dof)

    print(f"Aggiunta del campione al test set vero ({args.test_mat}) per la normalizzazione ...")
    real_test = scipy.io.loadmat(args.test_mat)
    mat_dict = {
        "xx": np.hstack([real_test["xx"], single_dict["xx"]]),
        "yy": np.hstack([real_test["yy"], single_dict["yy"]]),
        "T": real_test["T"], "E": real_test["E"],
        "params": np.vstack([real_test["params"], single_dict["params"]]),
    }
    if train_args.comp == 1:
        mat_dict["U"] = np.hstack([real_test["U"], single_dict["U"]])
    else:
        mat_dict["VX"] = np.hstack([real_test["VX"], single_dict["VX"]])
        mat_dict["VY"] = np.hstack([real_test["VY"], single_dict["VY"]])

    with tempfile.TemporaryDirectory() as tmp_dir:
        test_mat_path = str(Path(tmp_dir) / "extended_test.mat")
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

        # stessa normalizzazione min-max di mu usata in training.py (fit sui soli campioni di
        # training, salvata in train_meta.json) - la rete si aspetta input in [-1,1]
        mu_stats = {"min": np.array(meta["mu_min"]), "max": np.array(meta["mu_max"])}
        params_norm = normalize_minmax(params_np, mu_stats)
        params = torch.tensor(params_norm, dtype=torch.get_default_dtype())

        print("Predizione GNN sul punto richiesto ...")
        results_test, _, _ = testing.evaluate(VAR_test, model, val_loader, params, HyperParams, test_snapshots)

    # il campione nuovo e' l'ultimo del test set esteso - indice LOCALE (posizione dentro
    # test_snapshots, stesso ordine di VAR_test/results_test), non l'indice globale nel dataset
    local_idx = len(test_snapshots) - 1

    # M_full/A_diff sono in spazio DOF ridotto (Nh, come la POD) - pred/true della GNN
    # vivono su tutti i nodi mesh, quindi vanno ristretti ai soli DOF liberi prima del confronto
    if train_args.comp == 1:
        pred_full_all = inverse_scale_channel(results_test[:, :, 0], scaler_test, train_args.scaling_type).numpy()
        pred_full = pred_full_all[:, [local_idx]]
        true_full = single_dict["U"]
        pred = restrict_to_dof(pred_full, node_to_dof)
        true = restrict_to_dof(true_full, node_to_dof)
        err_l2 = relative_error(true, pred, M_full)
        err_h1 = relative_error(true, pred, A_diff)
        label = train_args.field or "campo"
        print(f"Errore {label} nel punto (mu1={args.mu1}, mu2={args.mu2}, mu_u={args.mu_u}):")
        print(f"  L2: {err_l2[0]:.4e}")
        print(f"  H1: {err_h1[0]:.4e}")
        plot_fields = [(label, true_full, pred_full)]
    else:
        pred_y_full_all = inverse_scale_channel(results_test[:, :, 0], scaler_test[0], train_args.scaling_type).numpy()
        pred_p_full_all = inverse_scale_channel(results_test[:, :, 1], scaler_test[1], train_args.scaling_type).numpy()
        pred_y_full, pred_p_full = pred_y_full_all[:, [local_idx]], pred_p_full_all[:, [local_idx]]
        true_y_full, true_p_full = single_dict["VX"], single_dict["VY"]
        pred_y, pred_p = restrict_to_dof(pred_y_full, node_to_dof), restrict_to_dof(pred_p_full, node_to_dof)
        true_y, true_p = restrict_to_dof(true_y_full, node_to_dof), restrict_to_dof(true_p_full, node_to_dof)
        err_y_l2 = relative_error(true_y, pred_y, M_full)
        err_y_h1 = relative_error(true_y, pred_y, A_diff)
        err_p_l2 = relative_error(true_p, pred_p, M_full)
        err_p_h1 = relative_error(true_p, pred_p, A_diff)
        print(f"Errore nel punto (mu1={args.mu1}, mu2={args.mu2}, mu_u={args.mu_u}):")
        print(f"  y - L2: {err_y_l2[0]:.4e}  H1: {err_y_h1[0]:.4e}")
        print(f"  p - L2: {err_p_l2[0]:.4e}  H1: {err_p_h1[0]:.4e}")
        plot_fields = [("y", true_y_full, pred_y_full), ("p", true_p_full, pred_p_full)]

    if args.plot:
        default_path = str(Path(tempfile.gettempdir()) / "gnn_single_solution.png")
        plot_comparison(mesh_data, plot_fields, args.mu1, args.mu2, args.mu_u, default_path)
        print(f"PLOT_PATH={default_path}")
        if args.save_plot:
            import shutil
            Path(args.save_plot).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(default_path, args.save_plot)
            print(f"Plot salvato anche in {args.save_plot}")


if __name__ == "__main__":
    main()
