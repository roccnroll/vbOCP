"""CLI: allena un GCA-ROM (gca-rom, libreria non modificata) su y, p, o entrambi combinati.

Bypassa pde.problem()/gui.hyperparameters_selection di gca-rom (pensati per i
benchmark del paper originale, con parametri costruiti a griglia via
itertools.product(mu_space)): qui i .mat sono gia' pronti (vedi
convert_to_gca_rom.py) con i mu1,mu2,mu_u REALI in dataset.params, letti
direttamente. HyperParams e' costruito a mano con lo stesso schema di argv
usato da gui.py (vedi preset "poisson" nel ramo headless), non tramite GUI.

Train e test sono due file .mat separati, generati dagli stessi snapshot
FOM usati per POD/PODNN (stesso split, stesso test set) - non lo split
random interno di gca-rom (SteadyDataProcessor.compute_indices), che
mischierebbe train/test in modo diverso da quello con cui e' stata
confrontata la POD.

Valutazione: la libreria (gca_rom.error/scaling.inverse_scaling) ha un bug
per comp=2 (usa sempre tensor[:,:,0], ignora il secondo canale) - qui non la
usiamo: inverse-transform scritto a mano per ciascun canale, poi errore in
norma L2 (M_full) e H1 (A_diff) per y/p (stessa formula di evaluate_pod_nn.py),
o euclidea per un confronto generico.

Uso (richiede la repo gca-rom clonata, path passato con --gca-rom-path):
    python -m src.gnn.train_gnn --config configs/test1.yaml \
        --gca-rom-path /path/to/gca-rom \
        --train-mat data/gnn/test1_300_y.mat --test-mat data/gnn/test1_test150_y.mat \
        --comp 1 --field y --net-name test1_gnn_y --net-dir data/gnn/models \
        --epochs 5000
"""
import argparse
import sys
import types
from pathlib import Path

import numpy as np
import scipy.io
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.full_order.mesh import load_mesh
from src.full_order.assembly import assemble_operators
from src.rom.inner_product import assemble_full_mass_matrix


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--gca-rom-path", required=True, help="path alla repo gca-rom clonata (per il sys.path)")
    parser.add_argument("--train-mat", required=True, help=".mat di training (da convert_to_gca_rom.py)")
    parser.add_argument("--test-mat", required=True, help=".mat di test (da convert_to_gca_rom.py)")
    parser.add_argument("--comp", type=int, choices=[1, 2], required=True)
    parser.add_argument("--field", choices=["y", "p"], default=None,
                         help="richiesto se --comp 1: quale campo (per il nome variabile e la norma)")
    parser.add_argument("--net-name", required=True, help="nome della run (sottocartella di --net-dir)")
    parser.add_argument("--net-dir", required=True, help="cartella dove gca-rom salva pesi/log")
    parser.add_argument("--bottleneck-dim", type=int, default=15)
    parser.add_argument("--ffn", type=int, default=200, help="nodi del feedforward nell'encoder/decoder")
    parser.add_argument("--map-nodes", type=int, default=50, help="nodi dell'MLP mu->latente")
    parser.add_argument("--in-channels", type=int, default=2, help="numero di layer GMMConv")
    parser.add_argument("--lambda-map", type=float, default=10.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--scaling-type", type=int, default=4, choices=[1, 2, 3, 4],
                         help="1=sample 2=feature 3=feature-sample 4=sample-feature (default gca-rom per poisson)")
    parser.add_argument("--scaler-number", type=int, default=3, choices=[1, 2, 3],
                         help="1=minmax 2=robust 3=standard")
    return parser.parse_args()


def build_combined_dataset(train_mat_path, test_mat_path, comp):
    """Carica i due .mat (train/test) e li concatena in un unico oggetto 'dataset'
    compatibile con gca_rom.preprocessing (stessa interfaccia di loader.LoadDataset),
    con indici espliciti train/test invece dello split casuale interno alla libreria."""
    train_mat = scipy.io.loadmat(train_mat_path)
    test_mat = scipy.io.loadmat(test_mat_path)

    n_train = train_mat["xx"].shape[1]
    n_test = test_mat["xx"].shape[1]

    dataset = types.SimpleNamespace()
    dataset.dim = 2
    dataset.xx = torch.tensor(np.concatenate([train_mat["xx"], test_mat["xx"]], axis=1))
    dataset.yy = torch.tensor(np.concatenate([train_mat["yy"], test_mat["yy"]], axis=1))
    dataset.T = torch.tensor(train_mat["T"].astype(int))
    dataset.E = torch.tensor(train_mat["E"].astype(int))

    if comp == 1:
        dataset.U = torch.tensor(np.concatenate([train_mat["U"], test_mat["U"]], axis=1))
    else:
        dataset.VX = torch.tensor(np.concatenate([train_mat["VX"], test_mat["VX"]], axis=1))
        dataset.VY = torch.tensor(np.concatenate([train_mat["VY"], test_mat["VY"]], axis=1))

    params = np.concatenate([train_mat["params"], test_mat["params"]], axis=0)
    train_snapshots = list(range(n_train))
    test_snapshots = list(range(n_train, n_train + n_test))

    return dataset, params, train_snapshots, test_snapshots


def inverse_scale_channel(tensor_2d, scale, scaling_type):
    """Come gca_rom.scaling.inverse_scaling, ma su un tensore 2D gia' estratto
    (un solo canale) invece di prendere sempre tensor[:,:,0] - fix del bug per comp=2."""
    arr = tensor_2d.detach().numpy()
    if scaling_type == 1:
        return torch.tensor(scale.inverse_transform(arr.T))
    elif scaling_type == 2:
        return torch.tensor(scale.inverse_transform(arr)).T
    elif scaling_type == 3:
        scaler_f, scaler_s = scale
        return torch.t(torch.tensor(scaler_f.inverse_transform(torch.tensor(scaler_s.inverse_transform(arr)))))
    elif scaling_type == 4:
        scaler_s, scaler_f = scale
        return torch.tensor(scaler_s.inverse_transform(torch.t(torch.tensor(scaler_f.inverse_transform(arr)))))


def relative_error(true, pred, norm_matrix):
    """Errore relativo per campione in norma indotta da norm_matrix (sparse-safe) - come evaluate_pod_nn.py."""
    diff = pred - true
    err_sq = np.sum(diff * (norm_matrix @ diff), axis=0)
    true_sq = np.sum(true * (norm_matrix @ true), axis=0)
    errors = np.sqrt(np.abs(err_sq))
    norms = np.sqrt(np.abs(true_sq))
    return errors / np.where(norms > 0, norms, 1.0)


def main():
    args = parse_args()
    sys.path.insert(0, args.gca_rom_path)
    from gca_rom import network, preprocessing, training, testing, initialization

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print(f"Caricamento mesh da {config['mesh']['path']} (per le norme L2/H1) ...")
    mesh_data = load_mesh(config["mesh"]["path"], config["boundary_markers"])
    operators = assemble_operators(mesh_data, config["problem"]["omega_obs"])
    M_full = assemble_full_mass_matrix(mesh_data)
    A_diff = operators["A_diff"]

    print(f"Caricamento train/test da {args.train_mat} / {args.test_mat} ...")
    dataset, params_np, train_snapshots, test_snapshots = build_combined_dataset(
        args.train_mat, args.test_mat, args.comp)
    n_param = params_np.shape[1]
    print(f"Campioni: {len(train_snapshots)} train, {len(test_snapshots)} test, {n_param} parametri")

    variable = args.field if args.comp == 1 else "yp"
    argv = [
        args.net_name, variable,
        args.scaling_type, args.scaler_number,
        1,  # skip connections
        80,  # rate - non usato: bypassiamo compute_indices, train/test sono espliciti
        args.ffn, args.map_nodes, args.bottleneck_dim,
        args.lambda_map, args.in_channels,
        n_param, args.epochs, args.comp,
    ]
    HyperParams = network.HyperParams(argv)
    HyperParams.net_dir = args.net_dir.rstrip("/") + "/"
    HyperParams.learning_rate = args.lr

    device = initialization.set_device()
    initialization.set_reproducibility(HyperParams)
    initialization.set_path(HyperParams)

    processor = preprocessing.SteadyDataProcessor()  # solo per riusare i metodi non-astratti
    xx, yy, zz, xyz, var, var1, var2, num_graphs, _ = processor.prepare_var(dataset, HyperParams)
    VAR_all, VAR_test, scaler_all, scaler_test = processor.scale(
        HyperParams, dataset, test_snapshots, var, var1, var2)
    graphs, train_dataset, test_dataset = processor.append_graphs(
        HyperParams, VAR_all, dataset, num_graphs, xx, yy, train_snapshots, test_snapshots, zz)
    graph_loader, train_loader, test_loader, val_loader = processor.return_loaders(
        HyperParams, graphs, train_dataset, test_dataset, len(train_snapshots), len(test_snapshots))

    model = network.Net(HyperParams).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=HyperParams.learning_rate, weight_decay=HyperParams.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=HyperParams.miles, gamma=HyperParams.gamma)

    params = torch.tensor(params_np, dtype=torch.get_default_dtype()).to(device)

    print("Training GCA-ROM ...")
    training.train(model, optimizer, device, scheduler, params, train_loader, test_loader,
                    train_snapshots, test_snapshots, HyperParams)

    print("Valutazione sul test set ...")
    model.to("cpu")
    params_cpu = params.to("cpu")
    results_test, _, _ = testing.evaluate(VAR_test, model, val_loader, params_cpu, HyperParams, test_snapshots)

    if args.comp == 1:
        pred = inverse_scale_channel(results_test[:, :, 0], scaler_test, args.scaling_type).numpy()
        true = dataset.U[:, test_snapshots].numpy()
        norm_matrix = M_full if args.field is None else M_full  # L2 per default
        err_l2 = relative_error(true, pred, M_full)
        err_h1 = relative_error(true, pred, A_diff)
        print(f"Errore {args.field or 'campo'} - L2: medio {err_l2.mean():.4e}  H1: medio {err_h1.mean():.4e}")
    else:
        pred_y = inverse_scale_channel(results_test[:, :, 0], scaler_test[0], args.scaling_type).numpy()
        pred_p = inverse_scale_channel(results_test[:, :, 1], scaler_test[1], args.scaling_type).numpy()
        true_y = dataset.VX[:, test_snapshots].numpy()
        true_p = dataset.VY[:, test_snapshots].numpy()
        err_y_l2 = relative_error(true_y, pred_y, M_full)
        err_y_h1 = relative_error(true_y, pred_y, A_diff)
        err_p_l2 = relative_error(true_p, pred_p, M_full)
        err_p_h1 = relative_error(true_p, pred_p, A_diff)
        print(f"Errore y - L2: medio {err_y_l2.mean():.4e}  H1: medio {err_y_h1.mean():.4e}")
        print(f"Errore p - L2: medio {err_p_l2.mean():.4e}  H1: medio {err_p_h1.mean():.4e}")


if __name__ == "__main__":
    main()
