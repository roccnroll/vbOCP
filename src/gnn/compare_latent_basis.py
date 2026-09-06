"""CLI: confronta i modi POD con i campi decodificati dalla GNN sulla base canonica
dello spazio ridotto (latente).

Idea: la GNN (rete combinata comp=2, un solo spazio latente bottleneck_dim per y e p
insieme) ha un decoder che mappa un vettore latente z (dimensione bottleneck_dim) in un
campo su tutta la mesh - esattamente come la base POD mappa un vettore di coefficienti
in un campo tramite combinazione lineare delle colonne della base. Decodificando i
vettori della base canonica e_i (1 nella posizione i, 0 altrove) invece dei coefficienti
veri prodotti da mapping(mu), si ottiene "il campo che il decoder associa alla direzione
i-esima dello spazio latente" - il confronto con basis_y[:,i]/basis_p[:,i] (i-esimo
modo POD) dice quanto le due rappresentazioni ridotte (lineare la POD, non lineare la
GNN) individuano direzioni simili nello spazio dei campi.

ATTENZIONE - il confronto e' solo qualitativo/esplorativo, non un errore rigoroso:
- Il decoder della GNN e' non lineare (mapping(mu) -> z non e' un cambio di base
  lineare come la POD): non c'e' garanzia teorica che le colonne "canoniche" del
  decoder abbiano un ordine o un segno paragonabili ai modi POD (ordinati per energia
  decrescente). Il confronto per indice i e' un'euristica, non un'identita' matematica.
- L'inverse-transform (denormalizzazione) applicato ai campi decodificati usa lo stesso
  scaler fittato sui dati REALI (scaler_test) - i vettori canonici non sono campioni
  reali, quindi la scala assoluta del risultato e' indicativa, non fisicamente esatta.
  Per il confronto si normalizzano comunque entrambi i lati a norma 1 prima del coseno.

Uso:
    python -m src.gnn.compare_latent_basis --config configs/test1.yaml \
        --gca-rom-path /path/to/gca-rom \
        --train-mat data/gnn/train_yp.mat --test-mat data/gnn/test_yp.mat \
        --net-dir data/gnn/models/test1_gnn_yp \
        --pod-model data/snapshots/test1_pod.npz \
        --n-modes 15 --output data/gnn/latent_basis_comparison.csv --save-plot
"""
import argparse
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
from src.full_order.assembly import assemble_operators
from src.rom.inner_product import assemble_full_mass_matrix
from src.gnn.train_gnn import build_combined_dataset, inverse_scale_channel, build_hyperparams
from src.gnn.convert_to_gca_rom import restrict_to_dof
from src.dl.common import normalize_minmax


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--gca-rom-path", required=True)
    parser.add_argument("--train-mat", required=True)
    parser.add_argument("--test-mat", required=True)
    parser.add_argument("--net-dir", required=True, help="cartella con pesi + train_meta.json (rete comp=2, y+p)")
    parser.add_argument("--pod-model", required=True, help="path al .npz da train_pod.py (basis_y, basis_p)")
    parser.add_argument("--n-modes", type=int, default=15,
                         help="quanti modi/direzioni confrontare (<= bottleneck_dim della rete e <= modi POD disponibili)")
    parser.add_argument("--output", default=None, help="opzionale: .csv con le similarita' coseno per modo")
    parser.add_argument("--save-plot", action="store_true",
                         help="salva un plot (norma L2 dei coefficienti sulla base FOM non serve qui: "
                              "confronto in norma euclidea su valori nodali) delle similarita' coseno per modo")
    return parser.parse_args()


def cosine_similarity(a, b):
    """Coseno tra due vettori, robusto al segno/scala arbitrari di entrambi i lati
    (sia i modi POD che le direzioni del decoder GNN sono definiti a meno di segno/scala)."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def main():
    args = parse_args()
    sys.path.insert(0, args.gca_rom_path)
    from gca_rom import network, preprocessing, initialization

    meta_path = Path(args.net_dir) / "train_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} non trovato - allenala prima con train_gnn.py")
    with open(meta_path) as f:
        meta = json.load(f)
    print(f"Iperparametri caricati da {meta_path}: {meta}")

    if meta["comp"] != 2:
        raise ValueError("Questo script confronta y e p insieme - serve una rete allenata con --comp 2 "
                          "(un solo spazio latente condiviso), non una rete a singolo campo")

    net_name = Path(args.net_dir.rstrip("/")).name
    train_args = SimpleNamespace(
        net_name=net_name, comp=2, field=None,
        scaling_type=meta["scaling_type"], scaler_number=meta["scaler_number"],
        ffn=meta["ffn"], map_nodes=meta["map_nodes"], bottleneck_dim=meta["bottleneck_dim"],
        lambda_map=meta["lambda_map"], in_channels=meta["in_channels"],
        epochs=meta["epochs"], batch_size=meta["batch_size"], minibatch=meta["minibatch"],
        lr=None,
    )
    bottleneck_dim = train_args.bottleneck_dim

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print(f"Caricamento mesh da {config['mesh']['path']} ...")
    mesh_data = load_mesh(config["mesh"]["path"], config["boundary_markers"])
    operators = assemble_operators(mesh_data, config["problem"]["omega_obs"])
    node_to_dof = operators["node_to_dof"]

    print(f"Caricamento train/test da {args.train_mat} / {args.test_mat} ...")
    dataset, params_np, train_snapshots, test_snapshots = build_combined_dataset(
        args.train_mat, args.test_mat, comp=2)
    n_param = params_np.shape[1]

    variable = "yp"
    HyperParams = build_hyperparams(network, train_args, variable, n_param, net_dir=args.net_dir)

    device = initialization.set_device()
    initialization.set_reproducibility(HyperParams)

    processor = preprocessing.SteadyDataProcessor()
    xx, yy, zz, xyz, var, var1, var2, num_graphs, _ = processor.prepare_var(dataset, HyperParams)
    VAR_all, VAR_test, scaler_all, scaler_test = processor.scale(
        HyperParams, dataset, test_snapshots, var, var1, var2)
    graphs, train_dataset, test_dataset = processor.append_graphs(
        HyperParams, VAR_all, dataset, num_graphs, xx, yy, train_snapshots, test_snapshots, zz)

    weights_path = Path(args.net_dir.rstrip("/") + "/") / f"{net_name}{HyperParams.net_run}.pt"
    print(f"Caricamento pesi da {weights_path} ...")
    model = network.Net(HyperParams).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to("cpu")

    if args.n_modes > bottleneck_dim:
        raise ValueError(f"--n-modes {args.n_modes} > bottleneck_dim della rete ({bottleneck_dim})")
    n_modes = args.n_modes

    # batch di n_modes grafi IDENTICI nella struttura (stessa mesh - il decoder usa solo
    # edge_index/edge_attr/num_graphs, non i valori nodali, vedi doc in cima al file) - i
    # primi n_modes grafi del test set bastano, i loro valori nodali non contano
    from torch_geometric.data import Batch
    graph_batch = Batch.from_data_list([test_dataset[i] for i in range(n_modes)])

    # base canonica dello spazio latente: e_i = 1 in posizione i, 0 altrove
    z_canonical = torch.eye(n_modes, bottleneck_dim, dtype=torch.get_default_dtype())

    print(f"Decodifica di {n_modes} vettori della base canonica (bottleneck_dim={bottleneck_dim}) ...")
    with torch.no_grad():
        decoded = model.solo_decoder(z_canonical, graph_batch)
    # solo_decoder su un batch di grafi restituisce i nodi concatenati lungo la prima
    # dimensione (n_modes * num_nodes, comp), non impilati in una dimensione a parte
    # (testing.evaluate() lo fa un grafo alla volta, batch_size=1, quindi non ha questo
    # problema) - li separiamo qui assumendo che Batch.from_data_list preservi l'ordine
    # dei nodi per grafo (vero per grafi con la stessa identica topologia)
    num_nodes = decoded.shape[0] // n_modes
    decoded = decoded.reshape(n_modes, num_nodes, HyperParams.comp)

    decoded_y_full = inverse_scale_channel(decoded[:, :, 0], scaler_test[0], train_args.scaling_type).numpy()
    decoded_p_full = inverse_scale_channel(decoded[:, :, 1], scaler_test[1], train_args.scaling_type).numpy()
    decoded_y = restrict_to_dof(decoded_y_full, node_to_dof)  # (Nh, n_modes)
    decoded_p = restrict_to_dof(decoded_p_full, node_to_dof)

    print(f"Caricamento base POD da {args.pod_model} ...")
    pod_data = np.load(args.pod_model)
    basis_y, basis_p = pod_data["basis_y"], pod_data["basis_p"]
    n_modes_pod = min(n_modes, basis_y.shape[1], basis_p.shape[1])
    if n_modes_pod < n_modes:
        print(f"Nota: la base POD ha solo {n_modes_pod} modi disponibili (< {n_modes} richiesti) - troncato")
        n_modes = n_modes_pod

    rows = []
    print(f"\n{'modo':>4}  {'cos(y_gnn, y_pod)':>18}  {'cos(p_gnn, p_pod)':>18}")
    for i in range(n_modes):
        cos_y = cosine_similarity(decoded_y[:, i], basis_y[:, i])
        cos_p = cosine_similarity(decoded_p[:, i], basis_p[:, i])
        rows.append({"mode": i + 1, "cos_y": cos_y, "cos_p": cos_p})
        print(f"{i + 1:>4}  {cos_y:>18.4f}  {cos_p:>18.4f}")

    if args.output is not None:
        import csv
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["mode", "cos_y", "cos_p"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nRisultati salvati in {args.output}")

    if args.save_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        modes = [r["mode"] for r in rows]
        ax.bar([m - 0.2 for m in modes], [abs(r["cos_y"]) for r in rows], width=0.4, label="|cos| stato (y)")
        ax.bar([m + 0.2 for m in modes], [abs(r["cos_p"]) for r in rows], width=0.4, label="|cos| aggiunto (p)")
        ax.set_xlabel("modo / direzione canonica")
        ax.set_ylabel("|similarita' coseno| con il modo POD corrispondente")
        ax.set_title("GNN (base canonica latente) vs modi POD")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plot_path = str(Path(args.output).with_suffix(".png")) if args.output else "latent_basis_comparison.png"
        plt.savefig(plot_path, dpi=100)
        print(f"Plot salvato in {plot_path}")


if __name__ == "__main__":
    main()
