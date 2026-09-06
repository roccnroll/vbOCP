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
from src.gnn.train_gnn import build_combined_dataset, build_hyperparams
from src.gnn.convert_to_gca_rom import restrict_to_dof, reconstruct_full_field
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
                         help="salva un plot a barre delle similarita' coseno per modo")
    parser.add_argument("--save-field-plots", action="store_true",
                         help="salva anche un confronto VISIVO (campo su mesh, POD vs GNN affiancati) "
                              "per i primi --plot-n-modes modi - piu' lento/pesante del plot a barre, "
                              "utile per vedere a occhio cosa significa un coseno alto o basso")
    parser.add_argument("--plot-n-modes", type=int, default=5,
                         help="quanti modi includere nel confronto visivo campo-su-mesh (default 5, "
                              "un plot per modo diventa pesante oltre una decina)")
    parser.add_argument("--plot-fields", default="y,p",
                         help="quali campi includere nel confronto visivo, separati da virgola "
                              "(default 'y,p', es. 'y' per il solo stato)")
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

    # ordine di "dominanza" delle direzioni canoniche: i vettori e_i restano quelli
    # (nessuna rotazione/PCA), ma li ordiniamo per quanto i codici latenti VERI (z =
    # mapping(mu) sui campioni di training) variano lungo quella coordinata - una
    # coordinata lungo cui z non varia mai e' una direzione che la rete non usa per
    # distinguere tra parametri diversi, analogo (ma piu' semplice della PCA completa,
    # che ruoterebbe gli assi) di un autovalore POD piccolo
    mu_stats = {"min": np.array(meta["mu_min"]), "max": np.array(meta["mu_max"])}
    params_train_norm = normalize_minmax(params_np[train_snapshots], mu_stats)
    params_train_t = torch.tensor(params_train_norm, dtype=torch.get_default_dtype())
    with torch.no_grad():
        z_train = model.mapping(params_train_t)
    variance_per_dim = z_train.var(dim=0).numpy()
    dominance_order = np.argsort(-variance_per_dim)  # indici originali, per varianza decrescente

    print(f"\nVarianza dei codici latenti per coordinata (ordine di dominanza, indice originale):")
    for rank, idx in enumerate(dominance_order):
        print(f"  rango {rank + 1:>2}: coordinata {idx + 1:>2}  varianza={variance_per_dim[idx]:.4e}")

    # decodifica UN vettore canonico alla volta (num_graphs=1 per chiamata) - stesso
    # identico pattern di testing.evaluate() (val_loader a batch_size=1), che assegna
    # model.solo_decoder(z_map, data) con un solo grafo in results[index, :, :]. Farlo
    # su un batch di piu' grafi insieme e' ambiguo (l'ordine di concatenazione dei nodi
    # lungo la prima dimensione non e' quello atteso - provato e sbagliato), un grafo
    # alla volta e' piu' lento ma inequivocabilmente corretto (15 forward pass, costo
    # trascurabile)
    # per ogni rango (0=piu' dominante) decodifica il vettore canonico e_idx con idx =
    # dominance_order[rango] - e_idx resta esattamente quello (nessuna rotazione), solo
    # l'ORDINE con cui li confrontiamo ai modi POD (anch'essi ordinati per energia
    # decrescente) segue la dominanza appena calcolata invece dell'indice grezzo 0..14
    from torch_geometric.data import Batch
    decoded_list = []
    for rank in range(n_modes):
        idx = dominance_order[rank]
        z_i = torch.zeros(1, bottleneck_dim, dtype=torch.get_default_dtype())
        z_i[0, idx] = 1.0
        single_graph = Batch.from_data_list([test_dataset[rank]])  # valori nodali non usati dal decoder
        with torch.no_grad():
            decoded_list.append(model.solo_decoder(z_i, single_graph))  # (num_nodes, comp)
    decoded = torch.stack(decoded_list, dim=0)  # (n_modes, num_nodes, comp)

    # inverse-transform PARZIALE: solo lo stadio dello scaler tarato per NODO (dimensione
    # = numero di nodi mesh, riusabile per un batch di qualunque dimensione) - verificato
    # empiricamente essere il SECONDO elemento della tupla (scaler_f), non il primo: il
    # primo (scaler_s) e' quello tarato esattamente sui 150 campioni REALI del test set
    # (stessa natura del bug di leak/popolazione-fissa gia' visto per
    # evaluate_gnn_single.py) - non puo' invertire un batch di n_modes vettori sintetici,
    # e infatti dava un errore di shape (scale_ di lunghezza 150). Per un confronto di
    # similarita' coseno (solo pattern spaziale, gia' normalizzato per norma) la
    # scala/offset per-campione che lo stadio saltato ripristinerebbe non e' comunque
    # significativa per un vettore canonico senza un "vero" campione a cui corrisponde.
    if train_args.scaling_type != 4:
        raise NotImplementedError("questo script assume scaling_type=4 (stesso usato in tutta la pipeline)")
    _, scaler_f_y = scaler_test[0]
    _, scaler_f_p = scaler_test[1]
    decoded_y_full = scaler_f_y.inverse_transform(decoded[:, :, 0].numpy()).T  # (num_nodes, n_modes)
    decoded_p_full = scaler_f_p.inverse_transform(decoded[:, :, 1].numpy()).T
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
    print(f"\n{'rango':>5}  {'coord.':>6}  {'cos(y_gnn, y_pod)':>18}  {'cos(p_gnn, p_pod)':>18}")
    for i in range(n_modes):
        cos_y = cosine_similarity(decoded_y[:, i], basis_y[:, i])
        cos_p = cosine_similarity(decoded_p[:, i], basis_p[:, i])
        rows.append({"mode": i + 1, "latent_coord": int(dominance_order[i]) + 1, "cos_y": cos_y, "cos_p": cos_p})
        print(f"{i + 1:>5}  {dominance_order[i] + 1:>6}  {cos_y:>18.4f}  {cos_p:>18.4f}")

    if args.output is not None:
        import csv
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["mode", "latent_coord", "cos_y", "cos_p"])
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

    if args.save_field_plots:
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

        n_plot = min(args.plot_n_modes, n_modes)
        base_path = str(Path(args.output).with_suffix("")) if args.output else "latent_basis_comparison"
        fields_to_plot = [f.strip() for f in args.plot_fields.split(",") if f.strip()]

        # segno e scala di entrambi i lati sono arbitrari (basi/direzioni, non campi fisici) -
        # normalizza a norma 1 e allinea il segno della GNN a quello del coseno gia' calcolato,
        # cosi' il confronto visivo non e' fuorviato da un colore invertito per puro segno
        def _prep(pod, gnn, cos):
            pod_n = pod / (np.linalg.norm(pod) or 1.0)
            gnn_n = gnn / (np.linalg.norm(gnn) or 1.0)
            if cos < 0:
                gnn_n = -gnn_n
            return pod_n, gnn_n

        for i in range(n_plot):
            # espande i modi POD (spazio DOF) a tutti i nodi mesh, come i campi decodificati
            # dalla GNN (gia' su tutti i nodi) - dirichlet_value=0 perche' un modo POD e' un
            # vettore di base, non un campo fisico con un vero valore al bordo
            field_data = {}
            if "y" in fields_to_plot:
                pod_y_full = reconstruct_full_field(basis_y[:, i], node_to_dof, dirichlet_value=0.0)
                field_data["y"] = _prep(pod_y_full, decoded_y_full[:, i], rows[i]["cos_y"])
            if "p" in fields_to_plot:
                pod_p_full = reconstruct_full_field(basis_p[:, i], node_to_dof, dirichlet_value=0.0)
                field_data["p"] = _prep(pod_p_full, decoded_p_full[:, i], rows[i]["cos_p"])

            fig, axes = plt.subplots(len(field_data), 3, figsize=(15, 4 * len(field_data)), squeeze=False)
            for row, label in enumerate(field_data):
                pod_field, gnn_field = field_data[label]
                # campi con segno (basi normalizzate, non quantita' fisiche positive) - palette
                # divergente centrata a zero (RdBu_r), non jet (non percettivamente uniforme e
                # senza uno zero neutro, fuorviante per dati con segno)
                vabs = max(abs(pod_field.min()), abs(pod_field.max()))
                levels = np.linspace(-vabs, vabs, 200) if vabs > 0 else 200
                cos_val = rows[i]["cos_y"] if label == "y" else rows[i]["cos_p"]

                ax = axes[row][0]
                tc = ax.tricontourf(triang, pod_field, levels=levels, cmap="RdBu_r")
                plt.colorbar(tc, ax=ax)
                ax.set_title(f"Modo POD {i + 1} ({label}, normalizzato)")
                ax.set_aspect("equal")

                ax = axes[row][1]
                tc = ax.tricontourf(triang, gnn_field, levels=levels, cmap="RdBu_r", extend="both")
                plt.colorbar(tc, ax=ax)
                ax.set_title(f"GNN e_{dominance_order[i] + 1} ({label}, |cos|={abs(cos_val):.3f})")
                ax.set_aspect("equal")

                ax = axes[row][2]
                # differenza assoluta: quantita' non negativa, palette sequenziale percettivamente
                # uniforme (viridis) invece di jet
                tc = ax.tricontourf(triang, np.abs(pod_field - gnn_field), levels=200, cmap="viridis")
                plt.colorbar(tc, ax=ax)
                ax.set_title(f"|differenza| ({label}, normalizzati)")
                ax.set_aspect("equal")

            fig.suptitle(f"Rango dominanza {i + 1} (coordinata latente {dominance_order[i] + 1}): "
                         f"POD vs GNN (base canonica)")
            plt.tight_layout()
            field_plot_path = f"{base_path}_mode{i + 1}.png"
            plt.savefig(field_plot_path, dpi=110)
            plt.close(fig)
            print(f"Confronto visivo modo {i + 1} salvato in {field_plot_path}")


if __name__ == "__main__":
    main()
