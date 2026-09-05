"""CLI: allena la FFNN mu -> coefficienti POD, per una variabile a scelta (y, p, u).

Script generico - sostituisce train_control_nn.py: funziona su qualunque
file .npz con chiavi f"coeffs_{field}", f"n_modes_{field}", mu1, mu2, mu_u
(prodotto da train_pod.py per y/p, o build_control_pod.py per u). Solo la
fase di training - separata dalla POD cosi' si puo' riallenare con
parametri diversi senza ricostruire la base.

Uso:
    python -m src.rom.train_reduced_nn --pod-model data/snapshots/test1_pod.npz \
        --field y --output data/snapshots/test1_y_nn.pt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.dl.common import (
    FFNN, train_ffnn,
    compute_minmax_stats, normalize_minmax,
    normalize_standard,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pod-model", required=True,
                         help="path al .npz da train_pod.py o build_control_pod.py")
    parser.add_argument("--field", required=True, choices=["y", "p", "u"],
                         help="quale variabile allenare (legge coeffs_<field>/n_modes_<field>)")
    parser.add_argument("--epochs", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--tol", type=float, default=1e-12,
                         help="soglia di stop anticipato sulla loss di training (default quasi disattivata: "
                              "1e-5 aveva senso solo con l'output standardizzato, su coefficienti grezzi "
                              "e' una soglia assoluta arbitraria che si raggiunge a scale diverse per y/p, "
                              "facendo fermare i due training a un numero di epoche diverso in modo spurio "
                              "- meglio lasciar decidere solo epochs/early stopping)")
    parser.add_argument("--hidden-dim", type=int, default=30)
    parser.add_argument("--n-hidden-layers", type=int, default=4)
    parser.add_argument("--n-modes", type=int, default=None,
                         help="numero di modi da usare (default: tutti quelli scelti dalla POD via "
                              "soglia di energia, n_modes_<field> nel pod-model). Se dato, deve essere "
                              "<= al numero disponibile - tronca ai primi N modi (i piu' energetici)")
    parser.add_argument("--output", required=True, help="path dove salvare i pesi della rete (.pt)")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Caricamento base POD da {args.pod_model} (campo: {args.field}) ...")
    pod_data = np.load(args.pod_model)
    coeffs = pod_data[f"coeffs_{args.field}"]  # (n_modes, n_samples)
    mu1, mu2, mu_u = pod_data["mu1"], pod_data["mu2"], pod_data["mu_u"]
    n_modes_available = int(pod_data[f"n_modes_{args.field}"])

    has_val = f"coeffs_{args.field}_test" in pod_data.files
    if has_val:
        coeffs_val = pod_data[f"coeffs_{args.field}_test"]
        mu1_val, mu2_val, mu_u_val = pod_data["mu1_test"], pod_data["mu2_test"], pod_data["mu_u_test"]
        print(f"Coefficienti di validazione trovati nel pod-model - early stopping attivo")
    else:
        print("Nessun coefficiente di validazione nel pod-model (rigenera con train_pod.py "
              "--test-snapshots per abilitare l'early stopping) - training senza early stopping, "
              "rischio di overfitting con molte epoche (vedi handout)")

    if args.n_modes is not None:
        if args.n_modes > n_modes_available:
            raise ValueError(
                f"--n-modes {args.n_modes} > modi disponibili nel pod-model ({n_modes_available}) - "
                f"ricostruisci la base POD con piu' modi se ne servono di piu'."
            )
        n_modes = args.n_modes
        coeffs = coeffs[:n_modes, :]  # primi N modi = i piu' energetici (ordinati per autovalore decrescente)
        if has_val:
            coeffs_val = coeffs_val[:n_modes, :]
        print(f"N modi: {n_modes} (esplicito, su {n_modes_available} disponibili)")
    else:
        n_modes = n_modes_available
        print(f"N modi: {n_modes} (automatico, dalla soglia di energia della POD)")

    x_raw = np.stack([mu1, mu2, mu_u], axis=1)
    y_raw = coeffs.T  # (n_samples, n_modes)

    # normalizzazione: input in [-1,1] (si abbina a Tanh). L'output (coefficienti POD)
    # NON viene standardizzato per modo: la MSE grezza pesa naturalmente di piu' i modi
    # a energia/varianza maggiore (i piu' importanti per la ricostruzione fisica e i piu'
    # puliti come dipendenza da mu), mentre standardizzare (varianza 1 per ogni modo)
    # da' lo stesso peso anche ai modi a bassa energia, spesso dominati da rumore/dipendenza
    # debole da mu - misurato empiricamente: standardizzare PEGGIORA l'errore relativo finale
    # di 2-4x sia su y che su p (vedi handout). y_stats resta a media 0/std 1 (identita')
    # solo per compatibilita' col formato del .norm.npz letto da evaluate_pod_nn.py.
    x_stats = compute_minmax_stats(x_raw)
    y_stats = {"mean": np.zeros(y_raw.shape[1]), "std": np.ones(y_raw.shape[1])}
    x_norm = normalize_minmax(x_raw, x_stats)
    y_norm = y_raw

    print(f"Training FFNN mu -> coefficienti POD({args.field}) (normalizzati) ...")
    x_train = torch.tensor(x_norm, dtype=torch.float32)
    y_train = torch.tensor(y_norm, dtype=torch.float32)

    x_val_t, y_val_t = None, None
    if has_val:
        x_val_raw = np.stack([mu1_val, mu2_val, mu_u_val], axis=1)
        x_val_t = torch.tensor(normalize_minmax(x_val_raw, x_stats), dtype=torch.float32)
        y_val_t = torch.tensor(normalize_standard(coeffs_val.T, y_stats), dtype=torch.float32)

    net = FFNN(input_dim=3, output_dim=n_modes, hidden_dim=args.hidden_dim, n_hidden_layers=args.n_hidden_layers)
    result = train_ffnn(net, x_train, y_train, epochs=args.epochs, lr=args.lr, tol=args.tol,
                         lr_drop_epoch=args.epochs // 2, return_history=True,
                         x_val=x_val_t, y_val=y_val_t)
    if has_val:
        net, loss_history, val_loss_history = result
    else:
        net, loss_history = result
        val_loss_history = None

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), args.output)

    # loss di training (e di validazione, se disponibile) ad ogni epoca, salvate accanto ai
    # pesi - permette di plottare la curva di convergenza senza dover riallenare
    loss_path = str(Path(args.output).with_suffix(".loss.npy"))
    np.save(loss_path, np.array(loss_history))
    print(f"Loss history salvata in {loss_path}")
    if val_loss_history is not None:
        val_loss_path = str(Path(args.output).with_suffix(".val_loss.npy"))
        np.save(val_loss_path, np.array(val_loss_history))
        print(f"Validation loss history salvata in {val_loss_path}")

    # statistiche di normalizzazione + architettura salvate accanto ai pesi - servono a
    # evaluate per de-normalizzare e per ricostruire la STESSA rete (stesse dimensioni)
    stats_path = str(Path(args.output).with_suffix(".norm.npz"))
    np.savez_compressed(
        stats_path,
        x_min=x_stats["min"], x_max=x_stats["max"],
        y_mean=y_stats["mean"], y_std=y_stats["std"],
        hidden_dim=args.hidden_dim, n_hidden_layers=args.n_hidden_layers,
        n_modes=n_modes,
    )
    print(f"Pesi rete salvati in {args.output}")
    print(f"Statistiche di normalizzazione salvate in {stats_path}")


if __name__ == "__main__":
    main()
