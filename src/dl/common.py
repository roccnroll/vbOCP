import numpy as np
import torch
import torch.nn as nn


def compute_minmax_stats(x):
    """Statistiche min-max per colonna, per normalize_minmax/denormalize_minmax.

    Args:
        x: array (n_samples, n_features)

    Returns:
        dict con 'min' e 'max', array (n_features,)
    """
    return {"min": x.min(axis=0), "max": x.max(axis=0)}


def normalize_minmax(x, stats):
    """Porta x in [-1, 1] per colonna, usando le statistiche di compute_minmax_stats.

    Adatto agli input di una rete con attivazioni Tanh (che satura per input grandi).
    """
    range_ = stats["max"] - stats["min"]
    range_ = np.where(range_ > 0, range_, 1.0)  # evita divisione per zero se una colonna e' costante
    return 2.0 * (x - stats["min"]) / range_ - 1.0


def denormalize_minmax(x_norm, stats):
    """Inverte normalize_minmax."""
    range_ = stats["max"] - stats["min"]
    return (x_norm + 1.0) / 2.0 * range_ + stats["min"]


def compute_standard_stats(x):
    """Statistiche media/deviazione standard per colonna, per normalize_standard/denormalize_standard.

    Args:
        x: array (n_samples, n_features)

    Returns:
        dict con 'mean' e 'std', array (n_features,)
    """
    return {"mean": x.mean(axis=0), "std": x.std(axis=0)}


def normalize_standard(x, stats):
    """Standardizza x per colonna (media 0, dev. standard 1).

    Adatto a target come i coefficienti POD, che possono avere scale molto
    diverse tra loro (i primi modi hanno coefficienti piu' grandi degli ultimi).
    """
    std = np.where(stats["std"] > 0, stats["std"], 1.0)  # evita divisione per zero
    return (x - stats["mean"]) / std


def denormalize_standard(x_norm, stats):
    """Inverte normalize_standard."""
    return x_norm * stats["std"] + stats["mean"]


class FFNN(nn.Module):
    """Rete feed-forward semplice: input -> N layer nascosti (Tanh) -> output lineare.

    Stesso pattern del notebook del prof (Lab9/PODnn.ipynb, classe Net):
    usata per imparare mu -> coefficienti ridotti (POD, o in futuro altri
    spazi latenti).
    """

    def __init__(self, input_dim, output_dim, hidden_dim=30, n_hidden_layers=4):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers += [nn.Linear(hidden_dim, output_dim)]  # output lineare, nessuna attivazione
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_ffnn(net, x_train, y_train, epochs=20000, lr=1e-3, lr_drop_epoch=None, lr_drop_factor=0.1,
               tol=1e-5, print_every=2000, device=None, return_history=False, x_val=None, y_val=None):
    """Allena una FFNN con Adam + MSE, full-batch, come nel notebook del prof.

    Usa la GPU se disponibile (rilevata automaticamente), altrimenti CPU -
    nessuna differenza di comportamento per chi chiama, solo piu' veloce se
    c'e' una GPU. Il modello viene riportato su CPU prima di essere
    restituito, cosi' salvataggio/valutazione altrove restano invariati
    (sempre su CPU).

    Args:
        net: modello (es. FFNN)
        x_train, y_train: tensori torch, stesso numero di righe (n_samples, ...)
        epochs: numero massimo di epoche
        lr: learning rate iniziale
        lr_drop_epoch: se dato, abbassa il lr a quell'epoca (come il prof a 20000)
        lr_drop_factor: fattore di riduzione del lr
        tol: soglia di loss sotto la quale ci si ferma prima delle epoche massime
        print_every: ogni quante epoche stampare la loss
        device: "cuda"/"cpu"/None (None = auto-rileva se la GPU e' disponibile)
        return_history: se True, restituisce anche la lista della loss ad ogni epoca
            (per diagnosticare se la loss e' ancora in discesa o e' in plateau - non
            impatta il training, solo cosa viene restituito)
        x_val, y_val: se dati (es. i coefficienti POD veri sul test set, stessa normalizzazione
            di y_train), registrano anche la loss di VALIDAZIONE ad ogni epoca - non entra mai
            nel training (nessun backward su questi dati), serve solo a diagnosticare overfitting
            (loss di training in discesa ma di validazione in salita)

    Returns:
        net allenato, riportato su CPU. Se return_history=True: (net, loss_history) oppure
        (net, loss_history, val_loss_history) se anche x_val/y_val sono dati.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    net = net.to(device)
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    track_val = x_val is not None and y_val is not None
    if track_val:
        x_val = x_val.to(device)
        y_val = y_val.to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    loss_value = float("inf")
    epoch = 0
    loss_history = []
    val_loss_history = []
    while loss_value >= tol and epoch < epochs:
        epoch += 1
        optimizer.zero_grad()

        output = net(x_train)
        loss = loss_fn(output, y_train)

        if lr_drop_epoch is not None and epoch == lr_drop_epoch:
            optimizer.param_groups[0]["lr"] *= lr_drop_factor

        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        if return_history:
            loss_history.append(loss_value)
        if track_val:
            with torch.no_grad():
                val_loss_value = loss_fn(net(x_val), y_val).item()
            if return_history:
                val_loss_history.append(val_loss_value)
        if epoch % print_every == 0:
            msg = f"  epoch {epoch}  loss {loss_value:.6e}"
            if track_val:
                msg += f"  val_loss {val_loss_value:.6e}"
            msg += f"  lr {optimizer.param_groups[0]['lr']:.1e}  device {device}"
            print(msg)

    net = net.to("cpu")
    if return_history:
        if track_val:
            return net, loss_history, val_loss_history
        return net, loss_history
    return net
