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
               tol=1e-5, print_every=2000):
    """Allena una FFNN con Adam + MSE, full-batch, come nel notebook del prof.

    Args:
        net: modello (es. FFNN)
        x_train, y_train: tensori torch, stesso numero di righe (n_samples, ...)
        epochs: numero massimo di epoche
        lr: learning rate iniziale
        lr_drop_epoch: se dato, abbassa il lr a quell'epoca (come il prof a 20000)
        lr_drop_factor: fattore di riduzione del lr
        tol: soglia di loss sotto la quale ci si ferma prima delle epoche massime
        print_every: ogni quante epoche stampare la loss

    Returns:
        net allenato (stesso oggetto, modificato in place)
    """
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    loss_value = float("inf")
    epoch = 0
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
        if epoch % print_every == 0:
            print(f"  epoch {epoch}  loss {loss_value:.6e}  lr {optimizer.param_groups[0]['lr']:.1e}")

    return net
