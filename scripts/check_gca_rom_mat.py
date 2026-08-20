"""Verifica ad-hoc di un .mat convertito per gca-rom: shape, indici, caricamento con LoadDataset.

Uso:
    python scripts/check_gca_rom_mat.py data/gnn/test1_300_y.mat
"""
import sys

import numpy as np
import scipy.io

sys.path.insert(0, "gca-rom")  # aggiusta se il path della repo gca-rom e' diverso


def main():
    path = sys.argv[1]
    mat = scipy.io.loadmat(path)

    print("Chiavi nel .mat:", [k for k in mat if not k.startswith("__")])
    xx, yy, T, E = mat["xx"], mat["yy"], mat["T"], mat["E"]
    print(f"xx shape: {xx.shape}  yy shape: {yy.shape}  T shape: {T.shape}  E shape: {E.shape}")

    if "U" in mat:
        print(f"U shape: {mat['U'].shape}")
    if "VX" in mat:
        print(f"VX shape: {mat['VX'].shape}  VY shape: {mat['VY'].shape}")
    print(f"params shape: {mat['params'].shape}")

    num_nodes = xx.shape[0]
    print(f"num_nodes (da xx): {num_nodes}")

    # T ed E sono 1-indicizzati - controlla che non sforino i limiti
    print(f"T min/max (atteso 1..{num_nodes}): {T.min()} / {T.max()}")
    print(f"E min/max (atteso 1..{num_nodes}): {E.min()} / {E.max()}")
    assert T.min() >= 1 and T.max() <= num_nodes, "T fuori dai limiti dei nodi"
    assert E.min() >= 1 and E.max() <= num_nodes, "E fuori dai limiti dei nodi"

    # xx/yy devono essere costanti per riga (stessa mesh per ogni snapshot)
    assert np.allclose(xx, xx[:, [0]]), "xx non costante tra le colonne - mesh diversa per snapshot?"
    assert np.allclose(yy, yy[:, [0]]), "yy non costante tra le colonne - mesh diversa per snapshot?"

    print("\nControlli di shape/indici OK. Provo a caricare con gca_rom.loader.LoadDataset ...")

    from gca_rom import loader
    n_comp = 2 if "VX" in mat else 1
    dataset = loader.LoadDataset(path, variable="U", dim_pde=2, n_comp=n_comp)
    print("LoadDataset OK.")
    print("dataset.xx shape:", dataset.xx.shape)
    if n_comp == 1:
        print("dataset.U shape:", dataset.U.shape)
    else:
        print("dataset.VX shape:", dataset.VX.shape, " dataset.VY shape:", dataset.VY.shape)
    print("dataset.params shape:", None if dataset.params is None else dataset.params.shape)


if __name__ == "__main__":
    main()
