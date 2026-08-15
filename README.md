# vbOCP — varying-boundary Optimal Control Problem

Ricostruzione ed estensione di Strazzullo & Vicini (2023) su vbOCP
parametrico: FOM, ROM (POD/DEIM), confronto con GNN e Autoencoder come
surrogati, sulla classe di problemi vbOCP — non un singolo caso.

## Obiettivo

Confrontare **GNN** e **AE** contro il benchmark **POD** per la classe di
problemi vbOCP. Il primo caso implementato è `test1` (geometria
rettangolare); la repo è pensata per aggiungere un secondo caso a
geometria più complessa senza riscrivere il codice, solo aggiungendo un
nuovo config.

## Requisiti

- **Solo Linux** (Colab, Docker, cluster). Su Windows: usare WSL.
  (`pypolydim` non pubblica wheel per Windows — vedi `handout.md` per i
  dettagli del test.)
- Python ≥ 3.11

## Setup

```bash
conda env create -f env/environment.yml
conda activate vbocp
```

## Struttura

```
src/
  snapshots/   # mesh, assembly, solve FOM — genera gli snapshot
  rom/         # POD, DEIM, (poi GNN/AE) — legge gli snapshot generati
notebooks/     # notebook interattivi (uso esplorativo/didattico)
configs/       # un file per caso (parametri geometrici, range mu, ...)
data/
  meshes/      # mesh generate (csv)
  snapshots/   # snapshot generati (.npz)
env/           # environment.yml
```

## Come lanciare una run

*(da completare quando gli script CLI saranno pronti)*

## Riferimenti

Vedi `bibbliografia/` e `handout.md` per il log delle decisioni prese
sessione per sessione.
