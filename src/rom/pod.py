import numpy as np


def compute_correlation_eigenvalues(snapshot_matrix, inner_product):
    """Calcola autovalori/autovettori della matrice di correlazione POD.

    Non dipende da N: serve per il plot di decadimento e per scegliere N,
    prima ancora di costruire la base vera e propria (build_pod_basis).

    Args:
        snapshot_matrix: array (Nh, M) - una colonna per snapshot
        inner_product: matrice (Nh, Nh) che definisce il prodotto scalare

    Returns:
        eigenvalues: array (M,), ordine decrescente
        eigenvectors: array (M, M), colonne = autovettori, stesso ordine
    """
    # matrice di correlazione M x M, normalizzata per M come nell'eq. 17 del
    # paper (l'errore medio di proiezione = somma degli autovalori solo se C
    # include il fattore 1/M) - non cambia la base, solo la scala di lambda
    M = snapshot_matrix.shape[1]
    C = (snapshot_matrix.T @ (inner_product @ snapshot_matrix)) / M

    eigenvalues, eigenvectors = np.linalg.eigh(C)

    # eigh ordina in modo crescente, la POD vuole i piu' grandi prima
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]

    return eigenvalues, eigenvectors


def select_n_modes(eigenvalues, energy_threshold=0.9999, max_modes=150):
    """Sceglie N automaticamente in base all'energia cumulata relativa.

    Stesso criterio del notebook del prof (Lab4/POD.ipynb, cella 19): il piu'
    piccolo N per cui la somma dei primi N autovalori supera la soglia
    dell'energia totale, con un tetto massimo di sicurezza.

    Args:
        eigenvalues: array di autovalori (ordine decrescente)
        energy_threshold: soglia di energia cumulata relativa (es. 0.9999)
        max_modes: numero massimo di modi, anche se la soglia non e' raggiunta

    Returns:
        N: numero di modi scelto
    """
    total_energy = np.sum(eigenvalues)
    cumulative_energy = np.cumsum(eigenvalues)
    relative_energy = cumulative_energy / total_energy

    if np.any(relative_energy >= energy_threshold):
        n = int(np.argmax(relative_energy >= energy_threshold)) + 1
    else:
        n = len(eigenvalues)

    return min(n, max_modes)


def build_pod_basis(snapshot_matrix, inner_product, n_modes):
    """Costruisce una base POD da una matrice di snapshot, dato un prodotto scalare.

    Segue la ricetta del paper (Strazzullo & Vicini 2023, eq. 15-17): matrice
    di correlazione pesata dal prodotto scalare, autovalori/autovettori,
    combinazione lineare degli snapshot normalizzata nella stessa norma.

    Args:
        snapshot_matrix: array (Nh, M) - una colonna per snapshot
        inner_product: matrice (Nh, Nh) che definisce il prodotto scalare
            (es. A_diff per H1-seminorma, A_diff+M_full per H1 completa)
        n_modes: N, numero di modi da tenere nella base ridotta

    Returns:
        basis: array (Nh, n_modes) - le colonne sono i modi POD, ortonormali
            nel prodotto scalare dato
        eigenvalues: array (M,) tutti gli autovalori, ordine decrescente
            (utile per il plot di decadimento, non solo i primi N)
    """
    eigenvalues, eigenvectors = compute_correlation_eigenvalues(snapshot_matrix, inner_product)

    basis = np.zeros((snapshot_matrix.shape[0], n_modes))
    for n in range(n_modes):
        omega_n = eigenvectors[:, n]
        chi_n = snapshot_matrix @ omega_n
        norm = np.sqrt(chi_n @ (inner_product @ chi_n))
        basis[:, n] = chi_n / norm

    return basis, eigenvalues


def plot_eigenvalue_decay(eigenvalues_y, eigenvalues_p, output_path=None, max_n=80):
    """Plotta il decadimento degli autovalori POD per stato (y) e aggiunto (p).

    Confrontabile con il plot di destra di Figura 4 nel paper.

    Args:
        eigenvalues_y: array di autovalori per lo stato
        eigenvalues_p: array di autovalori per l'aggiunto
        output_path: se dato, salva il PNG invece di mostrarlo (utile da script)
        max_n: numero massimo di autovalori mostrati nel plot (i piu' piccoli,
            in coda, sono spesso solo rumore numerico e non interessano)
    """
    import matplotlib
    if output_path is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eigenvalues_y = eigenvalues_y[:max_n]
    eigenvalues_p = eigenvalues_p[:max_n]

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.semilogy(range(1, len(eigenvalues_y) + 1), eigenvalues_y, "o-", label="Stato (y)")
    ax.semilogy(range(1, len(eigenvalues_p) + 1), eigenvalues_p, "s-", label="Aggiunto (p)")

    ax.set_xlabel("N")
    ax.set_ylabel(r"$\lambda_N$")
    ax.set_title("Decadimento autovalori POD")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=80)
        print(f"Plot salvato in {output_path}")
    else:
        plt.show()
