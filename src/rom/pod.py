import numpy as np


def compute_correlation_eigenvalues(snapshot_matrix, inner_product, normalize=True):
    """Calcola autovalori/autovettori della matrice di correlazione POD.

    Non dipende da N: serve per il plot di decadimento e per scegliere N,
    prima ancora di costruire la base vera e propria (build_pod_basis).

    Args:
        snapshot_matrix: array (Nh, M) - una colonna per snapshot
        inner_product: matrice (Nh, Nh) che definisce il prodotto scalare
        normalize: se True (default), divide la matrice di correlazione per M
            come nell'eq. 17 del paper (l'errore medio di proiezione = somma
            degli autovalori solo se C include il fattore 1/M). Non cambia la
            base ne' N scelto da select_n_modes (soglia = rapporto, il
            fattore si semplifica) - influisce solo sulla scala di lambda
            mostrata nei plot.

    Returns:
        eigenvalues: array (M,), ordine decrescente
        eigenvectors: array (M, M), colonne = autovettori, stesso ordine
    """
    C = snapshot_matrix.T @ (inner_product @ snapshot_matrix)
    if normalize:
        C = C / snapshot_matrix.shape[1]

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


def build_pod_basis_from_eigenvectors(snapshot_matrix, inner_product, eigenvectors, n_modes):
    """Costruisce la base POD dati gli autovettori gia' calcolati (senza rifare l'eigh).

    Stessa formula di build_pod_basis, ma separata cosi' si puo' costruire la
    base a piu' N diversi riusando la stessa decomposizione - utile per una
    curva di errore di ricostruzione al variare di N (vedi
    compute_reconstruction_error_curve), che altrimenti richiederebbe un eigh
    per ogni punto della curva.

    Args:
        snapshot_matrix: array (Nh, M) - una colonna per snapshot
        inner_product: matrice (Nh, Nh) che definisce il prodotto scalare
        eigenvectors: array (M, M) - da compute_correlation_eigenvalues, stessi snapshot
        n_modes: N, numero di modi da tenere nella base ridotta

    Returns:
        basis: array (Nh, n_modes), ortonormale nel prodotto scalare dato
    """
    basis = np.zeros((snapshot_matrix.shape[0], n_modes))
    for n in range(n_modes):
        omega_n = eigenvectors[:, n]
        chi_n = snapshot_matrix @ omega_n
        norm = np.sqrt(chi_n @ (inner_product @ chi_n))
        basis[:, n] = chi_n / norm

    return basis


def build_pod_basis(snapshot_matrix, inner_product, n_modes, normalize=True):
    """Costruisce una base POD da una matrice di snapshot, dato un prodotto scalare.

    Segue la ricetta del paper (Strazzullo & Vicini 2023, eq. 15-17): matrice
    di correlazione pesata dal prodotto scalare, autovalori/autovettori,
    combinazione lineare degli snapshot normalizzata nella stessa norma.

    Args:
        snapshot_matrix: array (Nh, M) - una colonna per snapshot
        inner_product: matrice (Nh, Nh) che definisce il prodotto scalare
            (es. A_diff per H1-seminorma, A_diff+M_full per H1 completa)
        n_modes: N, numero di modi da tenere nella base ridotta
        normalize: vedi compute_correlation_eigenvalues - non cambia la base
            (solo la scala di lambda), passato per coerenza col plot

    Returns:
        basis: array (Nh, n_modes) - le colonne sono i modi POD, ortonormali
            nel prodotto scalare dato
        eigenvalues: array (M,) tutti gli autovalori, ordine decrescente
            (utile per il plot di decadimento, non solo i primi N)
    """
    eigenvalues, eigenvectors = compute_correlation_eigenvalues(snapshot_matrix, inner_product, normalize)
    basis = build_pod_basis_from_eigenvectors(snapshot_matrix, inner_product, eigenvectors, n_modes)
    return basis, eigenvalues


def compute_reconstruction_error_curve(snapshot_matrix_train, snapshot_matrix_test, inner_product,
                                        eigenvectors, n_values):
    """Errore di ricostruzione solo-POD (proiezione di Galerkin) sul test set, per una lista di N.

    Riusa gli stessi autovettori (calcolati una volta da compute_correlation_eigenvalues
    sul training) per costruire basi via build_pod_basis_from_eigenvectors a ogni N -
    nessun eigh ripetuto, quindi economico anche per molti valori di N.

    Args:
        snapshot_matrix_train: array (Nh, M_train) - per costruire le basi
        snapshot_matrix_test: array (Nh, M_test) - su cui si misura l'errore
        inner_product: matrice (Nh, Nh) (sparse o densa - sparse-safe)
        eigenvectors: array (M_train, M_train) - da compute_correlation_eigenvalues sul training
        n_values: lista/array di N da valutare

    Returns:
        errors: array, stessa lunghezza di n_values - errore relativo medio per campione di test
    """
    errors = []
    for n_modes in n_values:
        basis = build_pod_basis_from_eigenvectors(snapshot_matrix_train, inner_product, eigenvectors, n_modes)
        coeffs_test = project_onto_basis(snapshot_matrix_test, basis, inner_product)
        reconstructed = basis @ coeffs_test

        diff = reconstructed - snapshot_matrix_test
        err_sq = np.sum(diff * (inner_product @ diff), axis=0)
        true_sq = np.sum(snapshot_matrix_test * (inner_product @ snapshot_matrix_test), axis=0)
        rel_err = np.sqrt(np.abs(err_sq)) / np.where(np.sqrt(np.abs(true_sq)) > 0, np.sqrt(np.abs(true_sq)), 1.0)
        errors.append(rel_err.mean())

    return np.array(errors)


def project_onto_basis(snapshot_matrix, basis, inner_product):
    """Proietta gli snapshot sulla base (proiezione di Galerkin), per il training della PODNN.

    Stesso pattern del notebook del prof (Lab9/PODnn.ipynb, cella 39): il
    target per la rete non e' il coefficiente "grezzo" della POD, ma la
    proiezione di Galerkin di ciascuno snapshot sulla base, risolvendo
    (B^T X B) u_rb = B^T X snapshot - piu' robusto della semplice B^T X
    snapshot se la base non fosse perfettamente ortonormale.

    Args:
        snapshot_matrix: array (Nh, M) - una colonna per snapshot
        basis: array (Nh, N) - base ridotta (da build_pod_basis)
        inner_product: matrice (Nh, Nh) del prodotto scalare

    Returns:
        coeffs: array (N, M) - coefficienti ridotti, una colonna per snapshot
    """
    reduced_inner_product = basis.T @ (inner_product @ basis)  # (N, N)
    rhs = basis.T @ (inner_product @ snapshot_matrix)  # (N, M)
    coeffs = np.linalg.solve(reduced_inner_product, rhs)
    return coeffs


def plot_eigenvalue_decay_curves(curves, output_path=None, max_n=80, title="Decadimento autovalori POD"):
    """Plotta il decadimento degli autovalori POD per una o piu' curve.

    Generica: usata sia per stato/aggiunto (due curve) sia per il controllo
    (una sola curva) - vedi plot_eigenvalue_decay per il caso a due curve.

    Args:
        curves: dict {etichetta: array di autovalori}
        output_path: se dato, salva il PNG invece di mostrarlo
        max_n: numero massimo di autovalori mostrati (la coda e' spesso rumore)
        title: titolo del plot
    """
    import matplotlib
    if output_path is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))

    markers = ["o-", "s-", "^-", "d-"]
    for i, (label, eigenvalues) in enumerate(curves.items()):
        eigenvalues = eigenvalues[:max_n]
        ax.semilogy(range(1, len(eigenvalues) + 1), eigenvalues, markers[i % len(markers)], label=label)

    ax.set_xlabel("N")
    ax.set_ylabel(r"$\lambda_N$")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=80)
        print(f"Plot salvato in {output_path}")
    else:
        plt.show()


def plot_eigenvalue_decay_with_error(eig_curves, error_curves, output_path=None, max_n=80,
                                      title="Decadimento autovalori POD"):
    """Come plot_eigenvalue_decay_curves, ma con un secondo pannello affiancato per
    l'errore di ricostruzione sul test set (stessa griglia di N, stesse etichette).

    Args:
        eig_curves: dict {etichetta: array di autovalori}
        error_curves: dict {etichetta: array di errori}, stesse chiavi di eig_curves -
            error_curves[label][i] e' l'errore con i+1 modi (N parte da 1)
        output_path: se dato, salva il PNG invece di mostrarlo
        max_n: numero massimo di autovalori mostrati nel pannello sinistro
        title: titolo del pannello sinistro (il destro ha titolo fisso)
    """
    import matplotlib
    if output_path is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    markers = ["o-", "s-", "^-", "d-"]

    for i, (label, eigenvalues) in enumerate(eig_curves.items()):
        ev = eigenvalues[:max_n]
        axes[0].semilogy(range(1, len(ev) + 1), ev, markers[i % len(markers)], label=label)
    axes[0].set_xlabel("N")
    axes[0].set_ylabel(r"$\lambda_N$")
    axes[0].set_title(title)
    axes[0].legend()
    axes[0].grid(True, which="both", alpha=0.3)

    for i, (label, errors) in enumerate(error_curves.items()):
        axes[1].semilogy(range(1, len(errors) + 1), errors, markers[i % len(markers)], label=label)
    axes[1].set_xlabel("N")
    axes[1].set_ylabel("errore relativo ricostruzione (test set)")
    axes[1].set_title("Errore di ricostruzione (solo POD)")
    axes[1].legend()
    axes[1].grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=80)
        print(f"Plot salvato in {output_path}")
    else:
        plt.show()


def plot_eigenvalue_decay(eigenvalues_y, eigenvalues_p, output_path=None, max_n=80):
    """Plotta il decadimento degli autovalori POD per stato (y) e aggiunto (p).

    Confrontabile con il plot di destra di Figura 4 nel paper. Wrapper di
    plot_eigenvalue_decay_curves per il caso a due curve fisse.

    Args:
        eigenvalues_y: array di autovalori per lo stato
        eigenvalues_p: array di autovalori per l'aggiunto
        output_path: se dato, salva il PNG invece di mostrarlo (utile da script)
        max_n: numero massimo di autovalori mostrati nel plot (i piu' piccoli,
            in coda, sono spesso solo rumore numerico e non interessano)
    """
    plot_eigenvalue_decay_curves(
        {"Stato (y)": eigenvalues_y, "Aggiunto (p)": eigenvalues_p},
        output_path=output_path, max_n=max_n,
    )
