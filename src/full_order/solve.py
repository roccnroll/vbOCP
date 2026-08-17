import numpy as np
import scipy.sparse
import scipy.sparse.linalg

def solve_otd(operators, control_matrix, dirichlet_data, mu1, mu2, alpha):
    """Risolve il sistema saddle-point OTD per una data terna di parametri.

    Args:
        operators: dict da assemble_operators() (A_diff, A_adv_y, A_adv_p, M_obs, ...)
        control_matrix: matrice C(mu_u) da assemble_control_matrix()
        dirichlet_data: dict da assemble_dirichlet_and_source() (u_D_y, u_D_p, yd_unit_vec)
        mu1: parametro di diffusione inversa
        mu2: ampiezza del desired state (yd_vec = mu2 * yd_unit_vec)
        alpha: penalizzazione del controllo

    Returns:
        (y, p, u): stato, aggiunto, controllo come array numpy
    """
    # estrae dagli assembly i pezzi necessari
    A_diff, A_diff_D   = operators["A_diff"], operators["A_diff_D"]
    A_adv_y, A_adv_y_D = operators["A_adv_y"], operators["A_adv_y_D"]
    A_adv_p, A_adv_p_D = operators["A_adv_p"], operators["A_adv_p_D"]
    M_obs, M_obs_D     = operators["M_obs"], operators["M_obs_D"]

    u_D_y       = dirichlet_data["u_D_y"]
    u_D_p       = dirichlet_data["u_D_p"]
    yd_unit_vec = dirichlet_data["yd_unit_vec"]

    C  = control_matrix
    Nh = M_obs.shape[0]

    # operatori mu-dipendenti: combinazione lineare economica di quantita' gia' assemblate (nessun riassembly FEM)
    A_mu   = (1.0 / mu1) * A_diff   + A_adv_y
    A_mu_D = (1.0 / mu1) * A_diff_D + A_adv_y_D

    A_adj   = (1.0 / mu1) * A_diff   + A_adv_p
    A_adj_D = (1.0 / mu1) * A_diff_D + A_adv_p_D

    # yd_vec e' affine in mu2: scala il vettore unitario invece di riassemblare
    yd_vec = mu2 * yd_unit_vec

    # scaling gamma per bilanciare gli ordini di grandezza tra i due blocchi del sistema
    gamma = abs(A_mu).max() / abs(M_obs).max()

    # termini noti con correzione Dirichlet
    rhs_sta = -A_mu_D @ u_D_y
    rhs_adj = yd_vec - M_obs_D @ u_D_y - A_adj_D @ u_D_p

    # sistema saddle-point a blocchi, formulazione OTD (vedi handout.md sez. 4/13)
    K = scipy.sparse.bmat([
        [gamma * M_obs, gamma * A_adj],
        [A_mu,          -1.0 / alpha * C]
    ], format='csr')

    rhs = np.concatenate([gamma * rhs_adj, rhs_sta])

    sol = scipy.sparse.linalg.spsolve(K, rhs)

    y = sol[:Nh]
    p = sol[Nh:]
    u = p / alpha

    return y, p, u


