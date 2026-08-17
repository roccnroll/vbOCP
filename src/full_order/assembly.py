import numpy as np
from scipy.sparse import csr_matrix
from pypolydim import polydim
import sys
sys.path.insert(1, '../..')
from src.utils import make_np_sparse

def assemble_operators(mesh_data, omega_obs_regions):
    """Assembla gli operatori indipendenti dai parametri mu.

    Args:
        mesh_data: dict restituito da load_mesh()
        omega_obs_regions: lista di rettangoli [[x_min,x_max],[y_min,y_max]]
            che definiscono la regione di osservazione Omega_obs

    Returns:
        dict con A_diff, A_diff_D, A_adv_y, A_adv_y_D, A_adv_p, A_adv_p_D,
        M_obs, M_obs_D, node_to_dof
    """
    # estrae dal dizionario di load_mesh i pezzi necessari all'assembly
    geometry_utilities   = mesh_data["geometry_utilities"]
    mesh                 = mesh_data["mesh"]
    mesh_geometric_data  = mesh_data["mesh_geometric_data"]
    trial_dofs_data      = mesh_data["trial_dofs_data"]
    test_dofs_data       = mesh_data["test_dofs_data"]
    trial_ref            = mesh_data["trial_ref"]
    test_ref             = mesh_data["test_ref"]

    assemble = polydim.pde_tools.assembler_utilities.pcc_2_d

    # operatore di diffusione, coefficiente costante = 1 (il fattore 1/mu1 si applica dopo, a runtime)
    def diff_coeff(x, y, z):
        return 1.0

    diff_result = assemble.assemble_diffusion_operator(
        geometry_utilities, mesh, mesh_geometric_data,
        trial_dofs_data, test_dofs_data, trial_ref, test_ref, diff_coeff)

    A_diff   = make_np_sparse(diff_result.operator_dofs)
    A_diff_D = make_np_sparse(diff_result.operator_strong)

    # avvezione stato: campo di Poiseuille +x2*(1-x2)
    def adv_coeff_y(x, y, z):
        return np.array([y * (1.0 - y), 0.0, 0.0])

    adv_result_y = assemble.assemble_advection_operator(
        geometry_utilities, mesh, mesh_geometric_data,
        trial_dofs_data, test_dofs_data, trial_ref, test_ref, adv_coeff_y)

    A_adv_y   = make_np_sparse(adv_result_y.operator_dofs)
    A_adv_y_D = make_np_sparse(adv_result_y.operator_strong)

    # avvezione aggiunto: stesso campo con segno invertito, richiesto dalla formulazione OTD 
    def adv_coeff_p(x, y, z):
        return np.array([-y * (1.0 - y), 0.0, 0.0])

    adv_result_p = assemble.assemble_advection_operator(
        geometry_utilities, mesh, mesh_geometric_data,
        trial_dofs_data, test_dofs_data, trial_ref, test_ref, adv_coeff_p)

    A_adv_p   = make_np_sparse(adv_result_p.operator_dofs)
    A_adv_p_D = make_np_sparse(adv_result_p.operator_strong)

    # indicatore di Omega_obs
    def chi_obs(x, y, z):
        for x_range, y_range in omega_obs_regions:
            if x_range[0] <= x <= x_range[1] and y_range[0] <= y <= y_range[1]:
                return 1.0
        return 0.0

    mass_obs_result = assemble.assemble_reaction_operator(
        geometry_utilities, mesh, mesh_geometric_data,
        trial_dofs_data, test_dofs_data, trial_ref, test_ref, chi_obs)

    M_obs   = make_np_sparse(mass_obs_result.operator_dofs)
    M_obs_D = make_np_sparse(mass_obs_result.operator_strong)

    # mappa ogni nodo della mesh al suo indice di DOF libero (-1 se e' un nodo Dirichlet)
    num_nodes   = mesh.cell0_d_total_number()
    node_to_dof = np.full(num_nodes, -1, dtype=int)

    dof_list = trial_dofs_data.cells_do_fs[0]
    for node_idx in range(num_nodes):
        dofs_on_node = dof_list[node_idx]
        if len(dofs_on_node) > 0:
            dof = dofs_on_node[0]
            if str(dof.type) == "Types.dof":
                node_to_dof[node_idx] = dof.global_index

    return {
        "A_diff": A_diff, "A_diff_D": A_diff_D,
        "A_adv_y": A_adv_y, "A_adv_y_D": A_adv_y_D,
        "A_adv_p": A_adv_p, "A_adv_p_D": A_adv_p_D,
        "M_obs": M_obs, "M_obs_D": M_obs_D,
        "node_to_dof": node_to_dof,
    }

def assemble_control_matrix(mesh_data, node_to_dof, mu_u, control_markers=(8, 10)):
    """Assembla la matrice di controllo C(mu_u) sul bordo Gamma_C.

    Args:
        mesh_data: dict restituito da load_mesh() (deve contenere node_to_dof
            calcolato da assemble_operators, va passato/unito dal chiamante)
        mu_u: valore del parametro di posizione del bordo di controllo

    Returns:
        matrice sparsa C, shape (Nh, Nh)
    """
    # x della soglia di controllo: il bordo di controllo Gamma_C parte da x=1+mu_u (specifico Test_1)
    mesh = mesh_data["mesh"]
    Nh   = mesh_data["Nh"]

    x_ctrl = 1.0 + mu_u

    node_coords_x = np.array([mesh.cell0_d_coordinate_x(i)
                               for i in range(mesh.cell0_d_total_number())])
    node_coords_y = np.array([mesh.cell0_d_coordinate_y(i)
                               for i in range(mesh.cell0_d_total_number())])

    rows, cols, vals = [], [], []

    # scorre i lati sui marker di controllo, costruisce la massa di bordo 1D solo dove x >= x_ctrl
    for e in range(mesh.cell1_d_total_number()):
        marker = mesh.cell1_d_marker(e)
        if marker not in control_markers:
            continue

        n0 = mesh.cell1_d_extremes(e)[0]
        n1 = mesh.cell1_d_extremes(e)[1]

        if node_coords_x[n0] < x_ctrl - 1e-10 and node_coords_x[n1] < x_ctrl - 1e-10:
            continue

        L  = np.sqrt((node_coords_x[n1] - node_coords_x[n0])**2 +
                     (node_coords_y[n1] - node_coords_y[n0])**2)
        d0 = node_to_dof[n0]
        d1 = node_to_dof[n1]

        for i, di in [(0, d0), (1, d1)]:
            for j, dj in [(0, d0), (1, d1)]:
                if di >= 0 and dj >= 0:
                    rows.append(di)
                    cols.append(dj)
                    vals.append(L / 6.0 * (2.0 if i == j else 1.0))

    return csr_matrix((vals, (rows, cols)), shape=(Nh, Nh))



    

