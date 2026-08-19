import numpy as np


def extract_boundary_control_trace(mesh_data, node_to_dof, U_snapshots, mu_u_samples, control_markers=(8, 10)):
    """Estrae la traccia di u sui nodi di bordo dove puo' esserci controllo, mascherata a zero fuori Gamma_C.

    u e' per definizione dy/dn (flusso al bordo dello stato), non un campo di
    dominio. Su Gamma_N (x < x_ctrl = 1+mu_u) lo stato ha Neumann omogeneo
    imposto direttamente, quindi u=dy/dn=0 li' per definizione - indipendente
    da cosa vale l'aggiunto p in quei punti.

    Args:
        mesh_data: dict da load_mesh()
        node_to_dof: mappa nodo->DOF, da assemble_operators()
        U_snapshots: array (Nh, n_samples) - u=p/alpha su tutti i DOF liberi
        mu_u_samples: array (n_samples,) - mu_u di ciascuno snapshot
        control_markers: marker dei lati dove passa x_ctrl (8, 10 per Test_1)

    Returns:
        boundary_x: array (n_boundary_nodes,) coordinate x, ordinate
        U_boundary: array (n_boundary_nodes, n_samples), mascherato a zero
            dove x < x_ctrl(mu_u) per ciascuno snapshot
    """
    mesh = mesh_data["mesh"]

    # trova i nodi unici sui lati marcati come possibile confine di controllo
    node_coords_x = np.array([mesh.cell0_d_coordinate_x(i)
                               for i in range(mesh.cell0_d_total_number())])

    boundary_nodes = set()
    for e in range(mesh.cell1_d_total_number()):
        if mesh.cell1_d_marker(e) not in control_markers:
            continue
        for n in mesh.cell1_d_extremes(e):
            boundary_nodes.add(n)

    boundary_nodes = sorted(boundary_nodes, key=lambda n: node_coords_x[n])
    boundary_x = node_coords_x[boundary_nodes]
    boundary_dofs = node_to_dof[boundary_nodes]

    if np.any(boundary_dofs < 0):
        raise ValueError("Trovato nodo Dirichlet (dof=-1) sui marker di controllo: inatteso per marker weak")

    # estrae i valori u sui nodi di bordo per ogni snapshot
    U_boundary = U_snapshots[boundary_dofs, :].copy()

    # maschera: zero dove x < x_ctrl(mu_u) - u e' zero li' per definizione (dy/dn=0 su Gamma_N)
    # stessa tolleranza di assemble_control_matrix, per coerenza sui nodi allineati a x_ctrl
    x_ctrl = 1.0 + mu_u_samples  # (n_samples,)
    mask = boundary_x[:, None] >= x_ctrl[None, :] - 1e-10  # (n_boundary_nodes, n_samples)
    U_boundary = U_boundary * mask

    return boundary_x, U_boundary
