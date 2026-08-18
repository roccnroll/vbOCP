import sys
from pathlib import Path

from pypolydim import polydim

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils import make_np_sparse


def assemble_full_mass_matrix(mesh_data):
    """Assembla la matrice di massa M_full = integrale(u*v) su tutto il dominio Omega.

    Usata per il prodotto scalare H1 completo nella POD: X = A_diff + M_full.
    Stesso pattern di M_obs in assembly.py, ma con indicatore costante 1.0
    invece che ristretto a Omega_obs.

    Args:
        mesh_data: dict restituito da load_mesh()

    Returns:
        M_full: matrice sparsa (Nh, Nh)
    """
    geometry_utilities  = mesh_data["geometry_utilities"]
    mesh                = mesh_data["mesh"]
    mesh_geometric_data = mesh_data["mesh_geometric_data"]
    trial_dofs_data     = mesh_data["trial_dofs_data"]
    test_dofs_data      = mesh_data["test_dofs_data"]
    trial_ref           = mesh_data["trial_ref"]
    test_ref            = mesh_data["test_ref"]

    assemble = polydim.pde_tools.assembler_utilities.pcc_2_d

    # indicatore costante su tutto il dominio (a differenza di chi_obs, sempre 1.0)
    def one_everywhere(x, y, z):
        return 1.0

    mass_result = assemble.assemble_reaction_operator(
        geometry_utilities, mesh, mesh_geometric_data,
        trial_dofs_data, test_dofs_data, trial_ref, test_ref, one_everywhere)

    return make_np_sparse(mass_result.operator_dofs)
