from pypolydim import polydim, gedim

def load_mesh(mesh_dir, boundary_markers,method_order=1):
    """Carica una mesh da CSV e prepara DOF/boundary info.

    Args:
        mesh_dir: path alla cartella con i CSV della mesh (Cell0Ds.csv, ...)
        boundary_markers: dict marker -> tipo ("strong"/"weak"), da config

    Returns:
        dict con mesh, mesh_geometric_data, trial_dofs_data, test_dofs_data, Nh
    """
    # tolleranze numeriche richieste da pypolydim per i calcoli geometrici 
    geometry_utilities_config = gedim.GeometryUtilitiesConfig()
    geometry_utilities_config.tolerance1_d = 1.0e-6
    geometry_utilities_config.tolerance2_d = 1.0e-12
    geometry_utilities = gedim.GeometryUtilities(geometry_utilities_config)
    mesh_utilities = gedim.MeshUtilities()

    # importa la mesh  
    mesh_type    = polydim.pde_tools.mesh.pde_mesh_utilities.MeshGenerator_Types_2D.csv_importer
    method_type  = polydim.pde_tools.local_space_pcc_2_d.MethodTypes.fem_pcc

    mesh_data_raw = gedim.MeshMatrices()
    mesh          = gedim.MeshMatricesDAO(mesh_data_raw)

    polydim.pde_tools.mesh.pde_mesh_utilities.import_mesh_2_d(
        geometry_utilities, mesh_utilities,
        mesh_type, mesh_dir, mesh)

    mesh_geometric_data = polydim.pde_tools.mesh.pde_mesh_utilities.compute_mesh_2_d_geometry_data(
        geometry_utilities, mesh_utilities, mesh)

    # boundary_info costruito dal config 
    BoundaryInfo = polydim.pde_tools.do_fs.DOFsManager.MeshDOFsInfo.BoundaryInfo
    BTypes       = BoundaryInfo.BoundaryTypes

    type_map = {"strong": BTypes.strong, "weak": BTypes.weak, "none": BTypes.none}

    boundary_info = {}
    for marker, record in boundary_markers.items():
        info = BoundaryInfo(type_map[record["type"]])
        info.marker = marker
        boundary_info[marker] = info

    # assegna indice numerico a ogni DOF libero, usando method_order 
    mesh_connectivity_data = polydim.pde_tools.mesh.MeshMatricesDAO_mesh_connectivity_data(mesh)

    trial_ref = polydim.pde_tools.local_space_pcc_2_d.create_reference_element(method_type, method_order)
    test_ref  = polydim.pde_tools.local_space_pcc_2_d.create_reference_element(method_type, method_order)

    dof_manager = polydim.pde_tools.do_fs.DOFsManager()

    trial_mesh_dofs_info = polydim.pde_tools.local_space_pcc_2_d.set_mesh_do_fs_info(
        trial_ref, mesh, boundary_info)
    trial_dofs_data = dof_manager.create_do_fs_2_d(trial_mesh_dofs_info, mesh_connectivity_data)

    test_mesh_dofs_info = polydim.pde_tools.local_space_pcc_2_d.set_mesh_do_fs_info(
        test_ref, mesh, boundary_info)
    test_dofs_data = dof_manager.create_do_fs_2_d(test_mesh_dofs_info, mesh_connectivity_data)

    Nh = trial_dofs_data.number_do_fs

    return {
        "geometry_utilities": geometry_utilities,
        "mesh_utilities": mesh_utilities,
        "mesh": mesh,
        "mesh_geometric_data": mesh_geometric_data,
        "trial_ref": trial_ref,
        "test_ref": test_ref,
        "trial_dofs_data": trial_dofs_data,
        "test_dofs_data": test_dofs_data,
        "Nh": Nh,
    }










