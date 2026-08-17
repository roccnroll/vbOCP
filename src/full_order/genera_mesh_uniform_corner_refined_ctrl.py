"""
genera_mesh_uniform_corner_refined_ctrl.py
=============================================
Variante: mesh UNIFORME su tutto il dominio (nessun infittimento di
Omega_obs) PIU' 4 infittimenti "a sfera" sui punti (1,0), (2,0), (1,1),
(2,1) -- stessi punti della variante precedente, ma qui la priorita' va
data alle sfere (il resto del dominio, incluso Omega_obs, e' uniforme).

Mantiene la conformita' a Omega_obs (linee interne, per compatibilita' con
M_obs) e i nodi di controllo forzati a x_ctrl = 1+mu_u.

Uso:
    python genera_mesh_uniform_corner_refined_ctrl.py <lc_uniform> <lc_corner> <dist_corner_max>
    -> crea cartella mesh_uniform_corner_refined_ctrl/ con i 4 CSV
"""

import gmsh
import numpy as np
import os
import sys

sys.path.insert(0, '.')
from genera_mesh_gedim import assign_marker


def build_and_export(lc_uniform=0.03, lc_corner=0.0015, dist_corner_max=0.15,
                      lc_neu=0.008, dist_neu_max=0.04,
                      out_folder="mesh_uniform_corner_refined_ctrl",
                      mu_u_values=(0.1, 0.5, 0.99)):
    gmsh.initialize()
    gmsh.model.add("test1_uniform_corner_refined_ctrl")

    x_ctrl_vals = sorted(1.0 + mu for mu in mu_u_values)

    bl  = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc_uniform)
    bm  = gmsh.model.geo.addPoint(1.0, 0.0, 0.0, lc_uniform)
    br  = gmsh.model.geo.addPoint(2.0, 0.0, 0.0, lc_uniform)
    rm1 = gmsh.model.geo.addPoint(2.0, 0.2, 0.0, lc_uniform)
    rm2 = gmsh.model.geo.addPoint(2.0, 0.8, 0.0, lc_uniform)
    tr  = gmsh.model.geo.addPoint(2.0, 1.0, 0.0, lc_uniform)
    tm  = gmsh.model.geo.addPoint(1.0, 1.0, 0.0, lc_uniform)
    tl  = gmsh.model.geo.addPoint(0.0, 1.0, 0.0, lc_uniform)
    im1 = gmsh.model.geo.addPoint(1.0, 0.2, 0.0, lc_uniform)
    im2 = gmsh.model.geo.addPoint(1.0, 0.8, 0.0, lc_uniform)

    ctrl_pts_bot = [gmsh.model.geo.addPoint(xc, 0.0, 0.0, lc_uniform) for xc in x_ctrl_vals]
    ctrl_pts_top = [gmsh.model.geo.addPoint(xc, 1.0, 0.0, lc_uniform) for xc in x_ctrl_vals]

    L_bot_l = gmsh.model.geo.addLine(bl, bm)

    bot_r_chain = [bm] + ctrl_pts_bot + [br]
    L_bot_r_segs = [gmsh.model.geo.addLine(bot_r_chain[i], bot_r_chain[i+1])
                     for i in range(len(bot_r_chain) - 1)]

    L_right_a = gmsh.model.geo.addLine(br,  rm1)
    L_right_b = gmsh.model.geo.addLine(rm1, rm2)
    L_right_c = gmsh.model.geo.addLine(rm2, tr)

    top_r_chain = [tr] + list(reversed(ctrl_pts_top)) + [tm]
    L_top_r_segs = [gmsh.model.geo.addLine(top_r_chain[i], top_r_chain[i+1])
                     for i in range(len(top_r_chain) - 1)]

    L_top_l = gmsh.model.geo.addLine(tm, tl)
    L_left  = gmsh.model.geo.addLine(tl, bl)

    L_v1 = gmsh.model.geo.addLine(bm,  im1)
    L_v2 = gmsh.model.geo.addLine(im1, im2)
    L_v3 = gmsh.model.geo.addLine(im2, tm)
    L_h1 = gmsh.model.geo.addLine(im1, rm1)
    L_h2 = gmsh.model.geo.addLine(im2, rm2)

    gmsh.model.geo.synchronize()

    cl = gmsh.model.geo.addCurveLoop([L_bot_l, L_v1, L_v2, L_v3, L_top_l, L_left])
    gmsh.model.geo.addPlaneSurface([cl])   # surf 1: Omega_left

    cl = gmsh.model.geo.addCurveLoop(L_bot_r_segs + [L_right_a, -L_h1, -L_v1])
    gmsh.model.geo.addPlaneSurface([cl])   # surf 2: Omega_obs_bot

    cl = gmsh.model.geo.addCurveLoop([L_h1, L_right_b, -L_h2, -L_v2])
    gmsh.model.geo.addPlaneSurface([cl])   # surf 3: Omega_center

    cl = gmsh.model.geo.addCurveLoop([L_h2, L_right_c] + L_top_r_segs + [-L_v3])
    gmsh.model.geo.addPlaneSurface([cl])   # surf 4: Omega_obs_top

    gmsh.model.geo.synchronize()

    # ── Base uniforme + 4 sfere sui punti (1,0),(2,0),(1,1),(2,1) ────────────
    # Nessun bisogno di Restrict: il fondo e' gia' uniforme = lc_uniform,
    # quindi anche se il campo delle sfere "esce" fino a lc_uniform ovunque
    # oltre DistMax, non cambia nulla (stesso valore del fondo).
    corner_points = [br, tr, bm, tm]

    dist_corner = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(dist_corner, "PointsList", corner_points)
    gmsh.model.mesh.field.setNumber(dist_corner, "Sampling", 100)

    thresh_corner = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(thresh_corner, "InField", dist_corner)
    gmsh.model.mesh.field.setNumber(thresh_corner, "SizeMin", lc_corner)
    gmsh.model.mesh.field.setNumber(thresh_corner, "SizeMax", lc_uniform)
    gmsh.model.mesh.field.setNumber(thresh_corner, "DistMin", 0.0)
    gmsh.model.mesh.field.setNumber(thresh_corner, "DistMax", dist_corner_max)

    # ── Banda lungo i bordi Neumann top/bottom (marker 8 e 10) ───────────────
    # Stesso schema: fondo gia' uniforme, nessun Restrict necessario.
    neumann_curves = L_bot_r_segs + L_top_r_segs

    dist_neu = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(dist_neu, "CurvesList", neumann_curves)
    gmsh.model.mesh.field.setNumber(dist_neu, "Sampling", 100)

    thresh_neu = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(thresh_neu, "InField", dist_neu)
    gmsh.model.mesh.field.setNumber(thresh_neu, "SizeMin", lc_neu)
    gmsh.model.mesh.field.setNumber(thresh_neu, "SizeMax", lc_uniform)
    gmsh.model.mesh.field.setNumber(thresh_neu, "DistMin", 0.0)
    gmsh.model.mesh.field.setNumber(thresh_neu, "DistMax", dist_neu_max)

    field_min = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(field_min, "FieldsList", [thresh_corner, thresh_neu])
    gmsh.model.mesh.field.setAsBackgroundMesh(field_min)

    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.Algorithm", 5)
    gmsh.model.mesh.generate(2)

    node_tags, coords_flat, _ = gmsh.model.mesh.getNodes()
    coords = coords_flat.reshape(-1, 3)
    tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}
    N = len(node_tags)

    all_tris = []
    for surf_tag in [1, 2, 3, 4]:
        r = gmsh.model.mesh.getElements(dim=2, tag=surf_tag)
        if len(r[2]) == 0 or len(r[2][0]) == 0:
            continue
        conn = r[2][0].reshape(-1, 3)
        for tri in conn:
            v0 = tag_to_idx[int(tri[0])]
            v1 = tag_to_idx[int(tri[1])]
            v2 = tag_to_idx[int(tri[2])]
            all_tris.append((v0, v1, v2))

    gmsh.finalize()

    node_markers = np.array(
        [assign_marker(coords[i, 0], coords[i, 1]) for i in range(N)],
        dtype=int
    )

    edge_dict = {}
    edges = []
    tri_edges = []
    for v0, v1, v2 in all_tris:
        eidxs = []
        for a, b in [(v0, v1), (v1, v2), (v2, v0)]:
            key = frozenset((a, b))
            if key not in edge_dict:
                edge_dict[key] = len(edges)
                edges.append((a, b))
            eidxs.append(edge_dict[key])
        tri_edges.append(eidxs)

    os.makedirs(out_folder, exist_ok=True)

    with open(os.path.join(out_folder, "Cell0Ds.csv"), 'w') as f:
        f.write("Id;Marker;Active;X;Y;Z\n")
        for i in range(N):
            x = coords[i, 0]
            y = coords[i, 1]
            f.write(f"{i};{node_markers[i]};1;"
                    f"{x:.16e};{y:.16e};0.0000000000000000e+00\n")

    with open(os.path.join(out_folder, "Cell1Ds.csv"), 'w') as f:
        f.write("Id;Marker;Active;Origin;End\n")
        for eid, (a, b) in enumerate(edges):
            mx = 0.5 * (coords[a, 0] + coords[b, 0])
            my = 0.5 * (coords[a, 1] + coords[b, 1])
            mk = assign_marker(mx, my)
            f.write(f"{eid};{mk};1;{a};{b}\n")

    with open(os.path.join(out_folder, "Cell2Ds.csv"), 'w') as f:
        f.write("Id;Marker;Active;NumVertices;Vertices;NumEdges;Edges\n")
        for i, ((v0, v1, v2), (e0, e1, e2)) in enumerate(zip(all_tris, tri_edges)):
            f.write(f"{i};0;1;3;{v0};{v1};{v2};3;{e0};{e1};{e2}\n")

    with open(os.path.join(out_folder, "Cell2DsMarker.csv"), 'w') as f:
        f.write("Cell2DId;VertexLocalId;Marker\n")
        for i, (v0, v1, v2) in enumerate(all_tris):
            for lid, v in enumerate([v0, v1, v2]):
                f.write(f"{i};{lid};{node_markers[v]}\n")

    print(f"Mesh esportata in: {out_folder}/  (lc_uniform={lc_uniform}, lc_corner={lc_corner}, dist_corner_max={dist_corner_max})")
    print(f"  Nodi:      {N}")
    print(f"  Triangoli: {len(all_tris)}")

    print("\n  Verifica nodi di controllo forzati:")
    for xc in x_ctrl_vals:
        on_bot = np.sum((np.abs(coords[:, 0] - xc) < 1e-9) & (np.abs(coords[:, 1] - 0.0) < 1e-9))
        on_top = np.sum((np.abs(coords[:, 0] - xc) < 1e-9) & (np.abs(coords[:, 1] - 1.0) < 1e-9))
        print(f"    x={xc}: y=0 -> {on_bot} trovato/i, y=1 -> {on_top} trovato/i")

    tris_arr = np.array(all_tris)
    y_tri = coords[:, 1][tris_arr]
    x_tri = coords[:, 0][tris_arr]
    eps = 1e-8
    xc_mean = x_tri.mean(axis=1)
    right = xc_mean > 1.0 + eps
    c02 = int(np.sum(np.any(y_tri[right] < 0.2 - eps, axis=1) & np.any(y_tri[right] > 0.2 + eps, axis=1)))
    c08 = int(np.sum(np.any(y_tri[right] < 0.8 - eps, axis=1) & np.any(y_tri[right] > 0.8 + eps, axis=1)))
    print(f"\n  Verifica conformita': attraversano y=0.2: {c02} (atteso 0), y=0.8: {c08} (atteso 0)")

    return N


if __name__ == "__main__":
    lc_uniform      = float(sys.argv[1]) if len(sys.argv) > 1 else 0.03
    lc_corner       = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0015
    dist_corner_max = float(sys.argv[3]) if len(sys.argv) > 3 else 0.15
    lc_neu          = float(sys.argv[4]) if len(sys.argv) > 4 else 0.008
    dist_neu_max    = float(sys.argv[5]) if len(sys.argv) > 5 else 0.04
    build_and_export(lc_uniform=lc_uniform, lc_corner=lc_corner, dist_corner_max=dist_corner_max,
                      lc_neu=lc_neu, dist_neu_max=dist_neu_max,
                      out_folder="mesh_uniform_corner_refined_ctrl")
