"""
genera_mesh_gedim.py
====================
Genera la mesh conforme ai sottodomini Omega_obs per il Test 1 saddle-point
e la esporta nel formato completo atteso da pypolydim.import_mesh_2_d.

Formato file:
  Cell0Ds.csv        Id;Marker;Active;X;Y;Z
  Cell1Ds.csv        Id;Marker;Active;Origin;End
  Cell2Ds.csv        Id;Marker;Active;NumVertices;V0;V1;V2;NumEdges;E0;E1;E2
  Cell2DsMarker.csv  Cell2DId;VertexLocalId;Marker

Marker nodi:
   1  = corner (0,0)        7  = bottom-left   y=0, x in (0,1)
   2  = corner (1,0)        8  = bottom-right  y=0, x in (1,2)
   3  = corner (2,0)        9  = destra        x=2
   4  = corner (2,1)       10  = top-right     y=1, x in (1,2)
   5  = corner (1,1)       11  = top-left      y=1, x in (0,1)
   6  = corner (0,1)       12  = sinistra      x=0
   0  = interno

Uso:
    python genera_mesh_gedim.py 0.005
    -> crea cartella mesh_conforming/ con i 4 CSV
"""

import gmsh
import numpy as np
import os
import sys


def assign_marker(x, y, tol=1e-10):
    """
    Marker geometrico per un nodo/punto.
    1-6: corner; 7-12: lati; 0: interno.
    """
    on_bottom = abs(y)        < tol
    on_top    = abs(y - 1.0)  < tol
    on_left   = abs(x)        < tol
    on_right  = abs(x - 2.0)  < tol
    on_xmid   = abs(x - 1.0)  < tol   # linea interna x=1

    # 6 corner (ordine fisso: prima i corner, poi i lati puri)
    if on_left  and on_bottom:  return 1   # (0,0)
    if on_xmid  and on_bottom:  return 2   # (1,0)
    if on_right and on_bottom:  return 3   # (2,0)
    if on_right and on_top:     return 4   # (2,1)
    if on_xmid  and on_top:     return 5   # (1,1)
    if on_left  and on_top:     return 6   # (0,1)

    # Lati puri
    if on_bottom and x < 1.0 - tol:  return 7
    if on_bottom and x > 1.0 + tol:  return 8
    if on_right:                      return 9
    if on_top    and x > 1.0 + tol:  return 10
    if on_top    and x < 1.0 - tol:  return 11
    if on_left:                       return 12

    return 0  # interno


def build_and_export(mesh_size=0.01, out_folder="mesh_conforming"):
    gmsh.initialize()
    gmsh.model.add("test1_conforming")
    lc = mesh_size

    # ── Punti ─────────────────────────────────────────────────────────────────
    bl  = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc)
    bm  = gmsh.model.geo.addPoint(1.0, 0.0, 0.0, lc)
    br  = gmsh.model.geo.addPoint(2.0, 0.0, 0.0, lc)
    rm1 = gmsh.model.geo.addPoint(2.0, 0.2, 0.0, lc)
    rm2 = gmsh.model.geo.addPoint(2.0, 0.8, 0.0, lc)
    tr  = gmsh.model.geo.addPoint(2.0, 1.0, 0.0, lc)
    tm  = gmsh.model.geo.addPoint(1.0, 1.0, 0.0, lc)
    tl  = gmsh.model.geo.addPoint(0.0, 1.0, 0.0, lc)
    im1 = gmsh.model.geo.addPoint(1.0, 0.2, 0.0, lc)
    im2 = gmsh.model.geo.addPoint(1.0, 0.8, 0.0, lc)

    # ── Linee bordo esterno ────────────────────────────────────────────────────
    L_bot_l   = gmsh.model.geo.addLine(bl,  bm)
    L_bot_r   = gmsh.model.geo.addLine(bm,  br)
    L_right_a = gmsh.model.geo.addLine(br,  rm1)
    L_right_b = gmsh.model.geo.addLine(rm1, rm2)
    L_right_c = gmsh.model.geo.addLine(rm2, tr)
    L_top_r   = gmsh.model.geo.addLine(tr,  tm)
    L_top_l   = gmsh.model.geo.addLine(tm,  tl)
    L_left    = gmsh.model.geo.addLine(tl,  bl)

    # ── Linee interne (conformita' a Omega_obs) ────────────────────────────────
    L_v1 = gmsh.model.geo.addLine(bm,  im1)   # x=1, y: 0   -> 0.2
    L_v2 = gmsh.model.geo.addLine(im1, im2)   # x=1, y: 0.2 -> 0.8
    L_v3 = gmsh.model.geo.addLine(im2, tm)    # x=1, y: 0.8 -> 1
    L_h1 = gmsh.model.geo.addLine(im1, rm1)   # y=0.2, x: 1 -> 2
    L_h2 = gmsh.model.geo.addLine(im2, rm2)   # y=0.8, x: 1 -> 2

    gmsh.model.geo.synchronize()

    # ── 4 sottoregioni ────────────────────────────────────────────────────────
    cl = gmsh.model.geo.addCurveLoop([L_bot_l, L_v1, L_v2, L_v3, L_top_l, L_left])
    gmsh.model.geo.addPlaneSurface([cl])   # surf 1: Omega_left

    cl = gmsh.model.geo.addCurveLoop([L_bot_r, L_right_a, -L_h1, -L_v1])
    gmsh.model.geo.addPlaneSurface([cl])   # surf 2: Omega_obs_bot

    cl = gmsh.model.geo.addCurveLoop([L_h1, L_right_b, -L_h2, -L_v2])
    gmsh.model.geo.addPlaneSurface([cl])   # surf 3: Omega_center

    cl = gmsh.model.geo.addCurveLoop([L_h2, L_right_c, L_top_r, -L_v3])
    gmsh.model.geo.addPlaneSurface([cl])   # surf 4: Omega_obs_top

    gmsh.model.geo.synchronize()

    # ── Genera mesh ────────────────────────────────────────────────────────────
    gmsh.option.setNumber("Mesh.Algorithm", 5)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
    gmsh.model.mesh.generate(2)

    # ── Estrai nodi ───────────────────────────────────────────────────────────
    node_tags, coords_flat, _ = gmsh.model.mesh.getNodes()
    coords = coords_flat.reshape(-1, 3)
    tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}
    N = len(node_tags)

    # ── Estrai triangoli da tutte e 4 le superfici ────────────────────────────
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

    # ── Marker nodi ───────────────────────────────────────────────────────────
    node_markers = np.array(
        [assign_marker(coords[i, 0], coords[i, 1]) for i in range(N)],
        dtype=int
    )

    # ── Tabella spigoli (frozenset deduplication) ─────────────────────────────
    edge_dict = {}   # frozenset({a,b}) → edge_idx
    edges     = []   # list of (origin, end)
    tri_edges = []   # per triangolo: [e0, e1, e2]

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

    # ── Cell0Ds.csv: Id;Marker;Active;X;Y;Z ──────────────────────────────────
    with open(os.path.join(out_folder, "Cell0Ds.csv"), 'w') as f:
        f.write("Id;Marker;Active;X;Y;Z\n")
        for i in range(N):
            x = coords[i, 0]
            y = coords[i, 1]
            f.write(f"{i};{node_markers[i]};1;"
                    f"{x:.16e};{y:.16e};0.0000000000000000e+00\n")

    # ── Cell1Ds.csv: Id;Marker;Active;Origin;End ──────────────────────────────
    with open(os.path.join(out_folder, "Cell1Ds.csv"), 'w') as f:
        f.write("Id;Marker;Active;Origin;End\n")
        for eid, (a, b) in enumerate(edges):
            mx = 0.5 * (coords[a, 0] + coords[b, 0])
            my = 0.5 * (coords[a, 1] + coords[b, 1])
            mk = assign_marker(mx, my)
            f.write(f"{eid};{mk};1;{a};{b}\n")

    # ── Cell2Ds.csv: Id;Marker;Active;NumVertices;V0;V1;V2;NumEdges;E0;E1;E2 ─
    with open(os.path.join(out_folder, "Cell2Ds.csv"), 'w') as f:
        f.write("Id;Marker;Active;NumVertices;Vertices;NumEdges;Edges\n")
        for i, ((v0, v1, v2), (e0, e1, e2)) in enumerate(zip(all_tris, tri_edges)):
            f.write(f"{i};0;1;3;{v0};{v1};{v2};3;{e0};{e1};{e2}\n")

    # ── Cell2DsMarker.csv: Cell2DId;VertexLocalId;Marker ─────────────────────
    with open(os.path.join(out_folder, "Cell2DsMarker.csv"), 'w') as f:
        f.write("Cell2DId;VertexLocalId;Marker\n")
        for i, (v0, v1, v2) in enumerate(all_tris):
            for lid, v in enumerate([v0, v1, v2]):
                f.write(f"{i};{lid};{node_markers[v]}\n")

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"Mesh esportata in: {out_folder}/")
    print(f"  Nodi:      {N}")
    print(f"  Spigoli:   {len(edges)}")
    print(f"  Triangoli: {len(all_tris)}")
    print()

    mk_names = {
        1: "corner(0,0)",    2: "corner(1,0)",    3: "corner(2,0)",
        4: "corner(2,1)",    5: "corner(1,1)",    6: "corner(0,1)",
        7: "bottom-left/D",  8: "bottom-right/N",  9: "destra/N",
       10: "top-right/N",   11: "top-left/D",    12: "sinistra/D",
        0: "interno",
    }
    print("Nodi per marker:")
    for mk in sorted(mk_names):
        cnt = int(np.sum(node_markers == mk))
        if cnt > 0:
            print(f"  marker {mk:2d} ({mk_names[mk]}): {cnt}")

    # ── Verifica conformita' ──────────────────────────────────────────────────
    tris_arr = np.array(all_tris)
    y_arr = coords[:, 1]
    x_arr = coords[:, 0]
    y_tri = y_arr[tris_arr]
    x_tri = x_arr[tris_arr]
    eps   = 1e-8
    xc    = x_tri.mean(axis=1)
    right = xc > 1.0 + eps
    c02 = int(np.sum(
        np.any(y_tri[right] < 0.2 - eps, axis=1) &
        np.any(y_tri[right] > 0.2 + eps, axis=1)
    ))
    c08 = int(np.sum(
        np.any(y_tri[right] < 0.8 - eps, axis=1) &
        np.any(y_tri[right] > 0.8 + eps, axis=1)
    ))
    print(f"\nVerifica conformita' (triangoli con centroide x>1):")
    print(f"  Attraversano y=0.2: {c02}  (atteso: 0)")
    print(f"  Attraversano y=0.8: {c08}  (atteso: 0)")

    # ── Verifica round-trip CSV ───────────────────────────────────────────────
    import csv as _csv

    node_coords_csv = {}
    with open(os.path.join(out_folder, "Cell0Ds.csv")) as f:
        rd = _csv.reader(f, delimiter=';')
        next(rd)
        for row in rd:
            node_coords_csv[int(row[0])] = (float(row[3]), float(row[4]))

    tri_verts_csv = {}
    with open(os.path.join(out_folder, "Cell2Ds.csv")) as f:
        rd = _csv.reader(f, delimiter=';')
        next(rd)
        for row in rd:
            tid = int(row[0])
            # row: Id;Marker;Active;NumVertices;V0;V1;V2;NumEdges;E0;E1;E2
            tri_verts_csv[tid] = (int(row[4]), int(row[5]), int(row[6]))

    errors = 0
    with open(os.path.join(out_folder, "Cell2DsMarker.csv")) as f:
        rd = _csv.reader(f, delimiter=';')
        next(rd)
        for row in rd:
            tid, lid, mk = int(row[0]), int(row[1]), int(row[2])
            gid = tri_verts_csv[tid][lid]
            xg, yg = node_coords_csv[gid]
            expected = assign_marker(xg, yg)
            if mk != expected:
                errors += 1
                if errors <= 5:
                    print(f"  ERR marker: tri={tid} local={lid} node={gid} "
                          f"({xg:.4f},{yg:.4f}) csv={mk} atteso={expected}")

    if errors == 0:
        print("\nVerifica round-trip CSV: OK — tutti i marker sono corretti.")
    else:
        print(f"\nVerifica round-trip CSV: {errors} errori trovati!")


if __name__ == "__main__":
    mesh_size = float(sys.argv[1]) if len(sys.argv) > 1 else 0.01
    print(f"Generazione mesh conforme (mesh_size={mesh_size})...")
    build_and_export(mesh_size=mesh_size, out_folder="mesh_conforming")
