"""
convert_dolphin_mesh.py
=======================
Converte la mesh del delfino (formato FEniCS/dolfin XML) nel formato CSV
atteso da pypolydim (Cell0Ds, Cell1Ds, Cell2Ds, Cell2DsMarker).

Marker (ereditati dal facet_region del file originale):
   0 = interno
   1 = bordo esterno (lati y=0, y=1, x=0)
   2 = bordo esterno x=1
   4 = bordo del delfino
Un nodo condiviso tra bordi 1 e 2 prende 1 (Dirichlet vince:
gli angoli (1,0),(1,1) -> 1). Il marker di un lato e' quello della facet.

Uso:
    python convert_dolphin_mesh.py [dir_xml] [out_dir]
"""

import os
import re
import sys
from collections import Counter

import numpy as np

# facet i di un triangolo dolfin e' quella opposta al vertice i
FACET_VERTS = [(1, 2), (0, 2), (0, 1)]


def convert(xml_dir, out_dir, base="mesh_dolphin"):
    txt = open(os.path.join(xml_dir, f"{base}.xml")).read()
    coords = np.array(re.findall(r'<vertex index="\d+" x="([^"]+)" y="([^"]+)"', txt), float)
    tris = np.array(re.findall(r'<triangle index="\d+" v0="(\d+)" v1="(\d+)" v2="(\d+)"', txt), int)

    ftxt = open(os.path.join(xml_dir, f"{base}_facet_region.xml")).read()
    facet_mk = {(int(c), int(l)): int(v)
                for c, l, v in re.findall(r'cell_index="(\d+)" local_entity="(\d)" value="(\d+)"', ftxt)}

    N = len(coords)

    # orientamento antiorario
    for i, (a, b, c) in enumerate(tris):
        area2 = ((coords[b, 0] - coords[a, 0]) * (coords[c, 1] - coords[a, 1])
                 - (coords[c, 0] - coords[a, 0]) * (coords[b, 1] - coords[a, 1]))
        if area2 < 0:
            tris[i] = (a, c, b)
    # (lo scambio cambia l'ordine locale dei lati: ricalcolo i marker per chiave)

    # marker dei lati per chiave {a,b} usando la numerazione originale
    raw = re.findall(r'<triangle index="\d+" v0="(\d+)" v1="(\d+)" v2="(\d+)"', txt)
    edge_marker = {}
    for ci, (a, b, c) in enumerate(raw):
        tv = (int(a), int(b), int(c))
        for le, (p, q) in enumerate(FACET_VERTS):
            mk = facet_mk[(ci, le)]
            key = frozenset((tv[p], tv[q]))
            # i lati interni hanno marker 3 -> 0; i lati di bordo mantengono 1/2/4
            edge_marker[key] = 0 if mk == 3 else mk

    # priorita' sui nodi condivisi: il Dirichlet (1) vince sul Neumann (2),
    # cosi' gli angoli (1,0),(1,1) restano nodi Dirichlet
    node_markers = np.zeros(N, dtype=int)
    for mk in (2, 1, 4):
        for key, emk in edge_marker.items():
            if emk == mk:
                for n in key:
                    node_markers[n] = mk

    edge_id, edges, tri_edges = {}, [], []
    for a, b, c in tris:
        ids = []
        for u, v in [(a, b), (b, c), (c, a)]:
            k = frozenset((u, v))
            if k not in edge_id:
                edge_id[k] = len(edges)
                edges.append((u, v))
            ids.append(edge_id[k])
        tri_edges.append(ids)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "Cell0Ds.csv"), "w") as f:
        f.write("Id;Marker;Active;X;Y;Z\n")
        for i in range(N):
            f.write(f"{i};{node_markers[i]};1;{coords[i,0]:.16e};{coords[i,1]:.16e};0.0000000000000000e+00\n")
    with open(os.path.join(out_dir, "Cell1Ds.csv"), "w") as f:
        f.write("Id;Marker;Active;Origin;End\n")
        for eid, (u, v) in enumerate(edges):
            f.write(f"{eid};{edge_marker[frozenset((u, v))]};1;{u};{v}\n")
    with open(os.path.join(out_dir, "Cell2Ds.csv"), "w") as f:
        f.write("Id;Marker;Active;NumVertices;Vertices;NumEdges;Edges\n")
        for i, ((a, b, c), (e0, e1, e2)) in enumerate(zip(tris, tri_edges)):
            f.write(f"{i};0;1;3;{a};{b};{c};3;{e0};{e1};{e2}\n")
    with open(os.path.join(out_dir, "Cell2DsMarker.csv"), "w") as f:
        f.write("Cell2DId;VertexLocalId;Marker\n")
        for i, tri in enumerate(tris):
            for lid, v in enumerate(tri):
                f.write(f"{i};{lid};{node_markers[v]}\n")

    print(f"Mesh scritta in {out_dir}: {N} nodi, {len(tris)} triangoli, {len(edges)} lati")
    print("Nodi per marker:", dict(sorted(Counter(node_markers.tolist()).items())))


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", ".."))
    xml_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, "data", "meshes", "delfino", "data")
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(root, "data", "meshes", "delfino_csv")
    convert(xml_dir, out_dir)
