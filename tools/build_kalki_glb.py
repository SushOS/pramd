#!/usr/bin/env python3
"""Build live/kalki/kalki.glb: the Kalki RP1's production meshes, decimated, with their textures baked into vertex colours.

The model is read through MuJoCo's own compiler (robot.xml + assets/), so mesh scale, recentring and texture orientation are
exactly what MuJoCo draws. Every visual mesh geom becomes one glTF node whose vertices are already in its body's frame;
live/kalki.js hangs each node on the body tree from live/kalki/model.json.

    python tools/build_kalki_glb.py [--model-dir DIR] [--out FILE] [--ratio 0.12] [--agg 3] [--angle 55]

Needs numpy, scipy, mujoco and fast-simplification. The production meshes are ~1.6M triangles and ~180 MB; the output is
~275k triangles in ~10 MB. Textures cannot survive decimation (the head texture is an atlas of hundreds of UV islands), so
each face's texture colour is baked into vertex colours instead: faces, eyes and the wordmark survive, thin trim lines
(e.g. the violet V on the chest) do not. The last column of the printed table is the two-sided surface-to-surface
distance between each original mesh and its decimated version.
"""
import argparse
import json
import struct
import time
from pathlib import Path

import fast_simplification
import mujoco
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO.parent / 'Kalki_RP1_MuJoCo' / 'Kalki_RP1_MuJoCo'
MIN_FACES = 600        # a mesh is never decimated below this
CREASE_DEG = 55.0      # edges sharper than this keep a hard normal


def srgb_to_linear(c):
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def sample(img, uv, nearest):
    """Sample an HxWx3 image the way MuJoCo does: row = v*H counted from row 0, repeat wrapping."""
    h, w, _ = img.shape
    x, y = uv[:, 0] * w - 0.5, uv[:, 1] * h - 0.5
    if nearest:
        return img[np.round(y).astype(int) % h, np.round(x).astype(int) % w]
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    x0, x1, y0, y1 = x0 % w, (x0 + 1) % w, y0 % h, (y0 + 1) % h
    return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x1] * fx * (1 - fy)
            + img[y1, x0] * (1 - fx) * fy + img[y1, x1] * fx * fy)


def visual_geoms(m):
    """Yield every mesh geom: vertices in the body frame, faces, and per-face sRGB colour (None for flat-colour materials)."""
    for g in range(m.ngeom):
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mi = int(m.geom_dataid[g])
        va, nv = int(m.mesh_vertadr[mi]), int(m.mesh_vertnum[mi])
        fa, nf = int(m.mesh_faceadr[mi]), int(m.mesh_facenum[mi])
        ta, nt = int(m.mesh_texcoordadr[mi]), int(m.mesh_texcoordnum[mi])
        V = m.mesh_vert[va:va + nv].astype(np.float64) @ quat_to_mat(m.geom_quat[g]).T + m.geom_pos[g]
        F = m.mesh_face[fa:fa + nf].astype(np.int64)
        assert 0 <= F.min() and F.max() < nv, 'mesh_face should index the mesh\'s own vertices'
        mat = int(m.geom_matid[g])
        face_srgb = None
        tex = int(m.mat_texid[mat][mujoco.mjtTextureRole.mjTEXROLE_RGB]) if mat >= 0 else -1
        if tex >= 0 and nt:
            w, h, c = int(m.tex_width[tex]), int(m.tex_height[tex]), int(m.tex_nchannel[tex])
            adr = int(m.tex_adr[tex])
            img = m.tex_data[adr:adr + w * h * c].reshape(h, w, c)[:, :, :3].astype(np.float64) / 255.0
            uv = m.mesh_texcoord[ta:ta + nt][m.mesh_facetexcoord[fa:fa + nf]].mean(axis=1)   # centroid of each face's UV triangle
            face_srgb = sample(img, uv, nearest=max(w, h) <= 256)
        yield {'name': m.geom(g).name, 'body': m.body(m.geom_bodyid[g]).name, 'material': mat,
               'V': V, 'F': F, 'face_srgb': face_srgb}


def weld(V, F, tol=1e-5):
    """Merge vertices closer than tol (chains included) into their mean: CAD exports leave near-coincident seams."""
    pairs = cKDTree(V).query_pairs(tol, output_type='ndarray')
    n = len(V)
    count, label = connected_components(coo_matrix((np.ones(len(pairs), dtype=np.int8), (pairs[:, 0], pairs[:, 1])), shape=(n, n)), directed=False)
    size = np.bincount(label, minlength=count)
    return np.stack([np.bincount(label, weights=V[:, k], minlength=count) / size for k in range(3)], axis=1), label[F]


def drop_degenerate(V, F):
    """Remove zero-area and duplicate faces and unused vertices; returns V, F and the mask of kept faces."""
    cr = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    keep = np.linalg.norm(cr, axis=1) > 1e-14
    first = np.unique(np.sort(F, axis=1), axis=0, return_index=True)[1]       # a face listed twice (e.g. both sides) keeps one copy
    once = np.zeros(len(F), dtype=bool)
    once[first] = True
    keep &= once
    used, inv = np.unique(F[keep], return_inverse=True)
    return V[used], inv.reshape(-1, 3), keep


def transfer_colours(V0, F0, face_lin0, V1):
    """Area-weighted average of the original face colours around each decimated vertex (linear RGB)."""
    centroid = V0[F0].mean(axis=1)
    area = np.linalg.norm(np.cross(V0[F0[:, 1]] - V0[F0[:, 0]], V0[F0[:, 2]] - V0[F0[:, 0]]), axis=1)
    owner = cKDTree(V1).query(centroid)[1]
    n = len(V1)
    wsum = np.bincount(owner, weights=area, minlength=n)
    out = np.stack([np.bincount(owner, weights=area * face_lin0[:, k], minlength=n) for k in range(3)], axis=1)
    has = wsum > 0
    out[has] /= wsum[has, None]
    if not has.all():                      # a vertex that owns no original face borrows its nearest owner's colour
        out[~has] = out[has][cKDTree(V1[has]).query(V1[~has])[1]]
    return out


def crease_split(V, F, C, angle_deg):
    """Duplicate vertices along edges sharper than angle_deg and compute smooth normals within each smooth patch."""
    nF = len(F)
    cr = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])          # area-weighted face normals
    unit = cr / np.maximum(np.linalg.norm(cr, axis=1), 1e-30)[:, None]
    ea, eb = F.reshape(-1), np.roll(F, -1, axis=1).reshape(-1)               # directed edge k of face f at index 3f+k
    key = np.minimum(ea, eb).astype(np.int64) * len(V) + np.maximum(ea, eb)
    order = np.argsort(key, kind='stable')
    ks = key[order]
    starts = np.r_[True, ks[1:] != ks[:-1]]
    size = np.bincount(np.cumsum(starts) - 1)
    first = np.flatnonzero(starts)[size == 2]                                # manifold edges only
    i0, i1 = order[first], order[first + 1]
    fa, fb = i0 // 3, i1 // 3
    smooth = (unit[fa] * unit[fb]).sum(axis=1) >= np.cos(np.radians(angle_deg))
    i0, fa, fb = i0[smooth], fa[smooth], fb[smooth]
    va, vb = ea[i0], eb[i0]

    def corner(face, vertex):
        return 3 * face + np.argmax(F[face] == vertex[:, None], axis=1)

    rows = np.concatenate([corner(fa, va), corner(fa, vb)])
    cols = np.concatenate([corner(fb, va), corner(fb, vb)])
    graph = coo_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(3 * nF, 3 * nF))
    count, label = connected_components(graph, directed=False)
    src = np.full(count, -1)
    src[label[::-1]] = np.arange(3 * nF)[::-1]                               # first corner of every smooth patch
    vert = F.reshape(-1)[src]
    w = np.repeat(cr, 3, axis=0)
    N = np.stack([np.bincount(label, weights=w[:, k], minlength=count) for k in range(3)], axis=1)
    N /= np.maximum(np.linalg.norm(N, axis=1), 1e-30)[:, None]
    return V[vert], N, C[vert], label.reshape(-1, 3)


def surface_points(V, F, n, rng):
    """n random points on the surface, area-weighted."""
    cr = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    area = np.linalg.norm(cr, axis=1)
    tri = rng.choice(len(F), size=n, p=area / area.sum())
    r1, r2 = rng.random(n), rng.random(n)
    flip = r1 + r2 > 1
    r1[flip], r2[flip] = 1 - r1[flip], 1 - r2[flip]
    a, b, c = V[F[tri, 0]], V[F[tri, 1]], V[F[tri, 2]]
    return a + (b - a) * r1[:, None] + (c - a) * r2[:, None]


def deviation_mm(V0, F0, V1, F1, rng):
    """Two-sided surface-to-surface distances (mm) between the original and the decimated mesh, from dense point samples."""
    s0 = surface_points(V0, F0, min(300000, 3 * len(F0) + 20000), rng)
    s1 = surface_points(V1, F1, min(60000, 5 * len(F1) + 5000), rng)
    return np.concatenate([cKDTree(s0).query(s1)[0], cKDTree(s1).query(s0)[0]]) * 1000


BOOST = (('waist_yaw', 2.5), ('brand_', 100.0), ('kalki-', 1.7), ('rp1-', 3.0))    # the torso and head get a bigger share; logos are kept whole


def target_faces(name, n, ratio):
    boost = next((b for prefix, b in BOOST if name.startswith(prefix)), 1.0)
    return int(min(n, max(MIN_FACES, n * ratio * boost)))


def write_glb(path, nodes, materials, extras):
    blob, views, accessors, meshes, gltf_nodes = bytearray(), [], [], [], []

    def view(data, target):
        while len(blob) % 4:
            blob.append(0)
        views.append({'buffer': 0, 'byteOffset': len(blob), 'byteLength': len(data), 'target': target})
        blob.extend(data)
        return len(views) - 1

    def accessor(v, ctype, count, kind, **kw):
        accessors.append({'bufferView': v, 'componentType': ctype, 'count': count, 'type': kind, **kw})
        return len(accessors) - 1

    for i, n in enumerate(nodes):
        P = np.ascontiguousarray(n['P'], dtype='<f4')
        attrs = {'POSITION': accessor(view(P.tobytes(), 34962), 5126, len(P), 'VEC3', min=P.min(0).tolist(), max=P.max(0).tolist()),
                 'NORMAL': accessor(view(np.ascontiguousarray(n['N'], dtype='<f4').tobytes(), 34962), 5126, len(P), 'VEC3')}
        if n['C'] is not None:             # linear RGB, 16-bit normalized: 8 bits would band the dark greys
            rgba = np.concatenate([np.round(np.clip(n['C'], 0, 1) * 65535), np.full((len(P), 1), 65535)], axis=1).astype('<u2')
            attrs['COLOR_0'] = accessor(view(rgba.tobytes(), 34962), 5123, len(P), 'VEC4', normalized=True)
        idx = n['I'].reshape(-1)
        small = len(P) < 65535
        indices = accessor(view(idx.astype('<u2' if small else '<u4').tobytes(), 34963), 5123 if small else 5125, len(idx), 'SCALAR')
        meshes.append({'name': n['name'], 'primitives': [{'attributes': attrs, 'indices': indices, 'material': n['material'], 'mode': 4}]})
        gltf_nodes.append({'name': n['name'], 'mesh': i, 'extras': {'body': n['body']}})
    gltf = {'asset': {'version': '2.0', 'generator': 'tools/build_kalki_glb.py', 'extras': extras},
            'scene': 0, 'scenes': [{'nodes': list(range(len(gltf_nodes)))}], 'nodes': gltf_nodes, 'meshes': meshes,
            'materials': materials, 'accessors': accessors, 'bufferViews': views, 'buffers': [{'byteLength': len(blob)}]}
    js = json.dumps(gltf, separators=(',', ':')).encode()
    js += b' ' * (-len(js) % 4)
    blob.extend(b'\0' * (-len(blob) % 4))
    with open(path, 'wb') as f:
        f.write(struct.pack('<4sII', b'glTF', 2, 12 + 8 + len(js) + 8 + len(blob)))
        f.write(struct.pack('<I4s', len(js), b'JSON') + js)
        f.write(struct.pack('<I4s', len(blob), b'BIN\0') + bytes(blob))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model-dir', type=Path, default=DEFAULT_MODEL, help='folder holding robot.xml and assets/ (default: %(default)s)')
    ap.add_argument('--out', type=Path, default=REPO / 'live' / 'kalki' / 'kalki.glb')
    ap.add_argument('--ratio', type=float, default=0.12, help='fraction of faces asked for on the body meshes (the torso, logos and head get more)')
    ap.add_argument('--angle', type=float, default=CREASE_DEG, help='crease angle in degrees for hard normals')
    ap.add_argument('--agg', type=float, default=3.0,
                    help='decimator aggressiveness. These CAD shells are long skinny strips: 5 and up shreds them, 3 keeps them within ~3 mm '
                         'but may stop above the requested face count')
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    m = mujoco.MjModel.from_xml_path(str(args.model_dir / 'robot.xml'))
    materials = []
    for k in range(m.nmat):
        textured = int(m.mat_texid[k][mujoco.mjtTextureRole.mjTEXROLE_RGB]) >= 0
        rgba = m.mat_rgba[k]
        spec, shin = float(m.mat_specular[k]), float(m.mat_shininess[k])
        base = [1.0, 1.0, 1.0, 1.0] if textured else [*srgb_to_linear(rgba[:3]).tolist(), float(rgba[3])]
        materials.append({'name': m.material(k).name, 'doubleSided': True,
                          'pbrMetallicRoughness': {'baseColorFactor': base, 'metallicFactor': 0.3 if spec >= 0.8 else 0.0,
                                                   'roughnessFactor': round(float(np.clip(1 - 0.9 * shin, 0.3, 1.0)), 3)}})
    nodes, totals = [], {'faces_in': 0, 'faces_out': 0, 'verts_out': 0}
    worst = 0.0
    print('%-32s %9s -> %-7s %7s %6s  %s' % ('geom', 'faces', 'faces', 'verts', 'secs', 'surface deviation from the original (mm): p95 / max'))
    for geom in visual_geoms(m):
        t0 = time.perf_counter()
        V0, F0 = weld(geom['V'], geom['F'])
        V0, F0, keep = drop_degenerate(V0, F0)
        face_lin = srgb_to_linear(geom['face_srgb'][keep]) if geom['face_srgb'] is not None else None
        goal = target_faces(geom['name'], len(F0), args.ratio)
        if goal < len(F0):
            V1, F1 = fast_simplification.simplify(V0, F0, target_count=goal, agg=args.agg)
            V1, F1, _ = drop_degenerate(V1, F1.astype(np.int64))
        else:
            V1, F1 = V0, F0
        C1 = transfer_colours(V0, F0, face_lin, V1) if face_lin is not None else None
        P, N, C, I = crease_split(V1, F1, C1 if C1 is not None else np.zeros((len(V1), 3)), args.angle)
        dev = deviation_mm(V0, F0, V1, F1, rng)
        worst = max(worst, dev.max())
        nodes.append({'name': geom['name'], 'body': geom['body'], 'material': geom['material'], 'P': P, 'N': N,
                      'C': C if C1 is not None else None, 'I': I})
        totals['faces_in'] += len(F0)
        totals['faces_out'] += len(I)
        totals['verts_out'] += len(P)
        print('%-32s %9d -> %-7d %7d %6.1f  %.2f / %.2f' % (geom['name'][:32], len(F0), len(I), len(P), time.perf_counter() - t0,
                                                            np.percentile(dev, 95), dev.max()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_glb(args.out, nodes, materials, {'source': 'Kalki_RP1_MuJoCo/robot.xml', 'ratio': args.ratio, **totals,
                                           'note': 'decimated production meshes; texture colours baked into vertex colours'})
    print('\n%d meshes: %d -> %d triangles (%.1f%%), %d vertices; %s is %.2f MB; largest deviation %.2f mm'
          % (len(nodes), totals['faces_in'], totals['faces_out'], 100 * totals['faces_out'] / totals['faces_in'], totals['verts_out'],
             args.out, args.out.stat().st_size / 1e6, worst))


if __name__ == '__main__':
    main()
