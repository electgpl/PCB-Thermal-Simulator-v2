"""
kicad_loader.py
===============
Frontend KiCad: lee un .kicad_pcb (KiCad 6 a 9) y devuelve un Board.

No necesita KiCad instalado: el archivo es una S-expression de texto y se
parsea directamente. Se extrae:
  - Stackup real (espesor de cada cobre y cada dieléctrico) de Board Setup.
  - Contorno de Edge.Cuts (rectángulos, círculos, polígonos, líneas y arcos).
  - Cobre real por capa: zonas RELLENAS, pistas, arcos, pads y anillos de vías.
  - Vías y pads pasantes (barriles metalizados).
  - Componentes con sus pads, courtyard y campos (Power_W, Theta_JB, ...).

Importante: las zonas se toman de los 'filled_polygon' guardados en el archivo.
Hay que rellenar zonas (tecla B) y guardar antes de simular.
"""

import math
import os
import re
from typing import Dict, List, Tuple

import numpy as np

from board_model import Board, Component, CopperLayer, Pad, Shape, Via

# ─── S-expression ────────────────────────────────────────────────────────────

_TOKEN = re.compile(r'\(|\)|"(?:\\.|[^"\\])*"|[^\s()]+')


def parse_sexpr(text: str):
    stack = [[]]
    for m in _TOKEN.finditer(text):
        t = m.group(0)
        if t == '(':
            stack.append([])
        elif t == ')':
            node = stack.pop()
            stack[-1].append(node)
        elif t[0] == '"':
            stack[-1].append(t[1:-1].replace('\\"', '"').replace('\\\\', '\\'))
        else:
            stack[-1].append(t)
    if len(stack) != 1 or not stack[0]:
        raise ValueError("Archivo S-expression mal formado (paréntesis desbalanceados)")
    return stack[0][0]


def kids(node, name):
    return [c for c in node if isinstance(c, list) and c and c[0] == name]


def kid(node, name):
    for c in node:
        if isinstance(c, list) and c and c[0] == name:
            return c
    return None


def _f(x) -> float:
    return float(x)


def _xy(node) -> Tuple[float, float]:
    return _f(node[1]), _f(node[2])


# ─── Geometría ───────────────────────────────────────────────────────────────

def rotate(pts: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotación con la convención de KiCad (ángulo positivo = antihorario en
    pantalla, con el eje Y hacia abajo)."""
    if not angle_deg:
        return pts
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    x, y = pts[:, 0], pts[:, 1]
    return np.column_stack((x * c + y * s, -x * s + y * c))


def circle_poly(cx, cy, r, n=32) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack((cx + r * np.cos(t), cy + r * np.sin(t)))


def roundrect_poly(w, h, r, n=6) -> np.ndarray:
    r = max(0.0, min(r, w / 2, h / 2))
    if r <= 1e-9:
        return np.array([[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]])
    pts = []
    for cx, cy, a0 in ((w/2 - r, -h/2 + r, -90), (w/2 - r, h/2 - r, 0),
                       (-w/2 + r, h/2 - r, 90), (-w/2 + r, -h/2 + r, 180)):
        for k in range(n + 1):
            a = math.radians(a0 + 90 * k / n)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return np.array(pts)


def arc_points(p0, pm, p1, seg_deg=10.0) -> np.ndarray:
    """Polilínea del arco que pasa por inicio, medio y fin."""
    (x1, y1), (x2, y2), (x3, y3) = p0, pm, p1
    d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(d) < 1e-12:
        return np.array([p0, p1])
    ux = ((x1**2 + y1**2) * (y2 - y3) + (x2**2 + y2**2) * (y3 - y1)
          + (x3**2 + y3**2) * (y1 - y2)) / d
    uy = ((x1**2 + y1**2) * (x3 - x2) + (x2**2 + y2**2) * (x1 - x3)
          + (x3**2 + y3**2) * (x2 - x1)) / d
    r = math.hypot(x1 - ux, y1 - uy)
    a0 = math.atan2(y1 - uy, x1 - ux)
    am = math.atan2(y2 - uy, x2 - ux)
    a1 = math.atan2(y3 - uy, x3 - ux)
    # elegir el sentido que pasa por el punto medio
    def ccw(a, b):
        return (b - a) % (2 * math.pi)
    if ccw(a0, am) <= ccw(a0, a1):
        sweep = ccw(a0, a1)
    else:
        sweep = -ccw(a1, a0)
    n = max(2, int(abs(math.degrees(sweep)) / seg_deg) + 1)
    t = a0 + sweep * np.linspace(0, 1, n + 1)
    return np.column_stack((ux + r * np.cos(t), uy + r * np.sin(t)))


def _pts(node) -> np.ndarray:
    """Lee (pts (xy ..) (arc ..) ...)."""
    out = []
    p = kid(node, 'pts')
    if p is None:
        return np.zeros((0, 2))
    for q in p[1:]:
        if not isinstance(q, list):
            continue
        if q[0] == 'xy':
            out.append(_xy(q))
        elif q[0] == 'arc':
            a = arc_points(_xy(kid(q, 'start')), _xy(kid(q, 'mid')), _xy(kid(q, 'end')))
            out.extend(map(tuple, a))
    return np.array(out, dtype=float).reshape(-1, 2)


# ─── Stackup ─────────────────────────────────────────────────────────────────

def _read_stackup(root, warnings) -> Tuple[List[CopperLayer], List[float]]:
    declared = [l[1] for l in kid(root, 'layers')[1:]
                if isinstance(l, list) and len(l) > 1 and l[1].endswith('.Cu')]

    def order(n):
        return 0 if n == 'F.Cu' else 10**6 if n == 'B.Cu' else int(re.sub(r'\D', '', n) or 1)
    declared.sort(key=order)

    setup = kid(root, 'setup')
    stackup = kid(setup, 'stackup') if setup else None
    if stackup:
        layers, diel, pending = [], [], 0.0
        for l in kids(stackup, 'layer'):
            tk = sum(_f(t[1]) for t in kids(l, 'thickness'))
            typ = (kid(l, 'type') or [None, ''])[1]
            if typ == 'copper':
                if layers:
                    diel.append(pending)
                pending = 0.0
                layers.append(CopperLayer(l[1], tk or 0.035))
            elif l[1].startswith('dielectric'):
                pending += tk
        if layers and len(diel) == len(layers) - 1 and all(d > 0 for d in diel):
            return layers, diel
        warnings.append("Stackup incompleto en el archivo; se usa uno genérico.")

    # Sin stackup definido: cobre de 1 oz y dieléctricos repartidos parejo
    total = _f(kid(kid(root, 'general'), 'thickness')[1]) if kid(root, 'general') else 1.6
    n = len(declared)
    layers = [CopperLayer(nm, 0.035) for nm in declared]
    diel = [max(0.05, (total - 0.035 * n) / (n - 1))] * (n - 1) if n > 1 else []
    warnings.append("El PCB no tiene stackup definido (Board Setup → Physical Stackup): "
                    "se asume 1 oz por capa y dieléctricos iguales.")
    return layers, diel


# ─── Contorno ────────────────────────────────────────────────────────────────

def _chain_loops(segments: List[np.ndarray], tol=0.02) -> List[np.ndarray]:
    """Une polilíneas sueltas (líneas y arcos de Edge.Cuts) en lazos cerrados."""
    segs = [s for s in segments if len(s) >= 2]
    loops = []
    while segs:
        cur = segs.pop(0)
        closed = False
        while not closed:
            if np.hypot(*(cur[0] - cur[-1])) < tol and len(cur) > 2:
                closed = True
                break
            for k, s in enumerate(segs):
                if np.hypot(*(s[0] - cur[-1])) < tol:
                    cur = np.vstack((cur, s[1:])); segs.pop(k); break
                if np.hypot(*(s[-1] - cur[-1])) < tol:
                    cur = np.vstack((cur, s[::-1][1:])); segs.pop(k); break
            else:
                break
        if closed:
            loops.append(cur[:-1])
    return loops


def _read_outline(root, warnings) -> List[np.ndarray]:
    loops, open_segs = [], []
    for g in root[1:]:
        if not isinstance(g, list) or not g[0].startswith('gr_'):
            continue
        lay = kid(g, 'layer')
        if not lay or lay[1] != 'Edge.Cuts':
            continue
        if g[0] == 'gr_rect':
            (x0, y0), (x1, y1) = _xy(kid(g, 'start')), _xy(kid(g, 'end'))
            loops.append(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]))
        elif g[0] == 'gr_circle':
            c, e = _xy(kid(g, 'center')), _xy(kid(g, 'end'))
            loops.append(circle_poly(c[0], c[1], math.hypot(e[0] - c[0], e[1] - c[1]), 64))
        elif g[0] == 'gr_poly':
            loops.append(_pts(g))
        elif g[0] == 'gr_line':
            open_segs.append(np.array([_xy(kid(g, 'start')), _xy(kid(g, 'end'))]))
        elif g[0] == 'gr_arc':
            open_segs.append(arc_points(_xy(kid(g, 'start')), _xy(kid(g, 'mid')),
                                        _xy(kid(g, 'end')), 5.0))
    loops += _chain_loops(open_segs)
    if not loops:
        raise ValueError("No se encontró un contorno cerrado en Edge.Cuts.")
    return loops


# ─── Footprints ──────────────────────────────────────────────────────────────

def _pad_local_polys(pad) -> List[np.ndarray]:
    shape = pad[3]
    w, h = _xy(kid(pad, 'size'))
    if shape == 'circle':
        base = circle_poly(0, 0, w / 2)
    elif shape == 'oval':
        base = roundrect_poly(w, h, min(w, h) / 2, 8)
    elif shape == 'roundrect':
        rr = kid(pad, 'roundrect_rratio')
        base = roundrect_poly(w, h, (_f(rr[1]) if rr else 0.25) * min(w, h))
    else:                       # rect, trapezoid, custom (ancla)
        base = roundrect_poly(w, h, 0)
    polys = [base]
    if shape == 'custom':
        opt = kid(pad, 'options')
        if opt and kid(opt, 'anchor') and kid(opt, 'anchor')[1] == 'circle':
            polys = [circle_poly(0, 0, w / 2)]
        prim = kid(pad, 'primitives')
        for g in (prim[1:] if prim else []):
            if isinstance(g, list) and g[0] == 'gr_poly':
                p = _pts(g)
                if len(p) >= 3:
                    polys.append(p)
    return polys


def _expand_layers(names: List[str], cu_names: List[str]) -> List[int]:
    out = set()
    for n in names:
        if n == '*.Cu':
            out.update(range(len(cu_names)))
        elif n == 'F&B.Cu':
            out.update((0, len(cu_names) - 1))
        elif n in cu_names:
            out.add(cu_names.index(n))
    return sorted(out)


def _read_footprint(fp, cu_names, shapes, vias, nets) -> Component:
    at = kid(fp, 'at')
    fx, fy = _f(at[1]), _f(at[2])
    frot = _f(at[3]) if len(at) > 3 else 0.0
    side = kid(fp, 'layer')[1]
    layer = cu_names.index(side) if side in cu_names else 0

    props = {p[1]: p[2] for p in kids(fp, 'property') if len(p) > 2 and isinstance(p[1], str)}
    # KiCad 6/7: (fp_text reference "U1" ...)
    for t in kids(fp, 'fp_text'):
        if t[1] == 'reference':
            props.setdefault('Reference', t[2])
        elif t[1] == 'value':
            props.setdefault('Value', t[2])
    attr = kid(fp, 'attr') or []

    def to_board(local: np.ndarray) -> np.ndarray:
        return rotate(local, frot) + (fx, fy)

    pads = []
    for pad in kids(fp, 'pad'):
        pat = kid(pad, 'at')
        px, py = _f(pat[1]), _f(pat[2])
        prot = _f(pat[3]) if len(pat) > 3 else 0.0     # ángulo absoluto
        center = to_board(np.array([[px, py]]))[0]
        polys = [rotate(p, prot) + center for p in _pad_local_polys(pad)]
        layers = _expand_layers(kid(pad, 'layers')[1:], cu_names)
        net = kid(pad, 'net')
        if not layers:
            continue                                    # pad sin cobre (NPTH, paste)
        pads.append(Pad(str(pad[1]), polys, layers, net[2] if net and len(net) > 2 else ""))
        for l in layers:
            shapes[l].extend(Shape('poly', p) for p in polys)
        drill = kid(pad, 'drill')
        if pad[2] == 'thru_hole' and drill and len(layers) > 1:
            nums = [_f(v) for v in drill[1:] if not isinstance(v, list) and v != 'oval']
            if nums:
                vias.append(Via(center[0], center[1], min(nums), min(layers), max(layers)))

    # Gráficos de cobre dentro del footprint
    for g in fp[1:]:
        if not isinstance(g, list) or g[0] not in ('fp_poly', 'fp_line', 'fp_rect'):
            continue
        lay = kid(g, 'layer')
        if not lay or lay[1] not in cu_names:
            continue
        l = cu_names.index(lay[1])
        if g[0] == 'fp_poly':
            shapes[l].append(Shape('poly', to_board(_pts(g))))
        else:
            a, b = _xy(kid(g, 'start')), _xy(kid(g, 'end'))
            wd = kid(kid(g, 'stroke') or [], 'width')
            if g[0] == 'fp_rect':
                shapes[l].append(Shape('poly', to_board(
                    np.array([a, (b[0], a[1]), b, (a[0], b[1])]))))
            else:
                shapes[l].append(Shape('line', to_board(np.array([a, b])),
                                       _f(wd[1]) if wd else 0.1))

    # Caja del componente: courtyard; si no hay, la de los pads
    cy = []
    for g in fp[1:]:
        if not isinstance(g, list) or not g[0].startswith('fp_'):
            continue
        lay = kid(g, 'layer')
        if not lay or not lay[1].endswith('.CrtYd'):
            continue
        if g[0] == 'fp_poly':
            cy.append(_pts(g))
        elif g[0] == 'fp_circle':
            c, e = _xy(kid(g, 'center')), _xy(kid(g, 'end'))
            cy.append(circle_poly(c[0], c[1], math.hypot(e[0]-c[0], e[1]-c[1]), 16))
        elif kid(g, 'start') and kid(g, 'end'):
            a, b = _xy(kid(g, 'start')), _xy(kid(g, 'end'))
            cy.append(np.array([a, (b[0], a[1]), b, (a[0], b[1])]) if g[0] == 'fp_rect'
                      else np.array([a, b]))
    if cy:
        pts = to_board(np.vstack(cy))
    elif pads:
        pts = np.vstack([p for pd in pads for p in pd.polys])
    else:
        pts = np.array([[fx, fy]])
    bbox = (pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max())

    return Component(ref=props.get('Reference', '?'), value=props.get('Value', ''),
                     footprint=str(fp[1]), layer=layer, x=fx, y=fy, bbox=bbox,
                     pads=pads, props=props,
                     dnp=('dnp' in attr) or kid(fp, 'dnp') is not None)


# ─── Carga principal ─────────────────────────────────────────────────────────

def load_kicad_pcb(path: str) -> Board:
    with open(path, encoding='utf-8') as fh:
        root = parse_sexpr(fh.read())
    if root[0] != 'kicad_pcb':
        raise ValueError(f"{path} no es un archivo .kicad_pcb")

    warnings: List[str] = []
    layers, diel = _read_stackup(root, warnings)
    cu = [l.name for l in layers]
    n = len(cu)
    shapes: List[List[Shape]] = [[] for _ in range(n)]
    vias: List[Via] = []
    nets = {}

    outline = _read_outline(root, warnings)

    # Pistas
    for s in kids(root, 'segment'):
        lay = kid(s, 'layer')[1]
        if lay in cu:
            shapes[cu.index(lay)].append(Shape(
                'line', np.array([_xy(kid(s, 'start')), _xy(kid(s, 'end'))]),
                _f(kid(s, 'width')[1])))
    for s in kids(root, 'arc'):
        lay = kid(s, 'layer')[1]
        if lay in cu:
            p = arc_points(_xy(kid(s, 'start')), _xy(kid(s, 'mid')), _xy(kid(s, 'end')))
            w = _f(kid(s, 'width')[1])
            shapes[cu.index(lay)].extend(
                Shape('line', p[k:k + 2], w) for k in range(len(p) - 1))

    # Vías
    for v in kids(root, 'via'):
        x, y = _xy(kid(v, 'at'))
        size = _f(kid(v, 'size')[1])
        drill = _f(kid(v, 'drill')[1])
        ls = _expand_layers(kid(v, 'layers')[1:], cu)
        if len(ls) < 2:
            continue
        l0, l1 = min(ls), max(ls)
        vias.append(Via(x, y, drill, l0, l1))
        for l in range(l0, l1 + 1):
            shapes[l].append(Shape('circle', np.array([[x, y]]), size))

    # Zonas (solo el relleno real)
    unfilled = 0
    for z in kids(root, 'zone'):
        if kid(z, 'keepout'):
            continue
        fills = kids(z, 'filled_polygon')
        if not fills:
            unfilled += 1
        for f in fills:
            lay = kid(f, 'layer')[1]
            p = _pts(f)
            if lay in cu and len(p) >= 3:
                shapes[cu.index(lay)].append(Shape('poly', p))
    if unfilled:
        warnings.append(f"{unfilled} zona(s) de cobre sin rellenar: en KiCad presioná B "
                        "(Fill All Zones), guardá y volvé a correr. Así como está, "
                        "ese cobre NO entra en la simulación.")

    # Gráficos de cobre sueltos
    for g in root[1:]:
        if isinstance(g, list) and g[0] == 'gr_poly':
            lay = kid(g, 'layer')
            if lay and lay[1] in cu:
                shapes[cu.index(lay[1])].append(Shape('poly', _pts(g)))

    comps = [_read_footprint(fp, cu, shapes, vias, nets)
             for fp in kids(root, 'footprint') + kids(root, 'module')]

    return Board(name=os.path.splitext(os.path.basename(path))[0],
                 outline=outline, layers=layers, dielectrics_mm=diel,
                 shapes=shapes, vias=vias, components=comps, warnings=warnings)
