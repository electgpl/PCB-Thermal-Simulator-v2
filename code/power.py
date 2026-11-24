"""
power.py
========
Fuentes de calor: qué componente disipa cuánto y por qué pads entra a la placa.

Los datos térmicos salen de dos lugares (el CSV pisa a los campos):
  1. Campos del componente en KiCad (se cargan en el esquemático y pasan al PCB
     con "Update PCB from Schematic"):  Power_W, Theta_JB, Theta_JC, Theta_JA,
     Thermal_Pads
  2. Un CSV:  Component,Power_W,Theta_JB,Theta_JC,Theta_JA,Thermal_Pads,Description

Modelo de componente:
  Toda la potencia entra a la placa por los pads (en convección natural y sin
  disipador, lo que sale por arriba del encapsulado es despreciable; es además
  la hipótesis conservadora). Reparto:
    - Thermal_Pads indicado (ej. "9" o "2 4")  → solo esos pads
    - si no, y el footprint tiene pad expuesto o aleta (TO-252, SOT-223) → solo ese pad
    - si no                                    → todos los pads, por área
  T_j = T_pads(simulada) + P · θ_JB

T_max es la temperatura máxima admisible del componente (Tj máx. de la hoja de
datos para integrados; temperatura máxima de operación para pasivos). Con ese
dato la simulación da un veredicto PASA / JUSTO / FALLA por componente.
"""

import csv
import io
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from board_model import Board, Component, Pad
from rasterizer import Grid, rasterize_polys

_ALIASES = {
    'component': 'ref', 'ref': 'ref', 'reference': 'ref', 'designator': 'ref',
    'power_w': 'power', 'power': 'power', 'thermal_power': 'power', 'p_w': 'power',
    'theta_jb': 'theta_jb', 'psi_jb': 'theta_jb', 'rth_jb': 'theta_jb',
    'theta_jc': 'theta_jc', 'rth_jc': 'theta_jc',
    'theta_ja': 'theta_ja', 'rth_ja': 'theta_ja',
    'thermal_pads': 'pads', 'pads': 'pads',
    't_max': 't_max', 'tmax': 't_max', 'tj_max': 't_max', 'tjmax': 't_max',
    't_max_c': 't_max', 'temp_max': 't_max',
    'description': 'desc',
}


@dataclass
class Source:
    ref: str
    power_W: float
    comp: Component
    theta_jb: Optional[float] = None
    theta_jc: Optional[float] = None
    theta_ja: Optional[float] = None
    t_max: Optional[float] = None           # límite de temperatura [°C]
    pads: List[Pad] = field(default_factory=list)
    pad_rule: str = ""
    weights: Optional[np.ndarray] = None    # (ny,nx), suma 1: reparto de P en celdas


def _num(s) -> Optional[float]:
    if s is None:
        return None
    s = str(s).strip().replace(',', '.')
    try:
        return float(s)
    except ValueError:
        return None


def read_csv(path: str) -> Dict[str, dict]:
    with open(path, encoding='utf-8-sig') as fh:
        text = fh.read()
    # Las líneas que empiezan con # son comentarios
    text = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith('#'))
    first = text.splitlines()[0] if text.strip() else ''
    delim = ';' if first.count(';') > first.count(',') else ','
    rows = {}
    for row in csv.DictReader(io.StringIO(text), delimiter=delim):
        rec = {}
        for k, v in row.items():
            key = _ALIASES.get((k or '').strip().lower())
            if key:
                rec[key] = (v or '').strip()
        if rec.get('ref'):
            rows[rec['ref']] = rec
    return rows


def _from_props(c: Component) -> dict:
    rec = {}
    for k, v in c.props.items():
        key = _ALIASES.get(k.strip().lower().replace(' ', '_'))
        if key and key not in ('ref', 'desc') and str(v).strip():
            rec[key] = str(v).strip()
    return rec


def _thermal_group(c: Component):
    """Detecta por dónde baja el calor cuando el footprint lo deja claro.
    Los pads se agrupan por número (un pad expuesto o una aleta suelen estar
    dibujados como varios pads con el mismo número). Devuelve (pads, motivo) o None.

      - Pad expuesto: grupo grande y centrado en un encapsulado de ≥5 pines.
      - Aleta (TO-252, SOT-223, ...): un grupo con ≥60 % del cobre del footprint.
    """
    mine = [p for p in c.pads if c.layer in p.layers]
    groups = {}
    for p in mine:
        groups.setdefault(p.number, []).append(p)
    if len(groups) < 3:
        return None

    def g_area(ps):         # sin contar dos veces pads superpuestos (sub-pads de pasta)
        big = max(ps, key=lambda p: p.area_mm2)
        v = np.vstack(big.polys)
        lo, hi = v.min(axis=0), v.max(axis=0)
        return big.area_mm2 + sum(p.area_mm2 for p in ps if p is not big
                                  and not np.all((p.center >= lo) & (p.center <= hi)))

    areas = {n: g_area(ps) for n, ps in groups.items()}
    num = max(areas, key=areas.get)
    others = [a for n, a in areas.items() if n != num]
    big = max(groups[num], key=lambda p: p.area_mm2)
    if len(groups) >= 5:
        ctr = np.mean([p.center for p in mine if p.number != num], axis=0)
        span = np.ptp(np.vstack([p.center for p in mine]), axis=0).max()
        if areas[num] > 2.5 * np.median(others) and \
                np.hypot(*(big.center - ctr)) < 0.15 * max(span, 1e-6):
            return groups[num], f"pad expuesto ({num})"
    if areas[num] >= 0.6 * sum(areas.values()):
        return groups[num], f"aleta (pad {num})"
    return None


def build_sources(board: Board, grid: Grid, csv_path: Optional[str] = None,
                  log=print) -> List[Source]:
    table = read_csv(csv_path) if csv_path else {}
    refs = {c.ref for c in board.components}
    for r in table:
        if r not in refs:
            log(f"  [aviso] '{r}' está en el CSV pero no en el PCB; se ignora.")

    sources = []
    for c in board.components:
        rec = _from_props(c)
        rec.update({k: v for k, v in table.get(c.ref, {}).items() if v != ''})
        P = _num(rec.get('power'))
        if not P or P <= 0:
            continue
        if c.dnp:
            log(f"  [aviso] {c.ref} está marcado DNP pero tiene potencia; se simula igual.")
        s = Source(c.ref, P, c, _num(rec.get('theta_jb')), _num(rec.get('theta_jc')),
                   _num(rec.get('theta_ja')), _num(rec.get('t_max')))

        mine = [p for p in c.pads if c.layer in p.layers]
        wanted = rec.get('pads', '').replace(';', ' ').replace(',', ' ').split()
        if wanted:
            s.pads = [p for p in mine if p.number in wanted]
            s.pad_rule = "pads " + " ".join(wanted)
            if not s.pads:
                log(f"  [aviso] {c.ref}: Thermal_Pads={wanted} no existe; uso todos los pads.")
        if not s.pads:
            tg = _thermal_group(c)
            if tg is not None:
                s.pads, s.pad_rule = tg
            else:
                s.pads, s.pad_rule = mine, "todos los pads"
        if not s.pads:
            log(f"  [aviso] {c.ref} no tiene pads de cobre; se ignora.")
            continue

        w = rasterize_polys(grid, [poly for p in s.pads for poly in p.polys]) * grid.board
        if w.sum() <= 0:
            log(f"  [aviso] {c.ref} cae fuera de la placa; se ignora.")
            continue
        s.weights = w / w.sum()
        sources.append(s)
    return sources


def power_map(sources: List[Source], grid: Grid, n_layers: int) -> np.ndarray:
    """q[l,j,i] en W por celda."""
    q = np.zeros((n_layers, grid.ny, grid.nx))
    for s in sources:
        q[s.comp.layer] += s.power_W * s.weights
    return q


def write_template(board: Board, path: str):
    """CSV con todos los componentes del PCB para completar potencias."""
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['Component', 'Power_W', 'T_max', 'Theta_JB', 'Theta_JC', 'Theta_JA',
                    'Thermal_Pads', 'Description'])
        for c in sorted(board.components, key=lambda c: c.ref):
            if c.pads and not c.dnp and len(c.pads) > 1:
                w.writerow([c.ref, 0, '', '', '', '', '', f"{c.value} ({c.footprint.split(':')[-1]})"])
