"""
thermal_solver.py
=================
Solver térmico estacionario 2.5D multicapa.

Modelo
------
Una grilla 2D por cada capa de cobre. En cada celda de cada capa hay un nodo.

  En el plano (dentro de una capa):
      conductancia de lámina  S = k_cu·t_cu·f + k_fr4_xy·t_fr4      [W/K]
      (f = fracción de cobre real de la celda; t_fr4 = dieléctrico asignado a
      esa capa). Entre celdas vecinas se usa la media armónica de S, así un
      corte en el cobre (clearance, isla) frena el calor como corresponde.

  Entre capas (vertical):
      G_z = k_fr4_z·A_celda/d  +  Σ barriles de vías en la celda     [W/K]

  Hacia el ambiente (caras top y bottom):
      G_s = (h_conv + h_rad)·A_celda
      h_conv: convección natural por correlación de placa (o valor fijo)
      h_rad = ε·σ·(T² + T_amb²)·(T + T_amb), celda por celda

Como h depende de T el problema es levemente no lineal: se resuelve el sistema
lineal disperso y se actualiza h unas pocas veces (Picard) hasta converger.

Qué NO modela: el aire como fluido (no es CFD), el calor que sale por el cuerpo
de los componentes, efecto Joule en pistas, ni transitorios.
"""

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, cg, splu

from rasterizer import Grid

SIGMA = 5.670374e-8     # Stefan-Boltzmann [W/m²K⁴]


@dataclass
class ThermalConfig:
    T_amb_C: float = 25.0
    k_cu: float = 385.0             # W/m·K
    k_fr4_xy: float = 0.8           # FR4 en el plano (la trama de vidrio ayuda)
    k_fr4_z: float = 0.3            # FR4 a través del espesor
    emissivity: float = 0.9         # máscara antisoldante ≈ 0.9; 0 = sin radiación
    orientation: str = "horizontal" # 'horizontal' | 'vertical'
    h_fixed: Optional[float] = None # W/m²K convectivo en ambas caras (aire forzado)
    max_iter: int = 30
    tol_C: float = 0.02


@dataclass
class Solution:
    T: np.ndarray                   # (nl, ny, nx) °C, NaN fuera de la placa
    P_in_W: float
    P_out_W: float
    h_conv_top: float
    h_conv_bot: float
    h_rad_mean: float
    iterations: int
    seconds: float
    rise_parts: Optional[list] = None   # por grupo de fuentes: (nl,ny,nx) °C sobre ambiente

    @property
    def T_max(self) -> float:
        return float(np.nanmax(self.T))

    @property
    def T_mean_surface(self) -> float:
        return float(np.nanmean(self.T[[0, -1]]))


def _air(T_film_C):
    """Propiedades del aire a presión atmosférica (ajuste lineal, 0–150 °C)."""
    k = 0.0242 + 7.3e-5 * T_film_C          # W/m·K
    nu = 1.33e-5 + 9.6e-8 * T_film_C        # m²/s
    return k, nu, 0.71                      # Pr


def natural_convection_h(dT, T_amb_C, L_m, kind):
    """h medio [W/m²K] por convección natural, régimen laminar.
       kind: 'up' cara caliente hacia arriba   Nu = 0.54·Ra^¼   (L = A/P)
             'down' cara caliente hacia abajo  Nu = 0.27·Ra^¼   (L = A/P)
             'vertical'                        Nu = 0.59·Ra^¼   (L = altura)
       Ref: Incropera, Fundamentals of Heat and Mass Transfer, cap. 9."""
    dT = max(dT, 0.05)
    Tf = T_amb_C + dT / 2
    k, nu, Pr = _air(Tf)
    Ra = 9.81 * dT * L_m**3 * Pr / ((Tf + 273.15) * nu**2)
    C = {'up': 0.54, 'down': 0.27, 'vertical': 0.59}[kind]
    return C * Ra**0.25 * k / L_m


def solve(grid: Grid, board, q: np.ndarray, cfg: ThermalConfig, log=print,
          q_parts=None) -> Solution:
    """q: potencia total [W/celda]. q_parts (opcional): lista de mapas de potencia
    cuya suma es q; se devuelve cuánto calienta cada uno por separado (con el h
    final congelado el problema es lineal, así que los aportes suman el total)."""
    t0 = time.time()
    nl, ny, nx = grid.copper.shape
    A = grid.cell_area_m2
    active = grid.board > 0.3
    n2 = int(active.sum())
    idx2 = -np.ones((ny, nx), dtype=np.int64)
    idx2[active] = np.arange(n2)
    N = n2 * nl

    # ── Conductancia de lámina por capa ──────────────────────────────────
    d = np.array(board.dielectrics_mm) * 1e-3
    rows, cols, vals = [], [], []

    def couple(a, b, g):
        rows.extend((a, b, a, b)); cols.extend((a, b, b, a)); vals.extend((g, g, -g, -g))

    for l in range(nl):
        t_fr4 = (d[l - 1] / 2 if l > 0 else 0) + (d[l] / 2 if l < nl - 1 else 0)
        if nl == 1:
            t_fr4 = 1.5e-3
        S = cfg.k_cu * board.layers[l].thickness_mm * 1e-3 * grid.copper[l] \
            + cfg.k_fr4_xy * t_fr4 * grid.board
        off = l * n2
        for ax in (0, 1):
            a = [slice(None)] * 2; b = [slice(None)] * 2
            a[ax] = slice(0, -1); b[ax] = slice(1, None)
            a, b = tuple(a), tuple(b)
            ok = active[a] & active[b]
            g = 2 * S[a] * S[b] / (S[a] + S[b] + 1e-30)
            couple(idx2[a][ok] + off, idx2[b][ok] + off, g[ok])

    # ── Acople vertical: FR4 + vías ──────────────────────────────────────
    for g_i in range(nl - 1):
        gz = cfg.k_fr4_z * A * grid.board / d[g_i] + grid.via_G[g_i]
        couple(idx2[active] + g_i * n2, idx2[active] + (g_i + 1) * n2, gz[active])

    K = sparse.coo_matrix((np.concatenate([np.atleast_1d(v) for v in vals]),
                           (np.concatenate([np.atleast_1d(r) for r in rows]),
                            np.concatenate([np.atleast_1d(c) for c in cols]))),
                          shape=(N, N)).tocsc()

    qv = np.concatenate([q[l][active] for l in range(nl)])
    P_in = float(qv.sum())
    area = grid.board[active] * A                   # área expuesta por celda
    top = slice(0, n2)
    bot = slice((nl - 1) * n2, nl * n2)

    # Longitudes características
    x0, y0, x1, y1 = board.bbox
    A_board = float(area.sum())
    perim = 2 * ((x1 - x0) + (y1 - y0)) * 1e-3
    L_h = A_board / perim
    L_v = (y1 - y0) * 1e-3
    Ta = cfg.T_amb_C
    TaK = Ta + 273.15

    T = np.full(N, Ta + P_in / (2 * 12.0 * A_board + 1e-12))    # arranque
    it, lu = 0, None
    for it in range(1, cfg.max_iter + 1):
        dT_top = max(float(np.average(T[top], weights=area)) - Ta, 0.0)
        dT_bot = max(float(np.average(T[bot], weights=area)) - Ta, 0.0)
        if cfg.h_fixed is not None:
            hc_top = hc_bot = cfg.h_fixed
        elif cfg.orientation == 'vertical':
            hc_top = natural_convection_h(dT_top, Ta, L_v, 'vertical')
            hc_bot = natural_convection_h(dT_bot, Ta, L_v, 'vertical')
        else:
            hc_top = natural_convection_h(dT_top, Ta, L_h, 'up')
            hc_bot = natural_convection_h(dT_bot, Ta, L_h, 'down')

        def h_rad(Tc):
            TK = Tc + 273.15
            return cfg.emissivity * SIGMA * (TK**2 + TaK**2) * (TK + TaK)

        gs = np.zeros(N)
        gs[top] += (hc_top + h_rad(T[top])) * area
        gs[bot] += (hc_bot + h_rad(T[bot])) * area

        M = K + sparse.diags(gs, format='csc')
        rhs = qv + gs * Ta
        if lu is None and N <= 1_500_000:
            # Se factoriza una sola vez; en las iteraciones siguientes solo cambia
            # un poco la diagonal (h), así que esa LU sirve de precondicionador.
            lu = splu(M, permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=0.0,
                      options=dict(SymmetricMode=True))
            T_new = lu.solve(rhs)
        else:
            pre = (LinearOperator((N, N), matvec=lu.solve) if lu is not None
                   else sparse.diags(1.0 / M.diagonal()))
            try:
                T_new, info = cg(M, rhs, x0=T, M=pre, rtol=1e-10, maxiter=20000)
            except TypeError:                       # SciPy < 1.12
                T_new, info = cg(M, rhs, x0=T, M=pre, tol=1e-10, maxiter=20000)
            if info != 0:
                log(f"  [aviso] CG no convergió (info={info})")
        err = float(np.max(np.abs(T_new - T)))
        T = T_new
        if err < cfg.tol_C:
            break

    P_out = float(np.sum(gs * (T - Ta)))
    hr = np.concatenate((h_rad(T[top]), h_rad(T[bot])))

    def unpack(v, fill):
        full = np.full((nl, ny, nx), fill)
        for l in range(nl):
            full[l][active] = v[l * n2:(l + 1) * n2]
        return full

    parts = None
    if q_parts:
        pre = (LinearOperator((N, N), matvec=lu.solve) if lu is not None
               else sparse.diags(1.0 / M.diagonal()))
        parts = []
        for qp in q_parts:
            b = np.concatenate([qp[l][active] for l in range(nl)])
            guess = (T - Ta) * (b.sum() / max(P_in, 1e-30))
            try:
                x, _ = cg(M, b, x0=guess, M=pre, rtol=1e-9, maxiter=20000)
            except TypeError:
                x, _ = cg(M, b, x0=guess, M=pre, tol=1e-9, maxiter=20000)
            parts.append(unpack(x, 0.0))

    return Solution(unpack(T, np.nan), P_in, P_out, hc_top, hc_bot, float(hr.mean()),
                    it, time.time() - t0, parts)
