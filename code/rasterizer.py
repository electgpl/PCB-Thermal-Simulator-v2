"""
rasterizer.py
=============
Convierte la geometría del Board en mapas sobre una grilla regular de celdas
cuadradas:

  board[j,i]        fracción de la celda que es placa (0–1)
  copper[l,j,i]     fracción de la celda cubierta por cobre en la capa l (0–1)
  via_G[g,j,i]      conductancia de barriles de vía entre capa g y g+1 [W/K]

El cobre se dibuja con supersampling (cada celda = ss×ss sub-píxeles) para que
pistas más finas que una celda aporten su fracción real de cobre.
"""

import math
from dataclasses import dataclass
from typing import List

import numpy as np
from PIL import Image, ImageDraw

from board_model import Board, Shape, polygon_area


@dataclass
class Grid:
    x0: float               # esquina superior izquierda [mm]
    y0: float
    cell_mm: float
    nx: int
    ny: int
    board: np.ndarray       # (ny, nx)
    copper: np.ndarray      # (nl, ny, nx)
    via_G: np.ndarray       # (nl-1, ny, nx) [W/K]

    @property
    def extent(self):       # para imshow (Y hacia abajo)
        return (self.x0, self.x0 + self.nx * self.cell_mm,
                self.y0 + self.ny * self.cell_mm, self.y0)

    @property
    def cell_area_m2(self) -> float:
        return (self.cell_mm * 1e-3) ** 2

    def cell_of(self, x, y):
        return int((x - self.x0) / self.cell_mm), int((y - self.y0) / self.cell_mm)


class _Canvas:
    def __init__(self, grid_x0, grid_y0, cell, nx, ny, ss):
        self.s = ss / cell
        self.x0, self.y0, self.nx, self.ny, self.ss = grid_x0, grid_y0, nx, ny, ss
        self.img = Image.new('L', (nx * ss, ny * ss), 0)
        self.d = ImageDraw.Draw(self.img)

    def _px(self, pts):
        return [((x - self.x0) * self.s, (y - self.y0) * self.s) for x, y in pts]

    def poly(self, pts, fill=255):
        if len(pts) >= 3:
            self.d.polygon(self._px(pts), fill=fill)

    def circle(self, c, diam, fill=255):
        (x, y), = self._px([c])
        r = diam * self.s / 2
        self.d.ellipse((x - r, y - r, x + r, y + r), fill=fill)

    def line(self, pts, width, fill=255):
        p = self._px(pts)
        self.d.line(p, fill=fill, width=max(1, int(round(width * self.s))))
        for c in pts:                       # extremos redondeados
            self.circle(c, width, fill)

    def shape(self, s: Shape):
        if s.kind == 'poly':
            self.poly(s.pts)
        elif s.kind == 'line':
            self.line(s.pts, s.width)
        elif s.kind == 'circle':
            self.circle(s.pts[0], s.width)

    def coverage(self) -> np.ndarray:
        a = np.asarray(self.img, dtype=np.float64) / 255.0
        return a.reshape(self.ny, self.ss, self.nx, self.ss).mean(axis=(1, 3))


def rasterize_polys(grid: Grid, polys: List[np.ndarray], ss: int = 4) -> np.ndarray:
    """Cobertura (0–1) de una lista de polígonos sobre la grilla."""
    c = _Canvas(grid.x0, grid.y0, grid.cell_mm, grid.nx, grid.ny, ss)
    for p in polys:
        c.poly(p)
    return c.coverage()


def rasterize(board: Board, cell_mm: float = 0.2, ss: int = 4,
              plating_um: float = 25.0, k_cu: float = 385.0) -> Grid:
    x0, y0, x1, y1 = board.bbox
    nx = max(2, int(math.ceil((x1 - x0) / cell_mm - 1e-9)))
    ny = max(2, int(math.ceil((y1 - y0) / cell_mm - 1e-9)))
    nl = len(board.layers)

    # Placa: lazos de mayor a menor; los contenidos en otro son recortes
    cv = _Canvas(x0, y0, cell_mm, nx, ny, ss)
    loops = sorted(board.outline, key=polygon_area, reverse=True)
    from matplotlib.path import Path
    for k, lp in enumerate(loops):
        depth = sum(Path(big).contains_point(lp.mean(axis=0)) for big in loops[:k])
        cv.poly(lp, 255 if depth % 2 == 0 else 0)
    board_frac = cv.coverage()

    copper = np.zeros((nl, ny, nx))
    for l in range(nl):
        cv = _Canvas(x0, y0, cell_mm, nx, ny, ss)
        for s in board.shapes[l]:
            cv.shape(s)
        copper[l] = cv.coverage() * board_frac

    # Barriles de vías: G = k_cu · A_corona / distancia entre capas
    via_G = np.zeros((max(nl - 1, 0), ny, nx))
    tp = plating_um * 1e-3                                  # mm
    for v in board.vias:
        i, j = int((v.x - x0) / cell_mm), int((v.y - y0) / cell_mm)
        if not (0 <= i < nx and 0 <= j < ny) or v.drill_mm <= tp:
            continue
        area_mm2 = math.pi * tp * (v.drill_mm - tp)
        for g in range(v.layer_from, v.layer_to):
            via_G[g, j, i] += k_cu * area_mm2 * 1e-6 / (board.dielectrics_mm[g] * 1e-3)

    return Grid(x0, y0, cell_mm, nx, ny, board_frac, copper, via_G)
