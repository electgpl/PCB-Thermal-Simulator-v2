"""
board_model.py
==============
Modelo intermedio de la placa, independiente del EDA.

Cualquier "frontend" (KiCad hoy; Gerber+Excellon+pos, IPC-2581 u ODB++ mañana)
solo tiene que devolver un objeto Board. El rasterizador, el solver y el
renderer no saben de qué programa vino la placa.

Convenciones:
  - Unidades: milímetros.
  - Ejes: X a la derecha, Y hacia abajo (igual que KiCad). Todo se ve "desde arriba".
  - Capas de cobre ordenadas de top a bottom: índice 0 = top, índice n-1 = bottom.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class CopperLayer:
    name: str               # "F.Cu", "In1.Cu", ...
    thickness_mm: float     # 0.035 = 1 oz


@dataclass
class Shape:
    """Primitiva de cobre. kind = 'poly' | 'line' | 'circle'."""
    kind: str
    pts: np.ndarray         # poly: (N,2) | line: (2,2) | circle: (1,2) centro
    width: float = 0.0      # line: ancho | circle: diámetro


@dataclass
class Via:
    """Barril metalizado (vía o pad pasante) entre dos capas de cobre."""
    x: float
    y: float
    drill_mm: float
    layer_from: int         # índice de capa (incluido)
    layer_to: int           # índice de capa (incluido)


@dataclass
class Pad:
    number: str
    polys: List[np.ndarray]     # polígonos en coordenadas de placa
    layers: List[int]           # índices de capas de cobre
    net: str = ""

    @property
    def area_mm2(self) -> float:
        return sum(polygon_area(p) for p in self.polys)

    @property
    def center(self) -> np.ndarray:
        return np.vstack(self.polys).mean(axis=0)


@dataclass
class Component:
    ref: str
    value: str
    footprint: str
    layer: int                  # índice de la capa donde está montado
    x: float
    y: float
    bbox: Tuple[float, float, float, float]   # xmin, ymin, xmax, ymax (courtyard)
    pads: List[Pad] = field(default_factory=list)
    props: Dict[str, str] = field(default_factory=dict)   # campos del EDA
    dnp: bool = False


@dataclass
class Board:
    name: str
    outline: List[np.ndarray]           # lazos cerrados de Edge.Cuts (contorno + recortes)
    layers: List[CopperLayer]
    dielectrics_mm: List[float]         # espesor entre capa i e i+1 (len = n-1)
    shapes: List[List[Shape]]           # cobre por capa (pistas, zonas, pads, anillos)
    vias: List[Via] = field(default_factory=list)
    components: List[Component] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        pts = np.vstack(self.outline)
        return (pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max())

    @property
    def thickness_mm(self) -> float:
        return sum(l.thickness_mm for l in self.layers) + sum(self.dielectrics_mm)

    def component(self, ref: str) -> Optional[Component]:
        for c in self.components:
            if c.ref == ref:
                return c
        return None

    def summary(self) -> str:
        x0, y0, x1, y1 = self.bbox
        stack = " | ".join(
            f"{l.name} {l.thickness_mm*1000:.0f}µm" for l in self.layers)
        return (f"{self.name}: {x1-x0:.1f} × {y1-y0:.1f} mm, "
                f"{len(self.layers)} capas, espesor {self.thickness_mm:.2f} mm\n"
                f"  Stackup: {stack}\n"
                f"  Dieléctricos [mm]: {', '.join(f'{d:.3f}' for d in self.dielectrics_mm)}\n"
                f"  {len(self.components)} componentes, {len(self.vias)} vías/pads pasantes")


def polygon_area(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
