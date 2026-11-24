"""
Validación del solver contra soluciones analíticas y del loader contra el
PCB de ejemplo.   Correr con:  python -m pytest tests  (o python tests/test_validation.py)
"""
import os
import sys

import numpy as np
from scipy.special import k0

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from board_model import Board, CopperLayer, Shape                       # noqa: E402
from kicad_loader import arc_points, load_kicad_pcb, rotate             # noqa: E402
from rasterizer import rasterize                                        # noqa: E402
from thermal_solver import ThermalConfig, solve                         # noqa: E402

EXAMPLE = os.path.join(os.path.dirname(__file__), "..", "examples",
                       "EMI_SS_Supply_Electgpl.kicad_pcb")
quiet = lambda *a, **k: None


def plate(size_mm, layers=1, diel=()):
    sq = np.array([[0, 0], [size_mm, 0], [size_mm, size_mm], [0, size_mm]], float)
    return Board("plate", [sq], [CopperLayer(f"L{i}", 0.035) for i in range(layers)],
                 list(diel), [[Shape('poly', sq)] for _ in range(layers)])


def test_uniform_heating():
    """Potencia uniforme en una placa: T = T_amb + P / (2·h·A), exacto."""
    b = plate(50); g = rasterize(b, 1.0)
    q = np.full((1, g.ny, g.nx), 2.0 / (g.nx * g.ny))
    cfg = ThermalConfig(h_fixed=10.0, emissivity=0.0)
    s = solve(g, b, q, cfg, quiet)
    expected = 25 + 2.0 / (2 * 10.0 * 0.05**2)
    assert abs(s.T_max - expected) < 1e-6 and abs(np.nanmin(s.T) - expected) < 1e-6
    assert abs(s.P_out_W - 2.0) < 1e-9


def test_point_source_spreading():
    """Fuente puntual en una placa grande con convección:
       T(r) − T_amb = P / (2π·k·t) · K0(r / L),   L = sqrt(k·t / 2h)."""
    b = plate(301); g = rasterize(b, 1.0)
    cfg = ThermalConfig(h_fixed=10.0, emissivity=0.0, k_fr4_xy=0.0)
    q = np.zeros((1, g.ny, g.nx)); c = g.nx // 2; q[0, c, c] = 1.0
    s = solve(g, b, q, cfg, quiet)
    kt = 385.0 * 35e-6
    L = np.sqrt(kt / 20.0)
    for r_cells in (10, 20, 40):
        exact = 1.0 / (2 * np.pi * kt) * k0(r_cells * 1e-3 / L)
        sim = s.T[0, c, c + r_cells] - 25
        assert abs(sim - exact) / exact < 0.02, (r_cells, sim, exact)


def test_two_layers_through_fr4():
    """Calor uniforme arriba, convección solo posible por ambas caras: el salto
       entre capas debe ser q_bot·d / k_z."""
    b = plate(20, 2, [1.5]); g = rasterize(b, 1.0)
    q = np.zeros((2, g.ny, g.nx)); q[0] = 1.0 / (g.nx * g.ny)
    cfg = ThermalConfig(h_fixed=10.0, emissivity=0.0)
    s = solve(g, b, q, cfg, quiet)
    A = 0.02**2
    flux_bot = 10.0 * A * (np.nanmean(s.T[1]) - 25)
    dT = np.nanmean(s.T[0]) - np.nanmean(s.T[1])
    assert abs(dT - flux_bot * 1.5e-3 / (0.3 * A)) < 1e-6
    assert abs(s.P_out_W - 1.0) < 1e-9


def test_gap_isolates_copper():
    """Dos mitades de cobre separadas por un corte: la mitad sin fuente queda
       más fría (y la fuente más caliente) que con el cobre continuo."""
    def run(gap):
        b = plate(40)
        if gap:
            b.shapes[0] = [Shape('poly', np.array([[0, 0], [19, 0], [19, 40], [0, 40]], float)),
                           Shape('poly', np.array([[21, 0], [40, 0], [40, 40], [21, 40]], float))]
        g = rasterize(b, 0.5)
        q = np.zeros((1, g.ny, g.nx)); q[0, 40, 10] = 1.0
        T = solve(g, b, q, ThermalConfig(h_fixed=10, emissivity=0), quiet).T[0]
        return T[40, 10], T[40, 70]
    (src_full, far_full), (src_cut, far_cut) = run(False), run(True)
    assert far_full - far_cut > 3.0 and src_cut - src_full > 3.0


def test_geometry_helpers():
    p = rotate(np.array([[1.0, 0.0]]), 90)          # KiCad: +90° lleva +X a −Y (arriba)
    assert np.allclose(p, [[0, -1]])
    a = arc_points((1, 0), (0, 1), (-1, 0))
    assert np.allclose(np.hypot(a[:, 0], a[:, 1]), 1) and np.allclose(a[-1], (-1, 0))


def test_kicad_example():
    b = load_kicad_pcb(EXAMPLE)
    assert [l.name for l in b.layers] == ["F.Cu", "In1.Cu", "In2.Cu", "B.Cu"]
    assert np.allclose(b.dielectrics_mm, [0.1, 1.24, 0.1])
    x0, y0, x1, y1 = b.bbox
    assert abs((x1 - x0) - 43.5) < 1e-6 and abs((y1 - y0) - 21.3) < 1e-6
    assert len(b.vias) == 216 and not b.warnings
    u1 = b.component("U1")
    assert len(u1.pads) == 9 and abs(u1.pads[-1].area_mm2 - 2.613 * 3.502) < 1e-6
    g = rasterize(b, 0.2)
    # todos los pads caen sobre cobre de su capa (valida rotaciones y offsets)
    for c in b.components:
        for p in c.pads:
            i, j = g.cell_of(*p.center)
            assert g.copper[c.layer, j, i] > 0.99, (c.ref, p.number)
    # C1 girado −90°: pad 1 (VIN) arriba, pad 2 (GND) abajo
    c1 = b.component("C1")
    assert c1.pads[0].center[1] < c1.pads[1].center[1]


def test_example_energy_balance():
    from main import run
    csvp = EXAMPLE.replace(".kicad_pcb", ".csv")
    board, grid, sol, sources, res = run(EXAMPLE, csvp, cell=0.4, log=quiet)
    assert abs(sol.P_in_W - sol.P_out_W) < 1e-6 * sol.P_in_W
    u1 = next(r for r in res if r['ref'] == 'U1')
    assert u1['inject'].startswith("pad expuesto") and u1['Tj'] > u1['T_pads'] > 25


def test_pass_fail_verdict():
    from main import FAIL, PASS, WARN, analyze, run, verdict
    from thermal_solver import ThermalConfig
    csvp = EXAMPLE.replace(".kicad_pcb", ".csv")
    cfg = ThermalConfig()
    board, grid, sol, sources, res = run(EXAMPLE, csvp, cell=0.4, cfg=cfg, log=quiet)
    u1 = next(r for r in res if r['ref'] == 'U1')
    assert u1['status'] == PASS and abs(u1['margin'] - (125 - u1['Tj'])) < 1e-9
    assert verdict(res)[0] == PASS
    # mismo resultado térmico, límites más exigentes
    u1s = next(s for s in sources if s.ref == 'U1')
    u1s.t_max = u1['Tj'] + 5
    assert verdict(analyze(board, grid, sol, sources, cfg, margin_C=10))[0] == WARN
    u1s.t_max = u1['Tj'] - 1
    res = analyze(board, grid, sol, sources, cfg)
    assert verdict(res)[0] == FAIL and res[0]['ref'] == 'U1'    # el peor va primero


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok ", name)
