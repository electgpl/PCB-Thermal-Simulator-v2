#!/usr/bin/env python3
"""
PCB Thermal Simulator v2
========================
Simulación térmica estacionaria de un PCB a partir del layout real.

Dos formas de usarlo:

  A) Desde Thonny / IDLE / VS Code: editá el bloque CONFIGURACIÓN de abajo y
     apretá Run (F5).

  B) Por línea de comandos (los argumentos pisan a la configuración):
        python main.py placa.kicad_pcb --csv potencias.csv
        python main.py placa.kicad_pcb --template potencias.csv
"""

# ══════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN  (se usa cuando se ejecuta sin argumentos, p. ej. desde Thonny)
#  Las rutas relativas se toman desde la carpeta donde está este archivo.
# ══════════════════════════════════════════════════════════════════════════
PCB_FILE     = "examples/EMI_SS_Supply_Electgpl.kicad_pcb"
CSV_FILE     = "examples/EMI_SS_Supply_Electgpl.csv"   # None = solo campos de KiCad
OUT_DIR      = "thermal_out"        # carpeta de resultados
MAKE_TEMPLATE = False               # True = solo genera CSV_FILE con todos los
                                    #        componentes para completar, y termina

T_AMB        = 25.0                 # temperatura ambiente [°C]
ORIENTATION  = "horizontal"         # "horizontal" | "vertical"
H_FIXED      = None                 # None = convección natural calculada
                                    # número = h fijo [W/m²K] (aire forzado: 20–60)
MARGIN       = 10.0                 # margen mínimo contra T_max para dar PASA [°C]
DEFAULT_LIMIT = None                # T_max [°C] para componentes sin límite propio
CELL_MM      = 0.2                  # tamaño de celda [mm]; más chico = más lento
SHOW_WINDOW  = True                 # abrir la imagen al terminar
# ══════════════════════════════════════════════════════════════════════════

import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

try:
    import numpy as np
    import scipy            # noqa: F401
    import matplotlib       # noqa: F401
    import PIL              # noqa: F401
except ImportError as e:
    raise SystemExit(
        f"Falta el paquete '{e.name}'.\n"
        "  En Thonny: Herramientas → Gestionar paquetes… e instalá "
        "numpy, scipy, matplotlib y Pillow.\n"
        "  Por consola: pip install -r requirements.txt")

from kicad_loader import load_kicad_pcb
from power import build_sources, power_map, write_template
from rasterizer import rasterize
from thermal_solver import ThermalConfig, solve


PASS, WARN, FAIL, NOLIM = "PASA", "JUSTO", "FALLA", "sin límite"


def analyze(board, grid, sol, sources, cfg, margin_C=10.0, default_limit=None):
    """Temperatura de pads y de juntura por componente, y veredicto contra su límite.

    Se compara la juntura (Tj) si hay θ para calcularla; si no, la temperatura
    de los pads. Estados:
        PASA   margen ≥ margin_C
        JUSTO  0 ≤ margen < margin_C
        FALLA  supera el límite
    """
    out = []
    for s in sources:
        Tl = np.nan_to_num(sol.T[s.comp.layer], nan=cfg.T_amb_C)
        T_pads = float((Tl * s.weights).sum())
        theta, note = s.theta_jb, "θJB"
        if theta is None and s.theta_jc is not None:
            theta, note = s.theta_jc, "θJC como θJB"
        Tj = T_pads + s.power_W * theta if theta is not None else None
        T_eval = Tj if Tj is not None else T_pads
        limit = s.t_max if s.t_max is not None else default_limit
        margin = limit - T_eval if limit is not None else None
        status = (NOLIM if margin is None else FAIL if margin < 0
                  else WARN if margin < margin_C else PASS)
        # ¿Cuánto del calentamiento de sus pads es propio y cuánto de los vecinos?
        contrib = {}
        if sol.rise_parts is not None:
            for name, part in zip(sol.part_names, sol.rise_parts):
                contrib[name] = float((part[s.comp.layer] * s.weights).sum())
        own = contrib.get(s.ref)
        out.append(dict(ref=s.ref, value=s.comp.value, P=s.power_W, inject=s.pad_rule,
                        contrib=contrib, rise_own=own,
                        rise_others=(T_pads - cfg.T_amb_C - own) if own is not None else None,
                        T_pads=T_pads, theta=theta, theta_src=note if theta is not None else "",
                        Tj=Tj, T_eval=T_eval, eval_what="Tj" if Tj is not None else "T pads",
                        limit=limit, margin=margin, status=status,
                        # El problema es casi lineal: subir el ambiente X °C sube todo ~X °C
                        T_amb_max=(cfg.T_amb_C + margin) if margin is not None else None,
                        Tj_jedec=(cfg.T_amb_C + s.power_W * s.theta_ja)
                        if s.theta_ja is not None else None))
    rank = {FAIL: 0, WARN: 1, PASS: 2, NOLIM: 3}
    return sorted(out, key=lambda r: (rank[r['status']],
                                      r['margin'] if r['margin'] is not None else -r['T_eval']))


def verdict(results, margin_C=10.0):
    """Veredicto global: (estado, frase)."""
    lim = [r for r in results if r['margin'] is not None]
    if not lim:
        return NOLIM, ("Sin veredicto: ningún componente tiene T_max. Cargá la temperatura "
                       "máxima de cada componente (columna T_max) para obtener PASA/FALLA.")
    worst = min(lim, key=lambda r: r['margin'])
    fails = [r['ref'] for r in lim if r['status'] == FAIL]
    w = (f"{worst['ref']}: {worst['eval_what']} {worst['T_eval']:.0f} °C, "
         f"límite {worst['limit']:.0f} °C")
    if fails:
        return FAIL, (f"FALLA — {', '.join(fails)} supera{'n' if len(fails) > 1 else ''} su límite. "
                      f"Peor caso {w} (excedido {-worst['margin']:.0f} °C)")
    if worst['status'] == WARN:
        return WARN, (f"JUSTO — pasa, pero con menos de {margin_C:g} °C de margen. "
                      f"Peor caso {w} (margen {worst['margin']:.0f} °C)")
    return PASS, (f"PASA — peor caso {w} (margen {worst['margin']:.0f} °C; "
                  f"aguanta hasta ~{worst['T_amb_max']:.0f} °C de ambiente)")


def run(pcb, csv_path=None, cell=0.2, cfg=None, plating_um=25.0, log=print,
        margin_C=10.0, default_limit=None):
    cfg = cfg or ThermalConfig()
    board = load_kicad_pcb(pcb)
    log(board.summary())
    for w in board.warnings:
        log(f"  [aviso] {w}")
    grid = rasterize(board, cell, plating_um=plating_um, k_cu=cfg.k_cu)
    log(f"Grilla: {grid.nx} × {grid.ny} celdas de {cell:g} mm × {len(board.layers)} capas")
    sources = build_sources(board, grid, csv_path, log)
    if not sources:
        raise SystemExit(
            "No hay ningún componente con potencia. Cargá el campo Power_W en los "
            "componentes de KiCad o pasá un CSV (--template genera uno para completar).")
    nl = len(board.layers)
    q = power_map(sources, grid, nl)
    # Para el desglose "propio vs. vecinos": una parte por fuente importante
    # (≥ 2 % de la potencia) y el resto agrupado.
    total = sum(s.power_W for s in sources)
    big = [s for s in sources if s.power_W >= 0.02 * total][:40]
    rest = [s for s in sources if s not in big]
    names = [s.ref for s in big] + (["otros"] if rest else [])
    parts = [power_map([s], grid, nl) for s in big] + ([power_map(rest, grid, nl)] if rest else [])
    sol = solve(grid, board, q, cfg, log, q_parts=parts)
    sol.part_names = names
    return board, grid, sol, sources, analyze(board, grid, sol, sources, cfg,
                                               margin_C, default_limit)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Simulación térmica de PCB desde KiCad")
    ap.add_argument("pcb", help="archivo .kicad_pcb (con las zonas rellenas)")
    ap.add_argument("--csv", help="CSV de potencias y resistencias térmicas")
    ap.add_argument("--template", metavar="CSV",
                    help="escribe un CSV con todos los componentes para completar y termina")
    ap.add_argument("--out", default="thermal_out", help="carpeta de salida")
    ap.add_argument("--cell", type=float, default=0.2, help="tamaño de celda [mm] (def. 0.2)")
    ap.add_argument("--T_amb", type=float, default=25.0, help="temperatura ambiente [°C]")
    ap.add_argument("--orientation", choices=("horizontal", "vertical"), default="horizontal",
                    help="posición de la placa para la convección natural")
    ap.add_argument("--h", type=float, default=None,
                    help="h convectivo fijo [W/m²K] en ambas caras (aire forzado: 20–60)")
    ap.add_argument("--emissivity", type=float, default=0.9, help="emisividad (0 = sin radiación)")
    ap.add_argument("--plating", type=float, default=25.0, help="metalizado de vías [µm]")
    ap.add_argument("--margin", type=float, default=10.0,
                    help="margen mínimo [°C] contra T_max para dar PASA (def. 10)")
    ap.add_argument("--limit", type=float, default=None,
                    help="T_max [°C] para los componentes que no tengan uno propio")
    ap.add_argument("--k_fr4_xy", type=float, default=0.8)
    ap.add_argument("--k_fr4_z", type=float, default=0.3)
    ap.add_argument("--cmap", default="inferno")
    ap.add_argument("--show", action="store_true", help="abrir la ventana de matplotlib")
    ide_mode = argv is None and len(sys.argv) == 1
    if ide_mode:                                    # Thonny / F5: usar CONFIGURACIÓN
        p = lambda f: f if f is None or os.path.isabs(f) else os.path.join(HERE, f)
        argv = [p(PCB_FILE), "--out", p(OUT_DIR), "--T_amb", str(T_AMB),
                "--orientation", ORIENTATION, "--margin", str(MARGIN), "--cell", str(CELL_MM)]
        if MAKE_TEMPLATE:
            argv += ["--template", p(CSV_FILE or "potencias.csv")]
        elif CSV_FILE:
            argv += ["--csv", p(CSV_FILE)]
        if H_FIXED is not None:
            argv += ["--h", str(H_FIXED)]
        if DEFAULT_LIMIT is not None:
            argv += ["--limit", str(DEFAULT_LIMIT)]
        if SHOW_WINDOW:
            argv += ["--show"]
    a = ap.parse_args(argv)
    for f in (a.pcb, a.csv):
        if f and not os.path.isfile(f):
            raise SystemExit(f"No encuentro el archivo: {f}")
    if not a.show:
        matplotlib.use("Agg")

    if a.template:
        write_template(load_kicad_pcb(a.pcb), a.template)
        print(f"Plantilla escrita en {a.template}. Completá Power_W y volvé a correr con --csv.")
        return 0

    cfg = ThermalConfig(T_amb_C=a.T_amb, k_fr4_xy=a.k_fr4_xy, k_fr4_z=a.k_fr4_z,
                        emissivity=a.emissivity, orientation=a.orientation, h_fixed=a.h)
    board, grid, sol, sources, res = run(a.pcb, a.csv, a.cell, cfg, a.plating,
                                         margin_C=a.margin, default_limit=a.limit)
    state, phrase = verdict(res, a.margin)

    print(f"\nResuelto en {sol.seconds:.1f} s ({sol.iterations} iteraciones)")
    print(f"  Potencia: {sol.P_in_W:.3f} W entra / {sol.P_out_W:.3f} W sale al ambiente")
    print(f"  h convección top/bottom: {sol.h_conv_top:.1f} / {sol.h_conv_bot:.1f} W/m²K"
          f"   h radiación medio: {sol.h_rad_mean:.1f} W/m²K")
    print(f"  T placa: máx {sol.T_max:.1f} °C, media {sol.T_mean_surface:.1f} °C "
          f"(ambiente {cfg.T_amb_C:g} °C)")
    print(f"  θ placa→ambiente (punto más caliente): "
          f"{(sol.T_max - cfg.T_amb_C) / sol.P_in_W:.1f} °C/W\n")

    f = lambda v, n=1: f"{v:.{n}f}" if v is not None else "—"
    print(f"{'Ref':<6}{'P[W]':>7}{'T_pads':>8}{'Tj':>8}{'Límite':>8}{'Margen':>8}"
          f"{'Amb.máx':>9}  {'Estado':<11}Calor entra por")
    for r in res:
        print(f"{r['ref']:<6}{r['P']:>7.3f}{r['T_pads']:>8.1f}{f(r['Tj']):>8}"
              f"{f(r['limit'], 0):>8}{f(r['margin']):>8}{f(r['T_amb_max'], 0):>9}  "
              f"{r['status']:<11}{r['inject']}"
              + (f"  [{r['theta_src']}]" if r['theta_src'] not in ('', 'θJB') else ""))
    print("\nDe dónde viene el calentamiento de cada componente (°C sobre el ambiente):")
    for r in res:
        if r['rise_own'] is None or r['P'] < 0.02 * sol.P_in_W:
            continue
        others = sorted(((k, v) for k, v in r['contrib'].items() if k != r['ref']),
                        key=lambda kv: -kv[1])
        txt = " + ".join(f"{v:.1f} de {k}" for k, v in others if v >= 0.05)
        print(f"  {r['ref']:<5} pads a +{r['T_pads'] - cfg.T_amb_C:.1f} °C = "
              f"{r['rise_own']:.1f} propio" + (f" + {txt}" if txt else ""))
    print(f"\n>>> {phrase}")
    nolim = [r['ref'] for r in res if r['status'] == NOLIM]
    if nolim and state != NOLIM:
        print(f"    Sin T_max (no evaluados): {', '.join(nolim)}")
    print("    El límite se compara contra Tj cuando hay θJB/θJC; si no, contra la "
          "temperatura de los pads.")

    os.makedirs(a.out, exist_ok=True)
    from renderer import render_copper, render_thermal
    render_copper(board, grid, os.path.join(a.out, "copper_layers.png"))
    render_thermal(board, grid, sol, sources, res, cfg,
                   os.path.join(a.out, "thermal_map.png"), a.cmap, show=a.show,
                   verdict=(state, phrase))
    np.savez_compressed(os.path.join(a.out, "temperature.npz"), T=sol.T,
                        layers=[l.name for l in board.layers],
                        x0=grid.x0, y0=grid.y0, cell_mm=grid.cell_mm)
    with open(os.path.join(a.out, "report.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["Component", "Value", "Power_W", "T_pads_C", "Theta_used", "Theta_source",
                    "Tj_C", "T_max_C", "Margin_C", "Status", "T_amb_max_C",
                    "Rise_own_C", "Rise_from_neighbours_C",
                    "Tj_JEDEC_thetaJA_C", "Heat_enters_through"])
        rd = lambda v: round(v, 2) if v is not None else ""
        for r in res:
            w.writerow([r['ref'], r['value'], r['P'], rd(r['T_pads']), rd(r['theta']),
                        r['theta_src'], rd(r['Tj']), rd(r['limit']), rd(r['margin']),
                        r['status'], rd(r['T_amb_max']), rd(r['rise_own']), rd(r['rise_others']),
                        rd(r['Tj_jedec']), r['inject']])
        w.writerow([])
        w.writerow(["VERDICT", phrase])
    print(f"\nArchivos en {a.out}/: thermal_map.png, copper_layers.png, report.csv, temperature.npz")
    # Por consola: código de salida 1 si falla (útil en CI). En el IDE, siempre 0.
    return 1 if (state == FAIL and not ide_mode) else 0


if __name__ == "__main__":
    code = main()
    if code:
        sys.exit(code)
