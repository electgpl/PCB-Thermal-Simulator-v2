"""
renderer.py
===========
Salidas gráficas:
  - thermal_map.png    veredicto, temperatura de las caras top y bottom, y tabla
                       por componente con límite, margen y estado
  - copper_layers.png  el cobre y las vías tal como los "ve" el solver (control)
"""

from typing import List, Optional, Tuple

import matplotlib
import numpy as np

from board_model import Board
from power import Source
from rasterizer import Grid
from thermal_solver import Solution, ThermalConfig

STATUS_COLOR = {"PASA": "#2e9e4f", "JUSTO": "#e0a100", "FALLA": "#d62728",
                "sin límite": "#7f7f7f"}
MAX_ROWS = 14


def render_thermal(board: Board, grid: Grid, sol: Solution, sources: List[Source],
                   results: List[dict], cfg: ThermalConfig, path: str,
                   cmap: str = "inferno", dpi: int = 170, show: bool = False,
                   verdict: Optional[Tuple[str, str]] = None):
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe

    nl = len(board.layers)
    faces = [0] if nl == 1 else [0, nl - 1]
    w, h = grid.nx * grid.cell_mm, grid.ny * grid.cell_mm
    side_by_side = h > w                         # placa vertical: mapas lado a lado
    ncol = len(faces) if side_by_side else 1
    nrow = 1 if side_by_side else len(faces)

    FW = 13.0                                    # ancho de figura [in]
    map_h = (FW - 1.8) / ncol * h / w + 0.9      # alto de cada mapa [in]
    rows_tab = min(len(results), MAX_ROWS)
    tab_h = 0.36 * (rows_tab + 1) + 0.15
    head_h = 1.95
    fig = plt.figure(figsize=(FW, head_h + map_h * nrow + tab_h))
    gs = fig.add_gridspec(nrow + 1, ncol + 1,
                          width_ratios=[1] * ncol + [0.025],
                          height_ratios=[map_h] * nrow + [tab_h],
                          left=0.065, right=0.93, bottom=0.02,
                          top=1 - head_h / fig.get_figheight(), hspace=0.34, wspace=0.08)

    # ── Encabezado: veredicto ────────────────────────────────────────────
    state, phrase = verdict or ("sin límite", "")
    fig.text(0.5, 1 - 0.42 / fig.get_figheight(), phrase, ha="center", va="center",
             fontsize=14, fontweight="bold", color="white", wrap=True,
             bbox=dict(boxstyle="round,pad=0.5", fc=STATUS_COLOR[state], ec="none"))
    conv = (f"h fijo {cfg.h_fixed:g} W/m²K" if cfg.h_fixed is not None else
            f"convección natural ({cfg.orientation}), h ≈ {sol.h_conv_top:.0f} arriba / "
            f"{sol.h_conv_bot:.0f} abajo")
    fig.text(0.5, 1 - 1.0 / fig.get_figheight(),
             f"{board.name}  ·  {sol.P_in_W:.2f} W  ·  ambiente {cfg.T_amb_C:g} °C  ·  {conv}"
             f" + radiación {sol.h_rad_mean:.0f} W/m²K",
             ha="center", va="center", fontsize=11)

    # ── Mapas ────────────────────────────────────────────────────────────
    vmin, vmax = float(np.nanmin(sol.T[faces])), sol.T_max
    by_ref = {r['ref']: r for r in results}
    halo = [pe.withStroke(linewidth=3, foreground="black")]
    xs = np.linspace(grid.extent[0], grid.extent[1], grid.nx)
    ys = np.linspace(grid.extent[3], grid.extent[2], grid.ny)
    im = None
    for k, l in enumerate(faces):
        ax = fig.add_subplot(gs[0, k] if side_by_side else gs[k, 0])
        im = ax.imshow(sol.T[l], extent=grid.extent, cmap=cmap, vmin=vmin, vmax=vmax,
                       interpolation="bilinear")
        ax.contour(xs, ys, grid.copper[l], levels=[0.5], colors="white",
                   linewidths=0.4, alpha=0.3)
        for lp in board.outline:
            p = np.vstack((lp, lp[:1]))
            ax.plot(p[:, 0], p[:, 1], color="white", lw=1.2)
        for s in sources:
            if s.comp.layer != l:
                continue
            r = by_ref[s.ref]
            col = STATUS_COLOR[r['status']] if r['status'] != "sin límite" else "#cfd8dc"
            x0, y0, x1, y1 = s.comp.bbox
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                       ec=col, lw=2.0))
            if s.power_W >= 0.05 * sol.P_in_W:
                ax.text((x0 + x1) / 2, y0 - 0.25,
                        f"{s.ref}  {r['T_eval']:.0f} °C", color="white", fontsize=12,
                        fontweight="bold", ha="center", va="bottom", path_effects=halo,
                        clip_on=False)
        Tl = sol.T[l]
        j, i = np.unravel_index(np.nanargmax(Tl), Tl.shape)
        ax.plot(grid.x0 + (i + .5) * grid.cell_mm, grid.y0 + (j + .5) * grid.cell_mm,
                marker="+", color="white", ms=12, mew=2)
        side = "" if l == 0 else " (vista desde arriba)"
        ax.set_title(f"{board.layers[l].name}{side} — máx {np.nanmax(Tl):.1f} °C, "
                     f"mín {np.nanmin(Tl):.1f} °C", fontsize=12, loc="left", pad=22)
        ax.set_xlabel("x [mm]", fontsize=10); ax.set_ylabel("y [mm]", fontsize=10)
        ax.tick_params(labelsize=10)
        ax.set_aspect("equal")
    cax = fig.add_subplot(gs[0:nrow, ncol])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Temperatura de la placa [°C]", fontsize=11)
    cax.tick_params(labelsize=10)

    # ── Tabla ────────────────────────────────────────────────────────────
    tax = fig.add_subplot(gs[nrow, :]); tax.axis("off")
    f = lambda v, n=0: f"{v:.{n}f}" if v is not None else "—"
    rows = [[r['ref'], r['value'][:16], f"{r['P']:g}", f(r['T_pads']), f(r['Tj']),
             f(r['limit']), f(r['margin']), f(r['T_amb_max']), r['status']]
            for r in results[:MAX_ROWS]]
    cols = ["Ref", "Valor", "P [W]", "T pads [°C]", "Tj [°C]", "Límite [°C]",
            "Margen [°C]", "Amb. máx ≈ [°C]", "Estado"]
    tb = tax.table(cellText=rows, colLabels=cols, loc="upper center", cellLoc="center",
                   colWidths=[0.07, 0.17, 0.08, 0.11, 0.09, 0.11, 0.11, 0.13, 0.11])
    tb.auto_set_font_size(False); tb.set_fontsize(11); tb.scale(1, 1.55)
    for (ri, ci), cell in tb.get_celld().items():
        cell.set_edgecolor("#bbbbbb")
        if ri == 0:
            cell.set_facecolor("#263238"); cell.set_text_props(color="white", fontweight="bold")
        elif ci == len(cols) - 1:
            cell.set_facecolor(STATUS_COLOR[rows[ri - 1][-1]])
            cell.set_text_props(color="white", fontweight="bold")
    if len(results) > MAX_ROWS:
        tax.text(0.5, 0.0, f"… y {len(results) - MAX_ROWS} componentes más en report.csv",
                 ha="center", fontsize=9, transform=tax.transAxes)

    fig.savefig(path, dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close(fig)


def render_copper(board: Board, grid: Grid, path: str, dpi: int = 160):
    import matplotlib.pyplot as plt

    nl = len(board.layers)
    w, h = grid.nx * grid.cell_mm, grid.ny * grid.cell_mm
    cols = 1 if nl == 1 else 2
    rows = (nl + cols - 1) // cols
    scale = 6.0 / w
    fig, axes = plt.subplots(rows, cols, squeeze=False,
                             figsize=(6.4 * cols, h * scale * rows + 0.8 * rows))
    for ax in axes.ravel()[nl:]:
        ax.axis("off")
    for l, ax in zip(range(nl), axes.ravel()):
        img = np.where(grid.board > 0.3, grid.copper[l], np.nan)
        ax.imshow(img, extent=grid.extent, cmap="copper", vmin=0, vmax=1,
                  interpolation="nearest")
        vs = [(v.x, v.y) for v in board.vias if v.layer_from <= l <= v.layer_to]
        if vs:
            vx, vy = zip(*vs)
            ax.scatter(vx, vy, s=2, c="deepskyblue", lw=0)
        cov = grid.copper[l].sum() / grid.board.sum() * 100
        ax.set_title(f"{board.layers[l].name} — {board.layers[l].thickness_mm*1000:.0f} µm, "
                     f"{cov:.0f} % de cobre, {len(vs)} vías", fontsize=10)
        ax.set_aspect("equal"); ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
