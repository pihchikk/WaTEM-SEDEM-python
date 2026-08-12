"""Multiple-flow-direction (MFD) routing.

The single-direction D8 grid in compute_dtm.py sends a cell's entire outflow to
one neighbour, so sediment collects into channels one cell wide. The original
WaTEM/SEDEM software spreads outflow across every downslope neighbour, which is
why its erosion maps are diffuse where ours are dendritic.

Measured against the two WaTEM/SEDEM software reference catchments (lom, spok),
switching D8 -> MFD moves top-decile erosion IoU from 0.478/0.556 to 0.744/0.730
and Spearman rho from 0.818/0.874 to 0.895/0.932, at p=3 with the Wischmeier LS
method and ktc=0.015. Pearson r does *not* improve -- the reference has heavy
tails and r is dominated by a few dozen extreme cells, not by the spatial
pattern. See docs/ for the full sweep.

Routing order here is descending elevation rather than a topological sort of the
flow graph. Outflow only ever goes to strictly lower neighbours, so that order is
always valid and no pit filling is needed; cells in a depression simply have no
outflow and act as sinks, same as D8's -1.
"""

from __future__ import annotations

import numpy as np

# Neighbour offsets, indexed by D8 code: 0 = N, then clockwise.
# Same convention as lateraldistribution.compute_erosion().
DR = np.array([-1, -1, 0, 1, 1, 1, 0, -1])
DC = np.array([0, 1, 1, 1, 0, -1, -1, -1])
_DIAG = np.sqrt(2.0)
_DIST = np.array([1.0, _DIAG, 1.0, _DIAG, 1.0, _DIAG, 1.0, _DIAG])


def mfd_weights(elevation: np.ndarray, cell_size: float, exponent: float = 3.0):
    """Fraction of each cell's outflow going to each of its 8 neighbours.

    Returns (w, has_outflow) where w has shape (8, nrow, ncol) and sums to 1
    over axis 0 wherever has_outflow is True, and to 0 elsewhere.

    Weights are proportional to the downslope gradient raised to `exponent`.
    exponent -> infinity reproduces D8; 1.0 is the Quinn et al. (1991) form.
    """
    H, W = elevation.shape
    dist = _DIST * cell_size
    drop = np.zeros((8, H, W))

    for k in range(8):
        nb = np.full((H, W), np.nan)
        i0, i1 = max(0, -DR[k]), H - max(0, DR[k])
        j0, j1 = max(0, -DC[k]), W - max(0, DC[k])
        nb[i0:i1, j0:j1] = elevation[i0 + DR[k]:i1 + DR[k], j0 + DC[k]:j1 + DC[k]]
        grad = (elevation - nb) / dist[k]
        drop[k] = np.where(np.isfinite(grad) & (grad > 0), grad, 0.0)

    raw = drop ** exponent
    total = raw.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        w = np.where(total > 0, raw / total, 0.0)
    return w, total > 0


def elevation_order(elevation: np.ndarray) -> np.ndarray:
    """Cell indices sorted by descending elevation -- a valid processing order
    for strictly-downhill routing."""
    idx = np.argwhere(np.isfinite(elevation))
    return idx[np.argsort(-elevation[idx[:, 0], idx[:, 1]])]


def mfd_accumulation(elevation: np.ndarray, cell_size: float,
                     exponent: float = 3.0) -> np.ndarray:
    """Upslope contributing area (m^2), routed multi-directionally.

    Drop-in replacement for compute_flow_accumulation() when feeding
    compute_ls(), so the LS factor sees the same dispersion the sediment does.
    """
    w, has_out = mfd_weights(elevation, cell_size, exponent)
    H, W = elevation.shape
    acc = np.where(np.isfinite(elevation), cell_size * cell_size, np.nan)

    for (i, j) in elevation_order(elevation):
        if not has_out[i, j]:
            continue
        a = acc[i, j]
        for k in range(8):
            f = w[k, i, j]
            if f <= 0:
                continue
            ti, tj = i + DR[k], j + DC[k]
            if 0 <= ti < H and 0 <= tj < W and np.isfinite(acc[ti, tj]):
                acc[ti, tj] += a * f
    return acc


def route_mfd(ero_pot: np.ndarray, cap_m3: np.ndarray, elevation: np.ndarray,
              cell_size: float, exponent: float = 3.0):
    """Route sediment with the same per-cell capacity rule compute_cell() uses,
    but splitting outflow across all downslope neighbours.

    ero_pot : potential erosion per cell (m^3)
    cap_m3  : transport capacity per cell (m^3)

    Returns (SEDI_IN, SEDI_OUT, WATEREROS) with WATEREROS in metres, negative
    for erosion and positive for deposition -- matching compute_erosion().
    """
    w, has_out = mfd_weights(elevation, cell_size, exponent)
    H, W = elevation.shape
    area = cell_size * cell_size
    SEDI_IN = np.zeros((H, W))
    SEDI_OUT = np.zeros((H, W))
    WATEREROS = np.zeros((H, W))

    for (i, j) in elevation_order(elevation):
        total_in = SEDI_IN[i, j] + ero_pot[i, j]
        if total_in > cap_m3[i, j]:
            out = cap_m3[i, j]
            WATEREROS[i, j] = (SEDI_IN[i, j] - out) / area
        else:
            out = total_in
            WATEREROS[i, j] = -ero_pot[i, j] / area
        SEDI_OUT[i, j] = out

        if not has_out[i, j]:
            continue
        for k in range(8):
            f = w[k, i, j]
            if f <= 0:
                continue
            ti, tj = i + DR[k], j + DC[k]
            if 0 <= ti < H and 0 <= tj < W:
                SEDI_IN[ti, tj] += out * f

    return SEDI_IN, SEDI_OUT, WATEREROS
