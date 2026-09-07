"""Sediment routing schemes.

Three are available, all expressed as a weight per neighbour so the router and
the flow-accumulation pass share one definition:

  ``desmet_govers`` -- the default. Flow leaves a cell along its aspect, and the
  vector is decomposed onto the two cardinal neighbours it lies between, with
  weights |cos(aspect)| north/south and |sin(aspect)| east/west, normalised by
  their sum. This is Desmet & Govers (1996), the routing WaTEM/SEDEM uses.

  ``holmgren`` -- weights proportional to downslope gradient ** exponent.
  exponent 1 is Quinn et al. (1991) without contour weighting, 1.1 is Freeman
  (1991), and large exponents converge on D8.

  ``d8`` -- the whole outflow to the single steepest neighbour.

Identified by running all of them against the WaTEM/SEDEM software's own output
on the lom and spok reference catchments, with every other parameter fixed at
the software's settings. Desmet & Govers won on every metric on both, and not
narrowly:

                        lom                          spok
                rho    sign   dep/GT  bad  mass    rho    sign   dep/GT  bad  mass
  Desmet&Govers 0.945  0.986  51/51   1/36  1.0%   0.979  0.992  107/112 3/87  0.9%
  Quinn 1991    0.931  0.982  52/51   2/36  2.4%   0.966  0.985   97/112 6/87  2.2%
  Holmgren p=1  0.930  0.981  51/51   3/36  2.6%   0.964  0.984   93/112 8/87  2.7%
  Holmgren p=3  0.919  0.975  43/51   8/36  8.2%   0.960  0.982   77/112 13/87 3.9%
  D-infinity    0.917  0.978  38/51   8/36 12.0%   0.949  0.981   79/112 14/87 4.6%
  D8            0.850  0.976  34/51  10/36  9.1%   0.883  0.976   66/112 26/87 8.2%

("dep/GT" is deposition cells against the reference's count, "bad" the largest
reference cells carrying the wrong sign, "mass" the share of the reference's
erosion mass sitting in wrong-sign cells.) The median ratio to the reference is
1.00 on both catchments under Desmet & Govers and drifts to 1.22 and 1.31 under
D8, so this is an identification rather than a preference.

Routing order is descending elevation. Outflow only ever goes to strictly lower
neighbours, so that order is always valid and no pit filling is needed; cells in
a depression simply have no outflow.
"""

from __future__ import annotations

import numpy as np

# Neighbour offsets indexed by D8 code: 0 = N, then clockwise.
# Same convention as lateraldistribution.compute_erosion().
DR = np.array([-1, -1, 0, 1, 1, 1, 0, -1])
DC = np.array([0, 1, 1, 1, 0, -1, -1, -1])
_DIAG = np.sqrt(2.0)
_DIST = np.array([1.0, _DIAG, 1.0, _DIAG, 1.0, _DIAG, 1.0, _DIAG])

SCHEMES = ("desmet_govers", "holmgren", "d8")


def _downslope_gradients(elevation: np.ndarray, cell_size: float) -> np.ndarray:
    """(8, nrow, ncol) of positive downslope gradients; 0 where the neighbour is
    higher, missing, or off the grid."""
    H, W = elevation.shape
    out = np.zeros((8, H, W))
    for k in range(8):
        nb = np.full((H, W), np.nan)
        i0, i1 = max(0, -DR[k]), H - max(0, DR[k])
        j0, j1 = max(0, -DC[k]), W - max(0, DC[k])
        nb[i0:i1, j0:j1] = elevation[i0 + DR[k]:i1 + DR[k], j0 + DC[k]:j1 + DC[k]]
        grad = (elevation - nb) / (_DIST[k] * cell_size)
        out[k] = np.where(np.isfinite(grad) & (grad > 0), grad, 0.0)
    return out


def _normalise(raw: np.ndarray):
    total = raw.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        w = np.where(total > 0, raw / total, 0.0)
    return w, total > 0


def weights(elevation: np.ndarray, cell_size: float, scheme: str = "desmet_govers",
            aspect: np.ndarray | None = None, exponent: float = 1.0):
    """Fraction of each cell's outflow going to each of its 8 neighbours.

    Returns (w, has_outflow); w sums to 1 over axis 0 wherever has_outflow.
    """
    if scheme not in SCHEMES:
        raise ValueError(f"unknown routing scheme {scheme!r}; expected one of {SCHEMES}")

    grad = _downslope_gradients(elevation, cell_size)

    if scheme == "holmgren":
        return _normalise(grad ** exponent)

    if scheme == "d8":
        raw = np.zeros_like(grad)
        k = np.argmax(grad, axis=0)
        H, W = elevation.shape
        ii, jj = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        raw[k, ii, jj] = grad[k, ii, jj]
        return _normalise(raw)

    # desmet_govers
    if aspect is None:
        raise ValueError("desmet_govers routing needs the aspect raster")
    H, W = elevation.shape
    sin_a, cos_a = np.sin(aspect), np.cos(aspect)
    ns = np.where(cos_a >= 0, 0, 4)   # 0 = N (row-1), 4 = S
    ew = np.where(sin_a >= 0, 2, 6)   # 2 = E (col+1), 6 = W
    ii, jj = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    raw = np.zeros((8, H, W))
    raw[ns, ii, jj] = np.abs(cos_a)
    raw[ew, ii, jj] = np.abs(sin_a)
    raw = np.where(np.isfinite(raw), raw, 0.0)
    # A component only counts if that neighbour is actually downslope; without
    # this a cell on a ridge would push sediment uphill.
    raw = np.where(grad > 0, raw, 0.0)
    return _normalise(raw)


def elevation_order(elevation: np.ndarray) -> np.ndarray:
    idx = np.argwhere(np.isfinite(elevation))
    return idx[np.argsort(-elevation[idx[:, 0], idx[:, 1]])]


def accumulate(w: np.ndarray, has_outflow: np.ndarray, elevation: np.ndarray,
               cell_size: float) -> np.ndarray:
    """Upslope contributing area (m^2) under the same weights the sediment uses.

    Feeding compute_ls() a D8 area while routing sediment multi-directionally
    would make the LS factor describe a different flow field than the one the
    sediment sees.
    """
    H, W = elevation.shape
    acc = np.where(np.isfinite(elevation), cell_size * cell_size, np.nan)
    for (i, j) in elevation_order(elevation):
        if not has_outflow[i, j]:
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


def route(w: np.ndarray, has_outflow: np.ndarray, ero_pot: np.ndarray,
          cap_m3: np.ndarray, elevation: np.ndarray, cell_size: float):
    """Route sediment under `w`, applying compute_cell()'s capacity rule.

    ero_pot, cap_m3 : m^3 per cell.
    Returns (SEDI_IN, SEDI_OUT, WATEREROS) with WATEREROS in metres, negative
    for erosion and positive for deposition -- matching compute_erosion().
    """
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

        if not has_outflow[i, j]:
            continue
        for k in range(8):
            f = w[k, i, j]
            if f <= 0:
                continue
            ti, tj = i + DR[k], j + DC[k]
            if 0 <= ti < H and 0 <= tj < W:
                SEDI_IN[ti, tj] += out * f

    return SEDI_IN, SEDI_OUT, WATEREROS
