"""Tillage erosion -- the "T" in WaTEM (Water and Tillage Erosion Model).

Ploughing moves soil downslope regardless of water. The flux through a unit
contour width is taken proportional to the local gradient (Govers et al. 1994;
Van Oost, Govers & Desmet 2000):

    Q = ktil * sin(slope)                       kg m-1 yr-1

directed downslope, i.e. along the aspect. Soil is lost where that flux diverges
and gained where it converges, so the erosion rate is its negative divergence:

    E_til = -div(Q)                             kg m-2 yr-1

which makes tillage erosion a function of curvature, not of slope: it strips
convexities -- hilltops, shoulders, the crests between furrows -- and fills
concavities, including places water never reaches. That signature is the reason
it matters here; it is also why it cannot be folded into the water-erosion
routing, which only ever moves sediment downstream.

ktil is a length-scaled coefficient in kg m-1; the WaTEM/SEDEM desktop default,
and the value used for the reference runs in docs/REFERENCE_RUN_SETTINGS.md, is
600 kg m-1.

Sign convention matches WATEREROS: negative is erosion, positive deposition.

NOT validated against the reference rasters, because they do not contain it --
the modeller who produced them confirmed tillage was left off ("Нет, ее можно
отдельно посчитать"). What is checked instead is the invariant that makes the
formulation falsifiable: tillage moves soil without creating or destroying it,
so the sum over a catchment with no flux across its boundary must be zero. See
tests/test_tillage.py.
"""

from __future__ import annotations

import numpy as np


def tillage_flux(slope_rad: np.ndarray, aspect_rad: np.ndarray, ktil: float):
    """Downslope soil flux components (east, north) in kg m-1 yr-1."""
    magnitude = ktil * np.sin(slope_rad)
    qx = magnitude * np.sin(aspect_rad)   # east
    qy = magnitude * np.cos(aspect_rad)   # north
    return qx, qy


def tillage_erosion(slope_rad: np.ndarray, aspect_rad: np.ndarray,
                    ktil: float, cell_size: float,
                    valid: np.ndarray | None = None) -> np.ndarray:
    """Tillage erosion rate, kg m-2 yr-1, negative for erosion.

    `valid` marks cells inside the domain; flux is taken as zero outside, so the
    divergence at the edge represents real export across the boundary rather
    than a numerical artefact of the array ending.
    """
    qx, qy = tillage_flux(np.asarray(slope_rad, dtype=float),
                          np.asarray(aspect_rad, dtype=float), ktil)

    if valid is None:
        valid = np.isfinite(qx) & np.isfinite(qy)
    qx = np.where(valid & np.isfinite(qx), qx, 0.0)
    qy = np.where(valid & np.isfinite(qy), qy, 0.0)

    # Central differences. Rows increase southward, so north is row-1 and the
    # y-derivative is (up - down), not (down - up).
    dqx_dx = np.zeros_like(qx)
    dqy_dy = np.zeros_like(qy)
    dqx_dx[:, 1:-1] = (qx[:, 2:] - qx[:, :-2]) / (2.0 * cell_size)
    dqy_dy[1:-1, :] = (qy[:-2, :] - qy[2:, :]) / (2.0 * cell_size)

    # One-sided at the array edge so boundary cells still export.
    dqx_dx[:, 0] = (qx[:, 1] - qx[:, 0]) / cell_size
    dqx_dx[:, -1] = (qx[:, -1] - qx[:, -2]) / cell_size
    dqy_dy[0, :] = (qy[0, :] - qy[1, :]) / cell_size
    dqy_dy[-1, :] = (qy[-2, :] - qy[-1, :]) / cell_size

    ero = -(dqx_dx + dqy_dy)
    return np.where(valid, ero, np.nan)


def to_t_ha(ero_kg_m2: np.ndarray) -> np.ndarray:
    """kg m-2 yr-1 -> t ha-1 yr-1."""
    return ero_kg_m2 * 10.0
