"""
Core lateral sediment‐routing and erosion computations for WaTEM-SEDEM.
Uses D8 flow‐direction codes (0=N, 1=NE, …, 7=NW).
"""

import numpy as np
from typing import List, Tuple

import logging

# ─── logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level    = logging.INFO,
    format   = "%(levelname)s:%(name)s: %(message)s"
)
for _lib in ("rasterio","fiona","numexpr"):
    logging.getLogger(_lib).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
 
def transport_capacity(LS, Kfactor, R_factor, slope_rad, aspect_rad, ktc_m, cell_res):
    """Transport capacity per cell, m3/yr. Works on scalars or arrays alike.

        TC = ktc * R * K * (LS - 0.6*6.86*|sin(slope)|**0.8)      kg m-1 yr-1
        capacity = TC * flow width through the cell                kg yr-1

    Units, since getting these wrong is what this function exists to prevent:
    R is MJ mm ha-1 h-1 yr-1 and K is kg h MJ-1 mm-1, so R*K is kg ha-1 yr-1;
    /1e4 puts it in kg m-2 yr-1; ktc is a LENGTH in metres (confirmed with the
    modeller who produced the reference runs), which turns it into a flux per
    unit contour width. Multiplying by the width of flow through the cell gives
    kg yr-1, and dividing by bulk density gives m3 -- the units SEDI_IN and
    SEDI_OUT are carried in.

    The width uses ASPECT, not slope: it is the geometric width a flow line of
    the given bearing presents across a square cell, the same
    |sin| + |cos| correction Desmet & Govers apply in the L factor. The previous
    code took it from the slope raster and multiplied by cell area twice over,
    which left the "capacity" in m^5 and around cell_res**3 too large.

    `Kfactor` is the raw K, not K/1000.
    """
    bracket = LS - 0.6 * 6.86 * np.abs(np.sin(slope_rad)) ** 0.8
    tc_kg_per_m = ktc_m * R_factor * Kfactor * np.maximum(bracket, 0.0) / 1e4
    width = cell_res * (np.abs(np.sin(aspect_rad)) + np.abs(np.cos(aspect_rad)))
    return tc_kg_per_m * width


def topo_order(flow_direction: np.ndarray) -> List[Tuple[int, int]]:
    """
    Given a 2D array `flow_direction` of D8 codes 0–7 (or -1 for no‐flow),
    return a list of (row, col) indices so that each cell is visited
    before the cell it flows into.
    """
    # neighbor offsets for D8: N, NE, E, SE, S, SW, W, NW
    dr = np.array([-1, -1,  0, 1, 1, 1,  0, -1])
    dc = np.array([ 0,  1,  1, 1, 0,-1, -1, -1])

    nrow, ncol = flow_direction.shape
    logger.info("running erosion routing on a %dx%d grid", nrow, ncol)
    indegree = np.zeros((nrow, ncol), dtype=int)

    # 1) compute indegree of each cell's downstream target
    for d in range(8):
        mask    = (flow_direction == d)
        shifted = np.roll(np.roll(mask, dr[d], axis=0), dc[d], axis=1)
        # clear wrap‐around artifacts
        if dr[d] < 0: shifted[dr[d]:, :] = False
        if dr[d] > 0: shifted[:dr[d], :] = False
        if dc[d] < 0: shifted[:, dc[d]:] = False
        if dc[d] > 0: shifted[:, :dc[d]] = False
        indegree[shifted] += 1

    # 2) initialize queue with cells of indegree zero
    queue = [(i, j)
             for i in range(nrow) for j in range(ncol)
             if indegree[i, j] == 0]
    topo = []

    # 3) peel off leaves
    while queue:
        i, j = queue.pop()
        topo.append((i, j))
        d = flow_direction[i, j]
        if 0 <= d < 8:
            ti, tj = i + dr[d], j + dc[d]
            if 0 <= ti < nrow and 0 <= tj < ncol:
                indegree[ti, tj] -= 1
                if indegree[ti, tj] == 0:
                    queue.append((ti, tj))

    return topo


def compute_cell(
    LS_cell: float,
    Kfactor_cell: float,
    Cfactor_cell: float,
    Pfactor_cell: float,
    R_factor: float,
    bulk_density: float,
    slope_cell: float,
    aspect_cell: float,
    ktc_cell: float,
    cell_res: float,
    sedi_in_cell: float
) -> Tuple[float, float, float]:
    """
    Compute erosion and routing for one cell.

    Args:
      LS_cell      : LS factor (unitless)
      Kfactor_cell : K factor, raw (kg h MJ-1 mm-1)
      Cfactor_cell : cover-management factor (unitless)
      Pfactor_cell : support practice factor (unitless)
      R_factor     : rainfall-erosivity (MJ mm ha-1 h-1 yr-1)
      bulk_density : soil bulk density (kg/m3)
      slope_cell   : slope in RADIANS
      aspect_cell  : aspect in radians (sets the flow width in the capacity)
      ktc_cell     : transport coefficient, metres
      cell_res     : grid cell size (m)
      sedi_in_cell : incoming sediment volume (m3)

    Returns:
      (sedi_out_m3, watereros_m, capacity_m3)
    """
    area = cell_res ** 2

    # Slope arrives in radians -- SAGA is called with -UNIT_SLOPE 0 and
    # compute_ls() reads the same raster as radians. This used to deg2rad it
    # again, which divided every slope by 57 and flattened the capacity's
    # steepness correction to nothing.
    slope_rad = slope_cell

    rusle_kg_m2 = R_factor * Cfactor_cell * Pfactor_cell * Kfactor_cell * LS_cell / 1e4
    ero_pot = rusle_kg_m2 * area / bulk_density  # m3 potential

    cap_kg = transport_capacity(LS_cell, Kfactor_cell, R_factor,
                               slope_rad, aspect_cell, ktc_cell, cell_res)
    cap_m3 = max(cap_kg, 0.0) / bulk_density

    total_in = sedi_in_cell + ero_pot
    if total_in > cap_m3:
        out       = cap_m3
        watereros = (sedi_in_cell - out) / area
    else:
        out       = total_in
        watereros = -ero_pot / area

    return out, watereros, cap_m3


def compute_erosion(
    LS: np.ndarray,
    Kfactor: np.ndarray,
    Cfactor: np.ndarray,
    Pfactor: np.ndarray,
    R_factor: float,
    bulk_density: float,
    slope: np.ndarray,
    aspect: np.ndarray,
    ktc: np.ndarray,
    cell_res: float,
    flow_direction: np.ndarray,
    topo: List[Tuple[int, int]]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    whole-grid erosion by routing sediment downslope.
    
    Returns:
      SEDI_IN, SEDI_OUT, WATEREROS, CAPACITY (all 2D arrays)
    """
    nrow, ncol = flow_direction.shape
    SEDI_IN   = np.zeros((nrow, ncol), dtype=float)
    SEDI_OUT  = np.zeros((nrow, ncol), dtype=float)
    WATEREROS = np.zeros((nrow, ncol), dtype=float)
    CAPACITY  = np.zeros((nrow, ncol), dtype=float)

    dr = np.array([-1, -1, 0, 1, 1, 1, 0, -1])
    dc = np.array([ 0,  1, 1, 1, 0,-1,-1, -1])

    for (i, j) in topo:
        out, we, cap = compute_cell(
            LS[i, j],
            Kfactor[i, j],
            Cfactor[i, j],
            Pfactor[i, j],
            R_factor,
            bulk_density,
            slope[i, j],
            aspect[i, j],
            ktc[i, j],
            cell_res,
            SEDI_IN[i, j]
        )
        SEDI_OUT[i, j]   = out
        WATEREROS[i, j]  = we
        CAPACITY[i, j]   = cap

        d = flow_direction[i, j]
        if 0 <= d < 8:
            ti, tj = i + dr[d], j + dc[d]
            if 0 <= ti < nrow and 0 <= tj < ncol:
                SEDI_IN[ti, tj] += out

    return SEDI_IN, SEDI_OUT, WATEREROS, CAPACITY
