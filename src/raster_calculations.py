"""
Functions to compute raster-based factors from DEM, including LS factor.
"""

import numpy as np
from typing import Optional
import logging

# logging setup
logging.basicConfig(
    level    = logging.INFO,
    format   = "%(levelname)s:%(name)s: %(message)s"
)
for _lib in ("rasterio","fiona","numexpr"):
    logging.getLogger(_lib).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def compute_ls(
    slope,
    upslope_area,
    cell_size,
    method='mccool',
    slope_length=None,
    aspect=None,
    filled_dem=None,
    compute_aspect_internal=False,
    LScor=22.13
):
    """
    Compute LS factor using various methods:
      - standard: 'wischmeier','mccool','govers','nearing'
      - Pascal variants: 'pascal_vanoost2003','pascal_mccool1987','pascal_nearing1997'
    """
    # Handle aspect requirement
    if method.startswith('pascal_'):
        if compute_aspect_internal:
            if filled_dem is None:
                raise ValueError("filled_dem required to compute aspect internally")
            dy, dx = np.gradient(filled_dem, filled_dem.geotransform[5], filled_dem.geotransform[1])
            aspect = np.arctan2(dx, dy)
        elif aspect is None:
            raise ValueError("Aspect array required for Pascal methods")

    sin_s = np.sin(slope)
    tan_s = np.tan(slope)
    A     = upslope_area

    # Normalize slope_length
    if slope_length is None:
        slope_len = (A / cell_size) / LScor
    else:
        slope_len = slope_length / LScor

    # Pascal methods
    if method.startswith('pascal_'):
        if method == 'pascal_vanoost2003':
            locres = cell_size
            exp = np.where(A < 10000, 0.3 + (A/10000)**0.8, 0.72)
            exp = np.minimum(exp, 0.72)
        else:
            locres = cell_size if slope_length is None else slope_length
            B = (sin_s / 0.0896) / (3.0 * sin_s**0.8 + 0.56)
            exp = B / (B + 1.0)

        ADJ = np.abs(np.cos(aspect)) + np.abs(np.sin(aspect))
        num = ((A + locres**2)**(exp+1) - A**(exp+1))
        den = (ADJ**exp) * (locres**(exp+2))
        Lfactor = (num / den) / (LScor**exp)

        if method in ('pascal_nearing1997','nearing'):
            Sfactor = -1.5 + 17.0 / (1.0 + np.exp(2.3 - 6.1 * sin_s))
        else:
            pct = tan_s * 100.0
            Sfactor = np.where(pct < 9.0,
                               10.8 * sin_s + 0.03,
                               16.8 * sin_s - 0.50)

    # Standard methods
    else:
        Lfactor = (A / LScor)**0.4
        if method == 'wischmeier':
            Sfactor = 65.41 * sin_s**2 + 4.56 * sin_s + 0.065
        elif method == 'govers':
            Sfactor = (tan_s / 0.09) * 1.45
        elif method in ('pascal_nearing1997','nearing'):
            Sfactor = -1.5 + 17.0 / (1.0 + np.exp(2.3 - 6.1 * sin_s))
        else:
            pct = tan_s * 100.0
            Sfactor = np.where(pct < 9.0,
                               10.8 * sin_s + 0.03,
                               16.8 * sin_s - 0.50)

    # Combine and mask
    LS       = Sfactor * Lfactor
    valid    = np.isfinite(slope) & np.isfinite(A)
    LS_full  = np.full_like(slope, np.nan)
    LS_full[valid] = LS[valid]

    # report global mean
    mean_ls = float(np.nanmean(LS_full))
    logger.info("LS-factor computed, mean = %.5f", mean_ls)

    return LS_full
