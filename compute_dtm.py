"""Compute DTM covariates (slope, aspect, flow accumulation, flow direction, slope length) via SAGA-GIS with temp-file I/O, cleanup, and fallbacks."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from affine import Affine

from pywatemsedem.defaults import SAGA_FLAGS
from pywatemsedem.geo.utils import load_raster, clean_up_tempfiles

import logging
logger = logging.getLogger(__name__)

_NODATA = -9999.0


def safe_dem(elevation: np.ndarray, mask: np.ndarray, nodata: float = _NODATA) -> np.ndarray:
    """Set all cells outside mask to nodata (float32)."""
    out = np.where(mask, elevation, nodata).astype("float32", copy=False)
    return out


def _run_saga(module: str, tool: str, args: list[str]) -> None:
    """Invoke saga_cmd with module/tool and args."""
    flags = [SAGA_FLAGS] if isinstance(SAGA_FLAGS, str) else list(SAGA_FLAGS)
    cmd = ["saga_cmd", *flags, module, tool, *args]
    logger.debug("SAGA: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    logger.debug("  SAGA %s/%s finished", module, tool)


def _write_temp_dem(arr: np.ndarray,
                    cell_size: float,
                    transform: Optional[Affine] = None,
                    crs=None) -> str:
    """Write arr as float32 GeoTIFF with nodata; return path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
    tmp.close()

    if transform is None:
        # minimal georef if real transform not provided
        transform = Affine(cell_size, 0, 0, 0, -cell_size, 0)

    with rasterio.open(
        tmp.name, "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=_NODATA,
    ) as ds:
        ds.write(arr.astype("float32", copy=False), 1)

    return tmp.name


def _read_and_clean(path_root: str, force_dtype: Optional[str] = None) -> np.ndarray:
    """Read SAGA .sdat by root; replace nodata with NaN; cast if requested."""
    arr, rp = load_raster(path_root + ".sdat")
    nodata = rp.get("nodata", None)
    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr)
    if force_dtype is not None:
        arr = arr.astype(force_dtype, copy=False)
    clean_up_tempfiles(path_root, "saga")
    return arr


def compute_slope(elevation: np.ndarray, cell_size: float) -> np.ndarray:
    """Slope (radians) via SAGA TA_Morphometry tool 0."""
    dem_tif = _write_temp_dem(elevation, cell_size)
    out = dem_tif.replace(".tif", "_slope")
    try:
        _run_saga("ta_morphometry", "0",
                  ["-ELEVATION", dem_tif, "-SLOPE", out, "-UNIT_SLOPE", "0"])
        return _read_and_clean(out, force_dtype="float32")
    finally:
        try: os.remove(dem_tif)
        except OSError: pass


def compute_aspect(elevation: np.ndarray, cell_size: float) -> np.ndarray:
    """Aspect via SAGA TA_Morphometry tool 0."""
    dem_tif = _write_temp_dem(elevation, cell_size)
    out = dem_tif.replace(".tif", "_aspect")
    try:
        _run_saga("ta_morphometry", "0",
                  ["-ELEVATION", dem_tif, "-ASPECT", out])
        return _read_and_clean(out, force_dtype="float32")
    finally:
        try: os.remove(dem_tif)
        except OSError: pass


def compute_flow_accumulation(elevation: np.ndarray, cell_size: float) -> np.ndarray:
    """D8 flow accumulation (# of upslope cells) via SAGA ta_hydrology tool 0."""
    dem_tif = _write_temp_dem(elevation, cell_size)
    out = dem_tif.replace(".tif", "_fac")
    try:
        _run_saga("ta_hydrology", "0",
                  ["-ELEVATION", dem_tif, "-FLOW", out, "-METHOD", "0"])
        return _read_and_clean(out, force_dtype="float32")
    finally:
        try: os.remove(dem_tif)
        except OSError: pass


def compute_flow_direction(elevation: np.ndarray, cell_size: float) -> np.ndarray:
    """
    D8 flow direction via SAGA 'Channel Network and Drainage Basins' (ta_channels, tool 5).
    Output int16: 1–8 for directions, -1 for nodata/no-flow.
    """
    dem_tif = _write_temp_dem(elevation, cell_size)
    dem_path = Path(dem_tif)
    out_root = dem_path.with_name(dem_path.stem + "_flowdir")
    arr: np.ndarray

    try:
        cmd = [
            "saga_cmd", "ta_channels", "5",
            "-DEM", str(dem_path),
            "-DIRECTION", str(out_root),
        ]
        logger.debug("SAGA: %s", " ".join(cmd))
        subprocess.run(cmd, check=True, capture_output=True, text=True)

        arr = load_raster(str(out_root) + ".sdat")[0].astype(np.int16, copy=False)

        # mask where DEM invalid
        mask = np.isfinite(elevation)
        arr = np.where(mask, arr, -1).astype(np.int16, copy=False)

        # sanitize: valid D8 codes 1..8 only; else -1
        bad = (arr < 1) | (arr > 8)
        arr[bad] = -1

    except subprocess.CalledProcessError as e:
        logger.warning("SAGA D8 flow-direction failed (exit %d); using -1 fallback", e.returncode)
        arr = np.full(elevation.shape, -1, dtype=np.int16)

    finally:
        try: os.remove(dem_tif)
        except OSError: pass
        clean_up_tempfiles(str(out_root), "saga")

    return arr


def compute_slope_length(elevation: np.ndarray, cell_size: float) -> np.ndarray:
    """
    Approximate slope length via SAGA ta_channels tool 1. If SAGA fails,
    fallback to sqrt(flow_accumulation * cell_size).
    """
    dem_tif = _write_temp_dem(elevation, cell_size)
    out_root = dem_tif.replace(".tif", "_sl")

    try:
        _run_saga("ta_channels", "1",
                  ["-DEM", dem_tif, "-SLOPE_LENGTH", out_root, "-THRESHOLD", "1"])
        arr = _read_and_clean(out_root, force_dtype="float32")
        return arr

    except subprocess.CalledProcessError:
        logger.warning("SAGA slope-length failed; using sqrt(fac*cell_size) fallback")
        try: os.remove(dem_tif)
        except OSError: pass

        fac = compute_flow_accumulation(elevation, cell_size)
        return np.sqrt(fac * cell_size).astype("float32", copy=False)

    finally:
        if os.path.exists(dem_tif):
            try: os.remove(dem_tif)
            except OSError: pass
