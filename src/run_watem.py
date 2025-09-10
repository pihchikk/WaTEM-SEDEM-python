#!/usr/bin/env python3
"""Main driver for WaTEM-SEDEM flexible input and output files,
optional raster export, optional plot saving. CLI flags override config."""

import os
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
import rasterio

from data_loader import load_and_validate_config, merge_cli_overrides, load_inputs
from lateraldistribution import topo_order, compute_erosion

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")
for _lib in ("rasterio", "fiona", "numexpr"):
    logging.getLogger(_lib).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

_DRIVER_MAP = {  # rasterio driver per extension
    "tif": "GTiff", "tiff": "GTiff", "rst": "RST",
    "netcdf": "NetCDF", "nc": "NetCDF", "asc": "AAIGrid"
}

def outdir_for_mode(base_dir: str, mode: str) -> str:
    """Return <base>/results_<mode>, correcting if base already looks like results_*."""
    base = os.path.abspath(base_dir)
    leaf = os.path.basename(base)
    want = f"results_{mode}"
    if leaf == want:
        return base
    if leaf.startswith("results_"):
        return os.path.join(os.path.dirname(base), want)
    return os.path.join(base, want)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run WaTEM-SEDEM.")
    p.add_argument("-c", "--config", required=True, help="config.yaml")
    p.add_argument("--mode",
                   choices=["external", "user_dtm", "user_watem", "hybrid", "internal"],
                   help="override mode from config")
    p.add_argument("--sediment-unit", choices=["m3", "kg", "t/ha"])
    p.add_argument("--erosion-unit",  choices=["m", "mm", "kg", "t/ha"])
    p.add_argument("--save-rasters", action="store_true")
    p.add_argument("--save-plots",   action="store_true")
    p.add_argument("--raster-dir",   help="override cfg.raster_dir")
    return p.parse_args()

def write_raster(name: str, arr: np.ndarray, meta: dict, outdir: str, fmt: str) -> None:
    driver = _DRIVER_MAP.get(fmt.lower())
    if not driver:
        raise ValueError(f"Unknown raster format: {fmt}")
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{name}.{fmt}")
    m = meta.copy(); m.update({"driver": driver, "dtype": arr.dtype, "count": 1})
    with rasterio.open(path, "w", **m) as dst:
        dst.write(arr, 1)
    logger.info("wrote %s -> %s", name, path)

def main() -> None:
    args = parse_args()

    cfg = load_and_validate_config(args.config)
    cfg = merge_cli_overrides(cfg, args)
    if args.raster_dir:
        cfg.raster_dir = args.raster_dir

    mode   = args.mode or cfg.mode
    outdir = outdir_for_mode(cfg.output.directory, mode)
    os.makedirs(outdir, exist_ok=True)
    logger.info("output dir: %s", outdir)

    data = load_inputs(cfg)

    # expand scalar WaTEM layers to full grids
    shape = data["elevation"].shape
    for key in ("LS-factor", "Kfactor", "Cfactor", "Pfactor", "ktc"):
        if not isinstance(data[key], np.ndarray):
            data[key] = np.full(shape, data[key], dtype=float)

    m = cfg.calibration.ktc_multiplier
    data["ktc"] = (data["ktc"].astype(float) if isinstance(data["ktc"], np.ndarray) else float(data["ktc"])) * m

    logger.debug(
        "shapes: %s",
        {k: getattr(data[k], "shape", None) for k in ("LS-factor","flow_direction","Cfactor","Kfactor","Pfactor","slope","ktc")}
    )
    
    # Ensure D8 codes for flow_direction (-1 = no flow)
    fdir = data["flow_direction"]
    if np.ma.isMaskedArray(fdir):
        fdir = fdir.filled(-1)
    fdir = fdir.astype(np.int16, copy=False)
    bad = (fdir < 0) | (fdir > 7)
    fdir[bad] = -1
    data["flow_direction"] = fdir


    topo = topo_order(data["flow_direction"])
    SEDI_IN, SEDI_OUT, WATEREROS, CAPACITY = compute_erosion(
        LS=data["LS-factor"], Kfactor=data["Kfactor"], Cfactor=data["Cfactor"],
        Pfactor=data["Pfactor"], R_factor=data["Rfactor"], bulk_density=data["bulk-density"],
        slope=data["slope"], aspect=data["aspect"], ktc=data["ktc"],
        cell_res=data["cell_size"], flow_direction=data["flow_direction"], topo=topo
    )

    # unit conversion
    cell_area = data["cell_size"] ** 2
    bd = data["bulk-density"]
    sed_mass = bd / 1000.0 * (10000.0 / cell_area)   # m³→t/ha
    ero_mass = bd * (10000.0 / 1000.0)               # m→t/ha

    su = args.sediment_unit or cfg.output.sediment_unit
    eu = args.erosion_unit  or cfg.output.erosion_unit
    save_r = args.save_rasters or cfg.output.save_rasters
    save_p = args.save_plots   or cfg.output.save_plots

    if su == "m3":
        sed_arr, cap_arr = SEDI_OUT, CAPACITY
        sed_label = cap_label = "m³ per cell"
    elif su == "kg":
        sed_arr, cap_arr = SEDI_OUT * bd, CAPACITY * bd
        sed_label = cap_label = "kg per cell"
    else:
        sed_arr, cap_arr = SEDI_OUT * sed_mass, CAPACITY * sed_mass
        sed_label = cap_label = "t/ha"

    if eu == "m":
        ero_arr, ero_label = WATEREROS, "m"
    elif eu == "mm":
        ero_arr, ero_label = WATEREROS * 1000.0, "mm"
    elif eu == "kg":
        ero_arr, ero_label = WATEREROS * cell_area * bd, "kg per cell"
    else:
        ero_arr, ero_label = WATEREROS * ero_mass, "t/ha"
    
    sed_arr = np.asarray(sed_arr, dtype=np.float32)
    cap_arr = np.asarray(cap_arr, dtype=np.float32)
    ero_arr = np.asarray(ero_arr, dtype=np.float32)

    if save_r:
        fmt = cfg.output.format.lstrip(".")
        meta = data["meta"]
        write_raster("SEDI_OUT",  sed_arr.astype(np.float32), meta, outdir, fmt)
        write_raster("CAPACITY",  cap_arr.astype(np.float32), meta, outdir, fmt)
        write_raster("WATEREROS", ero_arr.astype(np.float32), meta, outdir, fmt)

    # quick plots if save_plots (save_p) is enabled
    sed_vmin, sed_vmax = np.nanpercentile(sed_arr, [2, 98])
    cap_vmin, cap_vmax = np.nanpercentile(cap_arr, [2, 98])
    ero_vmin, ero_vmax = np.nanpercentile(ero_arr, [2, 98])

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    im1 = ax1.imshow(sed_arr, vmin=sed_vmin, vmax=sed_vmax, interpolation="none")
    ax1.set_title(f"Sediment ({su})"); ax1.axis("off")
    fig.colorbar(im1, ax=ax1, label=sed_label, fraction=0.046, pad=0.04)

    im2 = ax2.imshow(cap_arr, vmin=cap_vmin, vmax=cap_vmax, interpolation="none")
    ax2.set_title(f"Capacity ({su})"); ax2.axis("off")
    fig.colorbar(im2, ax=ax2, label=cap_label, fraction=0.046, pad=0.04)

    im3 = ax3.imshow(ero_arr, vmin=ero_vmin, vmax=ero_vmax, interpolation="none")
    ax3.set_title(f"Erosion ({eu})"); ax3.axis("off")
    fig.colorbar(im3, ax=ax3, label=ero_label, fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_p:
        path = os.path.join(outdir, "maps.png")
        fig.savefig(path, dpi=150)
        logger.info("saved plot -> %s", path)
    else:
        plt.show()

if __name__ == "__main__":
    main()
