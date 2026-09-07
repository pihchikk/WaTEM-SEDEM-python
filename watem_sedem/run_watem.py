#!/usr/bin/env python3
"""Main driver for WaTEM-SEDEM flexible input and output files,
optional raster export, optional plot saving. CLI flags override config."""

import os
import argparse
import logging
import numpy as np
import rasterio

from watem_sedem.data_loader import load_and_validate_config, merge_cli_overrides, load_inputs
from watem_sedem import mfd
from watem_sedem import solver

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

    su = args.sediment_unit or cfg.output.sediment_unit
    eu = args.erosion_unit  or cfg.output.erosion_unit

    SEDI_IN, SEDI_OUT, WATEREROS, CAPACITY, TILLEROS = solver.solve(cfg, data)
    fields = solver.convert_units(cfg, data, SEDI_OUT, CAPACITY, WATEREROS, TILLEROS,
                                  sediment_unit=su, erosion_unit=eu)
    sed_arr, sed_label = fields["SEDI_OUT"]
    cap_arr, cap_label = fields["CAPACITY"]
    ero_arr, ero_label = fields["WATEREROS"]

    save_r = args.save_rasters or cfg.output.save_rasters
    save_p = args.save_plots   or cfg.output.save_plots

    if save_r:
        fmt = cfg.output.format.lstrip(".")
        meta = data["meta"]
        for name, (arr, _) in fields.items():
            write_raster(name, np.asarray(arr, dtype=np.float32), meta, outdir, fmt)

    # quick plots, only if save_plots (save_p) is enabled -- matplotlib is an
    # optional dependency (the `plots` extra), so it must not be imported,
    # even transitively, on a path a bare install is expected to support.
    if save_p:
        import matplotlib.pyplot as plt

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
        path = os.path.join(outdir, "maps.png")
        fig.savefig(path, dpi=150)
        logger.info("saved plot -> %s", path)

if __name__ == "__main__":
    main()
