"""
Loads WaTEM-SEDEM inputs for five modes:

  external   – require all WaTEM & DTM rasters
  hybrid     – prefer externals; compute missing DTM; WaTEM fallbacks to defaults; special Cfactor path
  user_dtm   – require external DTM covariates; WaTEM from files or defaults; may compute after preprocess_all()
  user_watem – require external WaTEM; compute missing DTM
  internal   – run preprocess_all() and load its outputs

Entry: load_inputs(cfg)
"""

import os
import argparse
import yaml
import logging
from typing import Any, List, Dict, Optional, Literal
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.features import rasterize

# Optional cleanup override used by pywatemsedem
import pywatemsedem.geo.utils as _utils
_utils.clean_up_tempfiles = lambda *args, **kwargs: None

from pydantic import BaseModel, field_validator, ConfigDict

from pywatemsedem.catchment import Catchment
from pywatemsedem.cfactor import create_cfactor_degerick2015

from raster_calculations import compute_ls
from compute_dtm import (
    compute_slope,
    compute_aspect,
    compute_flow_accumulation,
    compute_slope_length,
    compute_flow_direction,
)
from preprocess_watem import preprocess_all, _ensure_int16_categorical


# ── logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")
for _lib in ("rasterio", "fiona", "numexpr"):
    logging.getLogger(_lib).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ── configs ──────────────────────────────────────────────────────────────
class LayerConfig(BaseModel):
    required: bool
    source: Literal["external", "compute", "auto"] = "external"


class OutputConfig(BaseModel):
    directory: str
    format: str
    sediment_unit: Literal["m3", "kg", "t/ha"] = "m3"
    erosion_unit: Literal["m", "mm", "kg", "t/ha"] = "m"
    save_rasters: bool = False
    save_plots: bool = False


class Calibration(BaseModel):
    ktc_multiplier: float = 1.0


class Config(BaseModel):
    model_config = ConfigDict(extra='ignore')

    version: float
    mode: Literal["external", "user_dtm", "user_watem", "hybrid", "internal"]
    raster_dir: str
    raw_input_dir: str
    segment_tables_dir: str
    extensions: List[str]
    layers: Dict[str, LayerConfig]
    raw_inputs: Dict[str, str]
    defaults: Dict[str, Any]
    dtm_covariates: Dict[str, bool]
    output: OutputConfig
    calibration: Calibration
    config_path: Optional[str] = None
    scenario_year: Optional[int] = None
    scenario_nr: Optional[int] = None
    catchment_name: Optional[str] = None
    pyws_output_dir: Optional[str] = None

    field_validator("raster_dir", "segment_tables_dir", "pyws_output_dir", "raw_input_dir", mode="before")
    
    def _expand_paths(cls, v: Optional[str]) -> Optional[str]:
        return os.path.expanduser(v) if isinstance(v, str) else v


# ── CLI argparser ───────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Load WaTEM-SEDEM inputs.")
    p.add_argument("-c", "--config", required=True, help="Path to config.yaml")
    p.add_argument("--mode", choices=["external", "user_dtm", "user_watem", "hybrid", "internal"],
                   help="Override mode from config")
    return p.parse_args()


def load_and_validate_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = Config.model_validate(raw)
    cfg.config_path = path
    return cfg


def merge_cli_overrides(cfg: Config, args) -> Config:
    if args.mode:
        cfg.mode = args.mode
    return cfg


# ── file finding helpers ───────────────────────────────────────────────────────
def find_raster(raster_dir: str, basename: str, exts: List[str]) -> Optional[str]:
    for ext in exts:
        fp = os.path.join(raster_dir, f"{basename}{ext}")
        if os.path.exists(fp):
            return fp
    return None


_ALIASES: Dict[str, List[str]] = {
    "elevation": ["elevation", "dtm"],
    "LS-factor": ["LS-factor", "LS_factor", "lsfactor", "ls_factor"],
    "ktc": ["ktc", "ktc_map"],
    "flow_direction": ["flow_direction", "fdir", "fdir_SAGA_20m"],
    "aspect": ["aspect", "aspect_SAGA_radians20m"],
    "slope_length": ["slope_length", "slopelen", "slopelen_20m"],
    "flow_accumulation": ["flow_accumulation", "flowacc", "flowacc_SAGA_20m", "fac"],
}


def _find_any(raster_dir: str, key: str, exts: List[str]) -> Optional[str]:
    for name in _ALIASES.get(key, [key, key.lower()]):
        fp = find_raster(raster_dir, name, exts)
        if fp:
            return fp
    return None


def _check_watem_inputs(cfg: Config):
    """Quick preflight for preprocess_all prerequisites."""
    rdir = Path(cfg.raster_dir)
    raw = Path(cfg.raw_input_dir)

    have_dtm = bool(find_raster(str(rdir), "dtm", cfg.extensions)
                    or find_raster(str(rdir), "elevation", cfg.extensions)
                    or (raw / cfg.raw_inputs.get("elevation", "")).exists())
    have_catch = (raw / cfg.raw_inputs.get("catchment", "")).exists()
    have_lu = (raw / cfg.raw_inputs.get("landuse", "")).exists()
    have_k = bool(_find_any(str(rdir), "Kfactor", cfg.extensions))

    missing = []
    if not have_dtm:   missing.append("dtm/elevation")
    if not have_catch: missing.append("catchment")
    if not have_lu:    missing.append("landuse")
    if not have_k:     missing.append("Kfactor")

    return (len(missing) == 0, missing)


# ── main loader ────────────────────────────────────────────────────────────────
def load_inputs(cfg: Config):
    """
    Hybrid path (no preprocess_all):
      • elevation → external
      • DTM → load external; compute if flagged and missing
      • WaTEM → try external; fallback to defaults; special Cfactor path from landuse
    """
    data: Dict[str, Any] = {}
    missing: List[str] = []
    rdir = cfg.raster_dir          

    # 1) elevation
    elev_fp = find_raster(rdir, "elevation", cfg.extensions) or \
              find_raster(rdir, "dtm", cfg.extensions)
    if not elev_fp:
        raise FileNotFoundError(
            f"Missing base DTM 'elevation' or 'dtm' in {rdir} or {cfg.raw_input_dir}"
        )
    with rasterio.open(elev_fp) as src:
        data["elevation"] = src.read(1, masked=True)
        data["meta"] = src.meta
        data["cell_size"] = abs(src.transform.a)
    logger.info("elevation ← %s", elev_fp)

    # Optional preprocess for user_dtm
    preprocessed = False
    if cfg.mode == "user_dtm":
        ok, wt_missing = _check_watem_inputs(cfg)
        if ok:
            logger.info("user_dtm: running preprocess_all()")
            preprocess_all(cfg_path=cfg.config_path, year=cfg.scenario_year, scenario_nr=cfg.scenario_nr)
            scen_dir = Path(cfg.pyws_output_dir) / cfg.catchment_name / f"scenario_{cfg.scenario_nr}" / "modelinput"
            rdir = str(scen_dir)       
            preprocessed = True
            dtm_fp = find_raster(rdir, "dtm", cfg.extensions)
            if dtm_fp:
                with rasterio.open(dtm_fp) as src:
                    data["elevation"] = src.read(1, masked=True)
                    data["meta"] = src.meta
                    data["cell_size"] = abs(src.transform.a)
                logger.info("elevation ← %s (preprocessed)", dtm_fp)
        else:
            logger.warning("user_dtm: skip preprocess_all; missing: %s", ", ".join(wt_missing))

    allow_watem_files = cfg.mode in ("external", "user_watem", "hybrid", "user_dtm")

    # Internal → preprocess everything and load
    if cfg.mode == "internal":
        preprocess_all(cfg_path=cfg.config_path, year=cfg.scenario_year, scenario_nr=cfg.scenario_nr)
        scen_dir = Path(cfg.pyws_output_dir) / cfg.catchment_name / f"scenario_{cfg.scenario_nr}" / "modelinput"
        rdir = str(scen_dir)           
        dtm_fp = find_raster(rdir, "dtm", cfg.extensions)
        with rasterio.open(dtm_fp) as src:
            data["elevation"] = src.read(1, masked=True)
            data["meta"] = src.meta
            data["cell_size"] = abs(src.transform.a)
        logger.info("elevation ← %s (internal)", dtm_fp)

    # 2) DTM covariates
    dtm_covs = cfg.dtm_covariates
    to_compute: set = set()

    # user_dtm: strict on DTM covariates unless preprocessed
    if cfg.mode == "user_dtm" and not preprocessed:
        for n, flag in dtm_covs.items():
            if not flag:
                continue
            lay = cfg.layers.get(n)
            if lay and lay.source in ("auto", "compute"):
                if not _find_any(rdir, n, cfg.extensions):
                    logger.debug("user_dtm missing DTM covariate %s in %s (exts=%s)", n, rdir, cfg.extensions)
                    missing.append(f"DTM covariate '{n}'")

    if cfg.mode == "internal":
        to_compute = {n for n, f in dtm_covs.items() if f}
    elif cfg.mode in ("hybrid", "user_watem") or (cfg.mode == "user_dtm" and preprocessed):
        for n, flag in dtm_covs.items():
            if not flag:
                continue
            lay = cfg.layers.get(n)
            if not lay:
                continue
            if lay.source == "compute":
                to_compute.add(n)
            elif lay.source == "auto":
                if not _find_any(rdir, n, cfg.extensions):
                    to_compute.add(n)

    # Load externals for non-computed DTM covariates
    for n, flag in dtm_covs.items():
        if n in to_compute:
            continue
        fp = _find_any(rdir, n, cfg.extensions)
        if fp:
            with rasterio.open(fp) as src:
                data[n] = src.read(1, masked=True)
            logger.info("DTM '%s' ← %s", n, fp)
        elif cfg.mode in ("external", "user_dtm") and cfg.layers[n].required:
            missing.append(f"DTM covariate '{n}'")

    # Compute DTM covariates (except LS-factor)
    for cov in to_compute - {"LS-factor", "LS_factor"}:
        if cov == "slope":
            data["slope"] = compute_slope(data["elevation"], data["cell_size"])
        elif cov == "aspect":
            data["aspect"] = compute_aspect(data["elevation"], data["cell_size"])
        elif cov == "flow_accumulation":
            data["flow_accumulation"] = compute_flow_accumulation(data["elevation"], data["cell_size"])
        elif cov == "slope_length":
            data["slope_length"] = compute_slope_length(data["elevation"], data["cell_size"])
        elif cov == "flow_direction":
            data["flow_direction"] = compute_flow_direction(data["elevation"], data["cell_size"])
        else:
            raise ValueError(f"Unknown DTM covariate '{cov}'")
        logger.info("DTM '%s' ← computed", cov)

    # Compute LS-factor last
    if ("LS-factor" in to_compute) or ("LS_factor" in to_compute):
        if "aspect" not in data:
            data["aspect"] = compute_aspect(data["elevation"], data["cell_size"])
        data["LS-factor"] = compute_ls(
            slope=data["slope"],
            upslope_area=data["flow_accumulation"],
            cell_size=data["cell_size"],
            method="pascal_mccool1987",
            aspect=data.get("aspect"),
        )
        logger.info("DTM 'LS-factor' ← computed")

    # 3) WaTEM covariates
    for n in ("Kfactor", "Cfactor", "Pfactor", "ktc"):
        if n in data:
            continue

        if cfg.mode == "internal":
            fp = _find_any(rdir, n, cfg.extensions)
            if not fp:
                raise FileNotFoundError(
                    f"[internal] Missing computed WaTEM raster '{n}' in {rdir}."
                )
            with rasterio.open(fp) as src:
                data[n] = src.read(1, masked=True)
            logger.info("WaTEM '%s' ← %s (internal)", n, fp)
            continue

        if cfg.mode == "user_watem":
            fp = _find_any(rdir, n, cfg.extensions)
            if fp:
                with rasterio.open(fp) as src:
                    data[n] = src.read(1, masked=True)
                logger.info("WaTEM '%s' ← %s", n, fp)
            elif cfg.layers.get(n, LayerConfig(required=False, source="external")).required:
                missing.append(f"WaTEM covariate '{n}'")
            continue

        if allow_watem_files:
            fp = _find_any(rdir, n, cfg.extensions)
            if fp:
                with rasterio.open(fp) as src:
                    data[n] = src.read(1, masked=True)
                logger.info("WaTEM '%s' ← %s", n, fp)
                continue

        # Hybrid: try to compute Cfactor from landuse
        if cfg.mode == "hybrid" and n == "Cfactor":
            try:
                lu_src = Path(cfg.raw_input_dir) / cfg.raw_inputs["landuse"]
                cat_src = Path(cfg.raw_input_dir) / cfg.raw_inputs["catchment"]
                dtm_src = Path(cfg.raw_input_dir) / cfg.raw_inputs["elevation"]
                if not (lu_src.exists() and cat_src.exists() and dtm_src.exists()):
                    raise FileNotFoundError("Missing landuse/catchment/elevation for Cfactor hybrid path")

                lu16 = _ensure_int16_categorical(lu_src, Path(rdir) / "__landuse16.tif")

                with rasterio.open(dtm_src) as ds:
                    grid_crs, grid_tr = ds.crs, ds.transform
                    H, W = ds.height, ds.width
                    cell_size = abs(ds.transform.a)

                gdf = gpd.read_file(cat_src).to_crs(grid_crs)
                bin_mask = rasterize(((geom, 1) for geom in gdf.geometry),
                                     out_shape=(H, W), transform=grid_tr,
                                     fill=0, all_touched=False, dtype="uint8")

                epsg = data["meta"]["crs"].to_epsg()
                catch = Catchment(
                    name=cfg.catchment_name, vct_catchment=cat_src, rst_dtm=dtm_src,
                    resolution=cell_size, epsg_code=epsg, nodata=0, results_folder=Path(rdir),
                )

                mask_r = catch.raster_factory(bin_mask, flag_mask=False, flag_clip=False)
                pr = mask_r.rp.rasterio_profile
                pr["dtype"], pr["nodata"], pr["count"] = "uint8", 0, 1

                with rasterio.open(lu16) as src:
                    lu_in = src.read(1)
                    lu_aligned = np.zeros((H, W), dtype=np.int32)
                    reproject(
                        source=lu_in, destination=lu_aligned,
                        src_transform=src.transform, src_crs=src.crs,
                        dst_transform=grid_tr, dst_crs=grid_crs,
                        resampling=Resampling.nearest,
                    )

                lut_cfg = cfg.defaults.get("C_LUT")
                default_agri_code = 10
                comp_lu = np.full_like(lu_aligned, default_agri_code, dtype=np.int32)
                if isinstance(lut_cfg, dict) and len(lut_cfg) > 0:
                    for k, v in lut_cfg.items():
                        comp_lu[lu_aligned == int(k)] = int(v)

                river_arr = np.zeros((H, W), dtype=np.uint8)
                infra_arr = np.zeros((H, W), dtype=np.uint8)

                river_r = catch.raster_factory(river_arr, flag_mask=False, flag_clip=False)
                infra_r = catch.raster_factory(infra_arr, flag_mask=False, flag_clip=False)
                comp_r = catch.raster_factory(comp_lu.astype(np.int32), flag_mask=False, flag_clip=False)
                for r, dt in [(river_r, "uint8"), (infra_r, "uint8"), (comp_r, "int32")]:
                    pr = r.rp.rasterio_profile
                    pr["dtype"], pr["nodata"], pr["count"] = dt, 0, 1

                empty_parcels_gdf = gpd.GeoDataFrame({"geometry": gpd.GeoSeries([], dtype="geometry")}, crs=grid_crs)
                vct_parcels = catch.vector_factory(empty_parcels_gdf, geometry_type="polygon", allow_empty=True)
                empty_grass_gdf = gpd.GeoDataFrame({"geometry": gpd.GeoSeries([], dtype="geometry"),
                                                    "width": pd.Series([], dtype=float)}, crs=grid_crs)
                vct_grass = catch.vector_factory(empty_grass_gdf, geometry_type="line", allow_empty=True)

                cf_agri = float(cfg.defaults.get("Cfactor", 0.37))
                _, cfactor = create_cfactor_degerick2015(
                    rivers=river_r, infrastructure=infra_r, composite_landuse=comp_r, mask=mask_r,
                    vct_parcels=vct_parcels, vct_grass_strips=vct_grass,
                    cfactor_aggriculture=cf_agri, use_source_oriented_measures=False,
                )

                data[n] = np.ma.masked_invalid(cfactor.astype("float32"))
                logger.info("Cfactor ← computed (hybrid)")
                continue
            except Exception as e:
                logger.warning("Cfactor hybrid path failed: %s — fallback to default", e)

        # Fallbacks
        if cfg.mode == "hybrid":
            if n in cfg.defaults:
                data[n] = cfg.defaults[n]
                logger.info("WaTEM '%s' ← default scalar (hybrid)", n)
            elif cfg.layers.get(n, LayerConfig(required=False, source="external")).required:
                missing.append(f"WaTEM covariate '{n}'")
            continue

        if cfg.mode == "user_dtm":
            if not preprocessed:
                if n in cfg.defaults:
                    data[n] = cfg.defaults[n]
                    logger.info("WaTEM '%s' ← default scalar (user_dtm, no preprocess)", n)
                elif cfg.layers.get(n, LayerConfig(required=False, source="external")).required:
                    missing.append(f"WaTEM covariate '{n}'")
            else:
                if cfg.layers.get(n, LayerConfig(required=False, source="external")).required:
                    missing.append(f"WaTEM covariate '{n}'")
            continue

        if cfg.mode == "external":
            if cfg.layers.get(n, LayerConfig(required=False, source="external")).required:
                missing.append(f"WaTEM covariate '{n}'")
            continue

        if n in cfg.defaults:
            data[n] = cfg.defaults[n]
            logger.info("WaTEM '%s' ← default scalar", n)
        elif cfg.layers.get(n, LayerConfig(required=False, source="external")).required:
            missing.append(f"WaTEM covariate '{n}'")

    # 4) Other layers declared in YAML
    for n in ("Rfactor", "bulk-density", "routing", "segments", "outlet"):
        if n in data:
            continue
        fp = _find_any(rdir, n, cfg.extensions)
        if fp:
            with rasterio.open(fp) as src:
                data[n] = src.read(1, masked=True)
            logger.info("'%s' ← %s", n, fp)
        else:
            if n in cfg.defaults:
                data[n] = cfg.defaults[n]
                logger.info("'%s' ← default scalar", n)
            elif cfg.layers.get(n, LayerConfig(required=False, source="external")).required:
                missing.append(f"required input '{n}'")

    # Normalize discrete int layers
    for n in ("routing", "segments", "outlet"):
        if n in data and np.ma.isMaskedArray(data[n]):
            data[n] = data[n].filled(-1).astype(np.int32)

    # Final fallbacks by YAML
    for n, lay in cfg.layers.items():
        if n == "elevation" or n in data:
            continue
        if n in cfg.defaults:
            data[n] = cfg.defaults[n]
            logger.info("'%s' ← final default scalar", n)
        elif lay.required and cfg.mode in ("external", "user_watem"):
            missing.append(f"required input '{n}'")

    # Deduplicate and raise if any
    if missing:
        kept = []
        for msg in missing:
            # msg looks like: "DTM covariate 'slope'"
            if ("'" in msg) and ("covariate" in msg):
                key = msg.split("'")[1]
                if key in data:
                    continue
            kept.append(msg)
        missing = kept

        logger.info("%s: all inputs ready.", cfg.mode)
    return data


# ── entrypoint ─────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    cfg = load_and_validate_config(args.config)
    cfg = merge_cli_overrides(cfg, args)
    _ = load_inputs(cfg)


if __name__ == "__main__":
    main()
