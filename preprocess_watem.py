"""Prepare WaTEM-SEDEM inputs from raw sources:
   build Catchment, cast landuse to int16, plug in simple dummies if vectors
   are missing, then let Scenario write rasters/tables/INI.
"""

from __future__ import annotations

import yaml
from pathlib import Path
import numpy as np
import rasterio
import geopandas as gpd
from shapely.geometry import LineString, Polygon
from affine import Affine
import logging

from pywatemsedem.catchment   import Catchment
from pywatemsedem.userchoices import UserChoices
from pywatemsedem.scenario    import Scenario
from pywatemsedem.geo.rasters import AbstractRaster

logger = logging.getLogger(__name__)


# helpers

def load_config(path: str | Path = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _ensure_int16_categorical(src: Path, dst: Path) -> Path:
    """Read `src`, cast band-1 to int16 with nodata=-9999, write GTiff to `dst`."""
    with rasterio.open(src) as ds:
        arr  = ds.read(1)
        prof = ds.profile
    prof.update(driver="GTiff", dtype="int16", nodata=-9999, count=1)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dst, "w", **prof) as out:
        out.write(arr.astype("int16", copy=False), 1)
    logger.debug("landuse cast → int16: %s", dst)
    return dst

def _infer_transform(rp) -> Affine:
    """Try to pull an Affine transform from pywatemsedem’s raster profile."""
    tr = getattr(rp, "transform", None)
    if tr is not None:
        return tr
    raw = rp.rasterio_profile.get("transform")
    return Affine(*raw[:6])

def _infer_resolution(rp, tr: Affine) -> float:
    res = getattr(rp, "resolution", None)
    return float(res) if res is not None else float(tr.a)


# main entry

def preprocess_all(
    cfg_path: str | Path = "config.yaml",
    year: int = 2025,
    scenario_nr: int = 1,
    show_preview: bool = False,
) -> None:
    """End-to-end prep: Catchment, optional dummies, Scenario, write outputs."""
    cfg = load_config(cfg_path)
    logger.info("Loaded config from %s", cfg_path)

    base    = Path(cfg["pyws_output_dir"]);     base.mkdir(exist_ok=True)
    raw_dir = Path(cfg["raw_input_dir"])
    ri      = cfg["raw_inputs"]

    rd = Path(cfg["raster_dir"]);         rd.mkdir(parents=True, exist_ok=True)
    sd = Path(cfg["segment_tables_dir"]); sd.mkdir(parents=True, exist_ok=True)
    logger.info("Output folders: rasters→%s, tables→%s", rd, sd)

    # Catchment
    catch = Catchment(
        name           = cfg["catchment_name"],
        vct_catchment  = raw_dir / ri["catchment"],
        rst_dtm        = raw_dir / ri["elevation"],
        resolution     = cfg.get("dtm_resolution", 20),
        epsg_code      = cfg.get("dtm_epsg", 32637),
        nodata         = cfg.get("dtm_nodata", -9999),
        results_folder = base,
    )
    logger.info("Constructed Catchment '%s'", cfg["catchment_name"])

    # K-factor (optional)
    raw_k = raw_dir / ri.get("Kfactor", "")
    if raw_k.exists():
        catch.kfactor = raw_k
        logger.info("Using provided Kfactor: %s", raw_k)
    else:
        logger.warning("Kfactor not found — default/derived value will be used")

    # Landuse → int16 (optional, but recommended)
    lu16: Path | None = None
    raw_lu = raw_dir / ri.get("landuse", "")
    if raw_lu.exists():
        lu16 = _ensure_int16_categorical(raw_lu, rd / "__landuse16.tif")
        catch.landuse = lu16
        logger.info("Landuse prepared: %s", lu16)
    else:
        logger.warning("Landuse missing — Cfactor will fall back to defaults")

    # Quick stats / optional preview
    if lu16 is not None:
        with rasterio.open(lu16) as src:
            arr = src.read(1)
            prof = src.profile
        logger.debug(
            "Landuse dtype=%s nodata=%s min=%s max=%s (nuniq≈%d)",
            arr.dtype, prof.get("nodata"), int(arr.min()), int(arr.max()),
            int(np.unique(arr[: min(arr.size, 100000)]).size),
        )
        if show_preview:
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(4, 4))
                plt.imshow(arr, interpolation="none")
                plt.title("Landuse (Int16)")
                plt.colorbar(label="class")
                plt.show()
            except Exception:
                logger.debug("preview skipped (matplotlib unavailable)")

    # River vector (or tiny dummy line)
    shp = raw_dir / ri.get("river", "")
    if shp.exists():
        catch.vct_river = shp
        logger.info("River: %s", shp)
    else:
        logger.warning("River shapefile missing — inserting tiny dummy line")
        mask_arr = catch.mask.arr
        rp       = catch.mask.rp
        tr       = _infer_transform(rp)
        res      = _infer_resolution(rp, tr)
        rows, cols = np.where(mask_arr == 1)
        i, j = (int(rows[0]), int(cols[0])) if rows.size else (0, 0)
        x, y = tr * (j, i)
        h = res / 2.0
        tiny = LineString([(x - h, y - h), (x + h, y + h)])
        catch.vct_river = gpd.GeoDataFrame({"geometry": [tiny]},
                                           crs=getattr(rp, "crs", rp.rasterio_profile.get("crs")))

    # Water vector (optional)
    shp = raw_dir / ri.get("water", "")
    if shp.exists():
        catch.vct_water = shp
        logger.info("Water: %s", shp)
    else:
        logger.warning("Water shapefile missing — skipping water rasterization")

    # Infrastructure (or 1-cell dummy)
    shp = raw_dir / ri.get("infrastructure", "")
    if shp.exists():
        try:
            catch.vct_infrastructure_buildings = shp
        except AttributeError:
            catch.vct_infrastructure = shp
        logger.info("Infrastructure: %s", shp)
    else:
        logger.warning("Infrastructure missing — inserting tiny dummy polygon")
        rp  = catch.mask.rp
        tr  = _infer_transform(rp)
        res = _infer_resolution(rp, tr)
        rows, cols = np.where(catch.mask.arr == 1)
        i, j = (int(rows[0]), int(cols[0])) if rows.size else (0, 0)
        x, y = tr * (j, i)
        h = res / 2.0
        poly = Polygon([(x - h, y - h), (x - h, y + h),
                        (x + h, y + h), (x + h, y - h)])
        gdf = gpd.GeoDataFrame({"geometry": [poly], "paved": [-2]},
                               crs=getattr(rp, "crs", rp.rasterio_profile.get("crs")))
        # assign to any of the supported attributes
        for attr in ("vct_infrastructure_buildings",
                     "vct_infrastructure_roads",
                     "vct_infrastructure"):
            try:
                setattr(catch, attr, gdf)
            except Exception:
                pass

    # User choices / INI
    choices     = UserChoices()
    user_ini    = Path("userchoices.ini")
    default_ini = Path(__file__).parent / "userchoices.ini"
    ini_path    = user_ini if user_ini.exists() else default_ini
    if not ini_path.exists():
        raise FileNotFoundError(f"userchoices.ini not found at {user_ini} or {default_ini}")
    logger.info("Loading UserChoices from %s", ini_path)
    choices.set_ecm_options(ini_path)
    choices.set_model_options(ini_path)
    choices.set_model_variables(ini_path)
    choices.set_output(ini_path)
    choices.set_model_version("WS")

    # Scenario
    scen = Scenario(catchm=catch, year=year, scenario_nr=scenario_nr, userchoices=choices)
    logger.info("Preparing input files for scenario %d (%d)…", scenario_nr, year)

    scen.composite_landuse = scen.create_composite_landuse()
    scen.cfactor           = scen.create_cfactor()

    if choices.dict_model_options.get("UserProvidedKTC", 1) == 1:
        ktc_arr = scen.create_ktc(
            choices.dict_variables["ktc low"],
            choices.dict_variables["ktc high"],
            choices.dict_variables["ktc limit"],
            choices.dict_model_options["UserProvidedKTC"],
        )
        scen.ktc = ktc_arr
        logger.info("Computed KTC raster")
    else:
        logger.info("Skipping KTC creation (UserProvidedKTC=0)")

    scen.prepare_input_files()
    scen.create_ini_file()
    logger.info("Preprocessing complete: rasters→%s, tables+INI→%s", rd, sd)


if __name__ == "__main__":
    # light console logging if run as a script
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")
    preprocess_all()
