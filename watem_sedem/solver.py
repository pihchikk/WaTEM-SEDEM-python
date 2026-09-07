"""The solve, in one place.

`run_watem.main()` and `BmiWaTEM.update()` both call `solve()` and
`convert_units()` here and do nothing model-related of their own. They used to
each carry their own copy, and the copies drifted: the CLI moved to
Desmet & Govers routing, ktc in metres and tillage while the BMI stayed on D8
with the retired `ktc_multiplier`, so the same config produced sums 4.7x apart
depending on which entry point you called. Everything validated in
docs/REFERENCE_RUN_SETTINGS.md went through the CLI; bmi-runner calls the BMI.

Keep it that way: if a new entry point needs the model, it calls these. There is
a test (tests/test_bmi_matches_cli.py) asserting the two agree to the byte, so a
second copy will be caught rather than shipped.
"""

from __future__ import annotations

import logging

import numpy as np

from watem_sedem import mfd
from watem_sedem.lateraldistribution import topo_order, compute_erosion, transport_capacity
from watem_sedem.tillage import tillage_erosion, to_t_ha

logger = logging.getLogger(__name__)

FACTOR_LAYERS = ("LS-factor", "Kfactor", "Cfactor", "Pfactor", "ktc")


def broadcast_scalars(data: dict, shape) -> None:
    """Expand any scalar factor layer to a full grid, in place."""
    for key in FACTOR_LAYERS:
        if key in data and not isinstance(data[key], np.ndarray):
            data[key] = np.full(shape, data[key], dtype=float)


def select_ktc(cfg, data: dict) -> np.ndarray:
    """ktc per cell, in metres.

    WaTEM/SEDEM picks ktc_low or ktc_high by comparing the C factor against
    ktc_limit -- "limit" is a threshold on C, not a coefficient. Cropland C sits
    well above it, so arable catchments run entirely on ktc_high.
    """
    cfac_raw = data["Cfactor"]
    valid = (~np.ma.getmaskarray(cfac_raw) if np.ma.isMaskedArray(cfac_raw)
             else np.ones(np.shape(cfac_raw), dtype=bool))
    cfac = np.asarray(cfac_raw, dtype=float)
    valid = valid & np.isfinite(cfac) & (cfac > -9000)   # nodata fills are not C values

    ktc = np.where(cfac >= cfg.calibration.ktc_limit,
                   cfg.calibration.ktc_high,
                   cfg.calibration.ktc_low).astype(float)
    logger.info("ktc: %.4g m below C=%.3g, %.4g m at or above (%.1f%% of cells high)",
                cfg.calibration.ktc_low, cfg.calibration.ktc_limit,
                cfg.calibration.ktc_high,
                100.0 * np.mean(cfac[valid] >= cfg.calibration.ktc_limit) if valid.any() else 0.0)
    return ktc


def _elevation(data: dict) -> np.ndarray:
    e = data["elevation"]
    return np.asarray(e.filled(np.nan) if np.ma.isMaskedArray(e) else e, dtype=float)


def _sanitise_flow_direction(fdir) -> np.ndarray:
    if np.ma.isMaskedArray(fdir):
        fdir = fdir.filled(-1)
    fdir = np.asarray(fdir).astype(np.int16, copy=True)
    fdir[(fdir < 0) | (fdir > 7)] = -1
    return fdir


def _solve_weighted(cfg, data: dict):
    """Every routing scheme but d8: weights from mfd.weights(), one router.

    Uses the same per-cell RUSLE and transport_capacity() as compute_cell(), so
    the physics has one home and only the routing differs.
    """
    bd = data["bulk-density"]
    res = data["cell_size"]
    area = res ** 2

    LS = np.nan_to_num(np.asarray(data["LS-factor"], dtype=float), nan=0.0,
                       posinf=0.0, neginf=0.0)
    C = np.nan_to_num(np.asarray(data["Cfactor"], dtype=float))
    K = np.nan_to_num(np.asarray(data["Kfactor"], dtype=float))
    P = np.nan_to_num(np.asarray(data["Pfactor"], dtype=float))
    ktc = np.nan_to_num(np.asarray(data["ktc"], dtype=float))
    slope = np.asarray(data["slope"], dtype=float)      # radians
    aspect = np.asarray(data["aspect"], dtype=float)
    R = data["Rfactor"]

    rusle_kg_m2 = R * C * P * K * LS / 1e4
    ero_pot = np.nan_to_num(rusle_kg_m2 * area / bd)

    cap_kg = transport_capacity(LS, K, R, slope, aspect, ktc, res)
    cap_m3 = np.nan_to_num(np.maximum(cap_kg, 0.0) / bd)

    elevation = _elevation(data)
    w, has_out = mfd.weights(elevation, res, scheme=cfg.routing_scheme,
                             aspect=aspect, exponent=cfg.mfd_exponent)
    logger.info("routing: %s%s", cfg.routing_scheme,
                f" (exponent {cfg.mfd_exponent:g})" if cfg.routing_scheme == "holmgren" else "")
    SEDI_IN, SEDI_OUT, WATEREROS = mfd.route(w, has_out, ero_pot, cap_m3, elevation, res)
    return SEDI_IN, SEDI_OUT, WATEREROS, cap_m3


def solve(cfg, data: dict):
    """Run one epoch. Returns (SEDI_IN, SEDI_OUT, WATEREROS, CAPACITY, TILLEROS).

    TILLEROS is None unless cfg.tillage.enabled; it is kg m-2 yr-1 and is not
    routed -- ploughing moves soil across the surface, it does not hand it to
    the flow -- so it stays a separate field.
    """
    broadcast_scalars(data, np.shape(data["elevation"]))
    # ktc is normally derived from Cfactor (select_ktc). A caller that pushed
    # its own ktc through BMI's set_value() (data["ktc_source"] == "external",
    # set by BmiWaTEM.update()) is deliberately overriding that derivation --
    # respect it. The CLI never sets ktc_source, so this is a no-op there.
    if data.get("ktc_source") != "external":
        data["ktc"] = select_ktc(cfg, data)
    data["flow_direction"] = _sanitise_flow_direction(data["flow_direction"])

    if cfg.routing_scheme != "d8":
        SEDI_IN, SEDI_OUT, WATEREROS, CAPACITY = _solve_weighted(cfg, data)
    else:
        logger.info("routing: d8")
        SEDI_IN, SEDI_OUT, WATEREROS, CAPACITY = compute_erosion(
            LS=data["LS-factor"], Kfactor=data["Kfactor"], Cfactor=data["Cfactor"],
            Pfactor=data["Pfactor"], R_factor=data["Rfactor"],
            bulk_density=data["bulk-density"], slope=data["slope"],
            aspect=data["aspect"], ktc=data["ktc"], cell_res=data["cell_size"],
            flow_direction=data["flow_direction"],
            topo=topo_order(data["flow_direction"]),
        )

    TILLEROS = None
    if cfg.tillage.enabled:
        elevation = _elevation(data)
        TILLEROS = tillage_erosion(np.asarray(data["slope"], dtype=float),
                                   np.asarray(data["aspect"], dtype=float),
                                   cfg.tillage.ktil, data["cell_size"],
                                   valid=np.isfinite(elevation))
        area = data["cell_size"] ** 2
        gross = np.nansum(np.abs(TILLEROS)) * area
        net = np.nansum(TILLEROS) * area
        logger.info("tillage: ktil %.4g kg/m, gross %.4g kg/yr moved, net %.3g kg/yr "
                    "(%.2f%% of gross -- should be near zero, it is export across "
                    "the boundary)", cfg.tillage.ktil, gross, net,
                    100.0 * abs(net) / gross if gross else 0.0)

    return SEDI_IN, SEDI_OUT, WATEREROS, CAPACITY, TILLEROS


def convert_units(cfg, data: dict, SEDI_OUT, CAPACITY, WATEREROS, TILLEROS=None,
                  sediment_unit=None, erosion_unit=None):
    """Map the solve's native units onto the configured output units.

    Returns {name: (array, label)}; TILLEROS and TOTALEROS appear only when
    tillage ran.
    """
    cell_area = data["cell_size"] ** 2
    bd = data["bulk-density"]
    sed_mass = bd / 1000.0 * (10000.0 / cell_area)   # m3 -> t/ha
    ero_mass = bd * (10000.0 / 1000.0)               # m  -> t/ha

    su = sediment_unit or cfg.output.sediment_unit
    eu = erosion_unit or cfg.output.erosion_unit

    if su == "m3":
        sed, cap, sed_label = SEDI_OUT, CAPACITY, "m3 per cell"
    elif su == "kg":
        sed, cap, sed_label = SEDI_OUT * bd, CAPACITY * bd, "kg per cell"
    else:
        sed, cap, sed_label = SEDI_OUT * sed_mass, CAPACITY * sed_mass, "t/ha"

    if eu == "m":
        ero, ero_label = WATEREROS, "m"
    elif eu == "mm":
        ero, ero_label = WATEREROS * 1000.0, "mm"
    elif eu == "kg":
        ero, ero_label = WATEREROS * cell_area * bd, "kg per cell"
    else:
        ero, ero_label = WATEREROS * ero_mass, "t/ha"

    out = {"SEDI_OUT": (np.asarray(sed, dtype=np.float32), sed_label),
           "CAPACITY": (np.asarray(cap, dtype=np.float32), sed_label),
           "WATEREROS": (np.asarray(ero, dtype=np.float32), ero_label)}

    if TILLEROS is not None:
        till = to_t_ha(TILLEROS) if eu == "t/ha" else TILLEROS * cell_area
        out["TILLEROS"] = (np.asarray(till, dtype=np.float32), ero_label)
        out["TOTALEROS"] = (np.asarray(ero + np.nan_to_num(till), dtype=np.float32),
                            ero_label)
    return out
