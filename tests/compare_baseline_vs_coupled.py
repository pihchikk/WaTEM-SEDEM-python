#!/usr/bin/env python3
"""
Runs WaTEM-SEDEM twice on the same catchment/config and diffs outputs:
  (A) baseline  — coupling_mode="static": current default scalar Cfactor/Rfactor
                   from config.yaml, exactly as today.
  (B) coupled   — coupling_mode="epoch": Cfactor/Rfactor derived from a real (or,
                   absent one, a documented synthetic stand-in) AquaCrop+LISEM
                   daily driver series via src/coupling_drivers.py.

Same DEM, K-factor, P-factor, ktc, cell size and flow topology are used in both
runs (Step 1's initialize()/update() split means only update() is re-executed for
each variant, and Step 3's flow-direction ingestion is left at whatever
`flow_direction_source` config.yaml specifies for both arms), so the only
difference under test is the driver-derivation path.

This repo ships no real AquaCrop/LISEM coupled-run output, so by default this
script generates a synthetic daily canopy-cover / rainfall series with a
realistic AquaCrop-style growth curve and storm rainfall (seed fixed for
reproducibility) — this is clearly labeled `synthetic` in summary.json.
Pass --canopy-csv/--rain-csv to use a real coupled run's daily series instead.

Output artifacts land in tests/tests_epoch_coupling/: SEDI_OUT/CAPACITY/WATEREROS
diff rasters (GeoTIFF), maps.png (before/after/diff panel), summary.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import rasterio

from watem_sedem import coupling_drivers as cd
from watem_sedem import data_loader
from watem_sedem.bmi_watem import BmiWaTEM

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTDIR = REPO_ROOT / "tests" / "tests_epoch_coupling"


# ── synthetic driver series (documented stand-in for a real coupled run) ──────
def synthetic_daily_series(n_days: int = 120, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """AquaCrop-style canopy cover growth/senescence curve + gamma-distributed
    intermittent storm rainfall, for demonstrating the coupling path when no
    real AquaCrop+LISEM coupled-run output is available. NOT a calibrated
    crop/climate scenario — for structural testing of the coupling only.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_days, dtype=float)
    growth = 90.0 / (1.0 + np.exp(-0.15 * (t - 30)))
    decay = 1.0 / (1.0 + np.exp(0.12 * (t - 90)))
    canopy = np.clip(growth * decay, 0.0, 100.0)

    storm_days = rng.random(n_days) < 0.22
    rain = np.where(storm_days, rng.gamma(shape=0.6, scale=9.0, size=n_days), 0.0)
    return canopy, rain


# ── running the model ──────────────────────────────────────────────────────
def run_static(config_path: str) -> tuple[dict[str, np.ndarray], BmiWaTEM]:
    m = BmiWaTEM()
    m.initialize(config_path)
    m.update()
    out = {k: m.get_value_ptr(k).copy() for k in ("SEDI_OUT", "CAPACITY", "WATEREROS")}
    return out, m


def run_coupled(
    config_path: str,
    canopy_daily: np.ndarray,
    rain_daily: np.ndarray,
    cfactor_override: float | None = None,
    rfactor_override: float | None = None,
) -> tuple[dict[str, np.ndarray], BmiWaTEM, float, float]:
    m = BmiWaTEM()
    m.initialize(config_path)

    cfactor = cfactor_override if cfactor_override is not None else cd.canopy_series_to_cfactor(canopy_daily)
    rfactor = rfactor_override if rfactor_override is not None else cd.rainfall_series_to_rfactor(rain_daily)

    shape = m._shape
    m.set_value("Cfactor", np.full(shape, cfactor, dtype=np.float32))
    m.set_value("Rfactor", np.full(shape, rfactor, dtype=np.float32))
    m.update()

    out = {k: m.get_value_ptr(k).copy() for k in ("SEDI_OUT", "CAPACITY", "WATEREROS")}
    return out, m, cfactor, rfactor


# ── summary metrics ────────────────────────────────────────────────────────
def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation without a scipy dependency."""
    a = a.ravel()
    b = b.ravel()
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    if a.size < 2:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def scalar_summary(out: dict[str, np.ndarray], model: BmiWaTEM) -> dict:
    watereros = out["WATEREROS"]
    sedi_out = out["SEDI_OUT"]
    fdir = model._flow_direction

    total_erosion = float(-np.nansum(watereros[watereros < 0]))
    total_deposition = float(np.nansum(watereros[watereros > 0]))

    # Outlet flux ≈ sediment leaving cells with no downstream target (D8 sinks).
    outlet_mask = fdir == -1
    outlet_flux = float(np.nansum(sedi_out[outlet_mask])) if outlet_mask.any() else float(np.nansum(sedi_out))
    potential_erosion = total_erosion if total_erosion > 0 else np.nan
    sdr = outlet_flux / potential_erosion if potential_erosion and potential_erosion > 0 else float("nan")

    return {
        "total_erosion": total_erosion,
        "total_deposition": total_deposition,
        "mean_sedi_out": float(np.nanmean(sedi_out)),
        "median_sedi_out": float(np.nanmedian(sedi_out)),
        "sediment_delivery_ratio": sdr,
        "erosion_unit": model._var_units["WATEREROS"],
        "sediment_unit": model._var_units["SEDI_OUT"],
    }


def pct_change(a: float, b: float) -> float:
    if a == 0 or not np.isfinite(a):
        return float("nan")
    return float((b - a) / abs(a) * 100.0)


def _rel_or_abs(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def write_raster(path: Path, arr: np.ndarray, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    m = dict(meta)
    m.update({"driver": "GTiff", "dtype": arr.dtype, "count": 1})
    with rasterio.open(path, "w", **m) as dst:
        dst.write(arr, 1)


def plot_panel(a: np.ndarray, b: np.ndarray, diff: np.ndarray, label: str, out_png: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    vmin, vmax = np.nanpercentile(np.concatenate([a.ravel(), b.ravel()]), [2, 98])

    im0 = axes[0].imshow(a, vmin=vmin, vmax=vmax, interpolation="none")
    axes[0].set_title(f"{label} (A) baseline/static"); axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(b, vmin=vmin, vmax=vmax, interpolation="none")
    axes[1].set_title(f"{label} (B) coupled/epoch"); axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    dmax = np.nanpercentile(np.abs(diff), 98) or 1.0
    im2 = axes[2].imshow(diff, cmap="RdBu_r", vmin=-dmax, vmax=dmax, interpolation="none")
    axes[2].set_title(f"{label} diff (B - A)"); axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-c", "--config", default=str(REPO_ROOT / "config.yaml"))
    p.add_argument("--canopy-csv", help="1-column CSV of daily canopy cover (%) from a real coupled run")
    p.add_argument("--rain-csv", help="1-column CSV of daily rainfall (mm) from a real coupled run")
    p.add_argument("--n-days", type=int, default=120)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", default=str(OUTDIR))
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    synthetic = args.canopy_csv is None or args.rain_csv is None
    if args.canopy_csv and args.rain_csv:
        canopy = np.loadtxt(args.canopy_csv, delimiter=",")
        rain = np.loadtxt(args.rain_csv, delimiter=",")
    else:
        canopy, rain = synthetic_daily_series(args.n_days, args.seed)

    cfg = data_loader.load_and_validate_config(args.config)
    baseline_cfactor = float(cfg.defaults.get("Cfactor", 0.37))
    baseline_rfactor = float(cfg.defaults.get("Rfactor", 350.0))

    # (A) baseline
    out_a, model_a = run_static(args.config)
    meta = model_a._data["meta"]

    # (B) coupled
    out_b, model_b, cfactor_b, rfactor_b = run_coupled(args.config, canopy, rain)

    summary_a = scalar_summary(out_a, model_a)
    summary_b = scalar_summary(out_b, model_b)

    scalar_table = {}
    for key in summary_a:
        if isinstance(summary_a[key], (int, float)):
            scalar_table[key] = {
                "A_baseline": summary_a[key],
                "B_coupled": summary_b[key],
                "pct_change": pct_change(summary_a[key], summary_b[key]),
            }
        else:
            scalar_table[key] = {"A_baseline": summary_a[key], "B_coupled": summary_b[key]}

    spearman_watereros = _spearman(out_a["WATEREROS"], out_b["WATEREROS"])

    # Spatial diff rasters
    for name in ("WATEREROS", "CAPACITY"):
        diff = out_b[name] - out_a[name]
        write_raster(outdir / f"{name}_diff.tif", diff.astype(np.float32), meta)

    for arm, out in (("A", out_a), ("B", out_b)):
        for name, arr in out.items():
            write_raster(outdir / f"{name}_{arm}.tif", arr.astype(np.float32), meta)

    # Sensitivity breakdown: revert one driver at a time to the baseline value
    sensitivity = {}
    out_c_only, _, cf_c, rf_c = run_coupled(
        args.config, canopy, rain, cfactor_override=None, rfactor_override=baseline_rfactor
    )
    out_r_only, _, cf_r, rf_r = run_coupled(
        args.config, canopy, rain, cfactor_override=baseline_cfactor, rfactor_override=None
    )

    def _delta_total_erosion(out: dict[str, np.ndarray]) -> float:
        we = out["WATEREROS"]
        return float(-np.nansum(we[we < 0]))

    total_a = summary_a["total_erosion"]
    total_b = summary_b["total_erosion"]
    total_c = _delta_total_erosion(out_c_only)  # Cfactor coupled, Rfactor baseline
    total_r = _delta_total_erosion(out_r_only)  # Rfactor coupled, Cfactor baseline

    sensitivity["cfactor_only"] = {
        "description": "Coupled Cfactor, baseline (static) Rfactor",
        "total_erosion": total_c,
        "pct_change_from_baseline": pct_change(total_a, total_c),
        "share_of_total_coupled_change": (
            (total_c - total_a) / (total_b - total_a) if total_b != total_a else float("nan")
        ),
    }
    sensitivity["rfactor_only"] = {
        "description": "Coupled Rfactor, baseline (static) Cfactor",
        "total_erosion": total_r,
        "pct_change_from_baseline": pct_change(total_a, total_r),
        "share_of_total_coupled_change": (
            (total_r - total_a) / (total_b - total_a) if total_b != total_a else float("nan")
        ),
    }
    sensitivity["flow_topology_only"] = {
        "description": (
            "Reruns (B) with A's flow topology instead of a reused/external one. "
            "Both arms in this harness use WaTEM-SEDEM's own internal D8 derivation "
            "(flow_direction_source=internal in config.yaml), so there is no topology "
            "difference to attribute here; set flow_direction_source=external with "
            "external_flow_direction_path to exercise this component."
        ),
        "skipped": True,
    }

    plot_panel(out_a["WATEREROS"], out_b["WATEREROS"], out_b["WATEREROS"] - out_a["WATEREROS"],
               "WATEREROS", outdir / "maps.png")

    summary = {
        "config": args.config,
        "coupling_mode_A": "static",
        "coupling_mode_B": "epoch",
        "driver_series_source": "synthetic" if synthetic else "user_provided",
        "rfactor_source": "derived_daily_depth (Renard & Freimund 1994 approximation — see docs/COUPLING_VARS_LEDGER.md §3)",
        "drivers": {
            "A_cfactor": baseline_cfactor,
            "A_rfactor": baseline_rfactor,
            "B_cfactor": cfactor_b,
            "B_rfactor": rfactor_b,
        },
        "scalar_summary": scalar_table,
        "spearman_correlation_watereros": spearman_watereros,
        "spatial_diff_rasters": [
            _rel_or_abs(outdir / "WATEREROS_diff.tif"),
            _rel_or_abs(outdir / "CAPACITY_diff.tif"),
        ],
        "sensitivity_breakdown": sensitivity,
        "maps_png": _rel_or_abs(outdir / "maps.png"),
    }

    with open(outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)

    model_a.finalize()
    model_b.finalize()

    print(json.dumps(summary["scalar_summary"], indent=2, default=float))
    print(f"\nSpearman correlation (A vs B, WATEREROS): {spearman_watereros:.4f}")
    print(f"Artifacts written to: {outdir}")


if __name__ == "__main__":
    main()
