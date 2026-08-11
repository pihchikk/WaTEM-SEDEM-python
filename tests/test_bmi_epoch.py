"""Tests for the epoch-coupling refactor of BmiWaTEM and coupling_drivers.

Module-scoped fixture: one BmiWaTEM.initialize() per module (not per test),
matching the existing project convention.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

from watem_sedem import bmi_watem
from watem_sedem import coupling_drivers as cd
from watem_sedem import lateraldistribution as ld
from watem_sedem.bmi_watem import BmiWaTEM

CONFIG_PATH = str(REPO_ROOT / "config.yaml")


@pytest.fixture(scope="module")
def model() -> BmiWaTEM:
    m = BmiWaTEM()
    m.initialize(CONFIG_PATH)
    yield m
    m.finalize()


# ── Step 1: initialize()/update() split ────────────────────────────────────
def test_initialize_caches_topology_and_static_rasters(model: BmiWaTEM):
    assert model._topo is not None
    assert model._flow_direction is not None
    assert model._data  # static rasters cached on self._data


def test_update_does_not_rerun_topo_order(monkeypatch):
    calls = {"n": 0}
    real_topo_order = ld.topo_order

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real_topo_order(*args, **kwargs)

    monkeypatch.setattr(bmi_watem, "topo_order", counting)

    m = BmiWaTEM()
    m.initialize(CONFIG_PATH)
    assert calls["n"] == 1

    m.update()
    m.update()
    m.update()
    assert calls["n"] == 1, "update() must not re-run topo_order()"
    m.finalize()


def test_update_with_changed_drivers_produces_different_output(model: BmiWaTEM):
    model.update()
    sedi_1 = model.get_value_ptr("SEDI_OUT").copy()
    epoch_1 = int(model.get_value_ptr("epoch_index").flat[0])

    shape = model._shape
    model.set_value("Cfactor", np.full(shape, 0.9, dtype=np.float32))
    model.set_value("Rfactor", np.full(shape, 900.0, dtype=np.float32))
    model.update()
    sedi_2 = model.get_value_ptr("SEDI_OUT").copy()
    epoch_2 = int(model.get_value_ptr("epoch_index").flat[0])

    assert epoch_2 == epoch_1 + 1
    assert not np.allclose(sedi_1, sedi_2, equal_nan=True)


def test_epoch_index_output_var_present(model: BmiWaTEM):
    assert "epoch_index" in model.get_output_var_names()
    assert model.get_var_grid("epoch_index") == 0


# ── Step 3: flow-topology mismatch ─────────────────────────────────────────
def test_external_flow_direction_shape_mismatch_raises(tmp_path):
    import rasterio
    from rasterio.transform import from_origin

    bad_path = tmp_path / "bad_flow_direction.tif"
    arr = np.zeros((5, 5), dtype=np.int16)
    transform = from_origin(0, 0, 20, 20)
    with rasterio.open(
        bad_path, "w", driver="GTiff", height=5, width=5, count=1,
        dtype="int16", crs="EPSG:32637", transform=transform,
    ) as dst:
        dst.write(arr, 1)

    import yaml
    with open(CONFIG_PATH) as f:
        cfg_dict = yaml.safe_load(f)
    cfg_dict["flow_direction_source"] = "external"
    cfg_dict["external_flow_direction_path"] = str(bad_path)
    override_path = tmp_path / "config_external.yaml"
    with open(override_path, "w") as f:
        yaml.safe_dump(cfg_dict, f)

    m = BmiWaTEM()
    with pytest.raises(ValueError, match="shape"):
        m.initialize(str(override_path))


# ── Step 2: coupling_drivers unit tests (hand-computed references) ─────────
def test_canopy_series_to_cfactor_flat_mean():
    canopy = np.array([0.0, 50.0, 100.0])
    cfactor = cd.canopy_series_to_cfactor(canopy)
    assert cfactor == pytest.approx(0.5)


def test_canopy_series_to_cfactor_weighted():
    canopy = np.array([0.0, 100.0])
    weights = np.array([1.0, 3.0])
    # SLR = [1.0, 0.0]; weighted mean = (1*1 + 0*3) / 4 = 0.25
    cfactor = cd.canopy_series_to_cfactor(canopy, stage_weights=weights)
    assert cfactor == pytest.approx(0.25)


def test_canopy_series_to_cfactor_clips_to_unit_range():
    canopy = np.array([-10.0, 110.0])
    cfactor = cd.canopy_series_to_cfactor(canopy)
    assert 0.0 <= cfactor <= 1.0


def test_rainfall_series_to_rfactor_hand_computed():
    # 20mm (<=76.2 branch) and 80mm (>76.2 branch) are erosive; 5mm is not (<12.7mm).
    rain = np.array([5.0, 20.0, 80.0])
    expected = 0.276 * 20.0**1.81 + 0.2483 * 80.0**2.2
    rfactor = cd.rainfall_series_to_rfactor(rain)
    assert rfactor == pytest.approx(expected, rel=1e-9)


def test_rainfall_series_to_rfactor_excludes_subthreshold_days():
    rain = np.array([1.0, 5.0, 10.0])  # all below 12.7mm threshold
    assert cd.rainfall_series_to_rfactor(rain) == 0.0


def test_event_rfactor_matches_daily_formula():
    assert cd.event_rfactor(20.0, 2.0) == pytest.approx(0.276 * 20.0**1.81)
    assert cd.event_rfactor(80.0, 5.0) == pytest.approx(0.2483 * 80.0**2.2)
    assert cd.event_rfactor(5.0, 1.0) == 0.0


def test_soil_moisture_to_ktc_multiplier_bounds():
    theta = np.array([0.0, 0.2, 1.0])
    theta_fc = np.array([0.3, 0.3, 0.3])
    mult = cd.soil_moisture_to_ktc_multiplier(theta, theta_fc)
    assert np.all(mult >= 0.5) and np.all(mult <= 1.5)


# ── Step 7: comparison harness smoke test ───────────────────────────────────
def test_comparison_harness_smoke(tmp_path):
    outdir = tmp_path / "epoch_coupling_smoke"
    script = REPO_ROOT / "tests" / "compare_baseline_vs_coupled.py"
    result = subprocess.run(
        [sys.executable, str(script), "-c", CONFIG_PATH, "--outdir", str(outdir), "--n-days", "60"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stderr

    summary_path = outdir / "summary.json"
    assert summary_path.exists()
    with open(summary_path) as f:
        summary = json.load(f)

    expected_keys = {
        "config", "coupling_mode_A", "coupling_mode_B", "driver_series_source",
        "rfactor_source", "drivers", "scalar_summary", "spearman_correlation_watereros",
        "spatial_diff_rasters", "sensitivity_breakdown", "maps_png",
    }
    assert expected_keys.issubset(summary.keys())
    assert (outdir / "maps.png").exists()
    assert (outdir / "WATEREROS_diff.tif").exists()
