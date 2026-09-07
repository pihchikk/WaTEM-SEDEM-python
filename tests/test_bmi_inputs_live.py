"""Every BMI input that set_value() accepts must actually change the output.

Two inputs used to fail that bar:

- ktc: solver.solve() called select_ktc(cfg, data) unconditionally, so
  set_value('ktc', ...) was silently discarded on every update().
- bulk-density: it advertised the full grid size via get_var_grid()/
  get_grid_size() but was stored as a single-element array, so a grid-shaped
  set_value() raised ValueError: cannot reshape.

Both are fixed in BmiWaTEM.set_value()/update() and solver.solve(). This file
pins that fix so a regression -- or a repeat of it going uncommitted, which is
exactly how it was lost once already -- fails CI immediately.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from watem_sedem import solver as solver_mod
from watem_sedem.bmi_watem import BmiWaTEM

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = str(REPO_ROOT / "config.yaml")


def _sedi_out_total(model: BmiWaTEM) -> float:
    grid = model.get_var_grid("SEDI_OUT")
    buf = np.empty(model.get_grid_size(grid), dtype=model.get_var_type("SEDI_OUT"))
    model.get_value("SEDI_OUT", buf)
    return float(np.nansum(buf))


def _run_with(**set_values: float) -> float:
    """initialize(), optionally set_value() each kwarg (broadcast to the
    grid), update() once, return total SEDI_OUT."""
    m = BmiWaTEM()
    m.initialize(CONFIG_PATH)
    grid_size = m.get_grid_size(m.get_var_grid("Cfactor"))
    for name, value in set_values.items():
        m.set_value(name, np.full(grid_size, value, dtype=np.float32))
    m.update()
    total = _sedi_out_total(m)
    m.finalize()
    return total


# ── bulk-density ─────────────────────────────────────────────────────────
def test_bulk_density_set_value_does_not_raise():
    m = BmiWaTEM()
    m.initialize(CONFIG_PATH)
    grid = m.get_var_grid("bulk-density")
    grid_size = m.get_grid_size(grid)

    # This is the exact call watem_execute.py's _apply_parameters() makes:
    # a grid_size-length array, matching what get_var_grid()/get_grid_size()
    # advertise. It used to raise ValueError: cannot reshape array of size
    # 22523 into shape (1,) because the value was stored as a 1-element array.
    m.set_value("bulk-density", np.full(grid_size, 1600.0, dtype=np.float32))

    buf = np.empty(grid_size, dtype=m.get_var_type("bulk-density"))
    m.get_value("bulk-density", buf)
    assert buf.size == grid_size
    assert np.all(buf == 1600.0)
    m.finalize()


def test_bulk_density_scales_raw_sediment_volume(monkeypatch, tmp_path):
    """On the model's native m3 output, SEDI_OUT must scale as 1/bulk_density
    (mass = volume * density is conserved). kg/t-ha outputs cancel bulk
    density back out by design -- see convert_units() -- so this checks the
    native volume rather than the default kg-per-cell config output, which
    would show no change even with the fix correctly applied.
    """
    # Force m3/m output regardless of what config.yaml's own defaults say.
    orig_convert = solver_mod.convert_units

    def forced_units(cfg, data, *args, **kwargs):
        kwargs["sediment_unit"] = "m3"
        kwargs["erosion_unit"] = "m"
        return orig_convert(cfg, data, *args, **kwargs)

    monkeypatch.setattr(solver_mod, "convert_units", forced_units)

    totals = {}
    for bd in (1300.0, 1600.0):
        m = BmiWaTEM()
        m.initialize(CONFIG_PATH)
        grid_size = m.get_grid_size(m.get_var_grid("bulk-density"))
        m.set_value("bulk-density", np.full(grid_size, bd, dtype=np.float32))
        m.update()
        totals[bd] = _sedi_out_total(m)
        m.finalize()

    assert totals[1300.0] != totals[1600.0], "bulk-density has no effect -- still inert"
    # mass = SEDI_OUT(m3) * bulk_density should be ~invariant across bulk_density
    mass_1300 = totals[1300.0] * 1300.0
    mass_1600 = totals[1600.0] * 1600.0
    assert mass_1300 == pytest.approx(mass_1600, rel=1e-4)


# ── ktc ──────────────────────────────────────────────────────────────────
def test_ktc_set_value_changes_output():
    low = _run_with(ktc=1.0)
    high = _run_with(ktc=500.0)
    assert low != high, "ktc has no effect -- still inert (set_value discarded by select_ktc)"


def test_ktc_untouched_matches_internal_selection():
    """A caller that never calls set_value('ktc', ...) must get exactly the
    old, CLI-matching behaviour: ktc derived internally from Cfactor via
    select_ktc(). This is what keeps test_bmi_matches_cli.py green."""
    m = BmiWaTEM()
    m.initialize(CONFIG_PATH)
    assert "ktc" not in m._externally_set
    m.update()
    assert m._data["ktc_source"] == "internal"
    m.finalize()


def test_ktc_units_are_metres():
    # ktc is a length (select_ktc's docstring: "ktc per cell, in metres"),
    # not the dimensionless "-" the BMI used to declare.
    m = BmiWaTEM()
    m.initialize(CONFIG_PATH)
    assert m.get_var_units("ktc") == "m"
    m.finalize()
