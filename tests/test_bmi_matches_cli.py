"""The BMI and the CLI must be the same model.

They were not. Each carried its own copy of the solve, and the copies drifted:
the CLI moved to Desmet & Govers routing, ktc in metres and tillage while the
BMI stayed on D8 and kept applying the retired ktc_multiplier. On the bundled
external-mode config their WATEREROS sums came out 4.69x apart -- and every
number in docs/REFERENCE_RUN_SETTINGS.md was measured through the CLI while
bmi-runner calls the BMI, so the validated path was not the shipped one.

Both now call watem_sedem.solver. This test is what stops that from happening
again: it asserts they agree to the byte on the same config, so a second copy
of the solve is caught rather than released.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

CONFIGS = ["tests_external/config.yaml", "tests_hybrid/config.yaml"]


def _cli_rasters(config: Path, outdir: Path, mode: str) -> dict[str, np.ndarray]:
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    cfg["output"]["directory"] = str(outdir)
    cfg["output"]["save_rasters"] = True
    cfg_path = outdir / "cli.yaml"
    outdir.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
                        encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, "-m", "watem_sedem.run_watem", "-c", str(cfg_path),
         "--mode", mode, "--save-rasters"],
        cwd=REPO, capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.fail(f"CLI failed:\n{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}")

    out = {}
    for tif in sorted((outdir / f"results_{mode}").glob("*.tif")):
        with rasterio.open(tif) as src:
            out[tif.stem] = src.read(1, masked=True).filled(np.nan).astype(np.float32)
    return out


def _bmi_rasters(config: Path) -> dict[str, np.ndarray]:
    # Imported here so collection does not require bmipy when only the CLI
    # tests are being run.
    from watem_sedem.bmi_watem import BmiWaTEM

    model = BmiWaTEM()
    cwd = Path.cwd()
    try:
        import os
        os.chdir(REPO)                    # config paths are repo-relative
        model.initialize(str(config))
        model.update()
        out = {}
        for name in model.get_output_var_names():
            if name == "epoch_index":
                continue
            grid = model.get_var_grid(name)
            buf = np.empty(model.get_grid_size(grid), dtype=np.float32)
            model.get_value(name, buf)
            out[name] = buf.reshape(model.get_grid_shape(grid, np.empty(2, dtype=int)))
        return out
    finally:
        os.chdir(cwd)


@pytest.mark.parametrize("config", CONFIGS)
def test_bmi_reproduces_cli_bitwise(config, tmp_path):
    cfg_path = HERE / config
    if not cfg_path.exists():
        pytest.skip(f"{config} missing")
    mode = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))["mode"]
    if mode != "external" and not shutil.which("saga_cmd"):
        pytest.skip("saga_cmd not on PATH")

    cli = _cli_rasters(cfg_path, tmp_path / "cli", mode)
    bmi = _bmi_rasters(cfg_path)

    shared = sorted(set(cli) & set(bmi))
    assert shared, f"no comparable rasters: CLI has {sorted(cli)}, BMI has {sorted(bmi)}"
    assert "WATEREROS" in shared

    mismatches = []
    for name in shared:
        a, b = cli[name], bmi[name]
        if a.shape != b.shape:
            mismatches.append(f"  {name}: shape {a.shape} vs {b.shape}")
            continue
        if not np.array_equal(a, b, equal_nan=True):
            diff = np.abs(a - b)
            n = int(np.nansum(a != b))
            mismatches.append(
                f"  {name}: {n} cells differ, max |diff| {np.nanmax(diff):.6g}, "
                f"sums {np.nansum(a):.6g} vs {np.nansum(b):.6g}")

    assert not mismatches, (
        "BMI and CLI have diverged -- they must run the same solver:\n"
        + "\n".join(mismatches))


def test_bmi_exposes_tillage_when_enabled(tmp_path):
    """Tillage outputs must be declared at initialize(), not appear after the
    first update(): a BMI's variable set is static and a caller is entitled to
    read it before stepping."""
    from watem_sedem.bmi_watem import BmiWaTEM

    src = HERE / "tests_external/config.yaml"
    if not src.exists():
        pytest.skip("tests_external/config.yaml missing")
    cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
    cfg["tillage"] = {"enabled": True, "ktil": 600}
    cfg["output"]["directory"] = str(tmp_path)
    path = tmp_path / "tillage.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
                    encoding="utf-8")

    import os
    cwd = Path.cwd()
    try:
        os.chdir(REPO)
        model = BmiWaTEM()
        model.initialize(str(path))
        names = model.get_output_var_names()
        assert "TILLEROS" in names and "TOTALEROS" in names, \
            f"tillage enabled but outputs not declared: {names}"
        model.update()
        grid = model.get_var_grid("TILLEROS")
        buf = np.empty(model.get_grid_size(grid), dtype=np.float32)
        model.get_value("TILLEROS", buf)
        assert np.isfinite(buf).any(), "TILLEROS is entirely nodata"
    finally:
        os.chdir(cwd)
