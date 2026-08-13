"""Byte-for-byte regression check against stored reference rasters.

These are characterisation tests, not correctness tests. They pin what the model
*currently* produces so that any change to it has to be noticed and accepted on
purpose. A failure here does not mean the new numbers are wrong -- it means they
moved, and someone has to say why.

Each reference set carries the exact config that produced it, in
`tests_<mode>/config.yaml`. That is the whole point: the previous references had
no config beside them, `calibration.ktc_multiplier` was later raised from 1 to
25, and every stored raster silently became unreproducible -- CAPACITY was off by
exactly 25x and nothing failed, because nothing read them.

Comparison is on raw bytes. The model is deterministic (no threading, no RNG,
fixed iteration order), so anything short of byte equality is a real change.
Where a run legitimately differs only in the last bit of float32 storage, the
assertion message reports the ULP distance so the reviewer can see the scale.
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

RASTERS = ("WATEREROS", "SEDI_OUT", "CAPACITY")

# `internal` is excluded: it calls pywatemsedem's preprocess_all(), which imports
# pywatemsedem.userchoices -- a module the currently released pywatemsedem does
# not have. The mode cannot run at all here, so there is nothing to pin.
MODES = ("external", "hybrid", "user_watem", "user_dtm")

# Modes that derive DTM covariates need SAGA on PATH. `external` reads
# everything from disk and does not.
NEEDS_SAGA = {"hybrid", "user_watem", "user_dtm"}


def _reference_dir(mode: str) -> Path:
    return HERE / f"tests_{mode}"


def _run(mode: str, outdir: Path) -> Path:
    """Run the real CLI entry point for `mode`, return its results directory."""
    ref_cfg = _reference_dir(mode) / "config.yaml"
    cfg = yaml.safe_load(ref_cfg.read_text(encoding="utf-8"))
    cfg["mode"] = mode
    cfg["output"]["directory"] = str(outdir)
    cfg["output"]["save_rasters"] = True

    cfg_path = outdir / "config.yaml"
    outdir.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
                        encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, "-m", "watem_sedem.run_watem", "-c", str(cfg_path),
         "--mode", mode, "--save-rasters"],
        cwd=REPO, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        pytest.fail(f"{mode} run failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    return outdir / f"results_{mode}"


def _describe_difference(expected: Path, actual: Path) -> str:
    a = rasterio.open(expected).read(1)
    b = rasterio.open(actual).read(1)
    if a.shape != b.shape:
        return f"grid changed: reference {a.shape}, run {b.shape}"

    A, B = a.astype(np.float64), b.astype(np.float64)
    ok = np.isfinite(A) & np.isfinite(B)
    if not np.array_equal(np.isfinite(A), np.isfinite(B)):
        return "nodata mask changed"

    diff = np.abs(A[ok] - B[ok])
    nz = ok & (np.abs(A) > 0)
    rel = (np.abs(A[nz] - B[nz]) / np.abs(A[nz])).max() if nz.any() else 0.0
    ulp = (np.abs(A[nz] - B[nz])
           / np.spacing(np.abs(A[nz]).astype(np.float32)).astype(np.float64)).max() \
        if nz.any() else 0.0
    return (f"{int((A[ok] != B[ok]).sum())} of {int(ok.sum())} cells differ; "
            f"max |diff| {diff.max():.6g}, max relative {rel:.3e}, "
            f"max {ulp:.0f} float32 ULP")


@pytest.mark.parametrize("mode", MODES)
def test_matches_stored_reference(mode, tmp_path):
    ref_dir = _reference_dir(mode)
    if not (ref_dir / "config.yaml").exists():
        pytest.skip(f"no pinned config for {mode}; reference cannot be reproduced")
    if mode in NEEDS_SAGA and not shutil.which("saga_cmd"):
        pytest.skip("saga_cmd not on PATH")

    results = _run(mode, tmp_path / mode)

    mismatches = []
    for name in RASTERS:
        expected = ref_dir / f"{name}.tif"
        actual = results / f"{name}.tif"
        assert actual.exists(), f"{mode}: run produced no {name}.tif"
        if expected.read_bytes() != actual.read_bytes():
            mismatches.append(f"  {name}: {_describe_difference(expected, actual)}")

    assert not mismatches, (
        f"{mode} no longer reproduces tests_{mode}/:\n" + "\n".join(mismatches) +
        "\nIf the change is intended, regenerate the reference set and its "
        "config.yaml in the same commit."
    )


def test_modes_that_read_the_same_inputs_agree(tmp_path):
    """external, hybrid and user_watem all end up loading the same pre-computed
    rasters from data/rasters, so on this dataset they are the same computation
    and must agree to the byte.

    This is here to keep that fact visible. If it ever fails, the modes have
    started to differ -- which may well be correct, but it means the bundled
    dataset no longer exercises them identically and the reference sets need
    revisiting rather than the assertion relaxing."""
    if not shutil.which("saga_cmd"):
        pytest.skip("saga_cmd not on PATH")

    digests = {}
    for mode in ("external", "hybrid", "user_watem"):
        if not (_reference_dir(mode) / "config.yaml").exists():
            pytest.skip(f"no pinned config for {mode}")
        results = _run(mode, tmp_path / mode)
        digests[mode] = (results / "WATEREROS.tif").read_bytes()

    assert digests["external"] == digests["hybrid"], "external and hybrid diverged"
    assert digests["external"] == digests["user_watem"], "external and user_watem diverged"
