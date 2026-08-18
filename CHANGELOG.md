# Changelog

## 0.2.0

Distribution renamed to **`bmi-WaTEM-SEDEM`** (was `watem-sedem-bmi`), matching
the trusted publisher registered on TestPyPI. The import name is unchanged:
`import watem_sedem`.

Aligns the model with the WaTEM/SEDEM desktop software it is meant to reproduce,
after obtaining that software's own run settings from the modeller who produced
the reference rasters. Full account, including the settings themselves and the
traps in the input rasters, in `docs/REFERENCE_RUN_SETTINGS.md`.

Against the software's output on three catchments — `lom`, `spok`, `lokna` —
with every parameter taken from its settings and **nothing fitted**:

| | median model | median reference | ratio | Spearman rho | sign agreement |
| --- | --- | --- | --- | --- | --- |
| lom | −11.79 | −11.45 | 1.00 | 0.945 | 0.986 |
| spok | −7.38 | −7.55 | 1.00 | 0.979 | 0.992 |
| lokna | −4.20 | −4.06 | 1.01 | 0.910 | 0.955 |

`lokna` was run blind: the settings were already fixed on the other two.

### Breaking

- **ktc is a length in metres**, not a dimensionless number. `calibration` now
  carries `ktc_low: 75`, `ktc_high: 250`, `ktc_limit: 0.1`; the limit is a
  threshold on the C factor deciding which of the two applies, not a
  coefficient. `ktc_multiplier` is still accepted so old configs load, but is
  no longer applied.
- **Default routing is `desmet_govers`**, not D8. The software spreads flow
  across directions; running all the plausible schemes against its output
  identified Desmet & Govers (1996) aspect decomposition on every metric on both
  tuning catchments. `holmgren` (Quinn/Freeman at exponent 1/1.1) and `d8`
  remain available via `routing_scheme`.
- **Transport capacity is dimensionally different.** It previously carried cell
  area twice and took the flow-width correction from slope rather than aspect,
  leaving "capacity" in m⁵. Existing results will not reproduce.
- `ls_method` is a config key, defaulting to the previous hardcoded
  `pascal_mccool1987`.

### Added

- Tillage erosion (`tillage.enabled`, default off) — WaTEM's "T". Flux
  proportional to gradient along the aspect, erosion its negative divergence, so
  it strips convexities and fills hollows. Writes `TILLEROS` and `TOTALEROS`.
- `routing_scheme`, `mfd_exponent`, `ls_method`, `tillage` config keys.
- Regression suite pinning the reference rasters byte for byte, each with the
  config that produced it.
- `tests/test_bmi_matches_cli.py`, asserting the two entry points agree to the
  byte.

### Fixed

- **The BMI and the CLI were different models.** Each carried its own copy of
  the solve and they drifted; on the bundled config their WATEREROS sums came
  out 4.69× apart. Both now call `watem_sedem/solver.py`.
- D8 code 0 — flow due north — was being discarded as invalid, turning ~6% of
  cells into artificial sinks that swallowed all sediment routed into them.
- The lazy `pywatemsedem` import was unreachable from the functions that needed
  it (module `__getattr__` does not fire for bare global lookups), so every
  SAGA-backed mode raised `NameError`.
- `compute_slope_length()` called SAGA's "Watershed Basins" tool and silently
  fell back to an approximation on every run. Nothing reads slope length at
  present, but the values differ threefold.
- `compute_cell()` applied `deg2rad` to a slope raster already in radians.

### Known limitations

- **Tillage erosion is unvalidated.** The reference rasters contain water
  erosion only, confirmed with the modeller, so there is nothing to compare it
  against. It is checked by invariant instead — mass conservation to a relative
  1e-9, plus the expected behaviour on cones, bowls and planes — and left off by
  default. Validating it needs a reference run with tillage enabled.
- **`internal` mode does not run.** `preprocess_all()` is written against the
  pywatemsedem 0.x API; the installed 1.2 has a different one. It raises a
  `NotImplementedError` explaining this and pointing at `hybrid`/`external`.
  There is also no data bundled to test a port against.
- **No parcel connectivity, roads, rivers or grass strips.** The desktop
  software routes sediment across parcel boundaries with trapping; this does
  not. On `lom`, 1 of the 36 largest reference cells still carries the wrong
  sign, and on `spok` 3 of 87 — all in the same direction, the software
  depositing in the talweg where this does not.
- **`lokna` is the weakest of the three** and the only one whose reference grid
  is not a window of its own DEM grid, so comparing it requires resampling. Its
  deposition-cell count overshoots by about half; the other two do not.

## 0.1.0

Initial packaging: `watem_sedem` as a real package, pip-installable, with the
SAGA dependency made optional.
