# Coupling Variables Ledger — WaTEM-SEDEM Epoch Coupling

Mirrors the format used in `aquacrop-bmi/docs/COUPLING_BALANCE_VARS_LEDGER.md` and
`lisem-bmi-babel/bmi_lisem/docs/COUPLING_VARS_LEDGER.md`. This document tracks the
BMI variables introduced or changed by the epoch-coupling work, the standard-name
targets they map to, and the modeling caveats that come with running WaTEM-SEDEM
downstream of a daily AquaCrop+LISEM loop.

## 1. Division of labor: LISEM vs. WaTEM-SEDEM

| Component | Temporal scale | Physical basis | Runs inside the daily BMI loop? |
|---|---|---|---|
| **LISEM** | Event-scale (single storm, sub-daily time steps) | Physically-based (kinematic wave overland flow + detachment/transport) | Yes — one of the daily-coupled models |
| **WaTEM-SEDEM** | Seasonal / annual / long-term epoch | Empirical (RUSLE erosion potential + transport-capacity redistribution) | No — called once per epoch, consuming aggregated drivers |

WaTEM-SEDEM has no internal time integration: `SEDI_IN` is reset and the flow
topology is (re-)established on every solve. It is not a per-day component and
must not be advanced inside the same daily loop as AquaCrop/LISEM. Instead it is
driven as an **epoch component**: one `update()` per growing season (default,
"epoch" mode) or, optionally, one `update()` per identified rainfall event
("event" mode) — see §4.

**Running WaTEM-SEDEM and LISEM over the same storm without care double-counts
that storm's erosion**: LISEM will already have produced a physically-based
erosion estimate for the event, and a WaTEM-SEDEM "event mode" run over the same
window produces an independent, empirical estimate for the same soil loss. The
only place these are intentionally run side-by-side is the explicit event-mode
comparison described in `tests/compare_baseline_vs_coupled.py`, where the intent
is exactly to compare the two estimates — not to sum them. In any other event-mode
run, `disable_if_lisem_covers_event` (default `true`) skips the WaTEM-SEDEM solve
for windows already covered by an active LISEM event.

Seasonal-epoch mode does not have this problem: it aggregates over a whole
growing season, a scale LISEM's per-storm accounting never claims to cover, so
the two are complementary rather than overlapping.

## 2. New / changed BMI variables

| Variable (bare name, current) | Standard name target (`object__property`) | Grid | Units | Notes |
|---|---|---|---|---|
| `epoch_index` | `model__epoch_index` | node, grid 0 | `-` (integer count) | New. Increments once per `update()` call; exposed for the comparison harness and any future multi-epoch degradation loop. |
| `SEDI_OUT` | `sediment__outflow_mass` (or `_volume`, unit-dependent — see `output.sediment_unit`) | node, grid 0 | `m3`/`kg`/`t ha-1` (config) | Unchanged interface; now recomputed on every `update()` instead of only in `initialize()`. |
| `WATEREROS` | `soil__erosion_deposition_rate` | node, grid 0 | `m`/`mm`/`kg`/`t ha-1` (config) | Unchanged interface; sign convention unchanged (negative = erosion). |
| `CAPACITY` | `sediment__transport_capacity` | node, grid 0 | matches `SEDI_OUT` units | Unchanged interface. |
| `Cfactor` (input) | `land__cover_management_factor` | node, grid 0 | `-` | Now mutable per-epoch via `set_value("Cfactor", ...)` before `update()`; seasonal value derived by `coupling_drivers.canopy_series_to_cfactor()`. |
| `Rfactor` (input) | `rainfall__erosivity_factor` | node, grid 0 (or scalar) | `MJ·mm/(ha·h·yr)` | Now mutable per-epoch; derived by `coupling_drivers.rainfall_series_to_rfactor()` (seasonal) or `event_rfactor()` (event mode). Flagged as the largest source of coupling uncertainty — see §3. |
| `ktc` (input) | `sediment__transport_coefficient` | node, grid 0 | `-` | Static by default; optionally scaled by `coupling_drivers.soil_moisture_to_ktc_multiplier()` — experimental, off by default (§5). |
| `flow_direction` (input) | `land_surface_water__flow_direction` | node, grid 0 | D8 code (-1 = no-flow) | New source path: `flow_direction_source: external` ingests a pre-routed grid (e.g. LISEM's) instead of deriving an independent D8 grid — see §6. |

The bare names (`Cfactor`, `SEDI_OUT`, etc.) remain the actual `get_value`/
`set_value` keys for backward compatibility with existing scripts (`run_watem.py`,
existing tests); the standard-name column above is the mapping convention used
when this component is registered alongside `aquacrop-bmi` and `lisem-bmi-babel`
in a multi-model coupling framework, matching their `object__property` style.

## 3. R-factor derivation — documented source of uncertainty

`coupling_drivers.rainfall_series_to_rfactor()` (seasonal) and `event_rfactor()`
(per-storm) both estimate EI30 from daily/event **rainfall depth only**, using the
Renard & Freimund (1994) daily approximation:

```
EI30 = 0.276 * P^1.81   (P ≤ 76.2 mm)
EI30 = 0.2483 * P^2.2   (P > 76.2 mm)
```

This has no sub-daily intensity information, so a short intense convective storm
and a long gentle rain of identical total depth get the same EI30 even though
their true erosivity differs substantially. **Any run using a coupling-derived
R-factor (`coupling_mode: epoch` or `event`) must be treated as carrying this
uncertainty** — it is not equivalent to a measured/pluviograph-derived R-factor,
and should be labeled as such wherever results are reported (e.g. in
`summary.json` from the comparison harness, which tags `rfactor_source:
"derived_daily_depth"` vs. `"static_config"`).

## 4. Coupling modes

- **`coupling_mode: static`** (default) — unchanged current behavior: scalar/raster
  `Cfactor`/`Rfactor` from `config.yaml` `defaults`, used as-is. This is the (A)
  baseline arm of the comparison harness.
- **`coupling_mode: epoch`** — one `update()` per AquaCrop growing season.
  `Cfactor` from `canopy_series_to_cfactor()` over the season's daily canopy
  cover; `Rfactor` from `rainfall_series_to_rfactor()` over the same period. This
  is the (B) coupled arm of the comparison harness and the primary coupling
  target of this work.
- **`coupling_mode: event`** — one `update()` per rainfall event (contiguous days
  above a depth threshold in the LISEM/AquaCrop daily rain series). Uses
  `event_rfactor()` and canopy cover *at event time* rather than a seasonal
  aggregate. Lower priority; primarily useful for comparing against LISEM's own
  per-event erosion output on the same storms (see §1's double-counting caveat).

## 5. Experimental: soil-moisture-scaled `ktc`

`coupling_drivers.soil_moisture_to_ktc_multiplier()` optionally scales the static
`ktc` raster by antecedent soil moisture (AquaCrop `soil__moisture` or LISEM
`domain_soil_water_storage__volume` relative to field capacity), bounded to
0.5–1.5×. **This is not part of standard WaTEM-SEDEM** — there is no published
calibration for it. It is off by default and should only be enabled for
exploratory sensitivity work, flagged as experimental in any output metadata.

## 6. Flow-topology reuse precondition

When LISEM has already routed the same catchment, WaTEM-SEDEM's own independent
D8 derivation (`preprocess_watem.py` / `compute_dtm.py`) can disagree with
LISEM's routing, causing sediment to be transported inconsistently between the
two models over the same domain. Setting `flow_direction_source: external` (with
`external_flow_direction_path` pointing at LISEM's routed grid, or a shared
preprocessing output) makes WaTEM-SEDEM ingest that topology instead of computing
its own. Grid shape and cell size are validated against the WaTEM-SEDEM grid on
`initialize()`; a mismatch raises immediately rather than silently
reprojecting/reshaping. **This is a required coupling precondition, not an
optional nicety, whenever both models cover the same catchment** — mismatched
topology is a correctness issue, not a style choice.
