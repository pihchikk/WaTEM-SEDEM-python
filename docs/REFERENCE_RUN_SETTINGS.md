# Settings behind the `erosion_soft_new.tif` reference rasters

The `lom` and `spok` reference catchments ship an `erosion_soft_new.tif` produced
by the original WaTEM/SEDEM desktop software. These are the settings that run
used, supplied by the modeller who produced them (screenshot of the Extra
Options tab plus a written note). Recording them here because nothing in the
shipped data states them, and without them the comparison is unfalsifiable —
every discrepancy can be blamed on an unknown parameter.

## The settings

| Setting | Value | Where it lands in this package |
| --- | --- | --- |
| R-factor | 320 MJ·mm·ha⁻¹·h⁻¹·yr⁻¹ | `defaults.Rfactor: 320` |
| LS | McCool (1987, 1989) | `ls_method: pascal_mccool1987` |
| Nearing slope-length exponent | McCool (1987,1989), rill=interrill | part of the same branch |
| kTc Low / High / Limit | 75 / 250 / 0.1 | see below |
| Tillage transport coef (ktil) | 600 kg/m | not modelled here |
| Bulk density | 1350 kg/m³ | `defaults.bulk-density: 1350` |
| Output units | Intensity, t/ha·yr | `output.erosion_unit: "t/ha"` |

## Three traps in the input rasters

**R is stored pre-divided by 10000.** `R_const.tif` holds 0.032 (lom) and 0.029
(spok) because that is how the desktop software wants it typed in. The physical
values are 320 and 290. Feeding 0.032 straight into `defaults.Rfactor` costs a
factor of 10⁴ and is exactly what we did at first.

**K is stored as the plain integer.** `K_const.tif` holds 67 and 35 in
kg·h·MJ⁻¹·mm⁻¹, no scaling. `compute_cell()` divides Kfactor by 1000 internally,
which combined with R in the units above yields t/ha — so K needs no adjustment,
despite looking like it does. (`K_const.tif.vat.dbf` is a value/count histogram,
not a lookup table; the stored number is the K value.)

**kTc Limit is a threshold on the C-factor, not a coefficient.** C below 0.1 uses
kTc Low (75), C at or above it uses kTc High (250). Both catchments have C = 0.4
and 0.45, so the reference run used 250 throughout — not 0.1. Since
`compute_cell()` divides ktc by 1000, that is `defaults.ktc: 250` with
`ktc_multiplier: 1`.

Note that the package's own defaults (`ktc: 1`, `ktc_multiplier: 25`) give an
effective 0.025 against the software's 0.25 — an order of magnitude apart. The
defaults are left alone here; this file is a record, not a migration.

## What agreement looks like on these settings

Both catchments, `routing_scheme: mfd` with `mfd_exponent: 3.0`, compared cell by
cell against the reference over its own footprint. Values in t/ha·yr, negative
for erosion, no fitted scale factor anywhere:

| | median model | median reference | median ratio | Spearman rho | sign agreement |
| --- | --- | --- | --- | --- | --- |
| lom, d8 | −9.04 | −11.45 | 1.22 | 0.776 | 0.962 |
| lom, mfd | −10.88 | −11.45 | 1.12 | 0.865 | 0.964 |
| spok, d8 | −5.70 | −7.55 | 1.19 | 0.845 | 0.965 |
| spok, mfd | −7.15 | −7.55 | 1.04 | 0.925 | 0.967 |

Catchment totals run high: −3495 vs −2875 on lom, −16492 vs −11416 on spok. The
medians agree to 4–12% while the sums do not, which locates the remaining
disagreement in the tails — we put more mass in the most-eroding cells than the
software does.

## Correction to an earlier result

Before these settings were known, a sweep over LS method / ktc / routing exponent
picked `ls_method: wischmeier` with an effective ktc of 0.015, scoring top-decile
IoU 0.729 (lom) and 0.741 (spok). That sweep ran with R = 0.032 and output in kg
per cell — both wrong — so it was fitting an artefact. On the correct settings
the matching LS formulation is `pascal_mccool1987`, which is what the software
actually uses: Desmet & Govers (1996) L with McCool's variable exponent
`m = B/(B+1)`, `B = (sinθ/0.0896)/(3·sinθ^0.8 + 0.56)`, over McCool's S factor
(`10.8·sinθ + 0.03` at ≤9% gradient, `16.8·sinθ − 0.50` above). Do not read the
IoU figures from that sweep as a result.

The `mccool` branch in `raster_calculations.compute_ls()` is *not* the right one
despite the name: its L is `(A/22.13)**0.4` with A an upslope **area** in m² and
a fixed exponent, where McCool's L is `(Xh/22.13)**m` with Xh a slope **length**
and m variable.
