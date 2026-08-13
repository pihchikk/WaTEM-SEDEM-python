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
| lokna, d8 | −3.63 | −4.06 | 1.10 | 0.873 | 0.949 |
| lokna, mfd | −3.97 | −4.06 | 1.05 | 0.896 | 0.949 |

`lokna` arrived after the settings above were already fixed on `lom` and
`spok`, and was run without touching them -- a different landscape, a 30 m SRTM
DEM instead of a 20 m one, its own R/C/K, and 114862 compared cells against
~1000-2500. It is the only one of the three whose reference grid is not a window
of its DEM grid (2.5/9.48-cell offset, 29.938 m against 30.0 m), so the model
output is resampled onto the reference grid with nearest neighbour before
comparison -- no smoothing, so the agreement is not flattered by the resampling.

Top-decile IoU is not comparable across these three: on lokna the top decile is
11486 cells against roughly 100-250 on the others, so the same score is a much
harder target. Median ratio and Spearman rho are the figures to read across
datasets.

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

## Where the remaining disagreement is: the talweg

Sign agreement of ~96% understates the problem. On `lom` the 37 disagreeing
cells carry **31.5% of the reference's total erosion mass**, and the direction is
one-sided: of the 36 largest reference cells, 23 disagree and all 23 are
"reference deposits, we erode". The reference has 51 deposition cells against
our 22. We fail to deposit exactly where deposition matters — the main talweg.

Four explanations were tested against the data and rejected:

  the `deg2rad` on an already-radian slope raster in `compute_cell()` (23 -> 22
  disagreeing large cells, i.e. nothing);

  the `cap_kg_raw if > 0 else ktc*rusle*area` fallback when `slope_term` goes
  negative (no change at all);

  parcel-bounded routing, i.e. restricting the domain to the crop polygon, on
  the theory that the software deposits sediment at the field edge — this
  produced *fewer* depositions, 16 against 22;

  dropping C, P and the surplus LS from the capacity expression to match the
  published `TC = kTC·R·K·(LS − 6.86·(sinθ)^0.8·0.6)` — also fewer depositions,
  13 against 22, and more large-cell disagreement.

What does explain it is the magnitude of transport capacity. Lowering the
effective ktc from 0.25 to 0.05:

| | large cells with wrong sign | deposition cells | sign | rho |
| --- | --- | --- | --- | --- |
| lom, ktc 0.25 | 23/36 | 22 (ref 51) | 0.964 | 0.865 |
| lom, ktc 0.05 | 8/36 | 48 (ref 51) | 0.980 | 0.909 |
| spok, ktc 0.25 | 43/87 | 31 (ref 112) | 0.967 | 0.925 |
| spok, ktc 0.05 | 16/87 | 134 (ref 112) | 0.969 | 0.943 |

Mass carried by wrong-sign cells on lom falls from 31.5% to 11.7%. At ktc 0.02
the large cells are nearly perfect (4/36, 2/87) but deposition is over-produced
threefold, so the useful range is 0.02–0.05.

**0.05 is fitted, not read off the software's settings**, and it cannot be
derived from them, because the units do not correspond:

```python
area     = cell_res**2                            # m2
cap_kg   = ktc * rusle_kg_m2 * slope_term * area  # kg
distcorr = area * (abs(sin) + abs(cos))           # m2   <- surplus area
cap_m3   = cap_kg * distcorr / bulk_density       # kg*m2/(kg/m3) = m5, not m3
```

`distcorr` should be the dimensionless `abs(sin) + abs(cos)`; as written it
carries an extra factor of cell area (400 m² on lom), and `cap_m3` is not a
volume. Compounding this, WaTEM/SEDEM's ktc has units of length (metres) —
TC is a flux per unit width — while ours is treated as dimensionless.

So the software's kTc = 250 has no defensible mapping onto our ktc, and the
earlier attempt to use 0.25 for it was meaningless from the start. The order of
work is: fix the dimensions first, then re-anchor ktc against 250 on units that
mean something. That changes every result by roughly the cell area, so it is not
a change to make quietly — hence documented here rather than applied to the
defaults.


## Resolved: kTc is a length, and the software routes multi-directionally

Both open questions were answered by the modeller.

**kTc low and high are in metres**, and kTc limit is a C-factor threshold — cells
with C at or above it take the high value. Their arable catchments therefore run
on 250 m throughout.

**The software spreads flow across several directions**, not one, "с выпуклого
склона с верхней точки будет течь в разные стороны".

Both confirm what the dimensional analysis above and the routing sweep had only
suggested, so the code now follows the software rather than approximating it:

  `transport_capacity()` in `lateraldistribution.py` computes
  `TC = ktc · R · K · (LS − 0.6·6.86·|sin θ|^0.8) / 1e4` in kg m⁻¹ yr⁻¹ and
  multiplies by the flow width `cell_res · (|sin(aspect)| + |cos(aspect)|)`.
  Capacity is now a volume. The old expression multiplied by cell area twice
  and took the width correction from the slope raster instead of aspect.

  `compute_cell()` no longer applies `deg2rad` to an already-radian slope.

  `calibration` carries `ktc_low: 75`, `ktc_high: 250`, `ktc_limit: 0.1` in
  metres; `ktc_multiplier` is retained so old configs load but is not applied.

  `routing_scheme` defaults to `mfd`.

Against the three references, with every parameter taken from the software's
settings and **nothing fitted**:

| | median model | median GT | ratio | Spearman rho | sign | deposition cells (ours/GT) | wrong-sign mass |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lom | −10.55 | −11.45 | 1.12 | 0.919 | 0.975 | 43 / 51 | 8.2% |
| spok | −6.33 | −7.55 | 1.14 | 0.959 | 0.982 | 77 / 112 | 3.9% |
| lokna | −3.94 | −4.06 | 1.05 | 0.902 | 0.954 | 3277 / 2705 | 5.3% |

This beats the earlier fitted `ktc = 0.05` on every catchment — Spearman 0.909 →
0.919, 0.943 → 0.959, 0.900 → 0.902, and wrong-sign mass on lom 11.7% → 8.2%,
on spok 11.3% → 3.9%. The fitted number is gone from the model entirely.

Using `sin` rather than `tan` in the capacity's steepness term changes results
below the second decimal on both catchments, so the reference cannot distinguish
them; `sin` is used because that is the published form.

What is still open: the largest cells. 8 of lom's 36 biggest reference cells and
13 of spok's 87 still carry the wrong sign, all in the same direction — the
software deposits in the talweg and we do not, only less often than before.
