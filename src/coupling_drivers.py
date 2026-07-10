"""Driver-aggregation glue between the daily AquaCrop/LISEM BMI loop and
WaTEM-SEDEM's annual/event-scale RUSLE terms.

WaTEM-SEDEM's `Cfactor` and `Rfactor` are empirical, seasonal/annual-average
regression terms (RUSLE — Renard et al. 1997, "Predicting Soil Erosion by
Water", USDA Agriculture Handbook 703). Feeding them raw daily canopy-cover
or rainfall values from a coupled AquaCrop/LISEM run misapplies the
regression: the functions here convert the daily BMI time series into the
scalars/rasters `compute_erosion()` actually expects.

None of this touches `lateraldistribution.py`'s erosion/routing math — it
only prepares its inputs.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "canopy_series_to_cfactor",
    "rainfall_series_to_rfactor",
    "event_rfactor",
    "soil_moisture_to_ktc_multiplier",
]


def canopy_series_to_cfactor(
    canopy_cover_daily: np.ndarray,
    stage_weights: np.ndarray | None = None,
) -> float:
    """Aggregate a daily AquaCrop `crop__canopy_cover` series (%, 0-100) into a
    single seasonal Cfactor using the USLE Soil-Loss-Ratio (SLR) approach
    (Wischmeier & Smith 1978; Renard et al. 1997 AH-703 ch.4).

    Each day's soil-loss ratio is approximated as the bare-soil-equivalent
    fraction ``SLR_t = 1 - canopy_cover_t / 100`` (a canopy that fully covers
    the soil drives local soil loss toward zero; bare soil drives it toward
    1). The seasonal Cfactor is the erosivity-weighted mean of daily SLR:

        Cfactor = sum(SLR_t * w_t) / sum(w_t)

    ``stage_weights`` lets the caller weight each day by its share of erosive
    rainfall (the SLR method properly weights by EI within each growth
    stage, since soil loss during a stage matters proportionally to the
    erosivity occurring in that stage — a bare day during a rainless month
    contributes little to annual soil loss). If ``stage_weights`` is None,
    this falls back to a flat arithmetic mean of daily SLR, i.e.
    ``mean(1 - canopy_cover/100)`` — a coarser approximation that ignores
    within-season erosivity distribution and should only be used when a
    rainfall series to derive weights from is unavailable.

    Parameters
    ----------
    canopy_cover_daily : 1D array of AquaCrop canopy cover, percent (0-100).
    stage_weights : optional 1D array, same length as canopy_cover_daily,
        of non-negative weights (e.g. daily erosivity EI30, or daily
        rainfall depth as a proxy). Need not be normalized.

    Returns
    -------
    float
        Dimensionless seasonal Cfactor, clipped to [0, 1].
    """
    canopy = np.asarray(canopy_cover_daily, dtype=float)
    if canopy.ndim != 1 or canopy.size == 0:
        raise ValueError("canopy_cover_daily must be a non-empty 1D array")

    slr = 1.0 - np.clip(canopy, 0.0, 100.0) / 100.0

    if stage_weights is None:
        cfactor = float(np.mean(slr))
    else:
        w = np.asarray(stage_weights, dtype=float)
        if w.shape != canopy.shape:
            raise ValueError("stage_weights must match canopy_cover_daily in shape")
        if np.all(w == 0):
            raise ValueError("stage_weights sum to zero — cannot weight SLR")
        cfactor = float(np.sum(slr * w) / np.sum(w))

    return float(np.clip(cfactor, 0.0, 1.0))


def rainfall_series_to_rfactor(rain_daily_mm: np.ndarray, method: str = "renard") -> float:
    """Convert a daily AquaCrop/LISEM rainfall series (mm/day) into an
    annual/seasonal R-factor (MJ·mm/(ha·h·yr)).

    ``method="renard"`` (default) uses the Renard & Freimund (1994) daily
    approximation of storm EI30 from total daily rainfall depth P (mm),
    developed as a simplified substitute for the standard (sub-hourly)
    USLE/RUSLE EI30 computation when only daily totals are available:

        EI30_day = 0.276 * P^1.81   (P <= 76.2 mm; extends the Foster et al.
                                       1981 / RUSLE relationship)
        EI30_day = 0.2483 * P^2.2   (P > 76.2 mm)

    (Renard, K.G. and Freimund, J.R., 1994. "Using monthly precipitation data
    to estimate the R-factor in the revised USLE." Journal of Hydrology,
    157(1-4), pp.287-306; the two-branch daily form is quoted from their
    Eq. 6/7 as used in RUSLE2 daily-erosivity estimators.)

    The seasonal/annual R-factor is the sum of daily EI30 over the series
    (each day below 12.7 mm is excluded — days that light are not considered
    erosive under standard USLE convention, matching RUSLE's erosive-event
    threshold).

    **Uncertainty flag:** this daily-depth approximation is the single
    biggest source of uncertainty in the coupled run. It has no sub-daily
    intensity information, so it cannot distinguish a short intense
    convective storm from a long gentle rain of the same total depth — both
    of which materially change true EI30. Any output produced with a
    derived R-factor MUST propagate a flag in metadata (see
    `docs/COUPLING_VARS_LEDGER.md`) rather than being treated as equivalent
    to a measured/pluviograph-derived R-factor.

    Parameters
    ----------
    rain_daily_mm : 1D array of daily rainfall depths (mm).
    method : currently only "renard" is implemented.

    Returns
    -------
    float
        R-factor in MJ·mm/(ha·h·yr) (or per given series length/season).
    """
    if method != "renard":
        raise NotImplementedError(f"Unknown R-factor method '{method}'")

    p = np.asarray(rain_daily_mm, dtype=float)
    if p.ndim != 1 or p.size == 0:
        raise ValueError("rain_daily_mm must be a non-empty 1D array")
    if np.any(p < 0):
        raise ValueError("rain_daily_mm must be non-negative")

    erosive = p >= 12.7  # USLE erosive-rainfall threshold (0.5 in)
    p_e = p[erosive]

    ei30 = np.where(
        p_e <= 76.2,
        0.276 * p_e**1.81,
        0.2483 * p_e**2.2,
    )
    return float(np.sum(ei30))


def event_rfactor(rain_event_mm: float, duration_h: float) -> float:
    """Single-storm erosivity for event-mode coupling.

    Uses the same Renard & Freimund (1994) daily-depth EI30 approximation as
    `rainfall_series_to_rfactor`, applied to one storm's total depth. The
    storm ``duration_h`` is accepted for interface symmetry with LISEM's
    per-event accounting and for future intensity-aware refinement, but the
    Renard & Freimund depth-only approximation does not use it — see the
    uncertainty note in `rainfall_series_to_rfactor`.

    Parameters
    ----------
    rain_event_mm : total storm rainfall depth (mm).
    duration_h : storm duration in hours (must be > 0; unused by this
        depth-only approximation, kept for interface/documentation clarity).

    Returns
    -------
    float
        Event EI30 (MJ·mm/(ha·h)), or 0.0 if the storm is below the 12.7 mm
        USLE erosive-rainfall threshold.
    """
    if rain_event_mm < 0:
        raise ValueError("rain_event_mm must be non-negative")
    if duration_h <= 0:
        raise ValueError("duration_h must be positive")

    if rain_event_mm < 12.7:
        return 0.0
    if rain_event_mm <= 76.2:
        return float(0.276 * rain_event_mm**1.81)
    return float(0.2483 * rain_event_mm**2.2)


def soil_moisture_to_ktc_multiplier(
    theta: np.ndarray,
    theta_fc: np.ndarray,
    low: float = 0.5,
    high: float = 1.5,
) -> np.ndarray:
    """EXPERIMENTAL: scale WaTEM-SEDEM's static `ktc` transport-capacity
    raster by antecedent soil moisture, drawn from AquaCrop `soil__moisture`
    or LISEM `domain_soil_water_storage__volume`.

    This is not part of standard WaTEM-SEDEM and has no published
    calibration — it is provided as an optional, off-by-default hook for
    experimentation, not a validated coupling term. Flag any run that uses
    it in output metadata (see `docs/COUPLING_VARS_LEDGER.md`).

    The multiplier is simply the wetness fraction ``theta / theta_fc``
    (saturation ratio relative to field capacity), clipped to
    ``[low, high]`` so that dry/near-zero moisture cannot collapse transport
    capacity to zero and saturated soil cannot runaway-amplify it — both of
    which would be physically implausible extrapolations from a term with
    no fitted range.

    Parameters
    ----------
    theta : soil moisture (volumetric water content), same shape as ktc.
    theta_fc : field-capacity soil moisture, same shape (or broadcastable).
    low, high : bounds for the returned multiplier (default 0.5-1.5x).

    Returns
    -------
    np.ndarray
        Multiplier array, same shape as ``theta``, clipped to [low, high].
    """
    theta = np.asarray(theta, dtype=float)
    theta_fc = np.asarray(theta_fc, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(theta_fc > 0, theta / theta_fc, 1.0)
    return np.clip(ratio, low, high)
