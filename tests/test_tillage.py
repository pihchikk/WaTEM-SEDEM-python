"""Tillage erosion cannot be checked against the reference rasters -- they hold
water erosion only. These check the invariants that make the formulation
falsifiable without one.
"""

from __future__ import annotations

import numpy as np
import pytest

from watem_sedem.tillage import tillage_erosion, tillage_flux, to_t_ha

KTIL = 600.0
RES = 20.0


def _cone(n=61, height=40.0, res=RES, fill=0.6):
    """A cone standing on a flat plain, its base well inside the array.

    `fill` keeps the base at 60% of the half-width so the surface is flat -- and
    the tillage flux therefore exactly zero -- for several cells before the array
    ends. Without that margin the cone is cut by the edge, soil legitimately
    leaves the domain, and the mass-balance check below would be testing the
    fixture rather than the model.
    """
    c = (n - 1) / 2.0
    y, x = np.mgrid[0:n, 0:n]
    r = np.hypot(x - c, y - c) * res
    radius = c * res * fill
    return np.maximum(height * (1.0 - r / radius), 0.0)


def _slope_aspect(z, res=RES):
    """Gradient-based slope and aspect, in radians, aspect measured clockwise
    from north the way SAGA reports it."""
    dz_dy_down, dz_dx = np.gradient(z, res, res)
    dz_dy = -dz_dy_down                      # north-positive
    slope = np.arctan(np.hypot(dz_dx, dz_dy))
    aspect = np.arctan2(-dz_dx, -dz_dy)      # points downslope
    return slope, aspect


def test_moves_soil_without_creating_it():
    """Tillage redistributes; it does not add or remove mass. On a surface whose
    flux vanishes at the domain edge, the net must cancel to rounding."""
    z = _cone()
    slope, aspect = _slope_aspect(z)
    ero = tillage_erosion(slope, aspect, KTIL, RES)
    net = np.nansum(ero)
    gross = np.nansum(np.abs(ero))

    assert gross > 0, "no tillage at all on a cone -- the flux is not being computed"
    assert abs(net) / gross < 1e-9, (
        f"tillage is not mass-conserving: net {net:.6g} against gross {gross:.6g}")


def test_strips_convexities_and_fills_concavities():
    z = _cone()
    slope, aspect = _slope_aspect(z)
    ero = tillage_erosion(slope, aspect, KTIL, RES)

    c = z.shape[0] // 2
    apex = ero[c - 1:c + 2, c - 1:c + 2]
    assert np.nanmean(apex) < 0, "the cone's apex should erode, not accumulate"

    # A bowl is the cone inverted; erosion and deposition must swap with it.
    bowl_slope, bowl_aspect = _slope_aspect(-z)
    bowl = tillage_erosion(bowl_slope, bowl_aspect, KTIL, RES)
    assert np.nanmean(bowl[c - 1:c + 2, c - 1:c + 2]) > 0, \
        "the bottom of a bowl should accumulate"


def test_flat_ground_does_nothing():
    z = np.full((40, 40), 100.0)
    slope, aspect = _slope_aspect(z)
    ero = tillage_erosion(slope, aspect, KTIL, RES)
    assert np.allclose(np.nan_to_num(ero), 0.0), "flat ground cannot till-erode"


def test_scales_linearly_with_ktil():
    z = _cone()
    slope, aspect = _slope_aspect(z)
    a = tillage_erosion(slope, aspect, KTIL, RES)
    b = tillage_erosion(slope, aspect, 2 * KTIL, RES)
    assert np.allclose(np.nan_to_num(b), 2 * np.nan_to_num(a))


def test_uniform_slope_is_transport_only():
    """A plane has constant flux, so nothing erodes or deposits on it even
    though soil is moving -- the classic tillage result, and a check that the
    divergence is not picking up the slope itself."""
    n = 40
    y, x = np.mgrid[0:n, 0:n]
    z = 100.0 - x * 0.1 * RES
    slope, aspect = _slope_aspect(z)
    ero = tillage_erosion(slope, aspect, KTIL, RES)
    interior = ero[2:-2, 2:-2]
    assert np.nanmax(np.abs(interior)) < 1e-9, \
        f"a uniform slope should be pure transport, got up to {np.nanmax(np.abs(interior)):.3g}"


def test_flux_points_downslope():
    z = _cone()
    slope, aspect = _slope_aspect(z)
    qx, qy = tillage_flux(slope, aspect, KTIL)
    c = z.shape[0] // 2
    # east of the apex the surface falls eastward, so the flux must run east
    assert qx[c, c + 6] > 0
    # west of it, west
    assert qx[c, c - 6] < 0


def test_unit_conversion():
    assert to_t_ha(np.array([1.0])) == pytest.approx([10.0])
