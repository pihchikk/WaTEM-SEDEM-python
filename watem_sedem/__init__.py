"""WaTEM-SEDEM: soil erosion and sediment transport, with a BMI wrapper.

The BMI entry point is :class:`watem_sedem.bmi_watem.BmiWaTEM`, re-exported
here for convenience::

    from watem_sedem import BmiWaTEM

`BmiWaTEM` is imported lazily: it pulls in the geospatial stack (rasterio,
geopandas) which is slow to import and unnecessary for callers that only want,
say, `watem_sedem.lateraldistribution`.
"""

__all__ = ['BmiWaTEM']


def __getattr__(name: str):
    if name == 'BmiWaTEM':
        from watem_sedem.bmi_watem import BmiWaTEM

        return BmiWaTEM
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
