"""BMI wrapper for WaTEM-SEDEM using bmipy BMI base class."""

from typing import Any
import logging
import numpy as np
from bmipy import Bmi
from numpy.typing import NDArray

from watem_sedem import data_loader # core module
from watem_sedem import solver
from watem_sedem.lateraldistribution import topo_order

logger = logging.getLogger(__name__)


def _to_plain(name: str, a: Any) -> NDArray[Any]:
    if np.ma.isMaskedArray(a):
        if np.issubdtype(a.dtype, np.floating):
            out = a.filled(np.nan).astype(np.float32, copy=False)
        elif name in ("flow_direction", "routing", "segments", "outlet"):
            out = a.filled(-1).astype(np.int16, copy=False)
        else:
            out = a.filled(0)
    else:
        out = np.asarray(a)
    return np.ascontiguousarray(out)


class BmiWaTEM(Bmi):
    """BMI implementation of the WaTEM-SEDEM erosion model.

    WaTEM-SEDEM is not a transient model: it has no internal time integration
    and SEDI_IN/topology are recomputed from scratch every solve. It is meant
    to be driven as an *epoch* component (once per season/year/storm), called
    repeatedly with updated drivers (Cfactor/Pfactor/ktc/Rfactor) set via
    set_value() before each update(). initialize() therefore only performs
    static, one-time setup (loading rasters, deriving flow topology); update()
    performs the actual RUSLE + transport-capacity solve and can be called
    many times without re-loading or re-sorting the grid.
    """

    _name = "WaTEM-SEDEM"
    _input_var_names = (
        "elevation", "slope", "aspect", "flow_accumulation",
        "slope_length", "flow_direction", "LS-factor",
        "Kfactor", "Cfactor", "Pfactor", "ktc", "Rfactor",
        "bulk-density"
    )
    _output_var_names = ("SEDI_OUT", "CAPACITY", "WATEREROS", "epoch_index")

    def __init__(self) -> None:
        super().__init__()
        self._values: dict[str, NDArray[Any]] = {}

        # default units (outputs set after conversion)
        self._var_units = {
            **{v: "m"   for v in ("elevation",)},
            **{v: "deg" for v in ("slope", "aspect")},
            "flow_accumulation": "cells",
            "slope_length": "m",
            "flow_direction": "D8 code (-1=no-flow)",  # clearer than "-"
            **{v: "-" for v in ("LS-factor", "Kfactor", "Cfactor", "Pfactor")},
            "ktc": "m",
            "Rfactor": "MJ·mm/(ha·h·yr)",
            "bulk-density": "kg/m³",
            "SEDI_OUT": "",
            "CAPACITY": "",
            "WATEREROS": "",
            "epoch_index": "-",
        }

        self._var_loc   = {v: "node" for v in (*self._input_var_names, *self._output_var_names)}
        self._grids     = {0: [*self._input_var_names, *self._output_var_names]}
        self._grid_type = {0: "uniform_rectilinear"}

        # WaTEM-SEDEM is a static epoch model: each update() advances one epoch
        self._start_time = 0.0
        self._time_step  = 1.0
        self._end_time   = 1.0
        self._time_units = "epoch"

        # epoch bookkeeping
        self._epoch = 0

        # set in initialize()
        self._shape: tuple[int, int]
        self._spacing: tuple[float, float]
        self._origin: tuple[float, float]
        self._cfg = None
        self._data: dict[str, Any] = {}
        self._topo = None
        self._flow_direction: NDArray[Any] | None = None

        # Names the caller has driven through set_value() at least once. Only
        # ktc needs this: solver.solve() otherwise always derives it from
        # Cfactor (see the note in update()), silently discarding anything
        # set_value('ktc', ...) wrote.
        self._externally_set: set[str] = set()

    # ── setup ────────────────────────────────────────────────────────────
    def initialize(self, filename: str | None = None) -> None:
        """Static, one-time grid setup: load config/inputs, normalize dtypes,
        derive (or ingest) flow topology, and cache everything needed to run
        update() repeatedly without re-loading rasters or re-sorting topology.
        """
        # load config and inputs
        cfg = data_loader.load_and_validate_config(filename or "config.yaml")
        data = data_loader.load_inputs(cfg)
        shape = data["elevation"].shape

        # broadcast WaTEM scalars (Rfactor and bulk-density included so
        # set_value() with a full-grid array works uniformly whether the
        # config provided a scalar default or a raster). Like Rfactor,
        # bulk-density is folded back to a representative scalar in update()
        # before the solve: the d8 router (lateraldistribution.compute_cell)
        # takes it as one unindexed float per cell, same as Rfactor, so a
        # per-cell raster is not honoured spatially by either entry point
        # today -- broadcasting here only exists to satisfy the BMI grid-size
        # contract for get_value()/set_value(), not to make it spatial.
        for key in ("LS-factor", "Kfactor", "Cfactor", "Pfactor", "ktc", "Rfactor", "bulk-density"):
            if not isinstance(data[key], np.ndarray):
                data[key] = np.full(shape, data[key], dtype=float)

        # ktc is no longer a dimensionless number scaled by ktc_multiplier; it
        # is a length in metres that solver.select_ktc() derives per cell from
        # the C factor against ktc_limit. Doing it here as well would apply it
        # twice.

        # flow direction: internal D8 (default) or externally supplied (e.g. LISEM)
        flow_source = getattr(cfg, "flow_direction_source", "internal") or "internal"
        if flow_source == "external":
            fdir = self._load_external_flow_direction(cfg, shape, data["cell_size"])
        else:
            fdir = data["flow_direction"]
            if np.ma.isMaskedArray(fdir):
                fdir = fdir.filled(-1)
            fdir = fdir.astype(np.int16, copy=False)
        bad = (fdir < 0) | (fdir > 7)
        fdir[bad] = -1
        data["flow_direction"] = fdir

        # A BMI's variable set is static, so tillage outputs are declared here
        # from the config rather than appearing after the first update().
        if cfg.tillage.enabled:
            self._output_var_names = (*self._output_var_names, "TILLEROS", "TOTALEROS")
            self._var_loc.update({v: "node" for v in ("TILLEROS", "TOTALEROS")})
            self._grids[0] = [*self._input_var_names, *self._output_var_names]

        # cache static grid/topology (Step 1: no re-derivation inside update())
        self._topo = topo_order(data["flow_direction"])
        self._flow_direction = data["flow_direction"]
        self._data = data
        self._cfg = cfg

        # inputs as plain arrays (drivers Cfactor/Pfactor/ktc/Rfactor are mutable
        # via set_value() before each update())
        for k in self._input_var_names:
            self._values[k] = _to_plain(k, data[k])

        # grid metadata
        self._shape = shape
        cs = data["cell_size"]
        self._spacing = (cs, cs)
        t = data["meta"]["transform"]
        nrows = shape[0]
        x0 = t[2]
        y0 = t[5] + t[4] * nrows
        self._origin = (x0, y0)

        # output placeholders until first update()
        self._values["SEDI_OUT"]  = np.zeros(shape, dtype=np.float32)
        self._values["CAPACITY"]  = np.zeros(shape, dtype=np.float32)
        self._values["WATEREROS"] = np.zeros(shape, dtype=np.float32)
        self._values["epoch_index"] = np.zeros(shape, dtype=np.int32)

        self._epoch = 0
        self._current_time = self._start_time

    def _load_external_flow_direction(
        self, cfg, shape: tuple[int, int], expected_cell_size: float
    ) -> NDArray[Any]:
        """Ingest a pre-computed flow-direction raster (e.g. LISEM's routed grid)
        instead of deriving an independent D8 grid, so both models agree on
        topology when they cover the same catchment (see docs/COUPLING_VARS_LEDGER.md).

        Grid shape and cell size must match the WaTEM-SEDEM grid exactly; a
        mismatch raises rather than silently reprojecting/reshaping, since a
        silent regrid would misalign sediment routing between the two models.
        """
        path = getattr(cfg, "external_flow_direction_path", None)
        if not path:
            raise ValueError(
                "flow_direction_source='external' requires 'external_flow_direction_path' "
                "in config.yaml pointing at a pre-computed flow-direction raster."
            )
        import rasterio
        with rasterio.open(path) as src:
            arr = src.read(1, masked=True)
            ext_shape = arr.shape
            ext_spacing = abs(src.transform.a)

        if ext_shape != tuple(shape):
            raise ValueError(
                f"External flow-direction grid shape {ext_shape} does not match "
                f"WaTEM-SEDEM grid shape {tuple(shape)}. Reproject/regrid the external "
                "source to match before coupling; automatic reprojection is not performed."
            )
        if not np.isclose(ext_spacing, expected_cell_size):
            raise ValueError(
                f"External flow-direction cell size ({ext_spacing}) does not match "
                f"WaTEM-SEDEM cell size ({expected_cell_size}). Grids must share resolution."
            )

        if np.ma.isMaskedArray(arr):
            arr = arr.filled(-1)
        return arr.astype(np.int16, copy=False)

    # ── solve ────────────────────────────────────────────────────────────
    def update(self) -> None:
        """Run one epoch of the RUSLE + transport-capacity solve using the
        current Cfactor/Pfactor/ktc/Rfactor values (set via set_value() before
        calling update()). Reuses the topology/static rasters cached by
        initialize() — no re-loading or re-sorting occurs here.
        """
        if self._topo is None:
            raise RuntimeError("update() called before initialize()")

        data = self._data
        cfg = self._cfg

        # Drivers may have been changed through set_value() since the last
        # update(); fold them back into the data dict the solver reads, so both
        # entry points run identical code on identical inputs.
        for key in self._input_var_names:
            if key in self._values:
                data[key] = self._values[key]

        Rfactor = self._values["Rfactor"]
        data["Rfactor"] = (float(np.mean(Rfactor)) if isinstance(Rfactor, np.ndarray)
                           else float(Rfactor))
        bulk_density = self._values["bulk-density"]
        data["bulk-density"] = (float(np.mean(bulk_density)) if isinstance(bulk_density, np.ndarray)
                                else float(bulk_density))
        data["flow_direction"] = self._flow_direction

        # solver.select_ktc() otherwise always overwrites ktc from Cfactor;
        # tell it to leave set_value('ktc', ...) alone when the caller used it.
        # The CLI never touches data['ktc_source'], so its behaviour -- and
        # test_bmi_matches_cli's byte-for-byte parity -- is unchanged.
        data["ktc_source"] = "external" if "ktc" in self._externally_set else "internal"

        SEDI_IN, SEDI_OUT, WATEREROS, CAPACITY, TILLEROS = solver.solve(cfg, data)

        # Same conversion the CLI writes its rasters with, from the same
        # function -- the two used to carry separate copies and drifted.
        fields = solver.convert_units(cfg, data, SEDI_OUT, CAPACITY, WATEREROS, TILLEROS)
        for name, (arr, label) in fields.items():
            self._values[name] = _to_plain(name, np.asarray(arr, dtype=np.float32))
            self._var_units[name] = label

        self._epoch += 1
        self._values["epoch_index"] = np.full(self._shape, self._epoch, dtype=np.int32)
        self._current_time = self._end_time

    def update_frac(self, frac: float) -> None:
        self.update()

    def update_until(self, then: float) -> None:
        self.update()

    def finalize(self) -> None:
        self._values.clear()

    # BMI: variable introspection
    def get_component_name(self) -> str:
        return self._name

    def get_input_item_count(self) -> int:
        return len(self._input_var_names)

    def get_output_item_count(self) -> int:
        return len(self._output_var_names)

    def get_input_var_names(self) -> tuple[str, ...]:
        return self._input_var_names

    def get_output_var_names(self) -> tuple[str, ...]:
        return self._output_var_names

    def get_var_units(self, name: str) -> str:
        return self._var_units[name]

    def get_var_type(self, name: str) -> str:
        return np.asarray(self.get_value_ptr(name)).dtype.name

    def get_var_nbytes(self, name: str) -> int:
        return self.get_value_ptr(name).nbytes

    def get_var_itemsize(self, name: str) -> int:
        return np.dtype(self.get_var_type(name)).itemsize

    def get_var_location(self, name: str) -> str:
        return self._var_loc[name]

    def get_var_grid(self, name: str) -> int | None:
        for gid, vars in self._grids.items():
            if name in vars:
                return gid
        return None

    # BMI: value access
    def get_value_ptr(self, name: str) -> NDArray[Any]:
        return self._values[name]

    def get_value(self, name: str, dest: NDArray[Any]) -> NDArray[Any]:
        arr = self.get_value_ptr(name).ravel()
        dest[:] = arr
        return dest

    def get_value_at_indices(
        self, name: str, dest: NDArray[Any], inds: NDArray[np.int_]
    ) -> NDArray[Any]:
        dest[:] = self.get_value_ptr(name).flat[inds]
        return dest

    def set_value(self, name: str, src: NDArray[Any]) -> None:
        val = self.get_value_ptr(name)
        val[:] = src.reshape(val.shape)
        if name == "ktc":
            self._externally_set.add(name)

    def set_value_at_indices(
        self, name: str, inds: NDArray[np.int_], src: NDArray[Any]
    ) -> None:
        arr = self.get_value_ptr(name)
        arr.flat[inds] = src
        if name == "ktc":
            self._externally_set.add(name)

    # BMI: time
    def get_start_time(self) -> float:
        return self._start_time

    def get_end_time(self) -> float:
        return self._end_time

    def get_current_time(self) -> float:
        return getattr(self, "_current_time", self._start_time)

    def get_time_step(self) -> float:
        return self._time_step

    def get_time_units(self) -> str:
        return self._time_units

    # BMI grid
    def get_grid_rank(self, grid: int) -> int:
        return 2

    def get_grid_size(self, grid: int) -> int:
        return int(self._shape[0] * self._shape[1])

    def get_grid_shape(self, grid: int, shape=None):
        shp = np.array(self._shape, dtype=int)
        if shape is None:
            return tuple(shp)
        shape[:] = shp
        return shape

    def get_grid_spacing(self, grid: int, spacing=None):
        sp = np.array(self._spacing, dtype=float)
        if spacing is None:
            return tuple(sp)
        spacing[:] = sp
        return spacing

    def get_grid_origin(self, grid: int, origin=None):
        org = np.array(self._origin, dtype=float)
        if origin is None:
            return tuple(org)
        origin[:] = org
        return origin

    def get_grid_type(self, grid: int) -> str:
        return self._grid_type[grid]

    # topology unsupported by the model design
    def get_grid_edge_count(self, grid: int) -> int:
        raise NotImplementedError

    def get_grid_edge_nodes(self, grid: int, edge_nodes: NDArray[np.int_]) -> None:
        raise NotImplementedError

    def get_grid_face_count(self, grid: int) -> int:
        raise NotImplementedError

    def get_grid_face_nodes(self, grid: int, face_nodes: NDArray[np.int_]) -> None:
        raise NotImplementedError

    def get_grid_node_count(self, grid: int) -> int:
        return self.get_grid_size(grid)

    def get_grid_nodes_per_face(self, grid: int, nodes_per_face: NDArray[np.int_]) -> None:
        raise NotImplementedError

    def get_grid_face_edges(self, grid: int, face_edges: NDArray[np.int_]) -> None:
        raise NotImplementedError

    def get_grid_x(self, grid: int, x: NDArray[np.float64]) -> None:
        raise NotImplementedError

    def get_grid_y(self, grid: int, y: NDArray[np.float64]) -> None:
        raise NotImplementedError

    def get_grid_z(self, grid: int, z: NDArray[np.float64]) -> None:
        raise NotImplementedError
