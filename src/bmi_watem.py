"""BMI wrapper for WaTEM-SEDEM using bmipy BMI base class."""

from typing import Any
import numpy as np
from bmipy import Bmi
from numpy.typing import NDArray

import data_loader # core module
from lateraldistribution import topo_order, compute_erosion


class BmiWaTEM(Bmi):
    """BMI implementation of the WaTEM-SEDEM erosion model (single 1-year step)."""

    _name = "WaTEM-SEDEM"
    _input_var_names = (
        "elevation", "slope", "aspect", "flow_accumulation",
        "slope_length", "flow_direction", "LS-factor",
        "Kfactor", "Cfactor", "Pfactor", "ktc", "Rfactor",
        "bulk-density"
    )
    _output_var_names = ("SEDI_OUT", "CAPACITY", "WATEREROS")

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
            **{v: "-" for v in ("LS-factor", "Kfactor", "Cfactor", "Pfactor", "ktc")},
            "Rfactor": "MJ·mm/(ha·h·yr)",
            "bulk-density": "kg/m³",
            "SEDI_OUT": "",
            "CAPACITY": "",
            "WATEREROS": "",
        }

        self._var_loc   = {v: "node" for v in (*self._input_var_names, *self._output_var_names)}
        self._grids     = {0: [*self._input_var_names, *self._output_var_names]}
        self._grid_type = {0: "uniform_rectilinear"}

        # WaTEM-SEDEM is a static 1-year model
        self._start_time = 0.0
        self._time_step  = 1.0
        self._end_time   = 1.0
        self._time_units = "year"

        # set in initialize()
        self._shape: tuple[int, int]
        self._spacing: tuple[float, float]
        self._origin: tuple[float, float]

    def initialize(self, filename: str | None = None) -> None:
        # conversion to ndarray
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


        # load config and inputs
        cfg = data_loader.load_and_validate_config(filename or "config.yaml")
        data = data_loader.load_inputs(cfg)
        shape = data["elevation"].shape

        # broadcast WaTEM scalars
        for key in ("LS-factor", "Kfactor", "Cfactor", "Pfactor", "ktc"):
            if not isinstance(data[key], np.ndarray):
                data[key] = np.full(shape, data[key], dtype=float)
        def _ktc_mult_from_cfg(cfg) -> float:
            # Pydantic-like
            md = getattr(cfg, "model_dump", None)
            if callable(md):
                return float(md().get("calibration", {}).get("ktc_multiplier", 1.0))
            # Attribute or dict
            calib = getattr(cfg, "calibration", None)
            if isinstance(calib, dict):
                return float(calib.get("ktc_multiplier", 1.0))
            return float(getattr(calib, "ktc_multiplier", 1.0))  

        m = _ktc_mult_from_cfg(cfg)

        data["ktc"] = (data["ktc"].astype(float) if isinstance(data["ktc"], np.ndarray)
                       else float(data["ktc"])) * m

        # "normalize" flow direction processing
        fdir = data["flow_direction"]
        if np.ma.isMaskedArray(fdir):
            fdir = fdir.filled(-1)
        fdir = fdir.astype(np.int16, copy=False)
        bad = (fdir < 0) | (fdir > 7)
        fdir[bad] = -1
        data["flow_direction"] = fdir

        # routing and erosion
        topo = topo_order(data["flow_direction"])
        SEDI_IN, SEDI_OUT, WATEREROS, CAPACITY = compute_erosion(
            LS             = data["LS-factor"],
            Kfactor        = data["Kfactor"],
            Cfactor        = data["Cfactor"],
            Pfactor        = data["Pfactor"],
            R_factor       = data["Rfactor"],
            bulk_density   = data["bulk-density"],
            slope          = data["slope"],
            aspect         = data["aspect"],
            ktc            = data["ktc"],
            cell_res       = data["cell_size"],
            flow_direction = data["flow_direction"],
            topo           = topo
        )

        # unit conversions
        cell_area       = data["cell_size"]**2
        bd              = data["bulk-density"]
        sed_mass_factor = bd/1000.0 * (10000.0 / cell_area)
        ero_mass_factor = bd * (10000.0 / 1000.0)

        su = cfg.output.sediment_unit
        eu = cfg.output.erosion_unit

        # sediment and capacity units 
        if su == "m3":
            sed_arr, cap_arr = SEDI_OUT, CAPACITY
            sed_label = cap_label = "m3 cell-1"
        elif su == "kg":
            sed_arr, cap_arr = SEDI_OUT * bd, CAPACITY * bd
            sed_label = cap_label = "kg cell-1"
        else:  # "t/ha"
            sed_arr, cap_arr = SEDI_OUT * sed_mass_factor, CAPACITY * sed_mass_factor
            sed_label = cap_label = "t ha-1"

        # erosion units + array
        if eu == "m":
            ero_arr, ero_label = WATEREROS, "m"
        elif eu == "mm":
            ero_arr, ero_label = WATEREROS * 1000.0, "mm"
        elif eu == "kg":
            ero_arr, ero_label = WATEREROS * cell_area * bd, "kg cell-1"
        else:  # "t/ha"
            ero_arr, ero_label = WATEREROS * ero_mass_factor, "t ha-1"
            
        sed_arr = sed_arr.astype(np.float32, copy=False)
        cap_arr = cap_arr.astype(np.float32, copy=False)
        ero_arr = ero_arr.astype(np.float32, copy=False)

        # store outputs and set units
        self._values.update({
            "SEDI_OUT":  _to_plain("SEDI_OUT",  sed_arr),
            "CAPACITY":  _to_plain("CAPACITY",  cap_arr),
            "WATEREROS": _to_plain("WATEREROS", ero_arr),
        })
        self._var_units["SEDI_OUT"]  = sed_label
        self._var_units["CAPACITY"]  = cap_label
        self._var_units["WATEREROS"] = ero_label

        # grid metadata
        self._shape = shape
        cs = data["cell_size"]
        self._spacing = (cs, cs)
        t = data["meta"]["transform"]
        nrows = shape[0]
        x0 = t[2]
        y0 = t[5] + t[4] * nrows
        self._origin = (x0, y0)

        # inputs as plain arrays
        for k in self._input_var_names:
            self._values[k] = _to_plain(k, data[k])
            
        # start at t0 
        self._current_time = self._start_time

    def update(self) -> None:
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

    def set_value_at_indices(
        self, name: str, inds: NDArray[np.int_], src: NDArray[Any]
    ) -> None:
        arr = self.get_value_ptr(name)
        arr.flat[inds] = src

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
