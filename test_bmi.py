import numpy as np
import pytest
import numpy.testing as npt

from bmi_watem import BmiWaTEM
import data_loader


# point at your on-disk config.yaml
CONFIG = "config.yaml"


@pytest.fixture(scope="module")
def model():
    """One BmiWaTEM instance, initialized once for all tests."""
    m = BmiWaTEM()
    m.initialize(CONFIG)
    yield m
    m.finalize()


@pytest.fixture
def fresh_model():
    """Fresh model for tests that finalize inside the test."""
    m = BmiWaTEM()
    m.initialize(CONFIG)
    return m


def test_initial_times(model):
    """Check that time metadata is 0 → 1 year."""
    assert model.get_start_time() == pytest.approx(0.0)
    assert model.get_time_step() == pytest.approx(1.0)
    assert model.get_end_time() == pytest.approx(1.0)
    # make sure you’ve set self._time_units = "year" in your wrapper
    assert model.get_time_units() == "year"


def test_update_advances_to_end(fresh_model):
    fresh_model.update()
    assert fresh_model.get_current_time() == pytest.approx(fresh_model.get_end_time())


def test_finalize_clears(fresh_model):
    """After finalize(), get_value_ptr should KeyError."""
    m = fresh_model
    m.finalize()
    first = m.get_input_var_names()[0]
    with pytest.raises(KeyError):
        _ = m.get_value_ptr(first)


def test_grid_metadata(model):
    gid = 0
    assert model.get_grid_rank(gid) == 2

    # with provided buffer
    buf = np.empty(2, int)
    ret = model.get_grid_shape(gid, buf)
    assert ret is buf
    assert tuple(buf) == tuple(model.get_grid_shape(gid))  # compare to tuple form
    assert len(buf) == 2

    assert model.get_grid_size(gid) == buf[0] * buf[1]

    spacing = model.get_grid_spacing(gid, np.empty(2, float))
    origin  = model.get_grid_origin(gid,  np.empty(2, float))
    assert spacing[0] == spacing[1] and spacing[0] > 0
    assert all(isinstance(o, float) for o in origin)
    assert model.get_grid_type(gid) == "uniform_rectilinear"


def test_var_lists_and_grids(model):
    """Inputs/outputs non-overlapping and all mapped to grid 0."""
    ins  = model.get_input_var_names()
    outs = model.get_output_var_names()
    assert set(ins).isdisjoint(outs)
    assert ins and outs
    for v in (*ins, *outs):
        assert model.get_var_grid(v) == 0


def test_get_set_value_and_pointer(model):
    """get_value_ptr, get_value, and set_value must work end-to-end."""
    var = model.get_output_var_names()[0]
    arr = model.get_value_ptr(var)
    assert isinstance(arr, np.ndarray)

    # copy-based getter
    dest = np.empty(arr.size, arr.dtype)
    out = model.get_value(var, dest)
    assert out is dest
    npt.assert_array_equal(dest.reshape(arr.shape), arr)

    # set to zeros
    zeros = np.zeros_like(arr)
    model.set_value(var, zeros.ravel())
    assert np.allclose(model.get_value_ptr(var), 0.0)


def test_units_match_requested_config(monkeypatch):
    orig = data_loader.load_and_validate_config
    def fake(cfgfile):
        cfg = orig(cfgfile)
        cfg.output.sediment_unit = "t/ha"
        cfg.output.erosion_unit  = "mm"
        return cfg
    monkeypatch.setattr(data_loader, "load_and_validate_config", fake)

    m = BmiWaTEM()
    m.initialize(CONFIG)

    # accept both "t/ha" and "t ha-1"
    assert m.get_var_units("SEDI_OUT")  in ("t/ha", "t ha-1")
    assert m.get_var_units("CAPACITY")  in ("t/ha", "t ha-1")
    assert m.get_var_units("WATEREROS") == "mm"


def test_flowdir_sanitized(monkeypatch):
    # minimal fake config with units
    class _Cfg: 
        class _Out: sediment_unit="kg"; erosion_unit="kg"
        output = _Out()
    def fake_cfg(_): return _Cfg()

    # tiny 3x3 dataset with bad codes: 255, 9, -9999 (masked)
    def fake_inputs(_):
        elev = np.ones((3,3), float)
        fd   = np.array([[0, 1, 255],
                         [9, -1, 3],
                         [4, 5, 6]], dtype=int)
        fd = np.ma.masked_where(fd==-1, fd)  # masked center
        data = {
            "elevation": elev,
            "slope": np.zeros_like(elev),
            "aspect": np.zeros_like(elev),
            "flow_accumulation": np.zeros_like(elev),
            "slope_length": np.zeros_like(elev),
            "flow_direction": fd,
            "LS-factor": np.ones_like(elev),
            "Kfactor": 1.0, "Cfactor": 1.0, "Pfactor": 1.0, "ktc": 1.0,
            "Rfactor": 1.0, "bulk-density": 1300.0,
            "cell_size": 10.0,
            "meta": {"transform": (10,0,0,0,-10,0)}  # dummy affine tuple
        }
        return data

    monkeypatch.setattr(data_loader, "load_and_validate_config", fake_cfg)
    monkeypatch.setattr(data_loader, "load_inputs", fake_inputs)

    m = BmiWaTEM()
    m.initialize("ignored.yaml")
    f = m.get_value_ptr("flow_direction")

    # bad/masked became -1, valid kept in [0..7]
    assert f[0,2] == -1 and f[1,0] == -1 and f[1,1] == -1
    assert np.all(((f >= 0) & (f <= 7)) | (f == -1))

def test_set_value_at_indices(model):
    var = model.get_output_var_names()[1]  # e.g., CAPACITY
    arr = model.get_value_ptr(var)
    inds = np.array([0, arr.size//2, arr.size-1], dtype=int)
    src  = np.array([111.0, 222.0, 333.0], dtype=arr.dtype)
    model.set_value_at_indices(var, inds, src)
    flat = model.get_value_ptr(var).ravel()
    np.testing.assert_allclose(flat[inds], src)

def test_var_locations(model):
    for v in (*model.get_input_var_names(), *model.get_output_var_names()):
        assert model.get_var_location(v) == "node"

def test_grid_queries_return_python_tuples(model):
    gid = 0
    assert isinstance(model.get_grid_shape(gid), tuple)
    assert isinstance(model.get_grid_spacing(gid), tuple)
    assert isinstance(model.get_grid_origin(gid), tuple)

def test_reinitialize_ok(fresh_model):
    fresh_model.finalize()
    m = BmiWaTEM()
    m.initialize(CONFIG)
    assert m.get_grid_size(0) > 0

def test_outputs_float32_contiguous(model):
    for v in ("SEDI_OUT", "CAPACITY", "WATEREROS"):
        arr = model.get_value_ptr(v)
        assert isinstance(arr, np.ndarray)          # not a masked array
        assert arr.dtype == np.float32              # float32 as promised
        assert arr.flags["C_CONTIGUOUS"]            # contiguous for BMI clients
        assert arr.flags["WRITEABLE"]               # can be set via set_value
