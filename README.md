# WaTEM-SEDEM (Python BMI Wrapper)

This repo lets you run the core WaTEM-SEDEM model via Python, wrapped with the Basic Model Interface (BMI) and simplified using the preprocessing logic from [pywatemsedem](https://github.com/watem-sedem/pywatemsedem).

It can be used in two ways:

- **As a standalone script** via `src/run_watem.py` — to run directly using a config file.  
- **As a BMI module** (`bmi_watem.py`) — within a wrrapper to integrate into larger model frameworks.

**Note:** This is not a full-featured version. It supports a single, default (“relief-based”) scenario—no tillage, strips, infrastructure, or multi-factor enhancements. Modes control *how inputs are sourced*, not alternative erosion logic.

---

## Project Structure

- `src/` — Python scripts for preprocessing, execution, and BMI access  
- `data/` — example input files under `rasters/` and `pywatemsedem_input/`  
- `tests/` — reference outputs per input mode  
- `metadata/` — JSON descriptors like `WaTEM_SEDEM_STD_extended.json` (schema + standard model metadata)  
- `config.yaml` — central run configuration

---

## Run Modes

Depending what input data you have, a few modes are available:

- **external** – all required rasters must be present in `data/rasters`. Missing files will cause the run to fail.  
- **hybrid** – preferred mode. Uses provided rasters where available, computes missing ones from DEM/landuse and catchment. Can resort to scalar values (defaults) where possible  
- **user_dtm** – all DTM-based covariates (slope, aspect, LS, etc.) must be provided; WaTEM factors can be from rasters or defaults.  
- **user_watem** – all WaTEM factors (K, C, P, ktc, etc.) must be provided; DTM covariates will be computed.  
- **internal** – ignores external rasters, computes everything internally from DEM and default constants.

None of these change the core model logic—just how data is supplied.

---

## How to Run

From the repo root:
```bash
python src/run_watem.py -c config.yaml --mode hybrid
```
Other flags include:

- `-c, --config PATH` — path to the YAML config (e.g., `config.yaml`).
- `--mode {external|hybrid|user_dtm|user_watem|internal}` — how inputs are sourced.
- `--save-rasters` — write GeoTIFF outputs.
- `--save-plots` — save png PNG maps.

_Notes:_ values from CLI override those in `config.yaml`.

## BMI quick start

```python
from src.bmi_watem import BmiWaTEM

m = BmiWaTEM()
m.initialize("config.yaml")
m.update()  # single annual step
sed = m.get_value("SEDI_OUT")
m.finalize()
print(type(sed), getattr(sed, "shape", None))
```


