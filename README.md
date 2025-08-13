# WaTEM-SEDEM (Python BMI Wrapper)

The repo lets you run the core WaTEM-SEDEM model using Python, wrapped with the Basic Model Interface (BMI) with the use of simplified input file preprocessing logic from [pywatemsedem](https://github.com/watem-sedem/pywatemsedem).

It can be used in two ways:

- **As a standalone script** via `src/run_watem.py` — to run directly using a config file.  
- **As a BMI module** (`bmi_watem.py`) — within a wrapper to integrate into larger model frameworks.

**Note:** This is not a full-featured version. It supports a single default scenario with no tillage, strips, infrastructure, or multi-factor options. Modes control *input files configuration* only.

---

## Structure

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

Some preprocessing steps (e.g., slope, LS-factor) call **SAGA GIS** (`saga_cmd`).  
Ensure SAGA is installed and available in your `PATH`. Usability has been checked for **SAGA GIS 8.5.1**  
([download here](https://sourceforge.net/projects/saga-gis/files/SAGA%20-%208/SAGA%20-%208.5.1/)).

### 1. Clone and install
```bash
git clone https://github.com/pihchikk/Watem-SEDEM-python.git
cd Watem-SEDEM-python
pip install -r requirements.txt
```

### 2. From the repo root:
```bash
python src/run_watem.py -c config.yaml --mode hybrid
```
Other flags include:

- `-c, --config PATH` — path to the YAML config (e.g., `config.yaml`).
- `--mode {external|hybrid|user_dtm|user_watem|internal}` — how inputs are sourced.
- `--save-rasters` — write GeoTIFF outputs.
- `--save-plots` — save png PNG maps.

Values from CLI override those in `config.yaml`.

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


