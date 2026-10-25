"""Project-wide configuration: paths, constants and modelling assumptions.

Every number that is an *assumption* (costs, lead times, SKU filters) lives here
so that it can be traced from the docs and swept in the analysis.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
OUT_TABLES = OUT / "tables"
OUT_FIGURES = OUT / "figures"
TESTS = ROOT / "tests"

for _p in (DATA_RAW, DATA_PROC, OUT_TABLES, OUT_FIGURES):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- raw data
URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
ZIP_NAME = "online_retail_ii.zip"
XLSX_NAME = "online_retail_II.xlsx"
RAW_ZIP = DATA_RAW / ZIP_NAME
RAW_XLSX = DATA_RAW / XLSX_NAME
MANIFEST = DATA_RAW / "manifest.json"

WEEKLY = DATA_PROC / "weekly_demand.csv"
FORECASTS = DATA_PROC / "forecasts.parquet"
RESIDUALS = DATA_PROC / "residual_quantiles.csv"
RESULTS = OUT / "results.json"

# sha256 of the artefacts as published on 2023-05-22 (verified locally)
ZIP_SHA256 = "572e36277c2390fbfde10664750731e0a86f55e33470d91919085f0408e67bfb"
XLSX_SHA256 = "bcbe73b35f5b7babf197fb0cb983a11f5d9ff929078d4aa53d171b1f2df2e980"
SHEETS = ["Year 2009-2010", "Year 2010-2011"]

# ---------------------------------------------------------------- filtering
MIN_POS_WEEKS_UNIVERSE = 8    # SKUs written to weekly_demand.csv
MIN_POS_WEEKS_MODEL = 12      # SKUs in the modelling sample (evaluated pre-origin)
POS_WEEK_SENSITIVITY = (8, 12, 26)

# non-product stock codes (charter §3.1). Anything not matching PRODUCT_RE is
# also dropped; missing Description is dropped too.
NON_PRODUCT_CODES = {
    "POST", "D", "M", "C2", "DOT", "S", "B", "BANK CHARGES", "PAD", "PADS",
    "CRUK", "AMAZONFEE", "ADJUST", "ADJUST2", "SP1002",
}
NON_PRODUCT_PREFIXES = ("TEST", "GIFT", "DCGS", "BANK")
PRODUCT_RE = r"^[0-9]{4,6}[A-Z]{0,2}$"

# ---------------------------------------------------------------- evaluation
TEST_WEEKS = 26
# rolling origins, as offsets from the number of weeks (index of first test week)
ORIGIN_OFFSETS = (-27, -19, -11)
HORIZONS_ACC = (1, 2, 4)          # charter §4
HORIZONS = tuple(range(1, 9))     # 1..8; 1/2/4 accuracy, 2/4/8 decision
CALIB_WEEKS = 13                  # conformal window (weeks immediately before origin)
LAGS = tuple(range(1, 9))
MIN_TRAIN_WEEKS = 16              # a pseudo-origin needs this much history

# ---------------------------------------------------------------- decisions
LEAD_TIMES = (2, 4, 8)            # weeks; also the aggregation window of demand
COST_RATIOS = (1, 3, 9, 19)       # Cu / Co, exogenous assumption
HOLD_COST_WEEK = 0.01             # Co = 1% of unit price per week held (assumption)
N_BOOTSTRAP = 500                 # CI resamples (kept small; CI is wide anyway)
N_BOOTSTRAP_TEST = 10_000         # focal p-value: resolution 1/(B+1) = 1e-4
N_BOOTSTRAP_TEST_CHUNK = 1_000    # draw the test bootstrap in bounded chunks
BOOTSTRAP_SEED = 20261024

# ---------------------------------------------------------------- misc
SEED = 20261024
N_QUANTILE_GRID = 199             # residual quantile grid, q = 0.005..0.995
CLASSES = ("fast", "medium", "slow", "intermittent")

