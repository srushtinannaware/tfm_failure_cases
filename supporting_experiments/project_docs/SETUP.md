# Setup

TabPFN-2.5, TabPFN-3, and TabICL v2 all download pretrained checkpoints
from Hugging Face Hub on first use, so this needs a machine (or Colab
instance) with normal internet access.

Current scope is exactly three models: `tabpfn_v2_5`, `tabpfn_v3`,
`tabicl_v2` (see `models.py`).

## 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

**TabPFN and TabICL use different devices by default** — see `models.py`'s
`TABPFN_DEVICE` / `TABICL_DEVICE`. This isn't an oversight, it's because
the two models behave differently under memory pressure:

- **TabPFN** uses the fastest device available (CUDA > MPS > CPU,
  auto-detected). Its per-estimator feature subsampling caps how many
  columns any single ensemble member processes at once (~150-200,
  regardless of total column count), which structurally bounds its
  memory use even at 5,000 columns. It never OOM'd during testing on
  Apple Silicon/MPS, so it's safe to run at full GPU speed.
- **TabICL** defaults to CPU. It has no equivalent per-estimator cap —
  its column-attention memory use scales with total feature count — and
  it crashed with "MPS backend out of memory" at 1,000 columns during
  testing, even after tuning its `batch_size` down to 1 (the minimum).
  CPU has no equivalent per-device memory ceiling, trading speed for
  actually completing the sweep without crashing.

Override either independently if you want to experiment:
```bash
export FAILURE_CASE_TABPFN_DEVICE=cpu
export FAILURE_CASE_TABICL_DEVICE=mps   # or cuda -- expect this to risk OOM again at high column counts
```
`run_benchmark.py` prints both devices at the start of every run so you
can always see what's actually being used.

## 2. Accept the TabPFN license and get a token

TabPFN-2.5 and TabPFN-3 model weights are gated on Hugging Face
(non-commercial research license).

1. Go to https://ux.priorlabs.ai, log in, and accept the license under the
   License tab.
2. For a headless/CI/notebook environment, set:
   ```bash
   export TABPFN_TOKEN=<token from your account>
   ```
   Otherwise, the first `.fit()` call will try to open a browser for you
   to log in interactively — fine on a laptop, won't work over SSH.

TabICL (`tabicl` package) is **not** gated — it downloads straight from
Hugging Face Hub with no login step, so it only needs plain internet
access.

## 3. Run it

```bash
# full sweep, 3 seeds, both tasks, all 3 in-scope models
python run_benchmark.py

# faster sanity check first (recommended before committing to the full sweep)
python run_benchmark.py --column-sweep 50 200 1000 --seeds 0 --models tabicl_v2 tabpfn_v2_5

# then the full thing
python run_benchmark.py

python plot_results.py
```

Results append to `results/results.csv` incrementally (one row per
task × model × column-count × seed), so if a run gets interrupted partway
you keep everything that already finished. Re-running `run_benchmark.py`
appends more rows rather than overwriting — delete or rename
`results/results.csv` first if you want a clean run.

## 4. What to expect

- `tabicl_v2` needs one checkpoint download (~few hundred MB), no login.
- `tabpfn_v2_5` and `tabpfn_v3` need the license acceptance step above,
  then download their own checkpoints.

First `.fit()` call per model downloads and caches its checkpoint
(`~/.cache/tabpfn/` and the Hugging Face cache respectively); subsequent
runs reuse the cache and are fast to start.

## 5. If you want more statistical confidence

`--seeds 0 1 2 3 4` gives you error bars (the plotting script already
shades ±1 std automatically once more than one seed is present). Given the
small sample sizes involved (100 train rows), the metrics have real
seed-to-seed variance — don't over-read a single seed's numbers,
especially past 1,000 columns.
