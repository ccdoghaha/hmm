# Statistics, Hidden Markov Models, and State Analysis

*A technical essay on `CNN_catBoost_Align_15min_o.py`, `summary.py`, and the
step-0.25 batch record `report0251/summary_step0.25_20260911T013630Z.csv`.*

---

## 0. Abstract

Three artifacts are examined as one measurement system:

1. **`CNN_catBoost_Align_15min_o.py`** — the estimator and experimental
   design: a CatBoost pipeline that turns raw 1-minute BTCUSD rows into
   binned multi-horizon labels, trains one or more regressors, sweeps
   threshold bins over out-of-sample predictions, and writes aggregated
   band statistics into a Postgres results table.
2. **`summary.py`** — the measurement layer: it reads those tables back,
   rebins thresholds, pools bins into ranges, recomputes drawdown along
   each band's window path, normalizes, classifies, and exports CSVs.
3. **`report0251/summary_step0.25_20260911T013630Z.csv`** — the empirical
   record: 2,181,519 band rows from 88 runs, steps of 0.25, loops 0–999.

The essay has three aims. First, to state precisely **what statistics the
pipeline computes** and what they mean (and do not mean). Second, to
introduce **Hidden Markov Models** as the natural formal language for the
kind of state-dependent behaviour this system probes — and to fit real
HMMs to the loop series found in the CSV. Third, to propose a **state
analysis** framework in which market state, model state, and signal-band
state are treated as three distinct layers, each observable in different
parts of the reviewed files.

---

## 1. The three artifacts, briefly

**The pipeline.** For each 1-minute row of the `model_BTCUSD_*_1440change15_m10_mtm_43_1min`
tables (43 causal features: RSI 14/21, MACD 12/26/9, realized volatility
5/15/60/240, Parkinson vol, volume ratio, OBV flow, VWAP deviation, candle
geometry, stochastic %K/%D, cyclic time encodings, MTM_z×vol signal, …),
the pipeline builds labels from the auxiliary `Y_change{N}` columns — the
forward price change over N minutes, N ∈ {60, 120, …, 1440}. Labels are
not raw returns but **binarized strength scores** mapped into [-1, 1]
(0.01, 0.05, 0.1, 0.2, 0.4, 0.8, 1.0 for positive moves; mirrored
negatives; a doubled edge set for the 24-hour horizon). A CatBoost
regressor (loss `MAE` for single target, `MultiRMSE` for multi-output)
trains on a rolling or frozen window. Predictions are then swept:
every prediction bin gets its own aggregate — count of rows, mean/median
of the realized `Y_change`, total, win rate — and each aggregate is one
row in the results DB.

**The summarizer.** `summary.py` reads a results table, re-bins the
`round_median*` threshold columns onto an exact grid, aggregates per band
(and per `loop_round`, which is never merged), computes `MaxDrawdown` with
a for-loop over each band's window equity path, normalizes totals so the
maximum band mean ≈ 0.999, tags a tier, filters (keep `total > 0` and
`frequency > 1000`), and exports per-target CSVs.

**The record.** The batch summary studied here is the step-0.25 view of
the whole results database as of 2026-09-11: **88 tables, 2,181,519 band
rows, 42 columns, frequency sums of 154.9 billion row-hits, loop indexes
0–999, model sizes 10–10,000 trees** (one shared 1000-loop family tops at
`trees = 10000 = 1000 loops × 10 trees/loop`).

---

## 2. Part I — The statistics inside the pipeline

### 2.1 From price change to binned labels

The label map is a **quantizer with hand-tuned edges**, not a statistical
transform. For the 15-minute base:

```
change >= 150  -> +1.00       -150 <= change < -100 -> -0.80
100 <= change < 150 -> +0.80  -100 <= change < -50  -> -0.40
 50 <= change < 100 -> +0.40   -50 <= change < -20  -> -0.20
 20 <= change <  50 -> +0.20   -20 <= change < -10  -> -0.10
 10 <= change <  20 -> +0.10   -10 <= change <  -5  -> -0.05
  5 <= change <  10 -> +0.05    -5 <= change <   0  -> -0.01
  0 <= change <   5 -> +0.01
```

and `Y_change1440` uses a doubled edge set (10/20/40/100/200/300). Three
statistical consequences matter:

* The target is **monotone but clipped**: extreme moves collapse into one
  value (±1.0), so the model learns *where in the distribution* a row
  sits, not the magnitude. This is a rank-like target.
* The bin edges make the target distribution **asymmetric**: most rows
  land in the tiny bins near zero, and the outer bins are rare events.
  A model that mostly predicts “near zero” is not wrong — it is matching
  the marginal.
* Because labels are bounded, `MAE` and `MultiRMSE` are bounded losses and
  train losses in the CSV (order 1e5 after the ×1e6 storage scaling) are
  comparable across runs.

### 2.2 The estimator and the loop design

CatBoost adds `CB_ITERATIONS` trees per fit; with `ACCUMULATED_EPOCH` /
`LOOP_TIME > 1` the previous model is **warm-continued** (`init_model`),
so a run grows an unbounded ensemble: loop k holds a model of about
`(k+1) × CB_ITERATIONS` trees. There is deliberately **no tree cap**.

This design makes the loop index a form of *pseudo-time*: each loop is a
distinct model state, snapshot in `accumulated_round_models`, evaluated
on the same test rows, and stored with its own `loop_round`. Whatever
drift exists between “young” and “old” models becomes measurable — and,
as Part III shows, it is large and monotone.

### 2.3 The threshold sweep: a non-parametric conditional table

For a prediction entry with columns `p_j` (one per target), the sweep
enumerates the product of threshold bins:

```
bin b of column j = rows with  b/ROUND_BINS <= p_j < (b+1)/ROUND_BINS
band     = a combination of bins, one per column (DFS with pruning)
record   = for each target j:
              mean/median/total of Y_change{N} over band rows
              win%      = share of those rows with positive outcome
              frequency = number of rows in the band
              holding   = N × 60 seconds
```

* `ROUND_STEP = 0.1` ⇒ bins of width 0.1 from −1.0 to +0.9 (20 bins).
* Joint mode (`TARGET_MODE > 1`) makes one model predict several horizons
  and sweeps the **joint** band (one threshold per output). `JOINT_ALL_CHANGE`
  relaxes this again: 24 outputs, but each horizon is swept **marginally**
  on its own column — because 24 correlated outputs make the joint bin
  intersection almost always non-empty and the DFS node count explodes.
* The long-horizon rule (implemented in the classic sweep for
  `Y_change{N}, N ≥ 120`): every band is visited, but inside a band only
  rows with **own prediction > 0** are kept. This is a *state-conditional
  subsample* — the aggregate statistics for long horizons describe
  “positive-signal” rows only. It must be read with that in mind.
* Each record is a **conditional moment estimate**:
  `E[Y_change{N} | quantized prediction ∈ band]` plus a hit rate. The
  frequency column is the sample size of that estimate.

### 2.4 Chain mode: a semi-Markov position engine

When `ACCUMULATED_HOLDING = True`, the sweep stops counting each
qualifying row as an independent 1-hour trade. Per band, positions open
serially and extend in whole hours while a **strict per-minute gate**
passes: every 1-minute row inside the hour just held must still clear the
band threshold (`p ≥ a`) and show its own `Y_change60 > 0`. The position
closes at the first failing hour (or at the 24-hour cap) and is recorded
as ONE trade whose outcome is the entry row's `Y_change{span}`:

```
position opens at row r0 (band b, pred>=a, Y_change60 finite)
for u = 1, 2, ..., 24:
    hour u held iff every minute in [t_u, t_u + 3600) has pred >= a
    and Y_change60 > 0
close at first failure; span = u hours
record: frequency 1, holding = u*3600 s, outcome = Y_change{u*60}(r0)
        letters A..X = the held hours' own binned predictions
        unreached hours = sentinel -1.0
```

Statistically this is a **discrete-time survival model**: the per-minute
gate is an empirical hazard, the held hours are a sojourn time S ∈ {1..24},
and the letters A..X are the observed trajectory of the binned signal
along the sojourn. That is precisely the structure of a **semi-Markov
(hidden semi-Markov) process** — see Part IV. Note that the currently
exported batch report **excludes** this data: single-trade chain rows have
`frequency = 1` and `summary.py`'s export filter keeps only
`frequency > 1000` (verified: the CSV's minimum frequency is 1001). So the
sojourn distribution of the chain engine is *not* visible in
`report0251`; it must be estimated from the results DB directly.

### 2.5 Risk statistics

Two drawdown definitions coexist:

* `_path_mdd(values)` — over an ordered sequence of trade outcomes:
  `eq += v; peak = max(peak, eq); mdd = min(mdd, eq - peak)` with the peak
  anchored at 0; always ≤ 0. This is what single-window sweep rows store.
* `_finalize_mdd` — the same for-loop, but over a band's **window equity
  path** (control_base × start_index). Only bands with more than one
  window row get overwritten; single-window bands keep the trade-path
  value, because min(0, total) would read 0 for every profitable band —
  the historical “MaxDrawdown is always 0” bug.

`ProfitNRisk = total / |MaxDrawdown|` (0 when MDD is 0) is the summary
layer's crude reward-to-pain ratio. The self-check that follows the
backfill (count negative-total rows vs rows with negative MaxDrawdown) is
good statistical hygiene: it converts a silent failure mode into a
printed warning.

### 2.6 Governance, versioning, reproducibility

Every run writes to `R{k}_<name>` if the base name exists — nothing is
ever dropped or overwritten. Practically, the results database becomes an
**append-only experiment ledger**: `CB0_` = first run, `R1_`–`R11_` =
re-runs, and every row carries `insert_time`, `loop_round`, `trees`,
`train_loss`, `start_date`/`end_date`. Statistical claims can therefore
always be traced to a specific run, window, and model size. The cost is
that the database accumulates old semantics (legacy 100-epoch rows next
to 10-tree chain rows) — filter by table family before comparing.

---

## 3. Part II — What `summary.py` measures

`summary.py` is the statistical pipeline that turns result rows into
comparable report rows. Its operations, in order:

1. **Rebin** — thresholds are re-snapped onto the 0.1/0.25/0.5 grid with
   `searchsorted(edges, v, side='right') - 1`; `-0.0` maps to `-step`,
   NaN is preserved. This makes runs with different `ROUND_BINS` joinable.
2. **Band aggregation** — group by all band columns (`round_median*`,
   `add_number`, `test_set`, `holding`, `loop_round`) and sum/count.
   `loop_round` in the identity guarantees **loops are never merged**.
3. **MaxDrawdown** — for-loop over each band's window path, identical
   semantics to `_finalize_mdd` so stored and reported values agree.
4. **Range pooling** — adjacent 0.1-bins are pooled into `RANGE_WIDTH`
   ranges (snapped so they partition the grid exactly) and emitted as
   extra rows tagged `agg = range0.5`. This is a smoothing device: bin
   rows are noisy at small samples, ranges trade resolution for power.
5. **Normalization** — divide `total`, `mean`, `MaxDrawdown` by
   `max(mean)/0.999`. Consequence: totals in the CSV are in **normalized
   units**, not USD — comparable across tables, meaningless as absolute
   PnL.
6. **Tiering** — `STRONG` if `total > -3000 and win > 5`; `MODERATE` if
   `total > -6000 or win > 2`; else `WEAK`; `FAIL` if `win == 0`. Because
   the export already keeps only `total > 0` and `frequency > 1000` rows,
   *every exported row is STRONG* (the CSV confirms: all 2,181,519 rows).
   The tier column is therefore informative only before the export
   filter — in the CSV it is a constant.
7. **Export filter** — `total > 0` and `frequency > 1000`, as above. This
   is the single most important caveat when reading the CSV: it is a
   **conditioned sample** (profitable, high-count bands only; single-trade
   chain rows excluded).

---

## 4. Part III — What the record shows

### 4.1 Scale

| quantity | value |
|---|---|
| rows × columns | 2,181,519 × 42 |
| runs (tables) | 88 (families: windows 131,400–2,102,400 min ≈ 3 mo … 4 y; `T1` single-horizon and `T24_J1` joint runs; versions R1–R11) |
| step | 0.25 |
| loop index | 0 – 999 (1000 loops ⇒ 10,000 trees in the longest run) |
| tree counts | 10 – 10,000 |
| frequency | min 1,001 (filter), median 52,685, max 3,389,871; total 154.9 billion |
| total (normalized) | median 2,139; p90 10,805; max 1,688,641 |
| win | p25 50.7, median 51.9, p75 54.1, p90 59.4, max 100 |
| win ≥ 60 | 9.38% of rows; win ≥ 65: 6.52% |
| MaxDrawdown | 0 in ≥75% of rows (single-window rows keep trade-path MDD); min −248,540 |
| ProfitNRisk | 0 in ≥75% of rows; p90 0.96; max 1,056,000 |

Largest tables: `R7_…525600…BUY` (552,502 rows / 1000 loops),
`R6_…1051200…` (390,731), `R6_…2102400…` (348,036),
`R6_…1576800…` (295,402).

### 4.2 The horizon gradient (holding_time = horizon × 60 s)

| horizon | rows | win % | total/row |
|---|---|---|---|
| 1 h | 80,052 | 53.22 | 4,322 |
| 2 h | 89,321 | 52.84 | 2,818 |
| 8 h | 133,189 | 54.09 | 5,960 |
| 15 h | 146,687 | 54.05 | 6,082 |
| 16 h | 87,006 | 54.58 | 4,202 |
| 17–23 h | ≈26,500 each | 58.1–59.7 | 6,000–9,600 |
| 24 h | 24,376 | 60.00 | 7,072 |

Read carefully: win rate and per-row total **rise with horizon** while
the sample counts for the longest horizons are an order of magnitude
smaller (tens of thousands vs hundreds of thousands). Longer horizons
partly “win” by construction (more time to drift up in a bull market)
and partly through the long-horizon positivity filter that keeps only
positive-prediction rows. This is a state-conditional edge, not a free
lunch.

### 4.3 The band gradient (round_medianA as signal state)

Aggregating all 2.18M rows by the first threshold letter:

| A bin | rows | win % | total/row |
|---|---|---|---|
| −1.00 … −0.60 | 22–25 k each | 51.5–52.9 | 2,600–3,100 |
| −0.50 … −0.10 | 34–56 k each | 51.0–51.4 | 2,500–8,450 |
| 0.00 | 114,488 | 52.94 | 7,874 |
| +0.10 | 108,463 | 54.59 | 6,350 |
| +0.20 | 100,424 | 55.36 | 4,342 |
| +0.40 | 76,065 | 56.33 | 3,855 |
| +0.60 | 59,511 | 56.27 | 2,398 |
| +0.80 | 50,132 | 58.41 | 4,590 |
| +0.95 | 43,644 | 58.69 | 24,626 |

The empirical conditional distribution is **monotone in the signal**: the
stronger the predicted move, the higher the hit rate, from ~51.3% in
negative bins to ~57–58.7% in the strongest positive bins. The extreme
total/row at A = +0.95 is a selection outlier (fewer, more profitable
rows) — see 4.6.

### 4.4 Loop dynamics: the clearest structure in the data

The 1000-loop run `R7_CB0_A1_525600_…_10_BUY` gives a clean model-state
trajectory (win averaged per loop):

* loop 0: 52.32 → peak **54.87 at loop 5** → plateau ≈ **52.73** (loops ≥ 200)
  → last 52.53.
* correlation(loop, win): **−0.815**; for loops ≥ 200: **−0.963**.
* per-100-loop segments: win 53.41, 53.09, 52.87, 52.80, 52.79, 52.77,
  52.74, 52.69, 52.62, 52.56 — a smooth monotone decay.
* rows per loop: 32 (loop 0, predictions concentrated in few bins) →
  median 572 → max 585 (all bins occupied).
* best loop total 1,692,202 (loop 21) vs worst 1,447,432 (loop 665):
  aggregate PnL varies by only ≈ −3% … +14% around its mean while the
  model grows 1000× in trees.

The same shape appears in the 2-year run `R6_…1051200…`: first 50.38,
peak 53.17 at loop 44, plateau (≥200) 51.13, last 50.87.

Interpretation: **most of the generalization is in the first few hundred
trees; afterwards the warm-continued ensemble drifts toward overfit and
the aggregate edge erodes to a stable floor.** Model size is a poor
return-on-investment lever beyond that point; feature/state engineering
is where the next edge must come from.

### 4.5 Selection and multiple-comparison caveats

* All rows are profitable (`total > 0`) because of the export filter.
* The top-1000 rows by total carry only **0.49% of all frequency** — the
  extreme band totals are thin slices of the data, not the main mass.
* 2.18M band rows × 88 runs is a massive multiple-comparison surface;
  the “best bands” are exploratory, not confirmatory.
* Totals are normalized per table; comparing totals across tables of
  different base scale is invalid.
* Chain (frequency = 1) trades are absent from this CSV (see 2.4) — the
  entire accumulated-holding side of the engine is unmeasured here.

---

## 5. Part IV — Hidden Markov Models, in the language of this system

A Hidden Markov Model describes a sequence of observations `o_1..o_T`
generated by a latent state sequence `s_1..s_T` on K states:

```
P(s_1)   = pi                       (initial distribution)
P(s_t | s_{t-1}) = A                (transition matrix, rows sum to 1)
o_t | s_t=k ~ B_k(·)                (emission; e.g. Gaussian N(mu_k, sigma_k^2))
```

Three classical problems: *filtering/smoothing* (`P(s_t | o_1..o_T)`, the
forward–backward recursions), *most likely state path* (Viterbi), and
*parameter estimation* (Baum–Welch = EM). Key quantities:

```
stationary distribution:   pi* = pi* A
expected sojourn in k:     E[T_k] = 1 / (1 - A_kk)
```

Two limitations matter here:

1. **Geometric sojourns.** A first-order HMM's state durations are
   geometric. Real regimes (and the chain engine's hold lengths) have
   duration structure that geometric lifetimes cannot capture — which is
   why *semi-Markov* / hidden semi-Markov models (HSMMs) exist: the
   transition is split into a state-transition kernel and an explicit
   duration distribution `d_k(u)`. The chain engine of §2.4 is precisely
   a hand-coded HSMM sampler: state = “position open”, duration = held
   hours, emission = the letters A..X.
2. **Stationarity.** Baum–Welch's likelihood assumes the process is
   stationary. If a series drifts (as the loop series do), the fitted
   model will soak the drift into near-absorbing states.

Why HMMs are the right lens for this pipeline:

* The sweep already **discretizes** the continuous signal into states
  (bins), and measures P(outcome | state) — the emission table of a
  hidden-state model where the state is, for now, *observed*.
* The `Y_change60..1440` ladder is a **multi-scale observation vector**
  of one latent market process; a Gaussian HMM over features like
  (return, realized vol, volume ratio, OBV flow) would let “trend”,
  “range”, “stress” become the hidden states behind those observations.
* The loop series is a **model-state chain**: warm continuation is a
  genuine Markov process on model parameters, and its transitions are
  what a state analysis should measure.

---

## 6. Part V — Real HMM fits on the loop series

**Method.** A from-scratch 1-D Gaussian HMM (scaled forward–backward,
Baum–Welch with 40 random restarts, Viterbi decoding) was validated on a
synthetic two-state chain (true means 0 and 6, persistence 0.95/0.90):
recovered means 0.01 / 5.99, persistence 0.951 / 0.893. It was then fitted
to the per-loop win series of the two largest tables.

### `R7_…525600…` (1000 loops)

| K | state means (win %) | Viterbi phases (loop ranges) | phase lengths |
|---|---|---|---|
| 2 | 52.73 / 53.23 | 53.23 for loops 0–209; 52.73 for 210–999 | 210 / 790 |
| 3 | 52.63 / 52.80 / 53.23 | 53.23 (0–215); 52.80 (216–666); 52.63 (667–999) | 216 / 451 / 333 |
| 4 | 52.64 / 52.78 / 52.86 / 53.24 | 53.24 (0–208); 52.86 (209–333); 52.78 (334–657); 52.63 (658–999) | 209 / 125 / 324 / 342 |

### `R6_…1051200…` (1000 loops)

| K | state means (win %) | Viterbi phases (loop ranges) |
|---|---|---|
| 2 | 50.97 / 51.83 | 51.83 (0–461); 50.97 (462–999) |
| 3 | 50.89 / 51.21 / 52.20 | 52.20 (8–262); 51.21 (263–696); 50.89 (697–999) |
| 4 | 50.89 / 51.08 / 51.39 / 52.20 | 52.20 (8–256); 51.39 (257–460); 51.08 (461–698); 50.89 (699–999) |

**How to read these fits.** The fitted transition matrices are
near-absorbing (diagonal ≈ 0.995–1.000), so expected state lifetimes run
into the hundreds of loops — the states are not “fast regimes” but
**phases of model maturity**; the phase lengths in the tables are Viterbi
run lengths. BIC keeps improving with K (each extra state carves off one
plateau), which confirms the series is a **piecewise-constant-like
drift**, not a two-regime switching process. Both runs agree: an early
high phase (0–~210 loops in R7; 0–~262 in R6), then a longer eroded
plateau. A naive median split of the same series reports P(stay) ≈ 0.948
and duration ≈ 19 loops — also true locally, but the HMM decomposition is
the better summary because it captures the global structure.

**Do not over-read.** For *market* state detection an HMM must be fitted
to market observations (returns/vol/flow) on a real time axis; the fits
above are state analysis of the **training process**, and they say what
§4.4 already suggested: the ensemble's useful information saturates early.

---

## 7. Part VI — State analysis: a three-layer model

The artifacts support a clean three-layer decomposition.

**Layer 1 — Market state (latent; not yet measured).**
The 43 m10 features + the 24-horizon `Y_change` ladder are the observation
vector. Proposed: a 2–4 state Gaussian HMM on standardized
(1-min return, realized vol 60, volume ratio vr60, MTM_z) to obtain per-row
regime probabilities; validate with Viterbi path stability and
state-conditional forward-return distributions; keep an explicit
semi-Markov duration term because crypto regimes have long tails.

**Layer 2 — Model state (observable: `loop_round`, `trees`, `train_loss`).**
This essay measured it: loop phases with means 53.2 → 52.6 (R7) and
51.8 → 50.9 (R6), transitions around loops 210/460/660; band occupancy
growing 32 → 585 rows per loop. Actionable: tail-stop or down-weight warm
continuation once a change-point detector (Page's CUSUM or an HMM
fitted *online*) declares the plateau entered; treat `trees` beyond that
point as uninformative. The `insert_time` + `loop_round` + `trees`
columns already carry everything needed to monitor this continuously.

**Layer 3 — Signal state (the bands; emission table already in the CSV).**
`round_medianA..X` are the discretized states; `win`, `mean`, `total`,
`frequency` are state-conditional statistics. This is exactly an emission
table `P(outcome | state)`, with sample sizes. The long-horizon positivity
filter is a state-conditional restriction that should be stated wherever
these numbers are consumed.

**Cross-layer uses.**

* *Regime-conditional sweeps*: join Layer-1 regime probabilities into the
  sweep (as an extra group-by column) and re-export;
  `report0251`'s structure (band × window × loop) already anticipates
  multi-factor conditioning.
* *Sojourn/hazard estimation*: export the chain tables' `frequency = 1`
  rows (the current `frequency > 1000` filter hides them), then estimate
  `h(u) = P(close at hour u | held u-1 hours)` per band — the empirical
  hazard of the per-minute gate. That is the HSMM duration distribution
  for the trading layer, and it is one query away.
* *Selection discipline*: choose bands on early loops, confirm on the
  plateau phase (R7 loops ≥ 210; R6 ≥ 462). Train HMMs / select rules on
  one run (e.g. R6) and verify on an independent one (R7) — the versioned
  table naming already gives independent samples.
* *Effect sizes, not stars*: with frequency sums of 154.9 billion rows,
  a 0.1 pp difference is “significant” in every schoolbook test. Use
  confidence intervals on win rate (binomial with n = frequency), effect
  magnitudes, and stability across tables instead.
* *Risk as state*: run `MaxDrawdown`/`ProfitNRisk` per regime and per
  model phase; a band that is profitable only in one regime should carry
  that dependency in its row, not hide it.

**Concrete next steps (small, in order).**

1. Export chain-trade rows (`frequency = 1`) from one results table and
   compute the hazard `h(u)` and E[sojourn] per band.
2. Fit the Layer-1 HMM on `model_BTCUSD_*_m10_mtm_43_1min` and persist a
   `regime_id` series; add it as a sweep grouping key.
3. Track a per-loop change-point (CUSUM on win rate) and record a
   `plateau_flag` on inserted rows.
4. When comparing runs, always split by table family (window size,
   T1 vs T24_J1, version) before aggregating — the CSV mixes all of them.

---

## Appendix A — reproduction

All numbers above come from the reviewed CSV and were computed with pandas
in `cuda130`:

```python
df = pd.read_csv("report0251/summary_step0.25_20260911T013630Z.csv",
                 usecols=[...])            # 2,181,519 × 42
# band state table:   df.groupby('round_medianA')[[...]].agg(...)
# horizon table:      df.groupby('holding_time')[[...]].agg(...)
# loop series:        df[df.table_name == T].groupby('loop_round')['win'].mean()
# HMM fits:           stats_hmm_loop_fit.py  (this repo, hand-rolled EM + Viterbi)
```

The HMM script (`stats_hmm_loop_fit.py`, next to this essay) reproduces
the synthetic validation and all K = 2/3/4 fits. The CSV's sibling under
`report0251/.ipynb_checkpoints/` is an editor copy; use the main file.

## Appendix B — glossary of the key columns

| column | meaning |
|---|---|
| `round_median*` | threshold bins (A = first output, … X = 24h chain letters) |
| `holding_time` | seconds held; horizon N ⇒ 60·N (classic) or chain span |
| `loop_round` | warm-continue loop index = model state (never merged) |
| `trees` | ensemble size at that loop (10 × loops) |
| `frequency` | band sample size (rows / trades) |
| `win` | share of band rows with positive outcome (%) |
| `total` | normalized sum of outcomes in the band |
| `MaxDrawdown` | ≤ 0; for-loop over the band's equity path (peak at 0) |
| `ProfitNRisk` | total / abs(MaxDrawdown) |
| `tier` | STRONG/MODERATE/WEAK/FAIL (constant STRONG after export filter) |

## Appendix C — the one-paragraph version

The pipeline is a conditional-moment measurement machine built on top of
a binned, multi-horizon quantizer; the summarizer turns its output into
normalized, drawdown-aware band tables; the step-0.25 record shows a
small but monotone signal-conditioned edge (51–59% win rates across
bands, rising with horizon and with signal strength) and a clear
model-state story: warm-continued ensembles gain most of their edge in
the first few hundred trees and then erode to a plateau, which HMM phase
decomposition dates at roughly loops 210 / 460 / 660 depending on the
run. The natural next moves are to make the hidden layer explicit —
market regimes via HMM/HSMM on the m10 features, chain sojourns via the
frequency-1 export, and regime-conditional band reports — instead of
buying more trees.
