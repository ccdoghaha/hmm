# HMM —  Hidden Markov Models & State Analysis with CatBoost

Point-form introduction to this repository.

## What this is
- A technical essay + repro script that read the CatBoost BTCUSD signal
  pipeline (`CNN_catBoost_Align_15min_o.py`), its results summarizer
  (`summary.py`), and the step-0.25 batch record (88 tables / 2,181,519 band
  rows, loops 0–999, trees 10–10,000) through a statistics → HMM → state
  analysis lens.
- Verified against real run data: frequency sums of 154.9 billion row-hits,
  horizons 1h–24h, windows 3 months – 4 years.

## Contents
- `Statistics_HMM_State_Analysis.md` — the essay (~4,200 words):
  - Part I — statistics inside the pipeline: binned multi-horizon labels,
    MultiRMSE fits, threshold-sweep conditional-moment tables, chain-mode
    (accumulated-holding) semi-Markov position engine, drawdown semantics.
  - Part II — what `summary.py` measures: rebin, range pooling, for-loop
    MaxDrawdown, normalization, tiers, export filters.
  - Part III — the empirical record: horizon gradient, band-state gradient,
    loop-phase erosion, selection caveats.
  - Part IV — HMM primer (forward–backward, Baum–Welch, Viterbi, durations,
    HSMM/semi-Markov link to the chain engine).
  - Part V — real Gaussian-HMM fits (hand-rolled EM, validated on synthetic
    data) on the per-loop win series of the two largest runs.
  - Part VI — three-layer state framework (market / model / signal-band) and
    concrete next steps.
- `stats_hmm_loop_fit.py` — self-contained repro: 1-D Gaussian HMM
  (scaled forward–backward, Baum–Welch with 40 restarts, Viterbi) fitted to
  the per-loop win series of `R7_…525600…` and `R6_…1051200…`.

## Key findings (point form)
- Signal-strength gradient: negative `round_medianA` bins win ≈51.0–51.4%;
  strongest positive bins win ≈57–58.7% — monotone conditional edge.
- Horizon gradient: win rate and per-row total rise with horizon
  (≈53.2% @1h → ≈59–60% @17–24h) on ~10x smaller samples.
- Model-state drift (clearest structure): the 1000-loop 1-year run peaks at
  54.87% win (loop 5), then erodes to a 52.73% plateau — HMM change-points at
  loops ≈210 / ≈667. corr(loop, win) = −0.815 (−0.963 beyond loop 200).
- The 2-year run agrees: peak 53.17% (loop 44) → plateau ≈51.1%.
- Band occupancy grows 32 → 585 rows/loop as the ensemble grows: dispersion
  up, edge down.
- Warm-continued ensembles gain most of their generalization in the first few
  hundred trees; beyond that, more trees mostly buy overfit.

## Caveats (point form)
- The batch CSV keeps only `total > 0` and `frequency > 1000` rows → every
  exported row is STRONG tier; chain-mode trades (frequency = 1) are NOT in
  that CSV (query the results DB directly for sojourn/hazard work).
- Totals are per-table normalized units, not USD.
- 2,181,519 band rows is a large multiple-comparison surface; top bands are
  exploratory, not confirmatory.
- The HMM fits describe model-maturity phases (training process), not market
  regimes; a market-regime HMM must be fitted on market observations
  (returns / realized vol / flow features).
- The pipeline source is not included here (it carries environment
  credentials); the essay documents its mechanics instead.

## Reproduce
- `pip install pandas numpy` (pandas for the CSV mining; the HMM script is
  numpy-only).
- `python stats_hmm_loop_fit.py` — point the CSV path at your
  `summary_step0.25_*.csv` batch file (see the top of the script).
- Essay numbers: see "Appendix A — reproduction" for the exact pandas
  snippets.

## License
MIT — see `LICENSE`. Analysis material; not investment advice.
