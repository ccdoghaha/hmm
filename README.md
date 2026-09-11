# HMM —  Hidden Markov Models & State Analysis with CatBoost

Formula-format introduction to the statistics implemented by
`CNN_catBoost_Align_15min_o.py` — the BTCUSD CatBoost signal pipeline this
repository analyses. Point form; every bullet is a statistic, an estimator,
or an identity the code computes.

> The pipeline source itself is not in this repo (it carries environment
> credentials); the formulas below are its specification.

## 0. Notation
- `t` = 1-minute `open_timestamp` index; `x_t ∈ R^43` = feature row
  (RSI/MACD/realized vol/OBV/VWAP/candle/cyclic clocks).
- `P(t)` = close at `t`; `N` = forecast horizon in minutes,
  `N ∈ {60, 120, …, 1440}`.
- `f_θ` = one CatBoost ensemble; `J` = number of outputs of a fit.

## 1. Label quantizer

$$\Delta_N(t)=P(t+N)-P(t)$$

$$y^{(N)}_t=B_N\!\left(\Delta_N(t)\right)\in\{-1.00,\dots,-0.01,\;+0.01,\dots,+1.00\}\subset[-1,1]$$

`B` = 15-minute map (active edges, 7 per side):

$$B(\delta)=\begin{cases}
+1.00 & \delta\ge 150\\
+0.80 & 100\le\delta<150\\
+0.40 & 50\le\delta<100\\
+0.20 & 20\le\delta<50\\
+0.10 & 10\le\delta<20\\
+0.05 & 5\le\delta<10\\
+0.01 & 0\le\delta<5\\
-0.01 & -5\le\delta<0\\
-0.05 & -10\le\delta<-5\\
-0.10 & -20\le\delta<-10\\
-0.20 & -50\le\delta<-20\\
-0.40 & -100\le\delta<-50\\
-0.80 & -150\le\delta<-100\\
-1.00 & \delta<-150
\end{cases}$$

`Y_change1440` uses the doubled-edge map `B₂ₓ` (edges 10/20/40/100/200/300).

- Consequence: labels are monotone, clipped *strength scores* — a rank-like
  transform of Δ, not the return itself.

## 2. Estimator

$$\hat p_t=f_\theta(x_t),\qquad f_\theta:\mathbb{R}^{43}\to\mathbb{R}^{J}$$

$$\mathcal{L}_{\mathrm{MAE}}(\theta)=\frac{1}{n}\sum_{t=1}^{n}\left|\hat y_t-y_t\right|,
\qquad
\mathcal{L}_{\mathrm{M}}(\theta)=\sqrt{\frac{1}{nJ}\sum_{t=1}^{n}\sum_{j=1}^{J}\left(\hat y_{tj}-y_{tj}\right)^{2}}$$

- Three fitting shapes: combined (J targets, L_M), per-target (one MAE fit
  per target), window-levels (one MAE fit per window scale).
- Warm continuation (`ACCUMULATED_EPOCH` / `LOOP_TIME`):
  $f_k=\mathrm{grow}\!\left(f_{k-1};\,+10\ \text{trees}\right)$; tree count after
  loop $k$ is $|f_k|=10\,(k+1)$ — no cap (1000-loop runs reach $|f|=10{,}000$).
- Reported `train_loss` = the same loss re-evaluated on the train window
  (stored scaled ×1e6).

## 3. Threshold sweep — conditional moments

$$\mathcal{B}_b=\left\{t:\ \frac{b}{20}\le \hat p_t < \frac{b+1}{20}\right\},
\qquad b\in\{-20,\dots,19\}\ \ (\text{step }0.1)$$

Per band (one row of the results table):

$$n_b=|\mathcal{B}_b|,\qquad
\bar y_b=\frac{1}{n_b}\sum_{t\in\mathcal{B}_b}y_t,\qquad
\tilde y_b=\mathrm{med}_{t\in\mathcal{B}_b}\,y_t,\qquad
T_b=\sum_{t\in\mathcal{B}_b}y_t,\qquad
\hat w_b=100\cdot\frac{1}{n_b}\sum_{t\in\mathcal{B}_b}\mathbf{1}\!\left[y_t>0\right],
\qquad h_N=60N\ \mathrm{s}$$

- Each row estimates $\mathbb{E}\left[y^{(N)}\mid \hat p^{(N)}\in\mathcal{B}_b\right]$
  and $P\left(y^{(N)}>0\mid \hat p^{(N)}\in\mathcal{B}_b\right)$, with sample
  size $n_b$ — a nonparametric conditional-moment table.
- Joint mode (J = 24): one model predicts all horizons; each horizon is
  swept *marginally* on its own column — a 24-fold joint DFS is avoided
  because strongly correlated outputs keep bin intersections non-empty and
  explode the node count.
- Long-horizon filter (`N` ≥ 120): the band scan keeps rows with
  $\hat p^{(N)}_t>0$ only ⇒ estimates become
  $\mathbb{E}\left[y^{(N)}\mid \hat p^{(N)}\in\mathcal{B}_b,\ \hat p^{(N)}>0\right]$.
- `TEST_SET_ALL`: the stacked out-of-sample set
  $O=\bigcup_c \mathrm{Test}_c$ is swept once per loop from the frozen
  snapshots $\{f_k\}$.

## 4. Chain mode — semi-Markov sojourn

A position opens at a band row $t_0$; hour $u$ is held iff every 1-minute
row inside it still passes the gate:

$$g_u=\prod_{s=t_0+3600(u-1)}^{t_0+3600u-1}
\mathbf{1}\!\left[\hat p_s\ge a\ \wedge\ y^{(60)}_s>0\right],
\qquad
S=1+\max\{k\ge 0:\ g_1=\dots=g_k=1\}\ \ (\le 24)$$

- Extension also requires the next hour's start row to exist on the
  1-minute grid; the position closes at the first failing hour (its loss
  included) or at the 24-hour cap.
- Record: frequency $=1$, holding $=3600S$ s, outcome $v=y^{(60S)}(t_0)$ —
  fallback $\sum_{u\le S}y^{(60)}\!\left(t_0+3600(u-1)\right)$ when the span
  column is NaN.
- Letters (schema-stable): $L_u=\lfloor 20\,\hat p_{t_0+3600(u-1)}\rfloor/20$
  for $u\le S$; sentinel $L_u=-1.0$ beyond; every row carries A..X (24
  letters), so insert batches never drift.
- Survival view: discrete hazard $h_u=P(S=u\mid S\ge u)$;
  $\mathbb{E}[S]=\sum_{u\ge0}P(S>u)$ — estimable from the frequency-1 chain
  rows of the results DB.

## 5. Risk statistics

$$E_k=\sum_{i\le k}v_i,\qquad
\Pi_k=\max\!\left(0,\ \max_{j\le k}E_j\right),\qquad
\mathrm{MDD}=\min_k\left(E_k-\Pi_k\right)\ \le 0$$

- Trade path: $v_i$ = band rows in `open_timestamp` order (single-window
  rows keep this value).
- Window path: $v_i$ = band totals per window
  $(control\_base,\ start\_index)$ — overwrites multi-window bands only.
- $\mathrm{PNR}=T_b/\lvert\mathrm{MDD}\rvert$ when $\mathrm{MDD}\ne0$, else 0.
- Self-check identity:
  $\#\left\{T_b<0\ \wedge\ \mathrm{MDD}=0\right\}=0$ — any violation is
  printed as a WARNING.

## 6. Run ledger

- Output table = smallest free version prefix $R_k(\text{base})$;
  append-only — nothing is dropped or overwritten, every batch shares one
  resolved table.
- Rows carry the loop index $k$ (`loop_round`), the tree count
  $|f_k|=10(k+1)$, the fit's `train_loss`, and its dates — every statistic
  is traceable to a model state.

## 7. What this repository adds

- **`Statistics_HMM_State_Analysis.md`** — the full statistics → HMM →
  state-analysis essay over this pipeline, its summarizer, and the
  step-0.25 batch record (2,181,519 rows, 88 runs): binned-label theory,
  sweep semantics, chain-mode survival, fitted Gaussian HMMs on the
  per-loop series, and a three-layer state framework.
- **`stats_hmm_loop_fit.py`** — self-contained repro for the HMM fits
  (validated EM + Viterbi, numpy only).
- Findings and caveats (signal-state gradient, loop-phase erosion around
  loops 210/667, export-filter blind spots) live in the essay.

## Reproduce

- CSV mining: `pandas`; HMM fits: `numpy` only — see the script header.
- The pipeline writes these rows to Postgres; `summary.py` exports the
  batch CSV the essay numbers come from.

## License

MIT — see `LICENSE`. Analysis material; not investment advice.
