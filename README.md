# Regime-Shift Asset Allocation

A regime-aware tactical asset allocation system. It first figures out the market's hidden
state (Bull / Bear / Crisis) using a Hidden Markov Model, then picks equity/gold/bond
portfolio weights for that regime through convex optimization (`cvxpy`) — all validated
with walk-forward testing so the backtest can't quietly cheat by peeking into the future.

## What's in this repo

- `Regime_Shift_Asset_Allocation.ipynb` — the full pipeline end to end, with explanations
  at each stage: data → features → regime detection → walk-forward validation →
  portfolio optimization → backtest → results.
- `README.md` — this file.

## How to run it

```bash
pip install yfinance hmmlearn cvxpy scikit-learn matplotlib pandas numpy scipy nbformat
jupyter notebook Regime_Shift_Asset_Allocation.ipynb
```

Run all cells top to bottom. **You'll need internet access to Yahoo Finance for the real
results.** If Yahoo Finance is unreachable (rate-limited, blocked network, sandboxed
environment), the notebook automatically falls back to a synthetic regime-switching market
generated on the fly. The pipeline still runs end to end and every output is genuine — it
just demonstrates the mechanics rather than actual market behavior. Whichever data source
is active is shown clearly at the top of Phase 1, e.g. `Active data source: yfinance
(live)` or `Active data source: synthetic (fallback)`.

## Data

| Asset | Ticker | Source |
|---|---|---|
| Equity | `^NSEI` (Nifty 50) | Yahoo Finance |
| Gold | `GOLDBEES.NS` | Yahoo Finance |
| Bonds | `LTGILTBEES.NS` | Yahoo Finance |
| Volatility | `^INDIAVIX` | Yahoo Finance |

## Key decisions and why

**Why 3 regimes (Bull / Bear / Crisis)?**
This matches the problem statement directly, and it's the minimum needed to tell "calm,"
"declining," and "acutely stressed" markets apart — each of which plausibly calls for a
different portfolio. Adding more states would let the model start chasing noise; fewer
would blur genuinely different market conditions together.

**Why these features (momentum + volatility + VIX + cross-asset gold vol)?**
Momentum captures the direction of the trend, volatility reflects how turbulent recent
moves have been, and VIX acts as the market's own forward-looking fear gauge. Gold's
realized volatility is included too, since a genuine flight-to-safety crisis usually shows
up there as well, not just in equities.

**Why log-transform the volatility-type features before the HMM?**
`vol_1m`, `gold_vol_1m`, and `vix` are strictly positive and right-skewed. A `GaussianHMM`
assumes each state's features are roughly Gaussian, so log-transforming volatility makes
that assumption much more reasonable — and noticeably sharpens how cleanly the model
separates regimes.

**Why multi-restart HMM fitting?**
EM (the algorithm `hmmlearn` uses to fit the HMM) only guarantees a *local* optimum. With a
single random initialization, it can converge to a degenerate solution — for example,
splitting the abundant "calm" days into two near-duplicate states while lumping the rarer
Bear and Crisis days into one, simply because that solution happens to achieve higher
likelihood when one true regime (Crisis) is far rarer than the others. The fix: fit the HMM
multiple times from different random seeds and keep whichever run converges to the highest
log-likelihood on the training data — essentially the HMM version of k-means++ restarts.

**Why regime-conditional mu/Sigma in the optimizer, instead of one trailing-window estimate
used for every regime?**
This was the single most important design choice in the whole project. If the optimizer's
expected-return and covariance inputs stayed the same regardless of the detected regime,
the risk-aversion parameter (`gamma`) alone couldn't turn a negative trailing-window return
estimate into an equity-heavy Bull-regime portfolio — the regime label would barely move
the final allocation. So instead, at every rebalance, expected returns and covariance are
estimated **separately for each regime**, drawing only on historically-labeled days of that
regime seen so far. Since Crisis days are scarce (especially early in the backtest), these
regime-conditional estimates are shrunk toward the unconditional (all-history) estimate
using `n / (n + k)` shrinkage, so a handful of extreme early days can't throw the estimate
wildly off balance.

**Why rebalance on regime-change-or-staleness, rather than every day?**
Rebalancing daily based on noisy day-to-day regime flips would generate unnecessary
turnover, letting transaction costs quietly eat away the strategy's entire edge — this trap
is explicitly flagged in the project brief. Weights only get a fresh optimization (and
incur a transaction cost) when the detected regime changes, or when `max_hold` trading days
have passed since the last rebalance (a staleness cap so a long-lived regime still gets its
estimates refreshed periodically). Between rebalances, weights simply drift with asset
returns.

## Avoiding lookahead bias — the checklist this notebook actually follows

1. Every engineered feature relies only on backward-looking rolling windows or same-day
   observable levels — never a future value.
2. The walk-forward harness uses an **expanding training window**: at each step, everything
   from the very start up to "now" is refit from scratch.
3. **Both** the feature scaler (`StandardScaler`, mean/std) **and** the HMM (means,
   covariances, transition matrix) are refit on **train-only** data at every step — fitting
   the scaler on the full dataset before splitting is a classic, easy-to-miss leak.
4. The model only ever labels the **held-out test block**, using parameters fit on
   strictly earlier data.
5. The portfolio optimizer's regime-conditional mu/Sigma at each rebalance date are
   estimated using only **strictly-past**, already-walk-forward-labeled days.
6. The exploratory full-sample HMM fit in Phase 3 is clearly marked as **not** the model
   used in the backtest — it's there purely as an initial visual sanity check.

## Results

See the notebook's "Results" section for the full performance table (Sharpe, Sortino, max
drawdown, Calmar, turnover), comparing the dynamic regime-shift strategy against a static
60/40 portfolio and an equal-weight portfolio, both with and without transaction costs.
Numbers will differ between a live-data run and the synthetic-fallback run — re-run with
internet access to Yahoo Finance to reproduce the actual submission results.

## Reproducing results

The notebook is deterministic given the same data: `np.random.seed(7)` is set at the top,
and the HMM's multi-restart fitting uses fixed, offset-per-window random seeds
(`random_state=100 + step`) in the walk-forward loop, so re-running produces the same
regime labels and backtest for the same input data.

## Tech stack

Python 3.9+, `hmmlearn`, `cvxpy`, `NumPy`, `Pandas`, `Matplotlib`, `yfinance`,
`scikit-learn`, `SciPy`.
