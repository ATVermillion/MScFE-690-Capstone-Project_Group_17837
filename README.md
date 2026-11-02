# Learned Market Regimes for a Core-Satellite Portfolio

MScFE 690 Capstone Project (WorldQuant University) · Group 17837 · cohort 26-09
Track 8.1: Machine Learning (Deep) Market Timing Strategies

**Research question.** Can a machine-learned market regime, forecast one week ahead, improve the risk-adjusted
outcome (Sharpe ratio, maximum drawdown) of a core-satellite portfolio by switching the satellite between an
aggressive and a defensive sleeve, compared with simple rules and static mixes?

> Status: Module 5 (draft). The proposal-stage design (autoencoder regimes, hand-picked stocks, 2015-2026) has
> been tested, corrected for look-ahead bias, and replaced by the design below after a peer review and an
> extended-sample study. The title will change if the group agrees; the deep component was tested and found to
> add nothing over simpler clustering.

---

## 1. Final design

| Step | Specification |
|---|---|
| Data | SPY, nine Select Sector SPDR ETFs, TLT, GLD, 3-month T-bill (FRED DGS3MO); daily, 1993-2026, from a committed snapshot |
| Features | Eight weekly SPY features at Friday close: returns (1, 4, 13 weeks), realised volatility (1, 13 weeks), distance from the 40-week average, 52-week drawdown, volume ratio |
| Regime labels | k-means (k = 2) on **return and trend features only** (1-, 4-, 13-week returns, distance from the 40-week average, 52-week drawdown); the weaker cluster is named Stressed. Refit every January on all earlier weeks (from 1994) |
| Forecast | XGBoost (200 depth-2 trees) on the eight features + current label + share of Stressed weeks in the last four → P(Stressed next week); refit every January; probabilities smoothed with a 2-week half-life |
| Decision | Defensive sleeve when the smoothed probability exceeds the break-even threshold (expected cost of a false alarm vs gain of a correct alarm, from earlier payoffs only) |
| Portfolio | 50% SPY core + 50% satellite. Aggressive sleeve: top-4 sector ETFs by 3-year beta, re-selected each January. Defensive sleeve: TLT + GLD (or T-bills). Drifting weights, 10-point rebalancing band, 5 bps per unit traded |
| Evaluation | Causal expanding-window walk-forward, 2006-2026 (2001-2026 with T-bills). Benchmarks: 40-week moving-average rule, persistence, oracle, slow/fast trend blend, static mixes, SPY. Excess-return Sharpe, drawdown, turnover; block-bootstrap CIs; Ledoit-Wolf Sharpe test; deflated Sharpe/information ratio over a registry of every strategy tried; attribution by regime transition, episode and tail weeks |

## 2. Main results (extended sample, setup A: sector ETFs vs TLT + GLD, 2006-2026, 1,081 weeks)

| Strategy | Ann. return | Sharpe (excess) | Max drawdown | Return diff vs 40-week rule (95% CI) |
|---|---|---|---|---|
| **Return-based labels + smoothed XGBoost** | 11.6% | **0.74** | −27.2% | +0.7% (−0.8%, +2.2%), LW p = 0.14 |
| Autoencoder labels + smoothed XGBoost | 11.1% | 0.70 | −28.7% | +0.2% |
| Autoencoder labels + XGBoost | 9.9% | 0.62 | −29.7% | −0.8% |
| 40-week moving-average rule | 10.7% | 0.66 | −28.7% | – |
| Static 50% SPY + 50% aggressive | 11.2% | 0.56 | −56.4% | +1.2% |
| SPY buy-and-hold | 11.0% | 0.58 | −54.6% | +0.8% |
| *Oracle (next week's return-based label known)* | *23.3%* | *1.43* | *−21.2%* | *upper bound* |

The same ranking holds with an S&P 100 stock sleeve (setup B: 0.87 vs 0.85 for the rule) and with T-bills as the
defensive sleeve from 2001 (setup C: 0.61 vs 0.55, p = 0.09).

## 3. Findings

1. **Regime timing is risk control, not extra return.** Every regime strategy roughly halves the maximum drawdown
   of SPY or a static mix (about −28% vs −55%) at a similar or better Sharpe ratio.
2. **The best ML design matches or modestly beats the 40-week rule, but not significantly.** After allowing for
   all strategies tried, the probability that it truly beats the rule is about 27-43% (deflated information ratio).
3. **Design choices matter more than the model.** Return-based labels and smoothed forecasts help in every setup
   and sub-period; the deep autoencoder adds nothing over plain k-means. Logistic regression, XGBoost and an LSTM
   run through the same causal walk-forward forecast about equally well (AUC 0.93-0.95 vs 0.86 for persistence;
   logistic regression is highest) and, once smoothed, give similar portfolios (setup A Sharpe 0.68-0.74);
   smoothed XGBoost is best or tied in all three setups (Extended notebook section 6).
4. **Gains come from entering stress; losses from leaving it.** Every model and rule exits too late; a faster
   exit (4-week trend) raises return but also volatility and drawdown (exploratory trial).
5. **Results depend on the kind of crisis.** The ML design does best in grinding bear markets (2001-02,
   2008-09), the slow rule in V-shaped crashes (2020, 2025); nothing helps in an inflation shock (2022), when
   bonds stop hedging.
6. **The proposal-stage results were partly an artefact.** Regime labels fitted once on 2015-2023 data leaked
   future information into the 2019-2023 walk-forward; refitting them causally lowered the XGBoost Sharpe ratio
   and made it significantly worse than the rule over 2019-2026 with the hand-picked sleeve.

## 4. Repository layout

```
Capstone_Extended_Sample.ipynb   Main evidence: 2000-2026, rule-based sleeves, final design (sections 1-5)
Capstone_Project_Develop.ipynb   Original study 2015-2026 + all diagnostics (guide at the top of the notebook)
Download_Long_Snapshot.ipynb     One-off Colab notebook that created data/snapshot_long/
notebooks/Capstone_Project.ipynb Proposal-stage notebook (Module 4), kept for reference
config/config.yaml               Settings of both studies
data/snapshot/                   Prices, volumes, FRED series 2015-2026 (used by the Develop notebook)
data/snapshot_long/              Prices, volumes, FRED series 1993-2026 (used by the Extended notebook)
results/                         Tables and figures of the Develop notebook
results/extended/                Tables and figures of the Extended notebook
src/data_loading.py              Proposal-stage data layer (not used by the current notebooks)
```

## 5. How to run

Both analysis notebooks read the committed data snapshots, so they run offline with no API keys.

```bash
git clone https://github.com/ATVermillion/deep-regime-detection-core-satellite-taa.git
cd deep-regime-detection-core-satellite-taa
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace Capstone_Extended_Sample.ipynb   # about 3 minutes on a CPU
jupyter nbconvert --to notebook --execute --inplace Capstone_Project_Develop.ipynb   # about 3 minutes on a CPU
```

In Colab, open a notebook from GitHub and run all; the first cell clones the repository. Re-downloading the data
(only needed to refresh it) requires the `FRED_API_KEY` Colab secret: set `REFRESH_SNAPSHOT = True` in the
Develop notebook, or run `Download_Long_Snapshot.ipynb`.

**Reproducibility.** Seed 42 for NumPy, PyTorch, k-means and XGBoost. Every forecast, label, threshold, basket
and model choice at a date uses only data before that date. All strategies tried on each walk-forward are listed
in the trial registries (Develop section 7.3, Extended section 4.3) and enter the deflated Sharpe ratios.

## 6. Data sources

Public data only: Yahoo Finance via `yfinance` (adjusted daily prices and volumes) and FRED (DGS10, T10Y2Y, DFF,
DGS3MO, VIXCLS). The S&P 100 stock sleeve uses today's index members, so it carries survivorship bias and is used
as a sensitivity check only.

## 7. Contributors

Weijie Zhu, Kevin Chang, Yuzhen Feng. See `CONTRIBUTORS.md`.
