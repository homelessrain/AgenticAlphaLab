# Memory Index

- [h60 label is overnight gap predictor](h60-label-is-overnight-gap-predictor.md) — h=60 dd=1.5% QQQ/down label fires on end-of-day bars; model predicts overnight gaps not intraday moves
- [DirectionModelV1 baseline](direction-model-v1-baseline.md) — Current best model: 122 features, mean AUC=0.813, mean PR-AUC=0.299 across 7 regimes; +59% annualized SQQQ backtest
- [Calibration bimodal pattern](calibration-bimodal-pattern.md) — Near-perfect calibration in flat/recovery months; inverted calibration in correction months (March 2026 calib_r=−0.058 is a deployment risk)
- [BTC/USD DirectionModelV1 failure](btc-direction-model-v1-result.md) — Test AUC=0.449 (below random); overnight gap signal absent in 24/7 crypto; label needs redesign for BTC
