---
name: btc-direction-model-v1-result
description: DirectionModelV1 on BTC/USD fails catastrophically — test AUC=0.449, below random; overnight gap signal absent in 24/7 crypto
metadata:
  type: project
---

DirectionModelV1 applied to BTC/USD (h=60, dd=1.5%) achieves test ROC-AUC=0.449 (below 0.5 random baseline) on April 2026 test period. Train AUC=0.998 — massive overfit with zero generalization.

**Why:** Two compounding causes: (1) The model's core mechanism is QQQ overnight gap prediction — there is no session break in 24/7 crypto, so `hour_of_day` has no privileged end-of-day window to anchor on. (2) Positive rate collapsed 2.21% (train) → 0.66% (test) due to BTC's April 2026 bull recovery; the model was calibrated for a bear environment.

**How to apply:** Do not apply DirectionModelV1 or the h=60, dd=1.5% label to BTC/USD without redesigning the label. Recommended next steps: (a) use dd=4–7% at h=60 to target similar positive rate, or (b) use a shorter horizon (h=10–20 min) with dd=1.5%, or (c) test on a BTC bear/correction period instead of April 2026. See `reports/local/btc_direction_model_v1_2026-05-31/report.md`.

Related: [[direction-model-v1-baseline]]
