"""
Step 2: Train cluster-specific downturn prediction models and compare against baseline.

Design:
  - Full data window: 2021-03-01 → 2026-03-31 (covered by a single cached file pair)
  - Train period: 2021-03-01 → 2024-12-31  (~4 years)
  - Test period:  2025-01-01 → 2026-03-31  (~15 months, matching prior regime eval window)

Three models evaluated on C0 (Sharp Down) test days:
  1. Baseline     — trained on ALL days in the train period
  2. C0 model     — trained only on C0 (Sharp Down) days
  3. C1 model     — trained only on C1 (Normal Bear) days (control: wrong regime)

Primary hypothesis: C0 model outperforms baseline on C0 test days.
Control:            C1 model should perform worse than baseline on C0 test days.
"""

import sys, time
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.calibration import calibration_curve

sys.path.insert(0, '.')
from utils.ml_data_processor import MinuteAndDailyMLDataProcessor
from utils.feature_transformer import OLHCVFeatureTransformer, DailyOLHCVFeatureTransformer

OUT       = "reports/regime_specific_model_v1"
TRAIN_END = '2024-12-31'
TEST_START= '2025-01-01'
TEST_END  = '2026-03-31'
DATA_START= '2021-03-01'   # earliest date with full cache coverage

CLUSTER_NAMES = {0: 'Sharp Down', 1: 'Normal Bear', 2: 'Normal Bull',
                 3: 'Crash/Crisis', 4: 'Volatile Up'}

# ── 0. Load cluster labels ────────────────────────────────────────────────────
cluster_labels = pd.read_parquet(f'{OUT}/daily_cluster_labels.parquet')
cluster_labels.index = pd.to_datetime(cluster_labels.index).normalize()
print(f"Cluster labels: {len(cluster_labels)} days")
print("Cluster distribution:")
print(cluster_labels['cluster'].value_counts().sort_index()
      .rename(CLUSTER_NAMES).to_string())

# ── 1. Load full ML dataset (single cache hit) ────────────────────────────────
print(f"\nLoading ML data {DATA_START} → {TEST_END}...")
t0 = time.time()
proc = MinuteAndDailyMLDataProcessor(
    'QQQ',
    minute_feature_transformer=OLHCVFeatureTransformer(),
    daily_feature_transformer=DailyOLHCVFeatureTransformer(),
)
full_df = proc.run(DATA_START, TEST_END, drop_label_na=True)
print(f"Loaded {len(full_df):,} rows in {time.time()-t0:.1f}s")
print(f"Label rate: {full_df['label'].mean():.3%}")

feature_cols = proc.feature_columns
label_col    = proc.label_column

# ── 2. Add cluster column ─────────────────────────────────────────────────────
full_df = full_df.copy()
full_df['trade_date'] = pd.to_datetime(full_df['datetime']).dt.tz_localize(None).dt.normalize()
full_df = full_df.merge(cluster_labels.rename(columns={'cluster': 'cluster_id'}),
                        left_on='trade_date', right_index=True, how='left')
print(f"\nCluster coverage: {full_df['cluster_id'].notna().mean():.1%} of rows have a cluster label")

# ── 3. Train / test split ─────────────────────────────────────────────────────
full_df['datestr'] = full_df['datetime'].dt.strftime('%Y-%m-%d')
train_df = full_df[full_df['datestr'] <= TRAIN_END].copy()
test_df  = full_df[full_df['datestr'] >= TEST_START].copy()
print(f"\nTrain: {len(train_df):,} rows ({train_df['datestr'].min()} → {train_df['datestr'].max()})")
print(f"Test:  {len(test_df):,} rows  ({test_df['datestr'].min()} → {test_df['datestr'].max()})")

# Test-set breakdown by cluster
print("\nTest rows by cluster:")
for c, name in CLUSTER_NAMES.items():
    n = (test_df['cluster_id'] == c).sum()
    n_pos = test_df.loc[test_df['cluster_id'] == c, 'label'].sum()
    print(f"  C{c} {name:15s}: {n:6,} rows, {int(n_pos):4d} positive ({n_pos/n:.2%} rate)")


def train_xgb(X_train: pd.DataFrame, y_train: pd.Series, tag: str) -> xgb.XGBClassifier:
    pos_rate = float(y_train.mean())
    spw = (1.0 - pos_rate) / max(pos_rate, 1e-6)
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        random_state=42,
        n_estimators=300,
        max_depth=4,
        scale_pos_weight=spw,
        enable_categorical=True,
    )
    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"  [{tag}] n_train={len(y_train):,}, pos_rate={pos_rate:.3%}, "
          f"scale_pos_weight={spw:.1f}, fit_time={elapsed:.1f}s")
    return model


def evaluate_on(model: xgb.XGBClassifier, test_rows: pd.DataFrame,
                feature_cols: list, label_col: str, tag: str) -> dict:
    if len(test_rows) == 0:
        print(f"  [{tag}] SKIP — no test rows")
        return {}
    X = test_rows[feature_cols]
    y = test_rows[label_col]
    prob = model.predict_proba(X)[:, 1]
    if y.sum() < 5:
        print(f"  [{tag}] SKIP — only {int(y.sum())} positives")
        return {}
    auc    = roc_auc_score(y, prob)
    pr_auc = average_precision_score(y, prob)
    pos_rate = float(y.mean())
    print(f"  [{tag}] n={len(y):,}, pos_rate={pos_rate:.3%}, "
          f"AUC={auc:.3f}, PR-AUC={pr_auc:.3f}")
    return {'auc': auc, 'pr_auc': pr_auc, 'n': len(y),
            'n_pos': int(y.sum()), 'pos_rate': pos_rate, 'prob': prob, 'y': y}


# ── 4. Train 3 models ─────────────────────────────────────────────────────────
print("\n── Training models ──")

# Baseline: all training data
X_base = train_df[feature_cols].fillna(0)
y_base = train_df[label_col]
print("Baseline:")
baseline_model = train_xgb(X_base, y_base, 'Baseline')

# C0 (Sharp Down) model
c0_train = train_df[train_df['cluster_id'] == 0]
X_c0 = c0_train[feature_cols].fillna(0)
y_c0 = c0_train[label_col]
print("C0 Sharp Down:")
c0_model = train_xgb(X_c0, y_c0, 'C0 Sharp Down')

# C2 (Normal Bull) model — control / contrast
c2_train = train_df[train_df['cluster_id'] == 2]
X_c2 = c2_train[feature_cols].fillna(0)
y_c2 = c2_train[label_col]
print("C2 Normal Bull:")
c2_model = train_xgb(X_c2, y_c2, 'C2 Normal Bull')

# C1 (Normal Bear) model — additional comparison
c1_train = train_df[train_df['cluster_id'] == 1]
X_c1 = c1_train[feature_cols].fillna(0)
y_c1 = c1_train[label_col]
print("C1 Normal Bear:")
c1_model = train_xgb(X_c1, y_c1, 'C1 Normal Bear')


# ── 5. Evaluate all models on C0 test days ────────────────────────────────────
print("\n── Evaluation on C0 (Sharp Down) test days ──")
c0_test = test_df[test_df['cluster_id'] == 0].copy().ffill().fillna(0)

results = {}
print("Baseline on C0 test days:")
results['baseline'] = evaluate_on(baseline_model, c0_test, feature_cols, label_col, 'Baseline')
print("C0 model on C0 test days:")
results['c0_model'] = evaluate_on(c0_model, c0_test, feature_cols, label_col, 'C0 model')
print("C2 model on C0 test days:")
results['c2_model'] = evaluate_on(c2_model, c0_test, feature_cols, label_col, 'C2 model')
print("C1 model on C0 test days:")
results['c1_model'] = evaluate_on(c1_model, c0_test, feature_cols, label_col, 'C1 model')


# ── 6. Also evaluate on ALL test days for full context ────────────────────────
print("\n── Evaluation on ALL test days (for reference) ──")
all_test = test_df.ffill().fillna(0)
all_results = {}
for tag, model in [('Baseline', baseline_model), ('C0 model', c0_model)]:
    X = all_test[feature_cols]
    y = all_test[label_col]
    prob = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, prob)
    pr_auc = average_precision_score(y, prob)
    print(f"  [{tag}] AUC={auc:.3f}, PR-AUC={pr_auc:.3f}, pos_rate={y.mean():.3%}")
    all_results[tag] = {'auc': auc, 'pr_auc': pr_auc}


# ── 7. Per-month breakdown on C0 test days ────────────────────────────────────
print("\n── Per-month AUC on C0 test days ──")
c0_test['yearmonth'] = c0_test['datetime'].dt.to_period('M')
months = sorted(c0_test['yearmonth'].unique())

monthly_rows = []
for ym in months:
    sub = c0_test[c0_test['yearmonth'] == ym]
    y = sub[label_col]
    if len(sub) < 20 or y.sum() < 3:
        continue
    row = {'month': str(ym), 'n': len(sub), 'n_pos': int(y.sum()),
           'pos_rate': float(y.mean())}
    X = sub[feature_cols]
    for tag, model in [('baseline', baseline_model), ('c0_model', c0_model)]:
        try:
            prob = model.predict_proba(X)[:, 1]
            row[f'auc_{tag}'] = roc_auc_score(y, prob)
            row[f'prauc_{tag}'] = average_precision_score(y, prob)
        except Exception:
            row[f'auc_{tag}'] = np.nan
            row[f'prauc_{tag}'] = np.nan
    monthly_rows.append(row)

monthly_df = pd.DataFrame(monthly_rows)
if not monthly_df.empty:
    print(monthly_df[['month', 'n', 'n_pos', 'pos_rate',
                       'auc_baseline', 'auc_c0_model',
                       'prauc_baseline', 'prauc_c0_model']].to_string(index=False))
else:
    print("  Not enough data for per-month breakdown")


# ── 8. Plots ──────────────────────────────────────────────────────────────────
y_true = results.get('baseline', {}).get('y')
if y_true is not None and len(y_true) > 0:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # ROC curves
    ax = axes[0]
    for tag, color in [('baseline', '#2196F3'), ('c0_model', '#F44336'),
                        ('c1_model', '#9C27B0'), ('c2_model', '#4CAF50')]:
        r = results.get(tag, {})
        if r.get('prob') is not None:
            RocCurveDisplay.from_predictions(
                r['y'], r['prob'], ax=ax,
                name=f"{tag.replace('_', ' ')} (AUC={r['auc']:.3f})",
                color=color, plot_chance_level=(tag == 'baseline'))
    ax.set_title('ROC — C0 (Sharp Down) Test Days')
    ax.legend(fontsize=8)

    # PR curves
    ax = axes[1]
    for tag, color in [('baseline', '#2196F3'), ('c0_model', '#F44336'),
                        ('c1_model', '#9C27B0'), ('c2_model', '#4CAF50')]:
        r = results.get(tag, {})
        if r.get('prob') is not None:
            PrecisionRecallDisplay.from_predictions(
                r['y'], r['prob'], ax=ax,
                name=f"{tag.replace('_', ' ')} (AP={r['pr_auc']:.3f})",
                color=color)
    ax.set_title('Precision-Recall — C0 Test Days')
    ax.legend(fontsize=8)

    # AUC bar chart summary
    ax = axes[2]
    tags = ['baseline', 'c0_model', 'c1_model', 'c2_model']
    names = ['Baseline\n(all data)', 'C0 model\n(Sharp Down)', 'C1 model\n(Normal Bear)', 'C2 model\n(Normal Bull)']
    colors = ['#2196F3', '#F44336', '#9C27B0', '#4CAF50']
    aucs    = [results.get(t, {}).get('auc', 0) for t in tags]
    praucs  = [results.get(t, {}).get('pr_auc', 0) for t in tags]
    x = np.arange(len(tags))
    w = 0.35
    ax.bar(x - w/2, aucs, w, label='AUC', color=colors, alpha=0.85)
    ax.bar(x + w/2, praucs, w, label='PR-AUC', color=colors, alpha=0.45, hatch='//')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='random')
    ax.legend(fontsize=9)
    ax.set_title('AUC / PR-AUC Summary\n(evaluated on C0 Sharp Down test days)')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'{OUT}/model_comparison_c0_test.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved comparison plot → {OUT}/model_comparison_c0_test.png")


# ── 9. Monthly AUC trend plot ─────────────────────────────────────────────────
if not monthly_df.empty and 'auc_baseline' in monthly_df.columns:
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    x = range(len(monthly_df))
    labels_x = monthly_df['month'].tolist()

    ax = axes[0]
    ax.plot(x, monthly_df['auc_baseline'], 'o-', color='#2196F3',
            label='Baseline (all data)', linewidth=2)
    ax.plot(x, monthly_df['auc_c0_model'], 's-', color='#F44336',
            label='C0 model (Sharp Down)', linewidth=2)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_x, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('AUC')
    ax.set_title('Per-Month AUC on C0 (Sharp Down) Test Days')
    ax.legend()
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    ax.plot(x, monthly_df['prauc_baseline'], 'o-', color='#2196F3',
            label='Baseline', linewidth=2)
    ax.plot(x, monthly_df['prauc_c0_model'], 's-', color='#F44336',
            label='C0 model', linewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_x, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('PR-AUC')
    ax.set_title('Per-Month PR-AUC on C0 (Sharp Down) Test Days')
    ax.legend()
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(f'{OUT}/monthly_auc_trend.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved monthly trend → {OUT}/monthly_auc_trend.png")


# ── 10. Summary table ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY — Evaluated on C0 (Sharp Down) test days (2025-01 → 2026-03)")
print("=" * 70)
print(f"{'Model':<25} {'AUC':>7} {'PR-AUC':>8} {'n_test':>8} {'n_pos':>7} {'pos%':>7}")
print("-" * 70)
for tag, label in [('baseline', 'Baseline (all data)'),
                   ('c0_model', 'C0 model (Sharp Down)'),
                   ('c1_model', 'C1 model (Normal Bear)'),
                   ('c2_model', 'C2 model (Normal Bull)')]:
    r = results.get(tag, {})
    if r:
        print(f"{label:<25} {r['auc']:>7.3f} {r['pr_auc']:>8.3f} "
              f"{r['n']:>8,} {r['n_pos']:>7} {r['pos_rate']:>6.2%}")
print("=" * 70)

print("\nReference — DirectionModelV1 from memory (all days, 7 regime months):")
print("  Mean AUC = 0.813, Mean PR-AUC = 0.299")
print("\nDone step 2.")
