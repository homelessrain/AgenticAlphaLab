"""
Step 1: Fit k-means on 2019-2025 daily features, then label all days
(including 2026) with a cluster id. Saves:
  - reports/regime_specific_model_v1/daily_cluster_labels.parquet
    columns: date (datetime.date), cluster (int 0-4)
  - reports/regime_specific_model_v1/cluster_stats.txt   (summary)
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

OUT = "reports/regime_specific_model_v1"
K = 5


def load_and_prep_daily(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df.xs('QQQ', level='symbol').copy()
    df.index = pd.DatetimeIndex(df.index).tz_localize(None)
    return df.sort_index()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)
    pc = df['close'].shift(1)
    feat['daily_return']   = (df['close'] - pc) / pc
    feat['intraday_range'] = (df['high'] - df['low']) / df['close']
    feat['open_gap']       = (df['open'] - pc) / pc
    feat['body_size']      = (df['close'] - df['open']).abs() / df['close']
    feat['upper_wick']     = (df['high'] - df[['open','close']].max(axis=1)) / df['close']
    feat['lower_wick']     = (df[['open','close']].min(axis=1) - df['low']) / df['close']
    feat['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
    feat['volume_ratio']   = df['volume'] / df['volume'].rolling(20).mean()
    feat['vol_5d']         = feat['daily_return'].rolling(5).std()
    feat['vol_20d']        = feat['daily_return'].rolling(20).std()
    feat['mom_5d']         = df['close'] / df['close'].shift(5) - 1
    feat['mom_20d']        = df['close'] / df['close'].shift(20) - 1
    return feat.dropna()


# ── 1. Load data ──────────────────────────────────────────────────────────────
# Fit on 2019-2025; extend with 2026 for full coverage
df_train_base = load_and_prep_daily('data/bars/QQQ/alpaca_2019-12-02_2025-12-31_day_mh.parquet')
df_2026 = load_and_prep_daily('data/bars/QQQ/alpaca_2025-01-01_2026-05-01_day_mh.parquet')

# Combine, deduplicate
df_all = pd.concat([df_train_base, df_2026])
df_all = df_all[~df_all.index.duplicated(keep='last')]
df_all = df_all.sort_index()
print(f"Combined daily data: {df_all.index[0].date()} → {df_all.index[-1].date()}, {len(df_all)} rows")

# ── 2. Build features on all data ─────────────────────────────────────────────
feat_all = build_features(df_all)
print(f"Features after dropna: {feat_all.shape}")

# ── 3. Fit scaler + k-means only on 2019-2025 data ───────────────────────────
fit_mask = feat_all.index <= pd.Timestamp('2025-12-31')
feat_fit = feat_all[fit_mask]

scaler = StandardScaler()
X_fit = scaler.fit_transform(feat_fit)

km = KMeans(n_clusters=K, random_state=42, n_init=50)
km.fit(X_fit)
print(f"K-means fit on {len(feat_fit)} days (2019-2025)")
print(f"Inertia: {km.inertia_:.2f}")

# ── 4. Predict on all available data (including 2026) ─────────────────────────
X_all = scaler.transform(feat_all)
feat_all = feat_all.copy()
feat_all['cluster'] = km.predict(X_all)

# ── 5. Cluster size + summary ─────────────────────────────────────────────────
cluster_counts = feat_all['cluster'].value_counts().sort_index()
print("\nAll-time cluster counts:")
print(cluster_counts)

# Cluster mean daily return to assign semantic names
cluster_means = feat_all.groupby('cluster')[['daily_return', 'intraday_range',
                                              'close_position', 'volume_ratio',
                                              'vol_5d', 'mom_20d']].mean()
print("\nCluster means:")
print(cluster_means.round(4))

# Show 2025-2026 distribution
recent = feat_all[feat_all.index >= pd.Timestamp('2025-01-01')]
print("\n2025-2026 cluster distribution:")
print(recent['cluster'].value_counts().sort_index())

# ── 6. Save cluster labels ────────────────────────────────────────────────────
labels = feat_all[['cluster']].copy()
labels.index.name = 'date'
labels.to_parquet(f'{OUT}/daily_cluster_labels.parquet')
print(f"\nSaved cluster labels: {len(labels)} rows → {OUT}/daily_cluster_labels.parquet")

# Also save the scaler and model for future use
with open(f'{OUT}/kmeans_model.pkl', 'wb') as f:
    pickle.dump({'scaler': scaler, 'km': km}, f)
print(f"Saved k-means model → {OUT}/kmeans_model.pkl")

# ── 7. Write cluster stats ────────────────────────────────────────────────────
CLUSTER_NAMES = {
    # Will be determined by the mean daily_return ordering
}
# Assign names based on return + vol profile
def cluster_name(row):
    ret = row['daily_return']
    vol = row['vol_5d']
    if vol > 0.03:
        return 'Crash/Crisis'
    elif ret > 0.012 and vol > 0.015:
        return 'Volatile Up'
    elif ret > 0.003:
        return 'Normal Bull'
    elif ret < -0.015:
        return 'Sharp Down'
    else:
        return 'Normal Bear'

cluster_means['name'] = cluster_means.apply(cluster_name, axis=1)
print("\nCluster semantic names:")
print(cluster_means[['daily_return', 'vol_5d', 'name']])

with open(f'{OUT}/cluster_stats.txt', 'w') as f:
    f.write("QQQ Daily Cluster Labels — Summary\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Data: {feat_all.index[0].date()} → {feat_all.index[-1].date()}\n")
    f.write(f"Total days: {len(feat_all)}\n\n")
    f.write("Cluster sizes:\n")
    f.write(str(cluster_counts) + "\n\n")
    f.write("Cluster profile:\n")
    f.write(cluster_means.to_string() + "\n\n")
    f.write("2025-2026 distribution:\n")
    f.write(str(recent['cluster'].value_counts().sort_index()) + "\n")

print("Done step 1.")
