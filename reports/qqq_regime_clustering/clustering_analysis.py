import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

OUT = "reports/qqq_regime_clustering"
DATA = "data/bars/QQQ/alpaca_2019-12-02_2025-12-31_day_mh.parquet"

# ── 1. Load data ──────────────────────────────────────────────────────────────
df = pd.read_parquet(DATA)
df = df.xs('QQQ', level='symbol').copy()
df.index = pd.DatetimeIndex(df.index).tz_localize(None)
df = df.sort_index()
print(f"Loaded {len(df)} trading days: {df.index[0].date()} → {df.index[-1].date()}")

# ── 2. Feature engineering ────────────────────────────────────────────────────
feat = pd.DataFrame(index=df.index)
prev_close = df['close'].shift(1)

feat['daily_return']    = (df['close'] - prev_close) / prev_close
feat['intraday_range']  = (df['high'] - df['low']) / df['close']
feat['open_gap']        = (df['open'] - prev_close) / prev_close
feat['body_size']       = (df['close'] - df['open']).abs() / df['close']
feat['upper_wick']      = (df['high'] - df[['open','close']].max(axis=1)) / df['close']
feat['lower_wick']      = (df[['open','close']].min(axis=1) - df['low']) / df['close']
feat['close_position']  = (df['close'] - df['low']) / (df['high'] - df['low'])
feat['volume_ratio']    = df['volume'] / df['volume'].rolling(20).mean()
feat['vol_5d']          = feat['daily_return'].rolling(5).std()
feat['vol_20d']         = feat['daily_return'].rolling(20).std()
feat['mom_5d']          = df['close'] / df['close'].shift(5) - 1
feat['mom_20d']         = df['close'] / df['close'].shift(20) - 1

feat = feat.dropna()
print(f"Feature matrix: {feat.shape} after dropping NaN rows")

FEATURE_COLS = feat.columns.tolist()

# ── 3. Scale ──────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X = scaler.fit_transform(feat)

# ── 4. Elbow + silhouette plot ────────────────────────────────────────────────
inertias, sil_scores = [], []
K_RANGE = range(2, 11)
for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = km.fit_predict(X)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X, labels))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(list(K_RANGE), inertias, 'bo-', linewidth=2, markersize=7)
axes[0].set(xlabel='k', ylabel='Inertia', title='Elbow Curve')
axes[0].grid(True, alpha=0.3)

axes[1].plot(list(K_RANGE), sil_scores, 'ro-', linewidth=2, markersize=7)
axes[1].set(xlabel='k', ylabel='Silhouette Score', title='Silhouette Score vs k')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT}/01_elbow_silhouette.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"Best silhouette at k={list(K_RANGE)[np.argmax(sil_scores)]}: {max(sil_scores):.4f}")

# ── 5. K-means with chosen k ──────────────────────────────────────────────────
K_OPT = 5
km = KMeans(n_clusters=K_OPT, random_state=42, n_init=50)
feat['cluster'] = km.fit_predict(X)
print(f"\nCluster sizes (k={K_OPT}):\n{feat['cluster'].value_counts().sort_index()}")

# ── 6. PCA visualization ──────────────────────────────────────────────────────
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)
print(f"PCA explained variance: {pca.explained_variance_ratio_}")

CLUSTER_COLORS = ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0']
CLUSTER_NAMES  = ['C0', 'C1', 'C2', 'C3', 'C4']

fig, ax = plt.subplots(figsize=(9, 7))
for c in range(K_OPT):
    mask = feat['cluster'] == c
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c=CLUSTER_COLORS[c], label=f'Cluster {c} (n={mask.sum()})',
               alpha=0.5, s=20, edgecolors='none')
ax.set(xlabel=f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
       ylabel=f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)',
       title='PCA of QQQ Daily Features — K-Means Clusters')
ax.legend(framealpha=0.9)
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(f"{OUT}/02_pca_clusters.png", dpi=150, bbox_inches='tight')
plt.close()

# ── 7. Cluster profiles ───────────────────────────────────────────────────────
profile_rows = []
for c in range(K_OPT):
    mask = feat['cluster'] == c
    row = {'cluster': c, 'n': int(mask.sum())}
    for col in FEATURE_COLS:
        row[f'{col}_mean'] = feat.loc[mask, col].mean()
        row[f'{col}_std']  = feat.loc[mask, col].std()
    row['date_min'] = feat.index[mask].min().date()
    row['date_max'] = feat.index[mask].max().date()
    profile_rows.append(row)

profile_df = pd.DataFrame(profile_rows).set_index('cluster')
print("\n── Cluster means ──")
mean_cols = [f'{c}_mean' for c in FEATURE_COLS]
print(profile_df[['n'] + mean_cols].to_string())

# Heatmap of cluster means (normalized for display)
mean_matrix = profile_df[[f'{c}_mean' for c in FEATURE_COLS]].copy()
mean_matrix.columns = FEATURE_COLS
fig, ax = plt.subplots(figsize=(14, 4))
sns.heatmap(mean_matrix, annot=True, fmt='.3f', cmap='RdYlGn',
            center=0, ax=ax, linewidths=0.5, cbar_kws={'shrink': 0.8})
ax.set_title('Cluster Feature Means (raw scale)')
ax.set_ylabel('Cluster')
plt.tight_layout()
plt.savefig(f"{OUT}/03_cluster_profile_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()

# ── 8. Time series cluster assignment ─────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [2.5, 1, 1]})

# Panel 1: QQQ price colored by cluster
ax = axes[0]
for c in range(K_OPT):
    mask = feat['cluster'] == c
    dates = feat.index[mask]
    prices = df.loc[dates, 'close']
    ax.scatter(dates, prices, c=CLUSTER_COLORS[c], s=15, alpha=0.7,
               label=f'Cluster {c}', zorder=3)
ax.set(ylabel='QQQ Close', title='QQQ Daily Close — Colored by Regime Cluster')
ax.legend(loc='upper left', ncol=K_OPT, fontsize=9, framealpha=0.9)
ax.grid(True, alpha=0.2)

# Add known regime annotations
annotations = [
    ('2020-02-15', '2020-04-01', 'COVID crash', 0.05),
    ('2022-01-01', '2022-12-31', '2022 bear', 0.05),
    ('2025-03-01', '2025-04-30', 'Tariff shock', 0.05),
]
for start, end, label, ypos in annotations:
    ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.10, color='gray')
    ax.text(pd.Timestamp(start), ax.get_ylim()[0] + (ax.get_ylim()[1]-ax.get_ylim()[0])*ypos,
            label, fontsize=7, color='black', style='italic')

# Panel 2: cluster label as scatter dots over time
ax = axes[1]
ax.scatter(feat.index, feat['cluster'],
           c=[CLUSTER_COLORS[c] for c in feat['cluster']], s=12, alpha=0.7)
ax.set(ylabel='Cluster', yticks=range(K_OPT),
       title='Cluster Assignment Over Time')
ax.grid(True, alpha=0.2)

# Panel 3: vol_20d overlay
ax = axes[2]
ax.fill_between(feat.index, feat['vol_20d'] * 100, alpha=0.4, color='steelblue')
ax.set(ylabel='20d Vol (%)', xlabel='Date',
       title='20-Day Realized Volatility')
ax.grid(True, alpha=0.2)

for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1,4,7,10]))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=8)

plt.tight_layout()
plt.savefig(f"{OUT}/04_timeseries_clusters.png", dpi=150, bbox_inches='tight')
plt.close()

# ── 9. Monthly cluster frequency heatmap ─────────────────────────────────────
feat['year_month'] = feat.index.to_period('M')
monthly_counts = (feat.groupby(['year_month', 'cluster'])
                  .size().unstack(fill_value=0))
monthly_pct = monthly_counts.div(monthly_counts.sum(axis=1), axis=0)

monthly_pct_str = monthly_pct.copy()
monthly_pct_str.index = [str(p) for p in monthly_pct.index]
fig, ax = plt.subplots(figsize=(20, 5))
monthly_pct_str.plot(kind='bar', stacked=True, ax=ax,
                     color=CLUSTER_COLORS[:K_OPT], legend=True, width=0.9, edgecolor='none')
ax.set(xlabel='Month', ylabel='Fraction of Days',
       title='Monthly Cluster Composition — QQQ Daily Regimes')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, [f'Cluster {l}' for l in labels], title='Cluster', loc='upper right', fontsize=8)
plt.xticks(rotation=45, ha='right', fontsize=6)
ax.grid(True, alpha=0.2, axis='y')
plt.tight_layout()
plt.savefig(f"{OUT}/05_monthly_cluster_composition.png", dpi=150, bbox_inches='tight')
plt.close()

# ── 10. Hierarchical dendrogram on monthly aggregated features ────────────────
monthly_feat = feat.drop(columns=['cluster', 'year_month']).resample('ME').median()
monthly_feat = monthly_feat.dropna()
month_labels = [str(p.to_period('M')) for p in monthly_feat.index]

X_monthly = StandardScaler().fit_transform(monthly_feat)
Z = linkage(X_monthly, method='ward')

fig, ax = plt.subplots(figsize=(18, 6))
dendrogram(Z, labels=month_labels, ax=ax, leaf_rotation=45, leaf_font_size=8,
           color_threshold=0.6 * max(Z[:, 2]))
ax.set(title='Hierarchical Clustering Dendrogram — Monthly QQQ Regime Aggregates',
       xlabel='Month', ylabel='Ward Distance')
ax.grid(True, alpha=0.2, axis='y')
plt.tight_layout()
plt.savefig(f"{OUT}/06_monthly_dendrogram.png", dpi=150, bbox_inches='tight')
plt.close()

# ── 11. Feature importance: cluster centroid spread ──────────────────────────
centroid_df = pd.DataFrame(km.cluster_centers_, columns=FEATURE_COLS)
spread = centroid_df.std(axis=0).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 4))
spread.plot(kind='bar', ax=ax, color='steelblue', edgecolor='navy', alpha=0.8)
ax.set(xlabel='Feature', ylabel='Std of Cluster Centroids (normalized scale)',
       title='Feature Discriminability Across Clusters')
ax.grid(True, alpha=0.3, axis='y')
plt.xticks(rotation=35, ha='right')
plt.tight_layout()
plt.savefig(f"{OUT}/07_feature_discriminability.png", dpi=150, bbox_inches='tight')
plt.close()

# ── 12. Print full profile for report ─────────────────────────────────────────
print("\n── FULL CLUSTER PROFILE ──")
for c in range(K_OPT):
    mask = feat['cluster'] == c
    print(f"\nCluster {c}  (n={mask.sum()}, {feat.index[mask].min().date()} – {feat.index[mask].max().date()})")
    for col in FEATURE_COLS:
        mu  = feat.loc[mask, col].mean()
        sig = feat.loc[mask, col].std()
        print(f"  {col:20s}: {mu:+.4f} ± {sig:.4f}")

print("\n── KNOWN REGIME COVERAGE ──")
known = {
    '2025-03': ('2025-03-01', '2025-03-31', 'Strong correction'),
    '2025-04': ('2025-04-01', '2025-04-30', 'Crisis crash+recovery'),
    '2025-11': ('2025-11-01', '2025-11-30', 'Mild correction'),
    '2025-12': ('2025-12-01', '2025-12-31', 'Normal/flat (est)'),
}
for period, (s, e, name) in known.items():
    sub = feat[(feat.index >= s) & (feat.index <= e)]
    if len(sub) > 0:
        dist = sub['cluster'].value_counts().to_dict()
        print(f"  {period} ({name}): {dist}")

print("\nDone. All plots saved to", OUT)
