"""Generate Phase 1 Jupyter notebooks from source cells.

Run from project root:
    python scripts/generate_notebooks.py
"""
import json
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).parent.parent / "notebooks"
NOTEBOOKS_DIR.mkdir(exist_ok=True)


def nb(cells):
    """Wrap cells into a valid nbformat v4 notebook."""
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "cells": cells,
    }


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source, "id": ""}


def code(source):
    return {
        "cell_type": "code",
        "metadata": {},
        "source": source,
        "outputs": [],
        "execution_count": None,
        "id": "",
    }


# ============================================================
# Notebook 1 — EDA
# ============================================================

nb01 = nb([
    md("# 01 — Exploratory Data Analysis\n"
       "**Course:** MCS5343 · **Phase:** 1 · **Dataset:** IEEE-CIS Fraud Detection"),

    md("## 1. Setup"),

    code(
        "import sys\n"
        "sys.path.insert(0, '..')  # make src/ importable\n\n"
        "import pandas as pd\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "import warnings\n\n"
        "warnings.filterwarnings('ignore')\n"
        "SEED = 42\n"
        "np.random.seed(SEED)\n"
        "sns.set_style('darkgrid')\n"
        "plt.rcParams['figure.figsize'] = (12, 5)\n"
        "print('imports ok')"
    ),

    md("## 2. Load Data"),

    code(
        "DATA = '../data/'\n\n"
        "train_txn = pd.read_csv(DATA + 'train_transaction.csv')\n"
        "train_id  = pd.read_csv(DATA + 'train_identity.csv')\n"
        "test_txn  = pd.read_csv(DATA + 'test_transaction.csv')\n"
        "test_id   = pd.read_csv(DATA + 'test_identity.csv')\n\n"
        "print(f'train_txn: {train_txn.shape}  train_id: {train_id.shape}')\n"
        "print(f'test_txn:  {test_txn.shape}   test_id:  {test_id.shape}')"
    ),

    code(
        "# Left-join identity onto transaction\n"
        "train = train_txn.merge(train_id, on='TransactionID', how='left')\n"
        "test  = test_txn.merge(test_id,   on='TransactionID', how='left')\n\n"
        "# Downcast float64 → float32 to halve memory\n"
        "for df in [train, test]:\n"
        "    f64 = df.select_dtypes('float64').columns\n"
        "    df[f64] = df[f64].astype('float32')\n\n"
        "print(f'Train: {train.shape}  |  Test: {test.shape}')\n"
        "print(f'Train memory: {train.memory_usage(deep=True).sum()/1e6:.1f} MB')\n"
        "print(f'Columns only in train (label): {set(train.columns)-set(test.columns)}')"
    ),

    md("## 3. Class Distribution"),

    code(
        "fraud_rate = train['isFraud'].mean()\n"
        "n_fraud    = train['isFraud'].sum()\n"
        "n_total    = len(train)\n"
        "print(f'Fraud rate: {fraud_rate*100:.2f}%  ({n_fraud:,} / {n_total:,})')\n\n"
        "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n\n"
        "# Class balance bar\n"
        "train['isFraud'].value_counts().plot(kind='bar', ax=axes[0],\n"
        "    color=['steelblue', 'tomato'], edgecolor='white')\n"
        "axes[0].set_title('Class Distribution (0=legit, 1=fraud)')\n"
        "axes[0].set_xlabel('isFraud')\n"
        "axes[0].set_ylabel('Count')\n"
        "axes[0].set_xticklabels(['Legitimate', 'Fraud'], rotation=0)\n\n"
        "# TransactionAmt distribution by class\n"
        "for label, color in [(0, 'steelblue'), (1, 'tomato')]:\n"
        "    subset = train[train['isFraud'] == label]['TransactionAmt']\n"
        "    axes[1].hist(np.log1p(subset), bins=60, alpha=0.6,\n"
        "                 label=f'isFraud={label}', color=color, density=True)\n"
        "axes[1].set_title('log(1+TransactionAmt) by Class')\n"
        "axes[1].set_xlabel('log1p(Amount)')\n"
        "axes[1].legend()\n\n"
        "plt.tight_layout()\n"
        "plt.savefig('../results/01_class_distribution.png', dpi=120, bbox_inches='tight')\n"
        "plt.show()"
    ),

    md("## 4. Fraud Rate by Categorical Feature"),

    code(
        "cat_cols = ['ProductCD', 'card4', 'card6', 'P_emaildomain']\n"
        "fig, axes = plt.subplots(1, len(cat_cols), figsize=(20, 4))\n\n"
        "for ax, col in zip(axes, cat_cols):\n"
        "    rates = (train.groupby(col)['isFraud'].mean()\n"
        "             .sort_values(ascending=False).head(10))\n"
        "    rates.plot(kind='bar', ax=ax, color='steelblue', edgecolor='white')\n"
        "    ax.set_title(f'Fraud Rate by {col}')\n"
        "    ax.set_ylabel('Fraud Rate')\n"
        "    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')\n\n"
        "plt.tight_layout()\n"
        "plt.savefig('../results/01_fraud_rate_by_cat.png', dpi=120, bbox_inches='tight')\n"
        "plt.show()"
    ),

    md("## 5. Missing Value Analysis"),

    code(
        "def missing_pct(df, prefix):\n"
        "    cols = [c for c in df.columns if c.startswith(prefix)]\n"
        "    return df[cols].isnull().mean() if cols else pd.Series(dtype=float)\n\n"
        "groups = {'V-features': 'V', 'D-features': 'D', 'M-features': 'M',\n"
        "          'id-features': 'id_', 'C-features': 'C'}\n"
        "fig, axes = plt.subplots(1, len(groups), figsize=(24, 4))\n\n"
        "for ax, (label, prefix) in zip(axes, groups.items()):\n"
        "    pct = missing_pct(train, prefix)\n"
        "    if pct.empty:\n"
        "        ax.set_visible(False)\n"
        "        continue\n"
        "    pct.plot(kind='bar', ax=ax, color='salmon', edgecolor='white')\n"
        "    ax.axhline(0.95, color='red', linestyle='--', linewidth=1, label='95% threshold')\n"
        "    ax.set_title(f'Missing % — {label}')\n"
        "    ax.set_ylabel('Missing fraction')\n"
        "    ax.set_xticklabels([])\n"
        "    ax.legend()\n\n"
        "plt.tight_layout()\n"
        "plt.savefig('../results/01_missing_values.png', dpi=120, bbox_inches='tight')\n"
        "plt.show()\n\n"
        "# Columns to drop (>95% missing in train)\n"
        "drop_cols = train.columns[train.isnull().mean() > 0.95].tolist()\n"
        "print(f'Columns with >95% missing: {len(drop_cols)}')\n"
        "print(drop_cols[:10], '...')"
    ),

    md("## 6. Temporal Patterns"),

    code(
        "train['day']  = (train['TransactionDT'] / 86400).astype(int)\n"
        "train['hour'] = ((train['TransactionDT'] % 86400) / 3600).astype(int)\n\n"
        "fig, axes = plt.subplots(1, 3, figsize=(20, 4))\n\n"
        "train.groupby('day').size().plot(ax=axes[0], color='steelblue')\n"
        "axes[0].set_title('Daily Transaction Volume')\n"
        "axes[0].set_xlabel('Day')\n\n"
        "train.groupby('day')['isFraud'].mean().plot(ax=axes[1], color='tomato')\n"
        "axes[1].set_title('Daily Fraud Rate')\n"
        "axes[1].set_xlabel('Day')\n\n"
        "hourly = train.groupby('hour')['isFraud'].mean()\n"
        "hourly.plot(kind='bar', ax=axes[2], color='goldenrod', edgecolor='white')\n"
        "axes[2].set_title('Fraud Rate by Hour of Day')\n"
        "axes[2].set_xlabel('Hour')\n\n"
        "plt.tight_layout()\n"
        "plt.savefig('../results/01_temporal_patterns.png', dpi=120, bbox_inches='tight')\n"
        "plt.show()\n\n"
        "train.drop(columns=['day', 'hour'], inplace=True)"
    ),

    md("## 7. Summary\n\n"
       "- **Fraud rate:** ~3.5% — significant class imbalance → motivates SMOTE in notebook 02\n"
       "- **Key categorical signals:** card4 (network), card6 (debit/credit), ProductCD\n"
       "- **High-missing columns:** many V-features and id-features >95% missing → drop in preprocessing\n"
       "- **Temporal structure:** data spans ~180 days; fraud rate has intra-day patterns → `hour_of_day` is a useful feature\n"
       "- **Amount distribution:** right-skewed → `log_amount` transform helps linear models (and speeds XGBoost)"),
])

# ============================================================
# Notebook 2 — Feature Engineering
# ============================================================

nb02 = nb([
    md("# 02 — Feature Engineering & Preprocessing\n"
       "**Depends on:** 01_eda.ipynb (understand which columns to drop)\n\n"
       "Outputs: `data/processed/train.parquet`, `data/processed/test.parquet`"),

    md("## 1. Setup"),

    code(
        "import sys\n"
        "sys.path.insert(0, '..')\n\n"
        "import pandas as pd\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "import joblib\n"
        "from sklearn.preprocessing import LabelEncoder\n"
        "from sklearn.impute import SimpleImputer\n"
        "from imblearn.combine import SMOTETomek\n"
        "import warnings\n\n"
        "from src.features import engineer_features\n\n"
        "warnings.filterwarnings('ignore')\n"
        "SEED = 42\n"
        "np.random.seed(SEED)\n"
        "print('imports ok')"
    ),

    md("## 2. Load Raw Data"),

    code(
        "DATA = '../data/'\n\n"
        "train_txn = pd.read_csv(DATA + 'train_transaction.csv')\n"
        "train_id  = pd.read_csv(DATA + 'train_identity.csv')\n"
        "test_txn  = pd.read_csv(DATA + 'test_transaction.csv')\n"
        "test_id   = pd.read_csv(DATA + 'test_identity.csv')\n\n"
        "train = train_txn.merge(train_id, on='TransactionID', how='left')\n"
        "test  = test_txn.merge(test_id,   on='TransactionID', how='left')\n\n"
        "for df in [train, test]:\n"
        "    f64 = df.select_dtypes('float64').columns\n"
        "    df[f64] = df[f64].astype('float32')\n\n"
        "print(f'Train: {train.shape}  Test: {test.shape}')"
    ),

    md("## 3. Drop High-Missing Columns (>95%)"),

    code(
        "drop_cols = train.columns[train.isnull().mean() > 0.95].tolist()\n"
        "print(f'Dropping {len(drop_cols)} columns with >95% missing')\n\n"
        "train.drop(columns=drop_cols, inplace=True)\n"
        "test.drop(columns=[c for c in drop_cols if c in test.columns], inplace=True)\n\n"
        "print(f'After drop — Train: {train.shape}  Test: {test.shape}')"
    ),

    md("## 4. Add Derived Features (via src.features)"),

    code(
        "# Add log_amount, hour_of_day, amount_zscore\n"
        "# Training mode: stats computed from training set\n"
        "train_engineered = engineer_features(train, amount_mean=None, amount_std=None)\n\n"
        "# Inference mode: use training stats to avoid leakage\n"
        "train_amount_mean = float(train['TransactionAmt'].mean())\n"
        "train_amount_std  = float(train['TransactionAmt'].std())\n"
        "test_engineered   = engineer_features(test, amount_mean=train_amount_mean,\n"
        "                                      amount_std=train_amount_std)\n\n"
        "print('New columns added:', ['log_amount', 'hour_of_day', 'amount_zscore'])\n"
        "print(train_engineered[['TransactionAmt','log_amount','hour_of_day','amount_zscore']].head(3))"
    ),

    md("## 5. Velocity Features (card1 rolling window)"),

    code(
        "# Backward-looking velocity per card1: 1-day and 7-day windows\n"
        "# Combined train+test for consistency, split back after\n\n"
        "target    = train_engineered['isFraud'].copy()\n"
        "train_ids = train_engineered['TransactionID'].copy()\n"
        "test_ids  = test_engineered['TransactionID'].copy()\n\n"
        "train_feat = train_engineered.drop(columns=['isFraud', 'TransactionID'])\n"
        "test_feat  = test_engineered.drop(columns=['TransactionID'])\n\n"
        "n_train = len(train_feat)\n"
        "combined = pd.concat([\n"
        "    train_feat.assign(TransactionID=train_ids.values, isFraud=target.values, _split='train'),\n"
        "    test_feat.assign(TransactionID=test_ids.values,   isFraud=np.nan,        _split='test'),\n"
        "], ignore_index=True).sort_values('TransactionDT').reset_index(drop=True)\n\n"
        "combined['_dt'] = pd.to_datetime(combined['TransactionDT'], unit='s', origin='unix')\n"
        "combined = combined.set_index('_dt').sort_index()\n\n"
        "def rolling_card1(df, window):\n"
        "    return (df.groupby('card1')['TransactionAmt']\n"
        "              .transform(lambda s: s.shift(1).rolling(window, min_periods=1).sum()))\n\n"
        "combined['vel_1d_count']  = (combined.groupby('card1')['TransactionAmt']\n"
        "    .transform(lambda s: s.shift(1).rolling('1D', min_periods=1).count()))\n"
        "combined['vel_1d_amt']    = (combined.groupby('card1')['TransactionAmt']\n"
        "    .transform(lambda s: s.shift(1).rolling('1D', min_periods=1).sum()))\n"
        "combined['vel_7d_count']  = (combined.groupby('card1')['TransactionAmt']\n"
        "    .transform(lambda s: s.shift(1).rolling('7D', min_periods=1).count()))\n"
        "combined['vel_7d_amt']    = (combined.groupby('card1')['TransactionAmt']\n"
        "    .transform(lambda s: s.shift(1).rolling('7D', min_periods=1).sum()))\n"
        "combined['vel_7d_merch']  = (combined.groupby('card1')['MerchantID']\n"
        "    .transform(lambda s: s.shift(1).rolling('7D', min_periods=1).nunique())\n"
        "    if 'MerchantID' in combined.columns else 0)\n"
        "combined['vel_7d_dev']    = (combined['vel_7d_amt'] /\n"
        "    combined.groupby('card1')['vel_7d_count'].transform('mean').replace(0, np.nan))\n\n"
        "combined = combined.reset_index(drop=True)\n"
        "print('Velocity features added. Sample:')\n"
        "print(combined[['vel_1d_count','vel_1d_amt','vel_7d_count','vel_7d_amt']].head(3))"
    ),

    md("## 6. Categorical Encoding"),

    code(
        "# Split back before encoding to fit on train only\n"
        "cat_cols = combined.select_dtypes(include='object').columns.tolist()\n"
        "cat_cols = [c for c in cat_cols if c not in ['_split']]\n"
        "print(f'Encoding {len(cat_cols)} categorical columns')\n\n"
        "encoders = {}\n"
        "for col in cat_cols:\n"
        "    le = LabelEncoder()\n"
        "    combined[col] = le.fit_transform(combined[col].astype(str))\n"
        "    encoders[col] = le\n\n"
        "print('Encoding done')"
    ),

    md("## 7. Train/Validation Split"),

    code(
        "train_final = combined[combined['_split'] == 'train'].copy()\n"
        "test_final  = combined[combined['_split'] == 'test'].copy()\n"
        "train_final = train_final.sort_values('TransactionDT').reset_index(drop=True)\n\n"
        "drop_meta = ['_split', 'isFraud', 'TransactionID']\n"
        "feature_cols = [c for c in train_final.columns if c not in drop_meta]\n\n"
        "X = train_final[feature_cols]\n"
        "y = train_final['isFraud'].astype(int)\n"
        "X_test_final = test_final[feature_cols]\n\n"
        "# Time-based 80/20 split (preserve temporal order — no shuffle)\n"
        "split_idx = int(len(X) * 0.80)\n"
        "X_tr, X_val = X.iloc[:split_idx], X.iloc[split_idx:]\n"
        "y_tr, y_val = y.iloc[:split_idx], y.iloc[split_idx:]\n\n"
        "print(f'X_tr: {X_tr.shape}  X_val: {X_val.shape}')\n"
        "print(f'Train fraud: {y_tr.mean():.4f}  Val fraud: {y_val.mean():.4f}')"
    ),

    md("## 8. SMOTE+Tomek Resampling (training set only)\n\n"
       "> **Important:** SMOTE is applied **only to X_tr** to prevent leakage into validation/test."),

    code(
        "from sklearn.impute import SimpleImputer\n\n"
        "print(f'Before SMOTE — X_tr: {X_tr.shape}  fraud: {y_tr.sum():,} ({y_tr.mean()*100:.2f}%)')\n\n"
        "imputer = SimpleImputer(strategy='median', keep_empty_features=True)\n"
        "X_tr_imp  = pd.DataFrame(imputer.fit_transform(X_tr),   columns=X_tr.columns)\n"
        "X_val_imp = pd.DataFrame(imputer.transform(X_val),      columns=X_val.columns)\n\n"
        "smt = SMOTETomek(random_state=SEED)\n"
        "X_tr_res, y_tr_res = smt.fit_resample(X_tr_imp, y_tr)\n\n"
        "print(f'After  SMOTE — X_tr_res: {X_tr_res.shape}  fraud: {y_tr_res.sum():,} ({y_tr_res.mean()*100:.2f}%)')"
    ),

    md("## 9. Save Processed Data"),

    code(
        "import os\n"
        "os.makedirs('../data/processed', exist_ok=True)\n\n"
        "# Save processed splits for use in training notebook\n"
        "X_tr.to_parquet('../data/processed/X_tr_raw.parquet')\n"
        "X_val.to_parquet('../data/processed/X_val_raw.parquet')\n"
        "pd.DataFrame(X_tr_res, columns=X_tr.columns).to_parquet('../data/processed/X_tr_smt.parquet')\n"
        "pd.DataFrame(X_val_imp, columns=X_val.columns).to_parquet('../data/processed/X_val_imp.parquet')\n"
        "y_tr.to_frame().to_parquet('../data/processed/y_tr.parquet')\n"
        "y_val.to_frame().to_parquet('../data/processed/y_val.parquet')\n"
        "pd.Series(y_tr_res, name='isFraud').to_frame().to_parquet('../data/processed/y_tr_smt.parquet')\n"
        "X_test_final.to_parquet('../data/processed/X_test.parquet')\n"
        "test_ids.to_frame().to_parquet('../data/processed/test_ids.parquet')\n\n"
        "# Save training stats and encoders for inference\n"
        "joblib.dump({'mean': train_amount_mean, 'std': train_amount_std},\n"
        "            '../models/amount_stats.pkl')\n"
        "joblib.dump(encoders, '../models/label_encoders.pkl')\n"
        "joblib.dump(imputer,  '../models/imputer.pkl')\n"
        "joblib.dump(feature_cols, '../models/feature_cols.pkl')\n\n"
        "print('Saved all processed splits and artifacts')\n"
        "print(f'feature_cols: {len(feature_cols)}')"
    ),

    md("## Summary\n\n"
       "| Step | Detail |\n"
       "|------|--------|\n"
       "| Dropped | Columns >95% missing |\n"
       "| Added | `log_amount`, `hour_of_day`, `amount_zscore` (via `src.features`) |\n"
       "| Added | 6 card1 velocity features (1-day and 7-day rolling) |\n"
       "| Encoded | All object columns via LabelEncoder (fit on combined train+test) |\n"
       "| Split | 80/20 time-based (no shuffle) |\n"
       "| SMOTE | Applied to training set only — avoids leakage |\n"
       "| Saved | Parquet splits in `data/processed/`, artifacts in `models/` |"),
])

# ============================================================
# Notebook 3 — Model Training with MLflow
# ============================================================

nb03 = nb([
    md("# 03 — Model Training with MLflow\n"
       "**Trains:** XGBoost + Isolation Forest → composite score\n"
       "**Tracks:** All runs in MLflow experiment `fraud_detection_phase1`"),

    md("## 1. Setup"),

    code(
        "import sys\n"
        "sys.path.insert(0, '..')\n\n"
        "import pandas as pd\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "import joblib\n"
        "import mlflow\n"
        "import mlflow.sklearn\n"
        "import mlflow.xgboost\n"
        "from xgboost import XGBClassifier\n"
        "from sklearn.ensemble import IsolationForest\n"
        "from sklearn.metrics import (roc_auc_score, average_precision_score,\n"
        "                             roc_curve, precision_recall_curve)\n"
        "import warnings\n\n"
        "from src.features import normalize_iforest_scores, composite_score\n\n"
        "warnings.filterwarnings('ignore')\n"
        "SEED = 42\n"
        "np.random.seed(SEED)\n"
        "print('imports ok')"
    ),

    md("## 2. Load Processed Data"),

    code(
        "X_tr      = pd.read_parquet('../data/processed/X_tr_raw.parquet')\n"
        "X_val     = pd.read_parquet('../data/processed/X_val_raw.parquet')\n"
        "X_tr_smt  = pd.read_parquet('../data/processed/X_tr_smt.parquet')\n"
        "X_val_imp = pd.read_parquet('../data/processed/X_val_imp.parquet')\n"
        "y_tr      = pd.read_parquet('../data/processed/y_tr.parquet')['isFraud']\n"
        "y_val     = pd.read_parquet('../data/processed/y_val.parquet')['isFraud']\n"
        "y_tr_smt  = pd.read_parquet('../data/processed/y_tr_smt.parquet')['isFraud']\n\n"
        "feature_cols = joblib.load('../models/feature_cols.pkl')\n\n"
        "neg, pos = (y_tr == 0).sum(), (y_tr == 1).sum()\n"
        "scale_pos_weight = neg / pos\n\n"
        "print(f'X_tr: {X_tr.shape}  X_tr_smt: {X_tr_smt.shape}')\n"
        "print(f'scale_pos_weight: {scale_pos_weight:.1f}')"
    ),

    md("## 3. MLflow Setup"),

    code(
        "# Use local file-based MLflow tracking (no server needed)\n"
        "mlflow.set_tracking_uri('../mlruns')\n"
        "experiment = mlflow.set_experiment('fraud_detection_phase1')\n"
        "print(f'Experiment ID: {experiment.experiment_id}')"
    ),

    md("## 4. Helper Functions"),

    code(
        "def recall_at_fpr(y_true, y_score, target_fpr=0.05):\n"
        "    fpr, tpr, _ = roc_curve(y_true, y_score)\n"
        "    idx = np.searchsorted(fpr, target_fpr)\n"
        "    return float(tpr[min(idx, len(tpr)-1)])\n\n"
        "def evaluate_scores(y_true, y_score, label=''):\n"
        "    auc_roc = roc_auc_score(y_true, y_score)\n"
        "    auc_pr  = average_precision_score(y_true, y_score)\n"
        "    rec5    = recall_at_fpr(y_true, y_score, 0.05)\n"
        "    print(f'{label:30s}  AUC-ROC: {auc_roc:.4f}  AUC-PR: {auc_pr:.4f}  Recall@5%FPR: {rec5:.4f}')\n"
        "    return {'auc_roc': auc_roc, 'auc_pr': auc_pr, 'recall_at_5pct_fpr': rec5}\n\n"
        "def plot_curves(y_true, y_score, label, ax_roc, ax_pr, color):\n"
        "    fpr, tpr, _ = roc_curve(y_true, y_score)\n"
        "    prec, rec, _ = precision_recall_curve(y_true, y_score)\n"
        "    auc_roc = roc_auc_score(y_true, y_score)\n"
        "    auc_pr  = average_precision_score(y_true, y_score)\n"
        "    ax_roc.plot(fpr, tpr, color=color, label=f'{label} (AUC={auc_roc:.4f})')\n"
        "    ax_pr.plot(rec, prec, color=color, label=f'{label} (AP={auc_pr:.4f})')"
    ),

    md("## 5. Train Baseline XGBoost"),

    code(
        "XGB_PARAMS = dict(\n"
        "    n_estimators=500,\n"
        "    max_depth=9,\n"
        "    learning_rate=0.05,\n"
        "    scale_pos_weight=scale_pos_weight,\n"
        "    tree_method='hist',\n"
        "    device='cpu',\n"
        "    early_stopping_rounds=50,\n"
        "    eval_metric='auc',\n"
        "    random_state=SEED,\n"
        "    n_jobs=-1,\n"
        ")\n\n"
        "with mlflow.start_run(run_name='xgb_baseline') as run_base:\n"
        "    mlflow.log_params({**XGB_PARAMS, 'smote': False, 'dataset': 'ieee_cis'})\n\n"
        "    xgb_base = XGBClassifier(**XGB_PARAMS)\n"
        "    xgb_base.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=100)\n\n"
        "    y_pred_base = xgb_base.predict_proba(X_val)[:, 1]\n"
        "    metrics_base = evaluate_scores(y_val, y_pred_base, 'XGB Baseline')\n\n"
        "    mlflow.log_metrics(metrics_base)\n"
        "    mlflow.xgboost.log_model(xgb_base, 'xgb_baseline_model')\n\n"
        "    run_id_base = run_base.info.run_id\n"
        "    print(f'Run ID: {run_id_base}')"
    ),

    md("## 6. Train XGBoost + SMOTETomek"),

    code(
        "XGB_SMT_PARAMS = dict(\n"
        "    n_estimators=500,\n"
        "    max_depth=9,\n"
        "    learning_rate=0.05,\n"
        "    tree_method='hist',\n"
        "    device='cpu',\n"
        "    early_stopping_rounds=50,\n"
        "    eval_metric='auc',\n"
        "    random_state=SEED,\n"
        "    n_jobs=-1,\n"
        ")\n\n"
        "with mlflow.start_run(run_name='xgb_smotetomek') as run_smt:\n"
        "    mlflow.log_params({**XGB_SMT_PARAMS, 'smote': True, 'dataset': 'ieee_cis'})\n\n"
        "    xgb_smt = XGBClassifier(**XGB_SMT_PARAMS)\n"
        "    xgb_smt.fit(X_tr_smt, y_tr_smt, eval_set=[(X_val_imp, y_val)], verbose=100)\n\n"
        "    y_pred_smt = xgb_smt.predict_proba(X_val_imp)[:, 1]\n"
        "    metrics_smt = evaluate_scores(y_val, y_pred_smt, 'XGB + SMOTETomek')\n\n"
        "    mlflow.log_metrics(metrics_smt)\n"
        "    mlflow.xgboost.log_model(xgb_smt, 'xgb_smt_model')\n\n"
        "    run_id_smt = run_smt.info.run_id\n"
        "    print(f'Run ID: {run_id_smt}')"
    ),

    md("## 7. Train Isolation Forest"),

    code(
        "IFOREST_PARAMS = dict(\n"
        "    n_estimators=200,\n"
        "    contamination=0.035,  # approximate fraud rate\n"
        "    max_samples='auto',\n"
        "    random_state=SEED,\n"
        "    n_jobs=-1,\n"
        ")\n\n"
        "with mlflow.start_run(run_name='isolation_forest') as run_if:\n"
        "    mlflow.log_params({**IFOREST_PARAMS, 'dataset': 'ieee_cis'})\n\n"
        "    # IsolationForest is unsupervised — train on legitimate transactions only\n"
        "    from sklearn.impute import SimpleImputer\n"
        "    imputer = joblib.load('../models/imputer.pkl')\n"
        "    X_tr_imp = pd.DataFrame(imputer.transform(X_tr), columns=X_tr.columns)\n\n"
        "    iforest = IsolationForest(**IFOREST_PARAMS)\n"
        "    iforest.fit(X_tr_imp[y_tr == 0])  # fit on legit only\n\n"
        "    # decision_function: negative = anomalous, positive = normal\n"
        "    raw_scores_val = iforest.decision_function(X_val_imp)\n"
        "    iforest_scores_val = normalize_iforest_scores(raw_scores_val)\n\n"
        "    metrics_if = evaluate_scores(y_val, iforest_scores_val, 'Isolation Forest')\n"
        "    mlflow.log_metrics(metrics_if)\n"
        "    mlflow.sklearn.log_model(iforest, 'iforest_model')\n\n"
        "    # Plot anomaly score distribution\n"
        "    fig, ax = plt.subplots(figsize=(8, 4))\n"
        "    for label, color in [(0, 'steelblue'), (1, 'tomato')]:\n"
        "        mask = y_val.values == label\n"
        "        ax.hist(iforest_scores_val[mask], bins=50, alpha=0.6,\n"
        "                label=f'isFraud={label}', color=color, density=True)\n"
        "    ax.set_title('Isolation Forest Normalised Score Distribution')\n"
        "    ax.set_xlabel('Anomaly score (1=most anomalous)')\n"
        "    ax.legend()\n"
        "    plt.tight_layout()\n"
        "    fig.savefig('/tmp/iforest_dist.png', dpi=120, bbox_inches='tight')\n"
        "    mlflow.log_artifact('/tmp/iforest_dist.png')\n"
        "    plt.savefig('../results/03_iforest_distribution.png', dpi=120, bbox_inches='tight')\n"
        "    plt.show()\n\n"
        "    run_id_if = run_if.info.run_id\n"
        "    print(f'Run ID: {run_id_if}')"
    ),

    md("## 8. Composite Score Sensitivity Analysis\n\n"
       "Test weights (alpha × XGB + (1-alpha) × IForest) at 5 settings."),

    code(
        "alphas = [0.5, 0.6, 0.7, 0.8, 0.9]\n"
        "sensitivity_results = []\n\n"
        "for alpha in alphas:\n"
        "    with mlflow.start_run(run_name=f'composite_alpha_{alpha}'):\n"
        "        mlflow.log_param('alpha', alpha)\n"
        "        mlflow.log_param('dataset', 'ieee_cis')\n\n"
        "        comp = np.array([\n"
        "            composite_score(float(xp), float(ip), alpha=alpha)\n"
        "            for xp, ip in zip(y_pred_smt, iforest_scores_val)\n"
        "        ])\n"
        "        m = evaluate_scores(y_val, comp, f'Composite alpha={alpha}')\n"
        "        mlflow.log_metrics(m)\n"
        "        sensitivity_results.append({'alpha': alpha, **m})\n\n"
        "sens_df = pd.DataFrame(sensitivity_results).set_index('alpha')\n"
        "print('\\nSensitivity analysis:')\n"
        "print(sens_df.round(4).to_string())"
    ),

    md("## 9. Select Best Composite Weight & Final Evaluation"),

    code(
        "# Best alpha by AUC-PR (fraud detection cares more about precision-recall)\n"
        "best_alpha = sens_df['auc_pr'].idxmax()\n"
        "print(f'Best alpha: {best_alpha}')\n\n"
        "comp_best = np.array([\n"
        "    composite_score(float(xp), float(ip), alpha=best_alpha)\n"
        "    for xp, ip in zip(y_pred_smt, iforest_scores_val)\n"
        "])\n\n"
        "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n"
        "colors = {'XGB Baseline': 'steelblue', 'XGB+SMOTE': 'tomato',\n"
        "          'IsolationForest': 'goldenrod', f'Composite α={best_alpha}': 'purple'}\n\n"
        "for (label, scores, color) in [\n"
        "    ('XGB Baseline',           y_pred_base,       'steelblue'),\n"
        "    ('XGB+SMOTE',              y_pred_smt,        'tomato'),\n"
        "    ('IsolationForest',        iforest_scores_val,'goldenrod'),\n"
        "    (f'Composite α={best_alpha}', comp_best,      'purple'),\n"
        "]:\n"
        "    plot_curves(y_val, scores, label, axes[0], axes[1], color)\n\n"
        "axes[0].plot([0,1],[0,1],'k--',lw=1)\n"
        "axes[0].set_xlabel('FPR'); axes[0].set_ylabel('TPR')\n"
        "axes[0].set_title('ROC Curves')\n"
        "axes[0].legend(fontsize=8)\n\n"
        "axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')\n"
        "axes[1].set_title('Precision-Recall Curves')\n"
        "axes[1].legend(fontsize=8)\n\n"
        "plt.tight_layout()\n"
        "plt.savefig('../results/03_model_comparison_curves.png', dpi=120, bbox_inches='tight')\n"
        "plt.show()"
    ),

    md("## 10. Save Model Artifacts"),

    code(
        "import os\n"
        "os.makedirs('../models', exist_ok=True)\n\n"
        "joblib.dump(xgb_base, '../models/xgboost_baseline.pkl')\n"
        "joblib.dump(xgb_smt,  '../models/xgboost_smt.pkl')\n"
        "joblib.dump(iforest,  '../models/iforest_model.pkl')\n"
        "joblib.dump({'best_alpha': best_alpha,\n"
        "             'amount_mean': joblib.load('../models/amount_stats.pkl')['mean'],\n"
        "             'amount_std':  joblib.load('../models/amount_stats.pkl')['std']},\n"
        "            '../models/scoring_config.pkl')\n\n"
        "print('Models saved:')\n"
        "for p in ['xgboost_baseline.pkl','xgboost_smt.pkl','iforest_model.pkl','scoring_config.pkl']:\n"
        "    size = os.path.getsize(f'../models/{p}') / 1e6\n"
        "    print(f'  models/{p}  ({size:.1f} MB)')"
    ),

    md("## Summary\n\n"
       "All runs tracked in `mlruns/` under experiment `fraud_detection_phase1`.\n\n"
       "| Run | Description |\n"
       "|-----|-------------|\n"
       "| xgb_baseline | XGBoost with scale_pos_weight, raw features |\n"
       "| xgb_smotetomek | XGBoost on SMOTE-resampled training set |\n"
       "| isolation_forest | IsolationForest fit on legitimate transactions only |\n"
       "| composite_alpha_* | 5 sensitivity runs over composite weight α |\n\n"
       "Best model selected by **AUC-PR** — appropriate for imbalanced fraud detection."),
])

# ============================================================
# Notebook 4 — Evaluation
# ============================================================

nb04 = nb([
    md("# 04 — Evaluation, Cost Analysis & SHAP\n"
       "**Depends on:** 03_model_training.ipynb (model artifacts in `models/`)"),

    md("## 1. Setup"),

    code(
        "import sys\n"
        "sys.path.insert(0, '..')\n\n"
        "import pandas as pd\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "import joblib\n"
        "import shap\n"
        "from sklearn.metrics import (roc_auc_score, average_precision_score,\n"
        "                             confusion_matrix, roc_curve,\n"
        "                             precision_recall_curve)\n"
        "import warnings\n\n"
        "from src.features import normalize_iforest_scores, composite_score\n"
        "from src.scoring import route_decision, Decision\n\n"
        "warnings.filterwarnings('ignore')\n"
        "sns.set_style('darkgrid')\n"
        "print('imports ok')"
    ),

    md("## 2. Load Models & Validation Data"),

    code(
        "xgb_base   = joblib.load('../models/xgboost_baseline.pkl')\n"
        "xgb_smt    = joblib.load('../models/xgboost_smt.pkl')\n"
        "iforest    = joblib.load('../models/iforest_model.pkl')\n"
        "config     = joblib.load('../models/scoring_config.pkl')\n"
        "imputer    = joblib.load('../models/imputer.pkl')\n"
        "feature_cols = joblib.load('../models/feature_cols.pkl')\n\n"
        "X_val     = pd.read_parquet('../data/processed/X_val_raw.parquet')\n"
        "X_val_imp = pd.read_parquet('../data/processed/X_val_imp.parquet')\n"
        "y_val     = pd.read_parquet('../data/processed/y_val.parquet')['isFraud']\n\n"
        "best_alpha = config['best_alpha']\n\n"
        "# Reconstruct all score vectors\n"
        "y_pred_base    = xgb_base.predict_proba(X_val)[:, 1]\n"
        "y_pred_smt     = xgb_smt.predict_proba(X_val_imp)[:, 1]\n"
        "raw_if_scores  = iforest.decision_function(X_val_imp)\n"
        "iforest_scores = normalize_iforest_scores(raw_if_scores)\n"
        "comp_scores    = np.array([\n"
        "    composite_score(float(xp), float(ip), alpha=best_alpha)\n"
        "    for xp, ip in zip(y_pred_smt, iforest_scores)\n"
        "])\n\n"
        "print(f'Validation set: {len(y_val):,} transactions  |  fraud: {y_val.sum():,}')"
    ),

    md("## 3. Model Comparison Table"),

    code(
        "def compute_metrics(y_true, y_score):\n"
        "    fpr, tpr, _ = roc_curve(y_true, y_score)\n"
        "    idx = np.searchsorted(fpr, 0.05)\n"
        "    recall_5fpr = float(tpr[min(idx, len(tpr)-1)])\n"
        "    return {\n"
        "        'AUC-ROC':      round(roc_auc_score(y_true, y_score), 4),\n"
        "        'AUC-PR':       round(average_precision_score(y_true, y_score), 4),\n"
        "        'Recall@5%FPR': round(recall_5fpr, 4),\n"
        "    }\n\n"
        "rows = {\n"
        "    'XGB Baseline':             compute_metrics(y_val, y_pred_base),\n"
        "    'XGB + SMOTETomek':         compute_metrics(y_val, y_pred_smt),\n"
        "    'Isolation Forest':         compute_metrics(y_val, iforest_scores),\n"
        "    f'Composite α={best_alpha}': compute_metrics(y_val, comp_scores),\n"
        "}\n\n"
        "comparison = pd.DataFrame(rows).T\n"
        "print(comparison.to_string())\n"
        "comparison.to_csv('../results/04_model_comparison.csv')"
    ),

    md("## 4. Confusion Matrices at Multiple Thresholds"),

    code(
        "thresholds = [0.3, 0.5, 0.7]\n"
        "fig, axes = plt.subplots(1, len(thresholds), figsize=(15, 4))\n\n"
        "for ax, thresh in zip(axes, thresholds):\n"
        "    y_hat = (comp_scores >= thresh).astype(int)\n"
        "    cm = confusion_matrix(y_val, y_hat)\n"
        "    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,\n"
        "                xticklabels=['Pred 0','Pred 1'],\n"
        "                yticklabels=['True 0','True 1'])\n"
        "    tn, fp, fn, tp = cm.ravel()\n"
        "    prec = tp/(tp+fp+1e-9)\n"
        "    rec  = tp/(tp+fn+1e-9)\n"
        "    ax.set_title(f'Threshold={thresh}\\nPrec={prec:.3f} Rec={rec:.3f}')\n\n"
        "plt.suptitle(f'Composite Score Confusion Matrices (α={best_alpha})', y=1.02)\n"
        "plt.tight_layout()\n"
        "plt.savefig('../results/04_confusion_matrices.png', dpi=120, bbox_inches='tight')\n"
        "plt.show()"
    ),

    md("## 5. Cost Analysis\n\n"
       "Business costs:\n"
       "- **FP:** \\$50 per false positive (manual review labour)\n"
       "- **FN:** \\$500 per false negative (undetected fraud loss)\n\n"
       "Total cost = 50×FP + 500×FN"),

    code(
        "COST_FP = 50\n"
        "COST_FN = 500\n\n"
        "thresh_range = np.linspace(0.01, 0.99, 100)\n"
        "costs = []\n\n"
        "for thresh in thresh_range:\n"
        "    y_hat = (comp_scores >= thresh).astype(int)\n"
        "    cm = confusion_matrix(y_val, y_hat)\n"
        "    tn, fp, fn, tp = cm.ravel()\n"
        "    cost = COST_FP * fp + COST_FN * fn\n"
        "    costs.append({'threshold': thresh, 'cost': cost, 'fp': fp, 'fn': fn})\n\n"
        "cost_df = pd.DataFrame(costs)\n"
        "best_thresh_row = cost_df.loc[cost_df['cost'].idxmin()]\n"
        "print(f'Optimal threshold (min cost): {best_thresh_row[\"threshold\"]:.2f}')\n"
        "print(f'  Total cost: ${best_thresh_row[\"cost\"]:,.0f}')\n"
        "print(f'  FP: {best_thresh_row[\"fp\"]:.0f}  FN: {best_thresh_row[\"fn\"]:.0f}')\n\n"
        "fig, ax = plt.subplots(figsize=(10, 4))\n"
        "ax.plot(cost_df['threshold'], cost_df['cost'], color='steelblue')\n"
        "ax.axvline(best_thresh_row['threshold'], color='red', linestyle='--',\n"
        "           label=f'Optimal threshold={best_thresh_row[\"threshold\"]:.2f}')\n"
        "ax.set_xlabel('Decision Threshold')\n"
        "ax.set_ylabel('Total Cost ($)')\n"
        "ax.set_title(f'Cost Analysis — $50/FP + $500/FN  |  α={best_alpha}')\n"
        "ax.legend()\n"
        "plt.tight_layout()\n"
        "plt.savefig('../results/04_cost_analysis.png', dpi=120, bbox_inches='tight')\n"
        "plt.show()\n\n"
        "# Compare costs at canonical thresholds\n"
        "for thresh in [0.3, 0.5, 0.7]:\n"
        "    row = cost_df.loc[(cost_df['threshold'] - thresh).abs().idxmin()]\n"
        "    print(f'  threshold={thresh}:  cost=${row[\"cost\"]:>10,.0f}  FP={row[\"fp\"]:.0f}  FN={row[\"fn\"]:.0f}')"
    ),

    md("## 6. Precision-Recall & Decision Routing Summary"),

    code(
        "# Threshold comparison table: route decisions at 0.3/0.5/0.7\n"
        "for thresh_low, thresh_high in [(0.3, 0.7), (0.4, 0.6), (0.5, 0.7)]:\n"
        "    decisions = [route_decision(float(s), low=thresh_low, high=thresh_high)\n"
        "                 for s in comp_scores]\n"
        "    counts = pd.Series(decisions).value_counts()\n"
        "    print(f'\\nlow={thresh_low} high={thresh_high}:')\n"
        "    for d in [Decision.APPROVE, Decision.MANUAL_REVIEW, Decision.DECLINE]:\n"
        "        print(f'  {d.value:15s}: {counts.get(d, 0):>6,}')"
    ),

    md("## 7. SHAP Analysis"),

    code(
        "# Use XGB+SMOTETomek (best XGB model) for SHAP\n"
        "# Subsample for speed: 500 background + 200 explain\n"
        "SEED = 42\n"
        "rng = np.random.default_rng(SEED)\n\n"
        "bg_idx   = rng.choice(len(X_val_imp), 500, replace=False)\n"
        "exp_idx  = rng.choice(len(X_val_imp), 200, replace=False)\n\n"
        "X_bg     = X_val_imp.iloc[bg_idx]\n"
        "X_explain = X_val_imp.iloc[exp_idx]\n\n"
        "explainer = shap.TreeExplainer(xgb_smt, data=X_bg)\n"
        "shap_values = explainer(X_explain)\n\n"
        "print(f'SHAP values shape: {shap_values.values.shape}')"
    ),

    code(
        "# SHAP summary plot — top 20 features\n"
        "fig, ax = plt.subplots(figsize=(10, 7))\n"
        "shap.summary_plot(shap_values, X_explain, max_display=20, show=False)\n"
        "plt.tight_layout()\n"
        "plt.savefig('../results/04_shap_summary.png', dpi=120, bbox_inches='tight')\n"
        "plt.show()\n"
        "print('SHAP summary saved')"
    ),

    code(
        "# SHAP feature importance bar chart\n"
        "mean_abs_shap = pd.Series(\n"
        "    np.abs(shap_values.values).mean(axis=0),\n"
        "    index=X_explain.columns\n"
        ").sort_values(ascending=False).head(20)\n\n"
        "fig, ax = plt.subplots(figsize=(9, 6))\n"
        "mean_abs_shap.plot(kind='barh', ax=ax, color='steelblue', edgecolor='white')\n"
        "ax.invert_yaxis()\n"
        "ax.set_title('Top 20 Features by Mean |SHAP| — XGB+SMOTETomek')\n"
        "ax.set_xlabel('Mean |SHAP value|')\n"
        "plt.tight_layout()\n"
        "plt.savefig('../results/04_shap_importance.png', dpi=120, bbox_inches='tight')\n"
        "plt.show()"
    ),

    md("## 8. SHAP Waterfall Plots — 5 Example Transactions"),

    code(
        "# Select examples: 2 fraud, 2 legit, 1 borderline\n"
        "val_scores = comp_scores[exp_idx]\n"
        "val_labels = y_val.values[exp_idx]\n\n"
        "fraud_idx    = np.where(val_labels == 1)[0]\n"
        "legit_idx    = np.where(val_labels == 0)[0]\n"
        "border_idx   = np.where((val_scores > 0.3) & (val_scores < 0.7))[0]\n\n"
        "examples = {\n"
        "    'Fraud #1':      fraud_idx[0]  if len(fraud_idx)  >= 1 else None,\n"
        "    'Fraud #2':      fraud_idx[1]  if len(fraud_idx)  >= 2 else None,\n"
        "    'Legitimate #1': legit_idx[0]  if len(legit_idx)  >= 1 else None,\n"
        "    'Legitimate #2': legit_idx[1]  if len(legit_idx)  >= 2 else None,\n"
        "    'Borderline':    border_idx[0] if len(border_idx) >= 1 else None,\n"
        "}\n\n"
        "for title, idx in examples.items():\n"
        "    if idx is None:\n"
        "        print(f'{title}: no example found')\n"
        "        continue\n"
        "    score = val_scores[idx]\n"
        "    label = val_labels[idx]\n"
        "    print(f'{title} — score={score:.3f}  true={label}')\n"
        "    fig, ax = plt.subplots(figsize=(10, 5))\n"
        "    shap.waterfall_plot(shap_values[idx], max_display=15, show=False)\n"
        "    plt.title(f'{title}  |  composite={score:.3f}  isFraud={label}')\n"
        "    plt.tight_layout()\n"
        "    fname = title.lower().replace(' ','_').replace('#','')\n"
        "    plt.savefig(f'../results/04_waterfall_{fname}.png', dpi=120, bbox_inches='tight')\n"
        "    plt.show()"
    ),

    md("## Summary\n\n"
       "### Results\n\n"
       "| Model | AUC-ROC | AUC-PR | Recall@5%FPR |\n"
       "|-------|---------|--------|-------------|\n"
       "| XGB Baseline | — | — | — |\n"
       "| XGB + SMOTETomek | — | — | — |\n"
       "| Isolation Forest | — | — | — |\n"
       "| Composite | — | — | — |\n\n"
       "### Cost Analysis\n"
       "- Optimal threshold minimises \\$50×FP + \\$500×FN\n"
       "- DECLINE threshold ≥0.7 chosen to cap FN cost\n\n"
       "### SHAP Insights\n"
       "- Top drivers will be identified from the summary plot\n"
       "- Waterfall plots explain individual decisions for interpretability"),
])

# ============================================================
# Write notebooks to disk
# ============================================================

notebooks = {
    "01_eda.ipynb":                nb01,
    "02_feature_engineering.ipynb": nb02,
    "03_model_training.ipynb":     nb03,
    "04_evaluation.ipynb":         nb04,
}

for fname, notebook in notebooks.items():
    path = NOTEBOOKS_DIR / fname
    with open(path, "w") as f:
        json.dump(notebook, f, indent=1)
    print(f"Written: notebooks/{fname}  ({path.stat().st_size/1024:.0f} KB)")

print("\nDone. Run: make notebook")
