"""Baseline model: logistic regression on the cleaned Credit Score data.

Target: Credit_Score == 'Poor' (1) versus Standard or Good (0).
Split: stratified and grouped by Customer_ID (a customer is never in both sets).
All preprocessing is fitted on the training set only (it lives in the pipeline).
"""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (brier_score_loss, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from scripts.clean_data import clean

SEED = 42


def psi(expected, actual, n_bins=10):
    """Population Stability Index between a reference sample and a new sample."""
    expected = pd.Series(expected).dropna()
    actual = pd.Series(actual).dropna()
    if expected.nunique() <= n_bins:  # few values: one bin per value
        cats = sorted(set(expected) | set(actual), key=str)
        e = expected.value_counts(normalize=True).reindex(cats).fillna(0).values
        a = actual.value_counts(normalize=True).reindex(cats).fillna(0).values
    else:  # bins from the quantiles of the reference sample
        edges = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        e = np.histogram(expected, edges)[0] / len(expected)
        a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def run(data_path, fig_dir, metrics_path):
    df = pd.read_csv(data_path, low_memory=False)
    df, num_col, cat_col, loan_col = clean(df)

    # Target and split
    y = (df['Credit_Score'] == 'Poor').astype(int)
    groups = df['Customer_ID']
    ord_col = ['Credit_Mix']
    ohe_col = [c for c in cat_col if c not in ord_col]
    num_all = num_col + loan_col
    X = df[num_all + ord_col + ohe_col]

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_idx, test_idx = next(sgkf.split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    shared = set(groups.iloc[train_idx]) & set(groups.iloc[test_idx])
    assert len(shared) == 0

    # Pipeline
    prep = ColumnTransformer([
        ('num', Pipeline([('imp', SimpleImputer(strategy='median')),
                          ('sc', StandardScaler())]), num_all),
        ('ord', Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                          ('enc', OrdinalEncoder(categories=[['Bad', 'Standard', 'Good']])),
                          ('sc', StandardScaler())]), ord_col),
        ('ohe', Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                          ('enc', OneHotEncoder(handle_unknown='ignore'))]), ohe_col),
    ])
    model = Pipeline([('prep', prep),
                      ('lr', LogisticRegression(max_iter=1000, random_state=SEED))])
    model.fit(X_train, y_train)

    p_train = model.predict_proba(X_train)[:, 1]
    p_test = model.predict_proba(X_test)[:, 1]

    # Discrimination
    auc_train = roc_auc_score(y_train, p_train)
    auc = roc_auc_score(y_test, p_test)
    fpr, tpr, _ = roc_curve(y_test, p_test)
    ks = float(np.max(tpr - fpr))
    pred = (p_test >= 0.5).astype(int)
    metrics = {
        'target': "Credit_Score == 'Poor' (1) vs Standard/Good (0)",
        'seed': SEED,
        'n_rows_train': int(len(train_idx)), 'n_rows_test': int(len(test_idx)),
        'n_customers_train': int(groups.iloc[train_idx].nunique()),
        'n_customers_test': int(groups.iloc[test_idx].nunique()),
        'share_poor_train': float(y_train.mean()), 'share_poor_test': float(y_test.mean()),
        'auc_train': float(auc_train), 'auc_test': float(auc),
        'gini_test': float(2 * auc - 1), 'ks_test': ks,
        'brier_test': float(brier_score_loss(y_test, p_test)),
        'brier_reference_constant': float(brier_score_loss(y_test, np.full(len(y_test), y_train.mean()))),
        'precision_test_at_0.5': float(precision_score(y_test, pred)),
        'recall_test_at_0.5': float(recall_score(y_test, pred)),
    }

    # Coefficients
    names = model.named_steps['prep'].get_feature_names_out()
    coef = model.named_steps['lr'].coef_[0]
    orig = []
    for n in names:
        n = n.split('__', 1)[1]
        orig.append(next((c for c in ohe_col if n.startswith(c + '_')), n))
    coefs = pd.DataFrame({'feature': [n.split('__', 1)[1] for n in names],
                          'variable': orig, 'coef': coef,
                          'odds_ratio': np.exp(coef)})
    coefs['abs_coef'] = coefs['coef'].abs()
    coefs = coefs.sort_values('abs_coef', ascending=False)
    coefs.drop(columns='abs_coef').round(4).to_csv(fig_dir.parent / 'coefficients.csv', index=False)

    # PSI: score and the 5 most important variables (largest |coef| of the variable)
    top_vars = coefs.groupby('variable')['abs_coef'].max().sort_values(ascending=False).head(5).index.tolist()
    metrics['psi_score_train_vs_test'] = psi(p_train, p_test)
    metrics['psi_top5_variables'] = {v: psi(X_train[v], X_test[v]) for v in top_vars}

    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    # Figures
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False positive rate'); plt.ylabel('True positive rate')
    plt.title('ROC curve (test set)'); plt.legend()
    plt.tight_layout(); plt.savefig(fig_dir / 'roc.png', dpi=130); plt.close()

    frac, mean_pred = calibration_curve(y_test, p_test, n_bins=10, strategy='quantile')
    plt.figure(figsize=(5, 5))
    plt.plot(mean_pred, frac, 'o-', label='Logistic regression')
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfect calibration')
    plt.xlabel('Mean predicted probability'); plt.ylabel('Observed share of Poor')
    plt.title(f"Reliability curve (Brier = {metrics['brier_test']:.3f})"); plt.legend()
    plt.tight_layout(); plt.savefig(fig_dir / 'calibration.png', dpi=130); plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist(p_train, bins=30, alpha=0.6, density=True, label='Train')
    plt.hist(p_test, bins=30, alpha=0.6, density=True, label='Test')
    plt.xlabel('Predicted probability of Poor'); plt.ylabel('Density')
    plt.title(f"Score distribution (PSI = {metrics['psi_score_train_vs_test']:.4f})"); plt.legend()
    plt.tight_layout(); plt.savefig(fig_dir / 'score_distribution.png', dpi=130); plt.close()

    top = coefs.head(15).iloc[::-1]
    plt.figure(figsize=(7, 5))
    plt.barh(top['feature'], top['coef'], color=np.where(top['coef'] > 0, 'tab:red', 'tab:blue'))
    plt.xlabel('Coefficient (log odds of Poor)')
    plt.title('15 largest coefficients')
    plt.tight_layout(); plt.savefig(fig_dir / 'coefficients.png', dpi=130); plt.close()

    return metrics, coefs, top_vars
