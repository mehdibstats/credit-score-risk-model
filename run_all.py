"""Runs the whole project: download, cleaning, split, training, evaluation, figures.

Usage: python run_all.py
"""
from pathlib import Path

from scripts.download_data import download
from scripts.train_baseline import run

ROOT = Path(__file__).resolve().parent

if __name__ == '__main__':
    download()
    (ROOT / 'reports' / 'figures').mkdir(parents=True, exist_ok=True)
    metrics, coefs, top_vars = run(ROOT / 'data' / 'raw' / 'train.csv',
                                   ROOT / 'reports' / 'figures',
                                   ROOT / 'reports' / 'metrics.json')
    for k, v in metrics.items():
        print(k, ':', v)
    print('Top 5 variables for PSI:', top_vars)
    print(coefs.head(15)[['feature', 'coef', 'odds_ratio']].round(3).to_string(index=False))
