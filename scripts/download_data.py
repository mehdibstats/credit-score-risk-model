"""Download the Kaggle dataset 'Credit score classification' into data/raw/.

Credentials are read by the Kaggle API from the environment (KAGGLE_USERNAME and
KAGGLE_KEY, or KAGGLE_API_TOKEN) or from ~/.kaggle. Nothing is stored in the repo.
If data/raw/train.csv already exists, nothing is downloaded.
"""
from pathlib import Path

DATASET = 'parisrohan/credit-score-classification'
RAW_DIR = Path(__file__).resolve().parent.parent / 'data' / 'raw'


def download():
    if (RAW_DIR / 'train.csv').exists():
        print('data/raw/train.csv already present, download skipped.')
        return
    from kaggle.api.kaggle_api_extended import KaggleApi  # imported here: it reads credentials on import
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(DATASET, path=str(RAW_DIR), unzip=True)
    print('Dataset downloaded in data/raw/')


if __name__ == '__main__':
    download()
