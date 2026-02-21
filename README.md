# Kaggle Titanic

End-to-end ML project for Kaggle's Titanic competition: predict passenger survival from tabular features.

## Problem

Given passenger information (`train.csv` / `test.csv`), build a model that predicts `Survived` for the unseen Kaggle test set.

## Repository Layout

- `data/`: competition train/test datasets
- `titanic_survival_NN.ipynb`: main notebook (EDA, preprocessing, modeling)
- `xgboost.py`: XGBoost experimentation script
- `solutions/`: generated CSV submission files

## Approach Summary

1. Data loading and inspection
2. Feature engineering (for example title extraction and scaling)
3. Missing-value handling (Embarked/Fare/Age strategies)
4. Model training and evaluation
5. Kaggle submission generation

## Result

Current best score in this repository: **0.78229** on Kaggle public leaderboard.

## How to Run

The project is notebook-first:

```bash
jupyter notebook titanic_survival_NN.ipynb
```

or:

```bash
jupyter lab
```

Then run notebook cells in order.

If you want to run the script experiment:

```bash
python xgboost.py
```

## Notes

- This repository focuses on explainable preprocessing and baseline modeling.
- It is intended as an educational competition project rather than a production pipeline.

## Next Improvements

- Add a pinned `requirements.txt` for reproducibility.
- Add cross-validation report and feature-importance artifacts.
- Add a small CLI to reproduce submissions without notebooks.
