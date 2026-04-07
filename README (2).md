# Titanic Survival Prediction

A machine learning project to predict passenger survival on the Titanic using classical classification algorithms.

---

## Overview

This project tackles the classic [Kaggle Titanic competition](https://www.kaggle.com/competitions/titanic) — predicting which passengers survived based on features like age, class, sex, and family size.

The pipeline covers everything from exploratory data analysis to feature engineering, model training, hyperparameter tuning, and final submission.

---

## Project Structure

```
titanic/
├── titanic_improved.ipynb   # Main notebook (EDA → modeling → submission)
├── train.csv                # Training data
├── test.csv                 # Test data
├── submission.csv           # Final Kaggle submission
└── README.md
```

---

## Pipeline

```
Load Data → EDA → Impute Missing → Feature Engineering → Encode → Split → Train → Tune → Predict
```

---

## Feature Engineering

| Feature | Description |
|---|---|
| `Title` | Extracted from passenger name (Mr, Miss, Mrs, Rare...) |
| `Family` | SibSp + Parch + 1 |
| `isAlone` | 1 if traveling alone |
| `AgeGroup` | Binned: Child / Young / Adult / Senior |
| `Age` (imputed) | Median grouped by Pclass + Sex |
| `Fare` (log) | log1p transform to reduce skew |

---

## Models

| Model | Notes | CV Accuracy |
|---|---|---|
| Logistic Regression | L2, liblinear, C=1.0 | ~80% |
| KNN | k=5, with StandardScaler | ~79% |
| Random Forest | 100 trees, max_depth=5 | ~82% |
| GBC (optimized) | GridSearch → n_estimators=200, max_depth=2 | ~84% |

Best model: Gradient Boosting Classifier after GridSearchCV tuning.

---

## Results

- Final model: Optimized GradientBoostingClassifier
- Cross-validation accuracy: ~84%
- Top features: Sex, Fare, Title, Pclass, Age

---

## How to Run

```bash
git clone https://github.com/amr-eleisawie/titanic-survival.git
cd titanic-survival

pip install pandas numpy scikit-learn matplotlib seaborn

jupyter notebook titanic_improved.ipynb
```

---

## Dependencies

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

---

## Author

Amr Eleisawie — [@amr-eleisawie](https://github.com/amr-eleisawie)
