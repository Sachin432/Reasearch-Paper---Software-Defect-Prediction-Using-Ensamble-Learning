# Software Defect Prediction Using Ensemble Learning

Research Paper Implementation

This repository contains the complete implementation, analysis, and supporting files for the research work titled **“Software Defect Prediction Using Ensemble Learning.”**
The project explores how various ensemble learning techniques can improve defect prediction accuracy across multiple software engineering datasets.

---

## Overview

Software Defect Prediction (SDP) aims to identify faulty modules before deployment, improving software reliability and reducing maintenance cost. This research investigates a stacking-based ensemble framework integrating multiple machine learning models to enhance predictive performance.

The repository includes all code, datasets (if allowed), evaluation metrics, plots, and the final IEEE-style research paper.

---

## Key Objectives

* Analyze the performance of individual machine learning models for defect prediction
* Build ensemble models including Bagging, Boosting, Voting, and Stacking
* Evaluate performance across multiple benchmark datasets
* Compare metrics such as AUC, F1-score, Precision, Recall, Accuracy, and MCC
* Provide reproducible experiments for research and academic use

---

## Features

* Comprehensive data preprocessing pipeline
* Multiple ML models including Random Forest, Extra Trees, XGBoost, LightGBM
* Stacking ensemble framework
* ROC curves, confusion matrices, and metric comparison tables
* Clean, modular, and reproducible code
* IEEE-style research paper included

---


## Methodology

1. Load and preprocess SDP datasets
2. Train baseline ML models
3. Build ensemble models (Bagging, Boosting, Stacking)
4. Evaluate metrics on each dataset
5. Compare and analyze best-performing model
6. Generate visualizations and tables for research paper

---

## Algorithms Used

* Logistic Regression
* K-Nearest Neighbors
* Decision Tree
* Random Forest
* Extra Trees
* XGBoost
* LightGBM
* Stacking Classifier (Meta-Learner Based)

---

## Performance Evaluation

The project uses the following metrics:

* Accuracy
* Precision
* Recall
* F1-score
* AUC
* MCC
* Confusion Matrix
* ROC Curve

All results are included inside the **results/** directory and integrated into the research paper.

---

## Setup and Installation

Install required packages:

```
pip install -r requirements.txt
```

Or manually install:

```
pip install numpy pandas scikit-learn xgboost lightgbm matplotlib seaborn
```



## Use Cases

* Academic research in software engineering
* Machine learning research on defect prediction
* Benchmarking ML/Ensemble models
* IEEE and conference paper implementation
* Comparative study for M.Tech/B.Tech projects

---

## Research Outcomes

* Stacking ensemble consistently performed better across datasets
* Improved AUC and F1-score compared to traditional ML models
* Demonstrated robustness across imbalanced datasets
* Provided strong empirical evidence supporting ensemble methods for SDP

