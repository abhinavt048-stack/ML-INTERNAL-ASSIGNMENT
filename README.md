# Machine Learning — Internal Assessment

**Student:** Abhinav Tiwari · **Submission:** 9–13 October 2026

| # | Component | Folder | Best Result |
|---|-----------|--------|-------------|
| 1 | **Dataset assignment:** Classification, Clustering, Boosting | [Assignment-1_Classification-Clustering-Boosting](Assignment-1_Classification-Clustering-Boosting/) | See table below |
| 2 | **Mini Project:** Customer Churn Prediction (Kaggle, batch of 3) | [Mini-Project_Customer-Churn-Prediction](Mini-Project_Customer-Churn-Prediction/) | Tuned XGBoost: ROC-AUC 0.844, Recall 74.9 % |
| 3 | Certificate course (NPTEL / equivalent) | Hard copy submitted separately | — |

## Assignment 1 — Result Table (best algorithm per dataset)

| Task | Dataset | Algorithms Compared | Best Algorithm | Result |
|------|---------|---------------------|----------------|--------|
| Classification | Social_Network_Ads.csv | Logistic Regression, KNN, Linear SVM, Kernel SVM, Naive Bayes, Decision Tree, Random Forest | **Kernel SVM (RBF)** | Accuracy 0.90, F1 0.868, ROC-AUC 0.963 |
| Clustering | Mall_Customers.csv | K-Means, Agglomerative, GMM, DBSCAN, BIRCH, Mean Shift, Spectral | **K-Means (k = 5)** | Silhouette 0.555, Davies-Bouldin 0.572 |
| Boosting | insurance_pre.csv | AdaBoost, Gradient Boosting, Hist GB, XGBoost, LightGBM, CatBoost | **XGBoost** | R² 0.884, RMSE 4,243.9 |

Full per-algorithm tables, plots and notebooks are in the [Assignment 1 README](Assignment-1_Classification-Clustering-Boosting/README.md).

## Mini Project — Customer Churn Prediction
11 algorithms compared on the Kaggle Telco Customer Churn dataset (7,043 customers). Class imbalance is handled with SMOTE, the top 3 models are tuned, and the decision threshold is optimised. The final tuned XGBoost catches **280 of 374 churners** in the unseen test set. See the [project README](Mini-Project_Customer-Churn-Prediction/README.md) and [report (PDF)](Mini-Project_Customer-Churn-Prediction/report/Mini_Project_Report_Customer_Churn.pdf).
