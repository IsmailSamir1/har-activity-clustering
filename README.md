# 🏃 Uncovering Patterns in Human Motion: Unsupervised Activity Clustering

Discovering types of human movement (walking, climbing stairs, sitting, standing, lying) from smartphone sensor data **without using labels**. Four clustering algorithms are compared on the Human Activity Recognition (HAR) dataset.

Introduction to Data Science course project, German International University (GIU), Cairo. Team of 4.

## Dataset

UCI **Human Activity Recognition** dataset: 7,352 samples × 561 accelerometer and gyroscope features recorded from 30 people. It has no missing values and no duplicates.

## Pipeline

1. **Scaling.** StandardScaler, because sensor features have very different ranges and K-Means relies on Euclidean distance.
2. **Redundancy removal.** Dropped features with absolute correlation > 0.95 to another feature.
3. **Choosing k.** The elbow method and silhouette analysis over k = 2–12 both point to **k = 6**, which matches the 6 real activities.
4. **K-Means and motion profiling.** Cluster centres split cleanly into **dynamic** clusters (high acceleration and gyroscope variability) and **static** clusters.
5. **PCA 2-D projection** to visualise how well the clusters separate.
6. **Anomaly detection.** The 5% of points furthest from their cluster centre sit on cluster boundaries. They are mostly transitions between activities, not sensor errors.
7. **Method comparison.** K-Means vs. Agglomerative (Ward) vs. Gaussian Mixture (full covariance) vs. BIRCH, compared by silhouette score. GMM also gives soft memberships, which suit gradual transitions between movements.

## Tech Stack

Python · pandas · NumPy · scikit-learn · Matplotlib · Seaborn · Google Colab

## Contents

| File | Description |
|---|---|
| [`har_clustering.ipynb`](har_clustering.ipynb) | Full analysis with outputs and figures |
