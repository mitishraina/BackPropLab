# Supervised vs Unsupervised

| Feature            | Supervised Learning                                  | Unsupervised Learning                         |
|--------------------|------------------------------------------------------|-----------------------------------------------|
| Data               | Labeled (X + y)                                      | Unlabeled (X only)                            |
| Objective          | Predict target/output                                | Discover hidden patterns/structure            |
| Common Tasks       | Regression, Classification                           | Clustering, Dimensionality Reduction          |
| Output             | Predicted value or class label                       | Clusters or grouped data                      |
| Example Algorithms | Linear Regression, Logistic Regression, SVM, Random Forest, XGBoost | K-Means, DBSCAN, PCA |
| Example Use Cases  | House price prediction, Spam detection               | Customer segmentation, Anomaly detection      |

## Regression vs Classification
Most algorithms can do both regression and classification.
The main ones that are task specific:
- Linear Regression -> Regression Only
- Logistic Regression -> Classification Only
- K-Means -> Unsupervised Clustering

## Regression vs Classification Algorithms

| Category        | Regression (Continuous Output) | Classification (Categorical Output) |
|----------------|---------------------------------|-------------------------------------|
| **Goal**       | Predict continuous numeric values | Predict class labels/categories |
| **Linear Models** | Linear Regression | Logistic Regression |
| **Tree-Based** | Decision Tree (Regressor) <br> Random Forest (Regressor) – Bagging <br> Gradient Boosting (Regressor) – Boosting <br> AdaBoost (Regressor) <br> XGBoost (Regressor) <br> LightGBM (Regressor) <br> CatBoost (Regressor) | Decision Tree (Classifier) <br> Random Forest (Classifier) – Bagging <br> Gradient Boosting (Classifier) – Boosting <br> AdaBoost (Classifier) <br> XGBoost (Classifier) <br> LightGBM (Classifier) <br> CatBoost (Classifier) |
| **Distance-Based** | KNN Regressor | KNN Classifier |
| **Margin-Based** | SVM (SVR – Support Vector Regression) | SVM (SVC – Support Vector Classification) |
| **Example Use Case** | House price prediction, Sales forecasting | Spam detection, Disease classification |




# Ensemble Methods - Bagging vs Boosting

| Feature | Bagging | Boosting |
|----------|----------|-----------|
| Full Form | Bootstrap Aggregating | Sequential Weak Learner Improvement |
| Main Idea | Train multiple models independently and combine their results | Train models sequentially, each correcting the previous model’s errors |
| Training Style | Parallel | Sequential |
| Goal | Reduce variance | Reduce bias (and variance) |
| Data Sampling | Random sampling with replacement (Bootstrap) | Adjusted sampling based on previous errors |
| Model Dependency | Independent models | Dependent models |
| Overfitting | Less prone (reduces variance) | Can overfit if not regularized |
| Best For | High variance models (e.g., Decision Trees) | Weak learners |
| Example Algorithms | Random Forest | AdaBoost, Gradient Boosting, XGBoost |
| Weight Adjustment | Equal weight to models | Higher weight to misclassified samples |
| Computation Speed | Faster (parallelizable) | Slower (sequential process) |
| Bias-Variance Impact | Mainly reduces variance | Mainly reduces bias |


## Bagging (Bootstrap Aggregating)

### Idea:
Train multiple models independently on different random samples of data and combine their predictions.

### Workflow Diagram:

```
                Original Dataset
                       │
        ┌──────────────┼──────────────┐
        │              │              │
   Bootstrap 1    Bootstrap 2    Bootstrap 3
   (Sampled)       (Sampled)       (Sampled)
        │              │              │
     Model 1        Model 2        Model 3
   (Tree 1)        (Tree 2)        (Tree 3)
        └──────────────┼──────────────┘
                       │
               Voting / Averaging
                       │
                 Final Prediction
```

### Key Points:
- Models are trained **in parallel**
- Each model is **independent**
- Reduces **variance**
- Example: Random Forest


## Boosting

### Idea:
Train models sequentially, where each new model focuses on correcting the errors of the previous one.

### Workflow Diagram:

```
        Original Dataset
               │
          Model 1
        (Weak Learner)
               │
         Errors Identified
               │
        Adjust Weights
               │
          Model 2
               │
         Errors Identified
               │
        Adjust Weights
               │
          Model 3
               │
        Weighted Combination
               │
        Final Strong Model
```

### Key Points:
- Models are trained **sequentially**
- Each model depends on the previous one
- Reduces **bias**
- Examples: AdaBoost, Gradient Boosting, XGBoost


## Core Difference Visual

```
Bagging  →  Parallel Models  →  Reduce Variance
Boosting →  Sequential Models →  Reduce Bias
```