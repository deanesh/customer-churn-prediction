#### Project: Customer Churn Prediction

##### Summary
* **Customer Churn Prediction with Logistic Regression and XGBoost**
    *  Built robust classification pipelines using scikit-learn with preprocessing (StandardScaler, OneHotEncoder, OrdinalEncoder), L1-based feature selection, and modeling.
    *  Trained and evaluated both Logistic Regression and XGBoost using 5-fold Stratified Cross-Validation.
    *  Achieved XGBoost Accuracy: ~XX%, ROC AUC: ~YY on test set.
    *  Developed detailed EDA and model evaluation reports including ROC curves, feature importances, and confusion matrices.
    *  Tools: pandas, scikit-learn, xgboost, matplotlib, seaborn
    * **ROC Curve (Receiver Operating Characteristic Curve)**
* **Logistic Regression** is a **classification algorithm** used to predict probability of a **binary** outcome (like Yes/No, 0/1, True/False).
* **XtremeGradientBoostClassifier:** classification model based on gradient boosting decision trees
    * **Penalties** are used to prevent overfitting by discouraging complex models (like large coefficients).
        * L1 - Adds **absolute value** of weights to loss, L2 - Adds **squared value** of weights to loss
    * **Solvers** are **optimization algorithms** used to find best model parameters (like weights) by minimizing loss function.
    * **Loss function** measures how far off a model's predictions are from actual values.
        * logloss/cross-entropy:  difference between the **actual label** and **predicted probability**
        * Cross-Entropy is just a general term (used in multi-class too), while Log Loss often refers to the binary case
    * **Gradient Descent** optimization algorithm used to minimize loss function by updating model parameters
    * **Model parameters** are internal values a machine learning model learns from training data
---
* **CSV File:** **`Customer-Churn.csv`**
* **Columns:**

| Column Name        | Description                                                       |
| ------------------ | ----------------------------------------------------------------- |
| `customerID`       | Unique identifier for each customer                               |
| `gender`           | Customer gender (`Male` or `Female`)                              |
| `SeniorCitizen`    | Indicates if the customer is a senior (1 = Yes, 0 = No)           |
| `Partner`          | Whether the customer has a partner (`Yes` or `No`)                |
| `Dependents`       | Whether the customer has dependents (`Yes` or `No`)               |
| `tenure`           | Number of months the customer has stayed with the company         |
| `PhoneService`     | Whether the customer has phone service (`Yes` or `No`)            |
| `MultipleLines`    | Whether the customer has multiple phone lines                     |
| `InternetService`  | Type of internet service (`DSL`, `Fiber optic`, `No`)             |
| `OnlineSecurity`   | Whether the customer has online security service                  |
| `OnlineBackup`     | Whether the customer has online backup service                    |
| `DeviceProtection` | Whether the customer has device protection service                |
| `TechSupport`      | Whether the customer has tech support service                     |
| `StreamingTV`      | Whether the customer has streaming TV service                     |
| `StreamingMovies`  | Whether the customer has streaming movies service                 |
| `Contract`         | Type of contract (`Month-to-month`, `One year`, `Two year`)       |
| `PaperlessBilling` | Whether the customer uses paperless billing (`Yes` or `No`)       |
| `PaymentMethod`    | Payment method (e.g., `Electronic check`, `Mailed check`, etc.)   |
| `MonthlyCharges`   | The amount charged to the customer monthly                        |
| `TotalCharges`     | Total amount charged (may need to convert from object to numeric) |
| `Churn`            | Whether the customer has left the company (`Yes` or `No`)         |
---

##### 📊 **Classification Metrics Summary**
| Metric     | Means                                                                 | Formula                                              | Best When                                               |
|------------|-----------------------------------------------------------------------|------------------------------------------------------|----------------------------------------------------------|
| Accuracy   | Overall, how many predictions were correct?                           | (TP + TN) / (TP + TN + FP + FN)                      | Classes are balanced and all errors matter equally.      |
| Precision  | Of all predicted positives, how many were actually positive?          | TP / (TP + FP)                                       | False positives are costly (e.g., flagging good emails). |
| Recall     | Of all actual positives, how many did we correctly predict?           | TP / (TP + FN)                                       | False negatives are costly (e.g., missing cancer cases). |
| F1 Score   | Balance between Precision and Recall                                  | 2 * (Precision * Recall) / (Precision + Recall)      | You need a balance and classes are imbalanced.           |

##### 🛠️ **How to Improve These Metrics**
| Technique                      | Helps Improve                          |
|-------------------------------|----------------------------------------|
| ✅ Tune classification threshold | Precision or Recall (adjust trade-off) |
| ✅ Use better features          | All metrics                            |
| ✅ Try different models         | All metrics                            |
| ✅ Resampling (SMOTE, etc.)     | Recall (especially for imbalanced data)|
| ✅ Class weighting              | Precision, Recall, F1 Score             |