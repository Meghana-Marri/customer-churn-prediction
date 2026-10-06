# Customer Churn Prediction

Predicts whether a telecom customer will leave (churn) using the Telco Customer Churn dataset.
Four models are compared, and Logistic Regression with a tuned threshold is used as the final model.

## Dataset
- IBM Telco Customer Churn sample dataset (7,043 customers, 21 columns)
- About 26.5% of customers churned, so the data is imbalanced

## What I did
- Cleaned the data: converted `TotalCharges` to numbers (11 blank rows had tenure 0, so they were filled with 0) and dropped `customerID`
- Explored churn patterns by contract type, tenure, internet service and payment method
- One-hot encoded the categorical columns and split the data 80/20 with stratification
- Trained Logistic Regression, Random Forest, Gradient Boosting and XGBoost
- Compared the models using precision, recall, F1 and ROC-AUC
- Tuned the decision threshold for Logistic Regression (0.30)
- Saved the model and scaler, and wrote a `predict_churn()` function for a single customer

## Model comparison (ROC-AUC on test data)
| Model | ROC-AUC |
|---|---|
| Logistic Regression | 0.842 |
| Random Forest | 0.828 |
| Gradient Boosting | 0.842 |
| XGBoost | 0.844 |

The scores are very close, so I chose Logistic Regression because it is simple and easy to explain.

## Final model (Logistic Regression, threshold 0.30)
| Metric | Score |
|---|---|
| Accuracy | 74.9% |
| Precision | 0.52 |
| Recall | 0.75 |
| F1 score | 0.62 |
| ROC-AUC | 0.842 |

Confusion matrix: 774 true negatives, 261 false positives, 92 false negatives, 282 true positives.

Recall is more important than accuracy here, because missing a customer who is about to leave is more costly than a false alarm. Lowering the threshold from 0.50 to 0.30 raised recall from 0.57 to 0.75, at the cost of lower accuracy and precision.

## Key findings
- Customers with short tenure churn more
- Fiber optic internet users churn more
- Customers on a two-year contract churn less

## Limitations
- The threshold was chosen using the test set, so the scores may be slightly optimistic
- No cross-validation or hyperparameter tuning yet

## Files
- `customer_churn_prediction.ipynb`: full notebook with outputs
- `WA_Fn-UseC_-Telco-Customer-Churn.csv`: dataset
- `churn_logistic_model.pkl`, `churn_scaler.pkl`: saved model and scaler

## How to run
1. Install the libraries: `pip install pandas numpy scikit-learn matplotlib xgboost joblib notebook`
2. Run `jupyter notebook` and open `customer_churn_prediction.ipynb`
3. Click Kernel > Restart & Run All

## Next steps
- Build a Streamlit web app for live predictions
- Add cross-validation and hyperparameter tuning
