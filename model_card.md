# Model Card: Credit Risk Default Prediction

## Model Details

- **Model type**: Logistic Regression (primary) + Gradient Boosting Classifier (challenger)
- **Framework**: scikit-learn 1.3+
- **Training data**: German Credit Dataset (OpenML ID 31, 1000 samples)
- **Task**: Binary classification (default vs. no default)
- **Version**: 0.1.0
- **License**: MIT


## Training Data

- **Source**: UCI German Credit Dataset via OpenML
- **Size**: 1,000 borrowers, 20 features
- **Class distribution**: ~70% no default, ~30% default
- **Features**: Duration, credit amount, employment, housing, savings, checking account status, purpose, and more

## Evaluation Metrics

| Metric | Logistic Regression | Gradient Boosting |
|--------|-------------------|-------------------|
| ROC-AUC | 0.787 ± 0.020 | 0.774 ± 0.023 |
| Recall | 0.713 | 0.457 |
| F1 | 0.606 | 0.523 |

*Metrics from 5-fold stratified cross-validation.*

**Best Model**: Logistic Regression

### Classification Report at Optimal Threshold

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| 0 (No Default) | 0.951 | 0.476 | 0.634 | 700 |
| 1 (Default) | 0.435 | 0.943 | 0.596 | 300 |

### Top Features (SHAP)

| Rank | Feature | Mean SHAP |
|------|---------|-----------|
| 1 | num__credit_amount | 0.5015 |
| 2 | cat__checking_status_no checking | 0.4922 |
| 3 | num__duration | 0.4044 |
| 4 | cat__credit_history_critical/other existing credit | 0.3634 |
| 5 | num__installment_commitment | 0.3112 |
| 6 | num__loan_burden | 0.3065 |
| 7 | cat__checking_status_<0 | 0.2695 |
| 8 | cat__purpose_new car | 0.2337 |
| 9 | cat__savings_status_<100 | 0.2260 |
| 10 | cat__personal_status_male single | 0.2242 |

## Cost-Sensitive Threshold

- **Optimal threshold**: 0.26 (tuned for 10:1 FN/FP cost ratio)
- **Minimum cost**: $537
- **Rationale**: Missing a defaulter (FN) costs ~10x more than denying a good borrower (FP)
- **Cost formula**: Cost = FN × 10 + FP × 1

## Limitations

- **Small dataset**: Only 1,000 samples limits generalization
- **Historical bias**: Dataset from 1994 German banking, may reflect outdated patterns
- **No protected attributes audit**: Fairness analysis not included in base pipeline
- **Feature simplicity**: Real credit scoring uses 100+ features (bureau data, payment history)

## Ethical Considerations

- Credit scoring directly affects people's access to financial services
- Models trained on historical data may perpetuate existing biases
- ECOA (Equal Credit Opportunity Act) requires adverse action reasons for denials
- FCRA (Fair Credit Reporting Act) requires model interpretability
- Logistic Regression chosen as primary model specifically for regulatory interpretability

## Monitoring

- PSI (Population Stability Index) for distribution drift detection
- Data quality checks for incoming predictions
- Thresholds: PSI < 0.10 (stable), 0.10-0.25 (investigate), > 0.25 (retrain)