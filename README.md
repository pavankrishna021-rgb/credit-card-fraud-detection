# Real-Time Credit Card Fraud Detection

Deep learning fraud detection system trained on 284,807 real European credit card transactions with extreme class imbalance (0.17% fraud).

## Live Demo
🔴 [Try it here](http://18.132.228.208:8502)

## The Problem
- Only 0.17% of transactions are fraudulent (492 out of 284,807)
- A naive model predicting "not fraud" every time achieves 99.83% accuracy but catches zero fraud
- Accuracy is meaningless — AUC-ROC is the correct metric for imbalanced classification

## Approach
1. *EDA* — Visualized extreme class imbalance
2. *Scaling* — StandardScaler on Time and Amount (V1-V28 already PCA-transformed)
3. *SMOTE* — Synthetic Minority Oversampling on training data only
4. *Deep Learning* — TensorFlow neural network with dropout regularization
5. *Model Comparison* — Evaluated Logistic Regression, Random Forest, and Deep Learning
6. *Deployment* — Optimized NumPy inference engine on AWS EC2

## Architecture
Input (30 features)
→ Dense(64, ReLU)
→ Dropout(0.3)
→ Dense(32, ReLU)
→ Dropout(0.2)
→ Dense(16, ReLU)
→ Dense(1, Sigmoid)

## Results

| Model | Recall | AUC-ROC | Flagged as Fraud | Actual Frauds |
|-------|--------|---------|-----------------|---------------|
| Logistic Regression | 0.918 | 0.970 | 1,548 | 98 |
| Random Forest | 0.816 | 0.962 | 94 | 98 |
| *Deep Learning* | *0.816* | *0.974* | *106* | *98* |

## Key Insight
Logistic Regression had the highest recall but flagged 1,548 transactions — over 1,400 false alarms. Deep Learning flagged 106 transactions (closest to the 98 actual frauds), providing the best precision-recall balance for production use.

## Tech Stack
- *Training:* TensorFlow, Scikit-learn, SMOTE (imbalanced-learn)
- *Inference:* NumPy forward propagation (500MB → 5MB deployment)
- *Dashboard:* Streamlit
- *Deployment:* AWS EC2
- *Data:* [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)

## Deployment Note
TensorFlow was too large for the EC2 instance. I extracted model weights as JSON and rebuilt the forward propagation using only NumPy — identical predictions at 1% of the deployment size. This demonstrates understanding of what deep learning frameworks do under the hood.

## Run Locally
```bash
pip install streamlit numpy
streamlit run app.py

Author
Pavan Krishna - ML Engineer
www.linkedin.com/in/pavankrishnapendlikal


