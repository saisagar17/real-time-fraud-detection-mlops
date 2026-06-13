# PaySim Dataset Exploration

## Project

**Real-Time Fraud Detection, Drift Observability & Model Performance Monitoring**

---

# Objective

The goal of Phase 02 is to understand the PaySim fraud dataset before building any machine learning models.

In production machine learning systems, understanding the dataset is often more important than selecting a model. Poor understanding of data leads to poor features, incorrect assumptions, unreliable models, and misleading evaluation results.

Therefore, before training any fraud detection model, we perform an extensive exploratory data analysis (EDA) to understand:

* Dataset structure
* Transaction types
* Fraud distribution
* Class imbalance
* Financial behavior
* Rule-based fraud detection
* Potential predictive features

---

# Dataset

Dataset Used:

**PaySim — Mobile Money Transaction Fraud Dataset**

Dataset Characteristics:

* Total Transactions: 6,362,620
* Total Features: 11
* Fraud Transactions: 8,213
* Legitimate Transactions: 6,354,407

Fraud Rate:

0.129082%

This means approximately:

```text
1 Fraud Transaction
per 775 Transactions
```

The dataset is highly imbalanced and closely resembles real-world fraud detection scenarios.

---

# Why We Performed This Analysis

Before building a fraud model we must answer:

```text
What does fraud look like?

Where does fraud occur?

Which features contain useful information?

How difficult is the detection problem?
```

Without answering these questions, model development becomes guesswork.

---

# Step 1 — Dataset Structure Analysis

We first inspected:

* Dataset Shape
* Data Types
* Memory Usage
* Available Features

Results:

```text
Rows      : 6,362,620
Columns   : 11
Memory    : ~534 MB
```

Feature Types:

Numeric Features:

* step
* amount
* oldbalanceOrg
* newbalanceOrig
* oldbalanceDest
* newbalanceDest
* isFraud
* isFlaggedFraud

Categorical Features:

* type
* nameOrig
* nameDest

Target Variable:

```text
isFraud
```

where:

```text
0 = Legitimate Transaction

1 = Fraudulent Transaction
```

---

# Step 2 — Transaction Type Analysis

The dataset contains five transaction categories.

| Transaction Type |     Count | Percentage |
| ---------------- | --------: | ---------: |
| CASH_OUT         | 2,237,500 |     35.17% |
| PAYMENT          | 2,151,495 |     33.81% |
| CASH_IN          | 1,399,284 |     21.99% |
| TRANSFER         |   532,909 |      8.38% |
| DEBIT            |    41,432 |      0.65% |

Observations:

* Most transactions are CASH_OUT and PAYMENT.
* DEBIT transactions are extremely rare.
* TRANSFER transactions represent only 8.38% of activity.

At this stage we suspected fraud would not be equally distributed across transaction types.

---

# Step 3 — Fraud Distribution Analysis

Fraud Distribution:

| Class      |     Count |
| ---------- | --------: |
| Legitimate | 6,354,407 |
| Fraud      |     8,213 |

Fraud Rate:

```text
0.129082%
```

Observations:

* Fraud is extremely rare.
* The dataset is highly imbalanced.
* Accuracy alone will be misleading.

Example:

A model predicting:

```text
Everything Legitimate
```

would achieve approximately:

```text
99.87% Accuracy
```

while detecting:

```text
0 Fraud Transactions
```

This demonstrates why fraud systems focus on:

* Precision
* Recall
* PR-AUC

instead of Accuracy.

---

# Step 4 — Fraud by Transaction Type

We investigated where fraud actually occurs.

Results:

| Type     | Legitimate | Fraud |
| -------- | ---------: | ----: |
| CASH_IN  |  1,399,284 |     0 |
| CASH_OUT |  2,233,384 | 4,116 |
| DEBIT    |     41,432 |     0 |
| PAYMENT  |  2,151,495 |     0 |
| TRANSFER |    528,812 | 4,097 |

Major Discovery:

Fraud occurs only in:

```text
TRANSFER

CASH_OUT
```

No fraud exists in:

```text
PAYMENT

DEBIT

CASH_IN
```

This immediately suggests transaction type is a highly informative feature.

---

# Step 5 — Fraud Risk by Transaction Type

We calculated fraud probability for each transaction type.

Results:

| Type     | Fraud Rate |
| -------- | ---------: |
| TRANSFER |    0.7688% |
| CASH_OUT |    0.1840% |
| PAYMENT  |    0.0000% |
| CASH_IN  |    0.0000% |
| DEBIT    |    0.0000% |

Observations:

TRANSFER transactions are approximately:

```text
4.2x
```

more likely to be fraudulent than CASH_OUT transactions.

This makes TRANSFER the highest-risk transaction type in the dataset.

---

# Step 6 — Fraud Composition

We analyzed the composition of fraud transactions.

Results:

| Fraud Type | Percentage |
| ---------- | ---------: |
| CASH_OUT   |     50.12% |
| TRANSFER   |     49.88% |

Observation:

Fraud is almost perfectly split between:

```text
TRANSFER

CASH_OUT
```

This suggests a typical fraud workflow:

```text
Stolen Funds
      ↓
TRANSFER
      ↓
Destination Account
      ↓
CASH_OUT
      ↓
Money Extraction
```

---

# Step 7 — Existing Rule Engine Analysis

The dataset contains an existing rule-based detector:

```text
isFlaggedFraud
```

Results:

```text
Flagged Transactions = 16
```

Out of:

```text
6,362,620 Transactions
```

Only:

```text
0.000251%
```

were flagged.

Cross-tabulation:

| Rule Flag   | Legitimate | Fraud |
| ----------- | ---------: | ----: |
| Not Flagged |  6,354,407 | 8,197 |
| Flagged     |          0 |    16 |

Observations:

Rule Precision:

```text
100%
```

Rule Recall:

```text
~0.19%
```

Interpretation:

The rule system catches only extremely obvious fraud cases.

It makes no mistakes but misses nearly all fraud.

This demonstrates why production systems combine:

```text
Rules
      +
Machine Learning
```

instead of relying on rules alone.

---

# Step 8 — Amount Analysis

Fraud Amount Statistics:

| Metric  |      Fraud |
| ------- | ---------: |
| Mean    |  1,467,967 |
| Median  |    441,423 |
| Maximum | 10,000,000 |

Legitimate Amount Statistics:

| Metric  | Legitimate |
| ------- | ---------: |
| Mean    |    178,197 |
| Median  |     74,685 |
| Maximum | 92,445,520 |

Observations:

Average fraud transactions are approximately:

```text
8x Larger
```

than legitimate transactions.

Median fraud transactions are approximately:

```text
6x Larger
```

than legitimate transactions.

This indicates that transaction amount contains useful predictive information.

---

# Step 9 — Log Transformation

The raw transaction amount distribution was highly skewed.

Characteristics:

* Large outliers
* Compressed boxplots
* Long-tailed financial distribution

To improve visualization and future modeling, we created:

```python
log_amount = np.log1p(amount)
```

Benefits:

* Reduces skewness
* Compresses extreme values
* Improves model stability
* Makes class differences easier to visualize

The transformed distribution showed clearer separation between fraud and legitimate transactions.

---

# Step 10 — Balance Behavior Investigation

We investigated sender balance consistency.

Formula:

```python
oldbalanceOrg
- amount
- newbalanceOrig
```

We expected most transactions to be near zero.

Instead we discovered:

```text
2,088,985 Transactions
```

where:

```text
oldbalanceOrg = 0

newbalanceOrig = 0
```

representing:

```text
32.8% of the dataset
```

We hypothesized this pattern might indicate fraud.

After testing:

| Pattern | Fraud Count |
| ------- | ----------: |
| False   |       8,172 |
| True    |          41 |

Result:

The pattern is actually associated with lower fraud risk.

This was an important lesson:

```text
Never assume

Always validate using data
```

---

# Step 11 — Correlation Analysis

Correlation with Fraud:

| Feature        | Correlation |
| -------------- | ----------: |
| amount         |      0.0767 |
| oldbalanceOrg  |      0.0102 |
| newbalanceDest |      0.0005 |
| oldbalanceDest |     -0.0059 |
| newbalanceOrig |     -0.0081 |

Observations:

* Amount has the strongest linear relationship with fraud.
* Correlations are small because fraud is extremely rare.
* Correlation should not be confused with feature importance.

---

# Key Findings

The most important discoveries from Phase 02 are:

1. Fraud Rate = 0.129%
2. Dataset is highly imbalanced.
3. Fraud occurs only in TRANSFER and CASH_OUT.
4. TRANSFER is the highest-risk transaction type.
5. Fraud transactions involve significantly larger amounts.
6. Rule-based detection has 100% precision but extremely poor recall.
7. Transaction amount is a useful predictive feature.
8. Log transformation improves financial feature representation.
9. Balance anomalies must be validated rather than assumed.
10. The dataset contains enough signal to support machine learning models.

---

# Outcome

At the end of Phase 02 we understand:

```text
What the data contains

Where fraud occurs

How fraud behaves

Which features are useful

Why rules alone are insufficient
```

This provides the foundation for Phase 03.

---

# Next Phase

## Baseline Fraud Model Development

Goals:

* Data Preparation
* Train/Test Split
* Handling Class Imbalance
* Logistic Regression Baseline
* Random Forest Baseline
* LightGBM Baseline
* Precision
* Recall
* F1 Score
* PR-AUC
* Feature Importance Analysis

The objective of Phase 03 is to establish the first machine learning benchmark for the fraud detection system.
