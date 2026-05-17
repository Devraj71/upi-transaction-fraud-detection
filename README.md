# 🔒 UPI Transaction Fraud Detection & Risk Analysis

Welcome to the **UPI Transaction Fraud Detection & Risk Analysis** repository! This project presents an end-to-end data science and machine learning pipeline trained to analyze, visualize, and detect fraudulent Unified Payments Interface (UPI) transactions across a 250,000-record dataset.

---

## 🚀 Key Features

* **Comprehensive Data Cleaning & Preprocessing:** Handles missing values, standardizes date-time parameters, and scales numerical values.
* **Deep Exploratory Data Analysis (EDA):** 
  * **Temporal Analysis:** Analyzes transaction patterns by hour of day and day of week.
  * **Geographic Demographics:** Visualizes transaction densities across Indian states.
  * **Bank Performance:** Compares transaction success and failure rates across major banks (SBI, HDFC, ICICI, etc.).
  * **Device & Network Analysis:** Uncovers patterns across device types (iOS vs. Android) and connection networks (4G, 5G, WiFi).
* **Advanced Machine Learning (Scikit-Learn):**
  * Explodes the dangerous **"Accuracy Paradox"** using a stratified 80/20 train-test split.
  * Implements a high-performance **Balanced Random Forest Classifier** with bootstrap weighting.
  * Trains a **Cost-Sensitive Logistic Regression** with probability threshold tuning.

---

## 📊 Model Performance & Findings

Our models were evaluated on a stratified, untouched 20% test set (50,000 transactions containing 129 fraud cases).

### Comparison Table

| Metric | Random Forest (Balanced) | Logistic Regression (Thresh = 0.50) | Logistic Regression (Thresh = 0.30) | Random Guess Baseline |
| :--- | :--- | :--- | :--- | :--- |
| **Raw Accuracy** | **99.9160%** | **96.7100%** | **95.8120%** | 99.7428% (Predict all 0) |
| **ROC-AUC Score** | **1.0000** | **0.9915** | **0.9915** | **0.5000** (Coin Toss) |
| **PR-AUC Score** | **0.9879** | **0.1670** | **0.1670** | **0.0026** (Class Ratio) |
| **Recall (Fraud Caught)** | **98.45%** (127 / 129) | **97.67%** (126 / 129) | **100.00%** (129 / 129) | 0.00% |
| **Precision** | **76.05%** | **7.13%** | **5.80%** | 0.00% |

### 🔍 Crucial Data Science Takeaways

1. **The "Accuracy Paradox" Exploded:** High raw accuracy (99.81%) is trivial to get by predicting all transactions as legitimate, but it results in a completely useless model (0% recall). We used custom class weights and stratified sampling to design useful models that actually catch fraud.
2. **Superiority of Non-Linear Ensemble Trees:** The **Random Forest Classifier** achieved a perfect **1.0000 ROC-AUC** and **76.05% Precision** (only 40 false alarms out of ~50,000 transactions) because it naturally models conditional decision rules (`IF amount >= X AND hour <= Y...`). 
3. **Threshold Tuning Sweep:** We sweep the decision threshold of Logistic Regression from `0.30` to `0.60` to show how risk managers can lower classification boundaries to catch **100% of fraud** at the cost of higher false alarms on legitimate users.

---

## 🛠️ Tech Stack & Libraries

* **Core Language:** Python 3.13 (Anaconda Environment)
* **Jupyter Notebooks:** For interactive analysis, cleaning, and visualization
* **Data Manipulation:** `pandas`, `numpy`
* **Plotting & Visualization:** `matplotlib`, `seaborn`
* **Machine Learning:** `scikit-learn` (StandardScaler, train_test_split, RandomForestClassifier, LogisticRegression)

---

## 📂 Project Structure

```
├── .gitignore                     # Configured to ignore large datasets & caches
├── README.md                      # Professional project home page
├── UPI_Analsys.ipynb              # Main interactive Jupyter Notebook (94 cells)
└── DATA SET/                      # (Ignored) Contains upi_transactions_2024.csv
```

---

## 💻 How to Run Locally

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Devraj71/upi-transaction-fraud-detection.git
   cd upi-transaction-fraud-detection
   ```
2. **Download the Dataset:**
   Place your `upi_transactions_2024.csv` inside the `DATA SET/` folder.
3. **Run the Notebook:**
   Launch Jupyter Notebook or VS Code, open `UPI_Analsys.ipynb`, and select **Run All Cells** to see the full EDA charts and train the machine learning models instantly!
