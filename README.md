# iPhone Purchase Prediction: Consumer Behavior Analytics 📱🛒

## 📌 Project Overview
This project builds a classification model using K-Nearest Neighbors (KNN) to predict whether a consumer will purchase a premium digital device (iPhone) based on their demographic and financial profile. Designed for retail and digital marketing analytics, this model acts as a profiling engine to help businesses optimize their ad spend by accurately identifying high-probability buyers.

## 📊 Dataset Description
The dataset contains demographic and financial records for prospective customers, alongside their historical purchase decisions.
* **Dataset Size:** ~400 records, 4 variables.
* **Target Variable:** `Purchase Iphone` (Binary: 0 = Not Purchased, 1 = Purchased)
* **Key Features:** `Gender`, `Age`, `Salary`.

## 🛠️ Tech Stack
* **Language:** Python
* **Data Manipulation:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn` (K-Nearest Neighbors Classifier, StandardScaler)
* **Data Visualization:** `matplotlib`, `seaborn`

## 🧠 Methodology & Model Development
1. **Exploratory Data Analysis (EDA):** Leveraged boxplots and correlation heatmaps to analyze the drivers of purchasing behavior. Discovered that `Age` and `Salary` strictly dictate premium purchasing power, while `Gender` demonstrated minimal to zero statistical variance regarding purchase likelihood.
2. **Data Preprocessing:** Handled categorical data (Gender) and applied **StandardScaler** to the numerical features. Scaling was a critical mathematical step to ensure the larger magnitude of `Salary` did not overpower the `Age` feature during KNN distance calculations.
3. **Train/Test Split:** An **80/20 train-test split was applied** to ensure unbiased model evaluation.
4. **Hyperparameter Tuning:** Iterated through various 'K' (neighbor) values to balance bias and variance, identifying **K=11** as the optimal architecture for maximum accuracy.

## 📈 Key Results & Performance
The KNN Classifier (K=11) successfully modeled consumer purchasing behavior with high reliability.

* **Overall Accuracy:** `89.47%`
* **Confusion Matrix & Metrics (For Purchasers - Class 1):**
  * **Recall: 1.00 (100%)** — The model successfully identified *every single customer* who bought an iPhone (Zero False Negatives).
  * **Precision: 0.78** — Out of all predicted purchasers, 78% actually bought the device (8 False Positives).
  * **F1-Score:** `0.88`

## 💼 Business Impact & Live Inference
* **Revenue Protection (100% Recall):** In a marketing context, a recall of 1.00 is highly favorable. It means the business captures 100% of the revenue opportunity, never missing a potential buyer, while absorbing only a minor efficiency loss in ad spend (the 0.78 precision).
* **The Profiling Engine:** Marketing teams can feed prospective customer profiles into the pipeline to receive a binary purchase prediction and a confidence score.
  * *Scenario:* Customer (Age: 54, Salary: $200,000, Gender: Female)
  * *Model Output:* **LIKELY TO PURCHASE (90.91% Confidence)**
* **Strategic Next Steps:** Deploy this logic to filter programmatic advertising audiences. Future iterations will test ensemble methods (like Random Forest) to improve the 0.78 precision and further reduce wasted ad spend without sacrificing the perfect recall.

