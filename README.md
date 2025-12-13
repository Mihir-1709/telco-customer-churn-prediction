# 📊 Telco Customer Churn Prediction

This project provides an **interactive Streamlit dashboard** and a **machine learning pipeline** to analyze and predict customer churn for a telecom company.  
Users can explore churn behavior, visualize key insights, and predict whether a customer is likely to churn using a **trained Random Forest model**.

---

## 🚀 Features

- **Data Exploration**
  - Interactive sidebar filters (gender, contract, payment method, etc.)
- **Key Performance Indicators (KPIs)**
  - Total customers
  - Churn rate
  - Average monthly charges
- **Visual Insights**
  - Churn distribution (bar & pie charts)
  - Contract type vs churn
  - Charges vs churn (boxplot)
- **Churn Prediction**
  - Form-based customer input
  - Churn probability and prediction using a trained ML model

---

## 🗂 Project Structure

```

telco-customer-churn-prediction/
│
├── train_model.py           # Model training & hyperparameter tuning
├── streamlit_app.py         # Streamlit dashboard & prediction app
├── Telco-Customer-Churn.csv # Dataset
├── best_rf_model.pkl        # Trained Random Forest pipeline
├── model_features.pkl       # Feature list used during training
├── requirements.txt         # Project dependencies
├── README.md

````

---

## ⚙️ Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Mihir-1709/telco-customer-churn-prediction
cd telco-customer-churn-prediction
````

---

### 2️⃣ Create & activate a virtual environment (recommended)

```bash
python -m venv churn_env
```

**Windows**

```bat
churn_env\Scripts\activate
```

**macOS / Linux**

```bash
source churn_env/bin/activate
```

---

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🧠 Model Training

Train the model **locally** (recommended before deployment):

```bash
python train_model.py
```

This will generate:

* `best_rf_model.pkl`
* `model_features.pkl`

---

## 🖥 Run the Streamlit App

Launch the dashboard locally:

```bash
streamlit run streamlit_app.py
```

### You can:

* Explore customer churn trends
* Apply filters dynamically
* Predict churn probability for new customers

---

## 📸 Screenshots

### 1️⃣ Dashboard Overview & Filters
Shows the main dashboard with:
- Sidebar filter options (Gender, Senior Citizen, Partner, Dependents, etc.)
- KPI cards for Total Customers, Churn Rate, and Average Monthly Charges

<img width="1919" height="910" alt="Dashboard Overview" src="https://github.com/user-attachments/assets/f04718aa-e1bc-4b9b-8f00-bcf5ee3cf05e" />


---

### 2️⃣ Filtered Data View
Demonstrates dynamic filtering of customer records based on selected criteria, with real-time KPI updates.

<img width="1919" height="909" alt="Filtered Data" src="https://github.com/user-attachments/assets/e8146461-7a44-4b6c-b963-ab3c76fd3e5f" />


---

### 3️⃣ Churn Analysis Visualizations
Visual insights including:
- Churn distribution (bar chart)
- Churn distribution (pie chart)
- Total charges by churn (box plot)
- Contract type vs churn

<img width="1919" height="910" alt="Churn Visualizations" src="https://github.com/user-attachments/assets/6260f9df-9cef-4a63-afc1-89c785dac9ae" />


---

### 4️⃣ Churn Prediction & Result
Interactive prediction form with customer inputs and final output showing:
- Churn prediction (Stay / Churn)
- Churn probability percentage

<img width="1919" height="871" alt="Churn Prediction" src="https://github.com/user-attachments/assets/6b2be5f0-ca71-4783-bd8e-e419ac5efb82" />

---

## 🌐 Live Demo

🔗 **Live Website:**
👉 https://telco-customer-churn-prediction-mihir.streamlit.app/

---

## 📊 Dataset

* **Source:** Kaggle – Telco Customer Churn Dataset
* Contains customer demographics, account information, service usage, and churn labels.

🔗 [https://www.kaggle.com/datasets/blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

## 🤖 Model Details

* **Algorithm:** Random Forest Classifier
* **Preprocessing:**

  * One-Hot Encoding for categorical variables
  * Median imputation for numeric features
* **Training:**

  * RandomizedSearchCV for hyperparameter tuning
  * Stratified data split
* **Output:**

  * Churn prediction (Yes / No)
  * Churn probability

---

## 🔧 Notes & Improvements

* Model training and inference feature sets are consistent
* XGBoost removed for simplicity and deployment stability
* Designed for **Streamlit Cloud compatibility**
* Future improvements:

  * Class imbalance handling
  * Feature importance visualization
  * SHAP / explainability
  * Additional ML models

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 👤 Author

**Mihir Modh**
📧 Email: [mihirmodh14@gmail.com](mailto:mihirmodh14@gmail.com)
🔗 LinkedIn / GitHub: *(add if you want)*

---

## 🙏 Acknowledgments

* Kaggle Telco Customer Churn Dataset
* Built with **Python, scikit-learn, Streamlit, Pandas, Matplotlib, Seaborn**
```
