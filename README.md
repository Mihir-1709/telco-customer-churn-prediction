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

### 1️⃣ Dashboard Overview & KPIs
Displays the main dashboard with key metrics including:
- Total Customers
- Churn Rate (%)
- Average Monthly Charges

![Dashboard Overview](screenshots/dashboard_overview.png)

---

### 2️⃣ Interactive Filters
Sidebar filters to dynamically slice customer data by:
- Gender
- Senior Citizen
- Partner
- Dependents
- Internet Service
- Contract
- Payment Method

![Filter Options](screenshots/filters.png)

---

### 3️⃣ Filtered Customer Data
Shows a live, filter-driven table of customer records based on selected criteria.

![Filtered Data Table](screenshots/filtered_data.png)

---

### 4️⃣ Churn Visualizations
Visual insights including:
- Churn distribution (bar chart)
- Churn distribution (pie chart)
- Total charges by churn (box plot)
- Contract type vs churn (bar chart)

![Churn Visualizations](screenshots/visualizations.png)

---

### 5️⃣ Churn Prediction Form
User-friendly form to input customer details such as:
- Demographics
- Tenure
- Services
- Billing and payment details

![Prediction Form](screenshots/prediction_form.png)

---

### 6️⃣ Prediction Result
Displays:
- Churn prediction (Stay / Churn)
- Churn probability percentage

![Prediction Result](screenshots/prediction_result.png)

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
