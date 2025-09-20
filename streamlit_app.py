import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load dataset
df = pd.read_csv("Telco-Customer-Churn.csv")
df.drop(['customerID'], axis=1, inplace=True)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})

st.set_page_config(page_title="Telco Churn Dashboard", layout="wide")
st.title("📊 Customer Churn Dashboard")
st.markdown("Explore churn behavior, KPIs, and predictions interactively.")

# Sidebar Multiselect helper
def sidebar_multiselect(column_name, key):
    options = sorted(df[column_name].unique().tolist())
    return st.sidebar.multiselect(
        label=column_name,
        options=options,
        default=st.session_state.get(key, []),
        key=key
    )

# Sidebar Filters
st.sidebar.header("🔎 Filter Options")
gender = sidebar_multiselect("Gender", "gender_key")
senior = sidebar_multiselect("SeniorCitizen", "senior_key")
partner = sidebar_multiselect("Partner", "partner_key")
dependents = sidebar_multiselect("Dependents", "dependents_key")
internet = sidebar_multiselect("InternetService", "internet_key")
contract = sidebar_multiselect("Contract", "contract_key")
payment = sidebar_multiselect("PaymentMethod", "payment_key")

def reset_filters_and_rerun():
    for key in ["gender_key", "senior_key", "partner_key", "dependents_key", "internet_key", "contract_key", "payment_key"]:
        st.session_state.pop(key, None)
    st.rerun()

if st.sidebar.button("🔄 Reset Filters"):
    reset_filters_and_rerun()

# Apply Filters
filtered_df = df.copy()
if gender:
    filtered_df = filtered_df[filtered_df["Gender"].isin(gender)]
if senior:
    filtered_df = filtered_df[filtered_df["SeniorCitizen"].isin(senior)]
if partner:
    filtered_df = filtered_df[filtered_df["Partner"].isin(partner)]
if dependents:
    filtered_df = filtered_df[filtered_df["Dependents"].isin(dependents)]
if internet:
    filtered_df = filtered_df[filtered_df["InternetService"].isin(internet)]
if contract:
    filtered_df = filtered_df[filtered_df["Contract"].isin(contract)]
if payment:
    filtered_df = filtered_df[filtered_df["PaymentMethod"].isin(payment)]

# KPI Cards
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Customers", len(filtered_df))
with col2:
    churn_rate = filtered_df['Churn'].mean() * 100 if len(filtered_df) > 0 else 0
    st.metric("Churn Rate (%)", f"{churn_rate:.2f}")
with col3:
    avg_monthly = filtered_df['MonthlyCharges'].mean() if len(filtered_df) > 0 else 0
    st.metric("Avg Monthly Charges", f"${avg_monthly:.2f}")

st.divider()

# Filtered data display
st.subheader("📄 Filtered Data")
st.dataframe(filtered_df, use_container_width=True)

# Download CSV
csv = filtered_df.to_csv(index=False).encode('utf-8')
st.download_button("⬇️ Download CSV", csv, "filtered_customer_churn.csv", "text/csv")

# Visualizations
st.divider()
st.subheader("📈 Visual Insights")
col1, col2 = st.columns(2)
with col1:
    fig1, ax1 = plt.subplots()
    sns.countplot(data=filtered_df, x='Churn', palette='Set2', ax=ax1)
    ax1.set_title("Churn Distribution")
    ax1.set_xticklabels(["No", "Yes"])
    st.pyplot(fig1)
with col2:
    fig2, ax2 = plt.subplots()
    counts = filtered_df['Churn'].value_counts()
    ax2.pie(counts, labels=["No", "Yes"], autopct='%1.1f%%', colors=['#66b3ff', '#ff9999'], startangle=90)
    ax2.set_title("Churn Distribution (Pie)")
    ax2.axis('equal')
    st.pyplot(fig2)

col3, col4 = st.columns(2)
with col3:
    fig3, ax3 = plt.subplots()
    sns.boxplot(data=filtered_df, x='Churn', y='TotalCharges', palette='Set1', ax=ax3)
    ax3.set_title("Total Charges by Churn")
    ax3.set_xticklabels(["No", "Yes"])
    st.pyplot(fig3)
with col4:
    fig4, ax4 = plt.subplots()
    sns.countplot(data=filtered_df, x='Contract', hue='Churn', palette='coolwarm', ax=ax4)
    ax4.set_title("Contract Type vs Churn")
    st.pyplot(fig4)

# Load feature list and model
try:
    model = joblib.load("best_rf_model.pkl")
    features = joblib.load("model_features.pkl")
except Exception as e:
    st.error(f"Model or features loading failed: {e}")
    st.stop()

st.divider()
st.header("🔮 Predict Customer Churn")

with st.form("prediction_form"):
    st.subheader("Customer Details")

    customer_input = {}

    for col in features:
        if col == 'Gender':
            customer_input[col] = st.selectbox("Gender", ["Male", "Female"])
        elif col == 'SeniorCitizen':
            customer_input[col] = st.selectbox("Senior Citizen", ["No", "Yes"])
        elif col == 'Partner':
            customer_input[col] = st.selectbox("Has Partner", ["No", "Yes"])
        elif col == 'Dependents':
            customer_input[col] = st.selectbox("Has Dependents", ["No", "Yes"])
        elif col == 'tenure':
            customer_input[col] = st.slider("Tenure (months)", 0, 72, 12)
        elif col == 'InternetService':
            customer_input[col] = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        elif col == 'Contract':
            customer_input[col] = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        elif col == 'PaperlessBilling':
            customer_input[col] = st.selectbox("Paperless Billing", ["Yes", "No"])
        elif col == 'PaymentMethod':
            customer_input[col] = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        elif col == 'MonthlyCharges':
            customer_input[col] = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
        elif col == 'TotalCharges':
            customer_input[col] = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)
        else:
            customer_input[col] = st.text_input(f"Input for {col}")

    submitted = st.form_submit_button("Predict Churn")

if submitted:
    input_df = pd.DataFrame([customer_input])
    input_df.columns = [col.strip() for col in input_df.columns]
    input_df = input_df.reindex(columns=features)

    # Convert numeric columns to numeric dtype
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    for col in numeric_cols:
        if col in input_df.columns:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

    # Convert categorical columns to string type
    categorical_cols = [col for col in features if col not in numeric_cols]
    for col in categorical_cols:
        if col in input_df.columns:
            input_df[col] = input_df[col].astype(str)

    try:
        prediction = model.predict(input_df)
        proba_array = model.predict_proba(input_df)
        if proba_array.shape[1] == 2:
            proba = proba_array[0][1]
        else:
            proba = proba_array[0]
    
        st.success("✅ Customer is likely to stay!" if prediction[0] == 0 else "⚠️ Customer is likely to churn!")
        st.info(f"Churn Probability: {proba * 100:.2f}%")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
